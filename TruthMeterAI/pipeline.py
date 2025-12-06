from typing import List, Type

import torch
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    from config import PipelineConfig
    from fact_checker import FactChecker
    from keyword_extractor import KeywordExtractor
    from schemas import ClaimAssessment, EvidenceChunk, FactCheckResult, Span
    from wiki_fetcher import AbstractWikiFetcher, WikiFetcher
else:
    from .config import PipelineConfig
    from .fact_checker import FactChecker
    from .keyword_extractor import KeywordExtractor
    from .schemas import ClaimAssessment, EvidenceChunk, FactCheckResult, Span
    from .wiki_fetcher import AbstractWikiFetcher, WikiFetcher


class Pipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        wiki_fetcher_cls: Type[AbstractWikiFetcher] = WikiFetcher,
    ):
        self.cfg = cfg
        self._llm_model_name = cfg.llm_model_name
        nlp = spacy.load(cfg.keyword.spacy_model_name)
        self.keyword_model = KeywordExtractor(cfg.keyword, nlp)
        self.wiki_fetcher = wiki_fetcher_cls(cfg.wiki, nlp)
        self._llm_tokenizer = AutoTokenizer.from_pretrained(self._llm_model_name)
        self._llm_model = AutoModelForCausalLM.from_pretrained(
            self._llm_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        gen_cfg = self._llm_model.generation_config
        gen_cfg.do_sample = False
        gen_cfg.temperature = 1.0
        gen_cfg.top_p = 1.0
        gen_cfg.top_k = 50
        self.fact_checker = FactChecker(cfg.checker, self._llm_call)

    def _llm_call(self, prompt: str) -> str:
        tokenizer = self._llm_tokenizer
        model = self._llm_model

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a factuality judge. "
                        "Follow the user's instructions and output exactly the "
                        "required LABEL, EVIDENCE, CONFIDENCE, and EXPLANATION "
                        "fields in the specified format."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = prompt

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0, input_ids.shape[1]:]

        return tokenizer.decode(gen_ids, skip_special_tokens=True)



    def run(self, text: str) -> FactCheckResult:
        spans = self.keyword_model.select_spans(text)

        evidence_cache: dict[str, List[EvidenceChunk]] = {}
        evidence_per_span: List[List[EvidenceChunk]] = []

        for span in spans:
            doc_sent = self.keyword_model.parse(span.text)
            entity_texts: List[str] = []

            for ent in doc_sent.ents:
                ent_text = ent.text.strip()
                if ent_text:
                    entity_texts.append(ent_text)

            if not entity_texts:
                base = span.text.strip()
                if base:
                    entity_texts.append(base)

            combined_chunks: List[EvidenceChunk] = []

            for ent_text in entity_texts:
                key = ent_text.lower()
                if key not in evidence_cache:
                    pseudo_span = Span(
                        span_id=-1,
                        text=ent_text,
                        char_start=0,
                        char_end=0,
                    )
                    evidence_cache[key] = self.wiki_fetcher.retrieve(pseudo_span, text)

                combined_chunks.extend(evidence_cache.get(key, []))

            evidence_per_span.append(combined_chunks)

        assessments = self.fact_checker.assess(
            text=text,
            spans=spans,
            evidence=evidence_per_span,
        )

        overall_label, overall_explanation = self._summarize_overall(assessments)
        llm_overall_explanation = self._combine_claim_explanations(assessments)

        return FactCheckResult(
            text=text,
            claims=assessments,
            overall_label=overall_label,
            overall_explanation=overall_explanation,
            llm_overall_explanation=llm_overall_explanation,
        )

    def _combine_claim_explanations(self, claims: List[ClaimAssessment]) -> str | None:
        unique = []
        seen = set()
        for c in claims:
            key = (c.label, c.explanation.strip())
            if key not in seen and c.explanation:
                seen.add(key)
                unique.append(c.explanation)
        return " ".join(unique) if unique else None


    def _summarize_overall(self, claims: List[ClaimAssessment]) -> tuple[str, str]:
        if not claims:
            return "unknown", "No factual claims were detected in the text."

        def importance(c: ClaimAssessment) -> int:
            return len(c.span_text or "")

        contradicted_all = [
            c for c in claims
            if c.label == "contradicted"
            and c.confidence >= 0.5
        ]
        contradicted = sorted(contradicted_all, key=importance, reverse=True)

        supported = [
            c
            for c in claims
            if c.label == "supported"
            and c.confidence >= 0.6
        ]

        if contradicted:
            main = contradicted[0]
            explanation = (
                "The statement is considered false because at least one important claim "
                "is contradicted by the evidence. For example: "
                f"\"{main.span_text}\"."
            )
            return "false", explanation

        uncertain_like = [
            c for c in claims if c.label in ("uncertain", "out_of_scope")
        ]

        if supported:
            if not uncertain_like:
                explanation = (
                    "The statement is considered true because all extracted claims "
                    "are supported by the available evidence."
                )
                return "true", explanation
            else:
                explanation = (
                    "The statement is partially supported: some claims are supported "
                    "by the evidence, while others could not be verified or are out of scope."
                )
                return "mixed", explanation

        explanation = (
            "The statement could not be clearly verified or refuted with the available evidence."
        )
        return "unknown", explanation
