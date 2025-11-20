from typing import List, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.keyword_model = KeywordExtractor(cfg.keyword)
        self.wiki_fetcher = wiki_fetcher_cls(cfg.wiki)
        self._llm_tokenizer = AutoTokenizer.from_pretrained(self._llm_model_name)
        self._llm_model = AutoModelForCausalLM.from_pretrained(
            self._llm_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
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
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0, input_ids.shape[1]:]

        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    def _sentences_for_spans(self, text: str, spans: List[Span]) -> List[str]:
        """
        Map each span back to the sentence that contains it, preserving ordering.
        """
        doc = self.keyword_model.parse(text)
        sentences = list(doc.sents)
        sent_bounds = [(sent.start_char, sent.end_char, sent.text.strip()) for sent in sentences]
        default_text = text.strip()

        def sentence_for_span(span: Span) -> str:
            midpoint = (span.char_start + span.char_end) // 2
            for start, end, sent_text in sent_bounds:
                if start <= midpoint < end:
                    return sent_text
            return default_text

        return [sentence_for_span(span) for span in spans]

    def run(self, text: str) -> FactCheckResult:
        # 1) Entity-level keyword spans (named entities / key noun phrases)
        spans = self.keyword_model.select_spans(text)

        # 2) Sentence text for each span (keeps local anchor) + full-text context
        sentences_for_span = self._sentences_for_spans(text, spans)
        contexts_for_span: List[str] = [text] * len(spans)

        # 3) Evidence retrieval per sentence-level entity set (cache duplicate queries)
        evidence_cache: dict[str, List[EvidenceChunk]] = {}
        evidence_per_span: List[List[EvidenceChunk]] = []

        for span, sent_text in zip(spans, sentences_for_span):
            # Parse the local sentence and collect all named entities in it.
            doc_sent = self.keyword_model.parse(sent_text)
            entity_texts: list[str] = []

            for ent in doc_sent.ents:
                ent_text = ent.text.strip()
                if ent_text:
                    entity_texts.append(ent_text)

            # Fallback: if no entities were found, use the span text itself.
            if not entity_texts:
                base = span.text.strip()
                if base:
                    entity_texts.append(base)

            combined_chunks: list[EvidenceChunk] = []

            for ent_text in entity_texts:
                key = ent_text.lower()
                if key not in evidence_cache:
                    # Wikipedia search only depends on the text, not on offsets.
                    pseudo_span = Span(
                        span_id=-1,
                        text=ent_text,
                        char_start=0,
                        char_end=0,
                    )
                    evidence_cache[key] = self.wiki_fetcher.retrieve(pseudo_span)

                combined_chunks.extend(evidence_cache.get(key, []))

            evidence_per_span.append(combined_chunks)

        # 4) Run the factuality checker on each span
        assessments = self.fact_checker.assess(
            text=text,
            spans=spans,
            evidence=evidence_per_span,
            sentences=sentences_for_span,
            contexts=contexts_for_span,
        )

        # 5) Summarize an overall label and explanation
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
            c
            for c in claims
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
