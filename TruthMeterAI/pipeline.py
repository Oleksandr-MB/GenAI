from typing import List, Optional, Type, Callable
from .config import PipelineConfig
from .schemas import FactCheckResult, Span, EvidenceChunk, ClaimAssessment
from .keyword_extractor import KeywordExtractor
from .wiki_fetcher import AbstractWikiFetcher, WikiFetcher
from .fact_checker import FactChecker
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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

        self._llm_tokenizer = AutoTokenizer.from_pretrained( self._llm_model_name)
        self._llm_model = AutoModelForCausalLM.from_pretrained(
            self._llm_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        llm_call = self._llm_call

        self.fact_checker = FactChecker(cfg.checker, llm_call)

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
            max_length=2048,
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


    def _evidence_mentions_span(self, span_text: str, snippet: str) -> bool:
        return span_text.lower() in snippet.lower()


    def _map_spans_to_subjects(self, text: str, spans: List[Span]):
        doc = self.keyword_model.parse(text)

        sent_bounds = []
        for idx, sent in enumerate(doc.sents):
            sent_bounds.append((idx, sent.start_char, sent.end_char))

        def sent_index_for_char(pos: int) -> int | None:
            for idx, start, end in sent_bounds:
                if start <= pos < end:
                    return idx
            return None

        span_list = list(spans)
        subjects_by_sentence: dict[int, Span] = {}

        pronouns = {"it", "he", "she", "they", "this", "that", "these", "those"}

        for idx, sent in enumerate(doc.sents):
            subj_token = None
            for tok in sent:
                if tok.dep_ in ("nsubj", "nsubjpass"):
                    subj_token = tok
                    break
            if subj_token is None:
                continue

            if subj_token.pos_ == "PRON" and subj_token.text.lower() in pronouns:
                if idx - 1 in subjects_by_sentence:
                    subjects_by_sentence[idx] = subjects_by_sentence[idx - 1]
                    continue

            ent_span = None
            for ent in doc.ents:
                if ent.start <= subj_token.i < ent.end:
                    ent_span = ent
                    break

            if ent_span is not None:
                subj_start = ent_span.start_char
                subj_end = ent_span.end_char
            else:
                left = subj_token.left_edge.i
                right = subj_token.right_edge.i + 1
                span = doc[left:right]
                subj_start = span.start_char
                subj_end = span.end_char

            best_span = None
            best_overlap = 0
            for s in span_list:
                overlap = max(
                    0,
                    min(s.char_end, subj_end) - max(s.char_start, subj_start),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_span = s

            if best_span is not None:
                subjects_by_sentence[idx] = best_span

        subject_for_span: dict[int, Span | None] = {}
        for s in span_list:
            mid = (s.char_start + s.char_end) // 2
            sent_idx = sent_index_for_char(mid)
            if sent_idx is not None and sent_idx in subjects_by_sentence:
                subject_for_span[s.span_id] = subjects_by_sentence[sent_idx]
            else:
                subject_for_span[s.span_id] = None

        return subjects_by_sentence, subject_for_span


    def _summarize_overall(self, claims: List[ClaimAssessment]) -> tuple[str, str]:
        if not claims:
            return "unknown", "No factual claims were detected in the text."

        def importance(c: ClaimAssessment) -> int:
            return len(c.span_text)

        contradicted_all = [
            c for c in claims
            if c.label == "contradicted"
            and c.evidence_used
        ]
        contradicted = sorted(contradicted_all, key=importance, reverse=True)

        supported = [
            c for c in claims
            if c.label == "supported"
            and c.evidence_used
        ]

        if contradicted:
            top = contradicted[:3]
            expl_parts = [
                f"Claim '{c.span_text}' is contradicted by Wikipedia."
                for c in top
            ]
            explanation = ("The statement is considered mostly false because at least one key claim is contradicted by the evidence. " + " ".join(expl_parts))

            return "false", explanation

        if supported:
            uncertain_like = [
                c for c in claims if c.label in ("uncertain", "out_of_scope")
            ]
            if not uncertain_like:
                explanation = (
                    "The statement is considered true because all extracted claims are supported by the evidence."
                )
                return "true", explanation
            else:
                explanation = (
                    "The statement is partially supported: some claims are supported by the evidence, while others could not be verified or are out of scope."
                )
                return "mixed", explanation

        explanation = (
            "The statement could not be clearly verified or refuted with the available evidence."
        )
        return "unknown", explanation


    def _combine_claim_explanations(self, claims: List[ClaimAssessment]) -> str | None:
        parts = []
        for idx, claim in enumerate(claims, start=1):
            if claim.explanation:
                parts.append(f"Claim {idx}: {claim.explanation}")
        return " ".join(parts) if parts else None

    def run(self, text: str) -> FactCheckResult:
        spans = self.keyword_model.select_spans(text)

        _, subject_for_span = self._map_spans_to_subjects(text, spans)

        subject_evidence_cache = {}
        span_evidence_cache = {}
        evidence_per_span: List[List[EvidenceChunk]] = []

        for s in spans:
            subj = subject_for_span.get(s.span_id)

            if subj is not None:
                if subj.span_id not in subject_evidence_cache:
                    subject_evidence_cache[subj.span_id] = self.wiki_fetcher.retrieve(subj)

                subj_ev = subject_evidence_cache[subj.span_id]

                if s.span_id == subj.span_id:
                    ev = subj_ev
                else:
                    filtered = [
                        chunk for chunk in subj_ev
                        if self._evidence_mentions_span(s.text, chunk.snippet)
                    ]
                    ev = filtered if filtered else []
            else:
                if s.span_id not in span_evidence_cache:
                    span_evidence_cache[s.span_id] = self.wiki_fetcher.retrieve(s)
                ev = span_evidence_cache[s.span_id]

            evidence_per_span.append(ev)

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
