from __future__ import annotations

from typing import List
import spacy

from .schemas import Span, ClaimFrame, ClaimArgument
from .config import KeywordModelConfig


class KeywordExtractor:

    def __init__(self, cfg: KeywordModelConfig):
        self.cfg = cfg
        self.nlp = spacy.load(cfg.spacy_model_name)

    def select_spans(self, text: str) -> List[Span]:
        """
        Return up to max_spans sentence-level spans.
        """
        doc = self.nlp(text)

        spans: List[Span] = []
        span_id = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            spans.append(
                Span(
                    span_id=span_id,
                    text=sent_text,
                    char_start=sent.start_char,
                    char_end=sent.end_char,
                )
            )
            span_id += 1

            if span_id >= self.cfg.max_spans:
                break

        return spans

    def parse(self, text: str):
        return self.nlp(text)

    def extract_claim_frames(self, text: str) -> List[ClaimFrame]:
        doc = self.nlp(text)
        frames: List[ClaimFrame] = []
        frame_id = 0

        def ent_type_for_span(start_char: int, end_char: int) -> str | None:
            for ent in doc.ents:
                if ent.start_char <= start_char and ent.end_char >= end_char:
                    return ent.label_
            return None

        for sent in doc.sents:
            subj_token = None
            for tok in sent:
                if tok.dep_ in ("nsubj", "nsubjpass"):
                    subj_token = tok
                    break
            if subj_token is None:
                continue

            subj_span = subj_token.subtree
            subj_text = subj_span.text
            subj_start = subj_span.start_char
            subj_end = subj_span.end_char
            subj_type = ent_type_for_span(subj_start, subj_end) or "NOUN_PHRASE"

            head = subj_token.head
            if head.pos_ not in ("VERB", "AUX"):
                head = sent.root
            relation_lemma = head.lemma_
            relation_text = head.text

            arguments: List[ClaimArgument] = []
            for child in head.children:
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            span = pobj.subtree
                            arg_text = span.text
                            arg_start = span.start_char
                            arg_end = span.end_char
                            ent_type = ent_type_for_span(arg_start, arg_end)
                            role = f"prep_{child.text.lower()}"
                            arguments.append(
                                ClaimArgument(
                                    role=role,
                                    text=arg_text,
                                    ent_type=ent_type,
                                )
                            )

                elif child.dep_ in ("dobj", "attr", "acomp", "obl"):
                    span = child.subtree
                    arg_text = span.text
                    arg_start = span.start_char
                    arg_end = span.end_char
                    ent_type = ent_type_for_span(arg_start, arg_end)
                    role = child.dep_
                    arguments.append(
                        ClaimArgument(
                            role=role,
                            text=arg_text,
                            ent_type=ent_type,
                        )
                    )

            frames.append(
                ClaimFrame(
                    frame_id=frame_id,
                    subject=subj_text,
                    subject_type=subj_type,
                    subject_start=subj_start,
                    subject_end=subj_end,
                    relation_lemma=relation_lemma,
                    relation_text=relation_text,
                    arguments=arguments,
                    sentence=sent.text,
                    sentence_start=sent.start_char,
                    sentence_end=sent.end_char,
                )
            )
            frame_id += 1

        return frames
