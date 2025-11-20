from __future__ import annotations

from typing import List
import spacy

from .schemas import Span, ClaimFrame, ClaimArgument
from .config import KeywordModelConfig


class KeywordExtractor:

    def __init__(self, cfg: KeywordModelConfig):
        self.cfg = cfg
        self.nlp = spacy.load(cfg.spacy_model_name)

class KeywordExtractor:

    def __init__(self, cfg: KeywordModelConfig):
        self.cfg = cfg
        self.nlp = spacy.load(cfg.spacy_model_name)

    def select_spans(self, text: str) -> List[Span]:
        """Select entity-level spans (named entities / key noun phrases) instead of
        whole sentences.

        Strategy:
        1. Take all named entities from the document as primary spans.
        2. For sentences that contain no named entities, fall back to a salient
           noun phrase (subject or main noun), or the whole sentence if nothing
           better is found.
        3. De-duplicate spans by character offsets and cap the total number to
           ``cfg.max_spans``.
        """
        doc = self.nlp(text)

        spans: List[Span] = []
        seen_ranges: set[tuple[int, int]] = set()
        span_id = 0

        def add_span(start_char: int, end_char: int, span_text: str) -> None:
            nonlocal span_id

            span_text = span_text.strip()
            if not span_text:
                return
            if span_id >= self.cfg.max_spans:
                return

            key = (start_char, end_char)
            if key in seen_ranges:
                return

            seen_ranges.add(key)
            spans.append(
                Span(
                    span_id=span_id,
                    text=span_text,
                    char_start=start_char,
                    char_end=end_char,
                )
            )
            span_id += 1

        # 1) Add all named entities as primary candidate spans
        for ent in doc.ents:
            add_span(ent.start_char, ent.end_char, ent.text)
            if span_id >= self.cfg.max_spans:
                break

        # 2) For sentences without any entity span, fall back to a noun-phrase anchor
        if span_id < self.cfg.max_spans:
            for sent in doc.sents:
                if span_id >= self.cfg.max_spans:
                    break

                sent_start, sent_end = sent.start_char, sent.end_char

                # Check if this sentence already contributed at least one span
                has_span_in_sentence = any(
                    (s_start >= sent_start and s_end <= sent_end)
                    for (s_start, s_end) in seen_ranges
                )
                if has_span_in_sentence:
                    continue

                # Try to pick a subject token as the anchor
                anchor = None
                for tok in sent:
                    if tok.dep_ in ("nsubj", "nsubjpass"):
                        anchor = tok
                        break

                # Fallback to the sentence root if it is a NOUN/PROPN
                if anchor is None and sent.root.pos_ in ("NOUN", "PROPN"):
                    anchor = sent.root

                # Fallback to the first NOUN/PROPN in the sentence
                if anchor is None:
                    for tok in sent:
                        if tok.pos_ in ("NOUN", "PROPN"):
                            anchor = tok
                            break

                if anchor is not None:
                    span = anchor.subtree
                    add_span(span.start_char, span.end_char, span.text)
                else:
                    # Ultimate fallback: use the whole sentence
                    add_span(sent_start, sent_end, sent.text)

        return spans

    def parse(self, text: str):
        """Small helper to expose the underlying spaCy Doc parser."""
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
