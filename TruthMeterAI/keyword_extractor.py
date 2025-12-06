from __future__ import annotations

from typing import List
import spacy

from schemas import Span
from config import KeywordModelConfig


class KeywordExtractor:

    def __init__(self, cfg: KeywordModelConfig, nlp):
        self.cfg = cfg
        self.nlp = nlp

    def select_spans(self, text: str) -> List[Span]:

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

        for ent in doc.ents:
            add_span(ent.start_char, ent.end_char, ent.text)
            if span_id >= self.cfg.max_spans:
                break

        if span_id < self.cfg.max_spans:
            for sent in doc.sents:
                if span_id >= self.cfg.max_spans:
                    break

                sent_start, sent_end = sent.start_char, sent.end_char

                has_span_in_sentence = any(
                    (s_start >= sent_start and s_end <= sent_end)
                    for (s_start, s_end) in seen_ranges
                )
                if has_span_in_sentence:
                    continue

                anchor = None
                for tok in sent:
                    if tok.dep_ in ("nsubj", "nsubjpass"):
                        anchor = tok
                        break

                if anchor is None and sent.root.pos_ in ("NOUN", "PROPN"):
                    anchor = sent.root

                if anchor is None:
                    for tok in sent:
                        if tok.pos_ in ("NOUN", "PROPN"):
                            anchor = tok
                            break

                if anchor is not None:
                    # anchor.subtree is a generator; build a concrete span around the subtree tokens
                    token_start = anchor.left_edge.idx
                    token_end = anchor.right_edge.idx + len(anchor.right_edge)
                    span_text = doc.text[token_start:token_end]
                    add_span(token_start, token_end, span_text)
                else:
                    add_span(sent_start, sent_end, sent.text)

        return spans

    def parse(self, text: str):
        return self.nlp(text)
