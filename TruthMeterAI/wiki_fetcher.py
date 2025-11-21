from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List
import re

import wikipedia

from .schemas import EvidenceChunk, Span
from .config import WikiFetcherConfig


class AbstractWikiFetcher(ABC):
    def __init__(self, cfg: WikiFetcherConfig):
        self.cfg = cfg

    @abstractmethod
    def retrieve(self, span: Span) -> List[EvidenceChunk]:
        pass


class WikiFetcher(AbstractWikiFetcher):
    def __init__(self, cfg: WikiFetcherConfig):
        super().__init__(cfg)
        wikipedia.set_lang(cfg.language)

    def _normalize_words(self, s: str) -> List[str]:
        s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
        return [w for w in s.split() if w]

    def _score_title(self, query_words: List[str], title: str) -> float:
        if not query_words:
            return 0.0
        title_words = self._normalize_words(title)
        matches = sum(1 for w in query_words if w in title_words)
        if title_words == query_words:
            return 1.0
        return matches / len(query_words)

    def _query(self, query: str) -> List[EvidenceChunk]:
        query_words = self._normalize_words(query)
        search_results = wikipedia.search(query, results=self.cfg.max_search_results)

        scored_pages = []
        for title in search_results:
            score = self._score_title(query_words, title)
            if score >= self.cfg.min_title_score:
                scored_pages.append((score, title))

        scored_pages.sort(reverse=True)
        selected_pages = scored_pages[: self.cfg.max_pages_to_fetch]

        evidence_chunks: List[EvidenceChunk] = []

        for _, title in selected_pages:
            try:
                page = wikipedia.page(title, auto_suggest=False)
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue

            snippet = page.content[: self.cfg.max_snippet_length].strip()

            evidence_chunks.append(
                EvidenceChunk(
                    doc_id=str(page.pageid),
                    source_title=page.title,
                    source_url=page.url,
                    snippet=snippet,
                )
            )

        return evidence_chunks

    def retrieve(self, span: Span) -> List[EvidenceChunk]:
        return self._query(span.text.strip())