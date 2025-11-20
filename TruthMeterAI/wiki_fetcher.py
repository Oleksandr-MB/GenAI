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
        t_words = self._normalize_words(title)
        if not query_words or not t_words:
            return 0.0

        common = len(set(query_words) & set(t_words))
        coverage = common / len(query_words)

        score = 2.0 * coverage
        if "(" not in title:
            score += 0.5

        score -= 0.01 * len(title)
        return score

    @lru_cache(maxsize=1024)
    def _retrieve_for_query(self, query: str) -> List[EvidenceChunk]:
        query = query.strip()
        n = query.count(".") + 1

        if not query:
            return []

        search_query = query
        try:
            suggestion = wikipedia.suggest(query)
            if suggestion:
                search_query = suggestion
        except Exception:
            search_query = query

        try:
            raw_titles = wikipedia.search(search_query)
        except Exception:
            raw_titles = []

        if not raw_titles and search_query != query:
            try:
                raw_titles = wikipedia.search(query)
                search_query = query
            except Exception:
                raw_titles = []

        if not raw_titles:
            return []

        q_words = self._normalize_words(search_query)
        scored = sorted(
            raw_titles,
            key=lambda title: self._score_title(q_words, title),
            reverse=True,
        )
        titles = scored[: self.cfg.top_k * 2]

        chunks: List[EvidenceChunk] = []
        for title in titles:
            if len(chunks) >= self.cfg.top_k:
                break

            try:
                page = wikipedia.page(title, auto_suggest=False)
            except Exception:
                continue

            content = page.content
            sentences = re.split(r"(?<=[.!?])\s+", content)
            if not sentences:
                continue

            head = " ".join(sentences[:n])

            q_lower = search_query.lower()
            hits = [s for s in sentences if q_lower in s.lower()]
            snippet = " ".join(hits) if hits else head
            snippet = snippet[: self.cfg.max_snippet_chars]

            chunks.append(
                EvidenceChunk(
                    doc_id=str(getattr(page, "pageid", page.title)),
                    source_title=page.title,
                    source_url=page.url,
                    snippet=snippet,
                )
            )

        return chunks

    def retrieve(self, span: Span) -> List[EvidenceChunk]:
        return self._retrieve_for_query(span.text.strip())