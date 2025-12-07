from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import re

import wikipedia
from wikipedia2vec import Wikipedia2Vec
from wikipedia2vec.dictionary import Entity

if __package__ in (None, ""):
    from schemas import EvidenceChunk, Span
    from config import WikiFetcherConfig
else:
    from .schemas import EvidenceChunk, Span
    from .config import WikiFetcherConfig

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AbstractWikiFetcher(ABC):
    def __init__(self, cfg: WikiFetcherConfig):
        self.cfg = cfg

    @abstractmethod
    def retrieve(self, span: Span) -> List[EvidenceChunk]:
        pass


class WikiFetcher(AbstractWikiFetcher):
    def __init__(self, cfg: WikiFetcherConfig, nlp):
        super().__init__(cfg)
        self.nlp = nlp
        self.page_tfidf_cache = {}
        wikipedia.set_lang(cfg.language)
        self.emb_model = Wikipedia2Vec.load(cfg.wikipedia2vec_path)

    def search(self, query: str) -> List[str]:
        
        entity = self.emb_model.dictionary.get_entity(query)
        if entity is not None:
            sims = self.emb_model.most_similar(entity, count=50)
            titles = [obj.title for obj, score in sims if isinstance(obj, Entity)]
            return titles[: self.cfg.max_pages_to_fetch]
        else:
            query_words = self._normalize_words(query)
            search_results = wikipedia.search(query, results=self.cfg.max_search_results)

            scored_pages = []
            for title in search_results:
                score = self._score_title(query_words, title)
                if score >= self.cfg.min_title_score:
                    scored_pages.append((score, title))

            scored_pages.sort(key=lambda x: (-x[0], len(x[1])))
            titles = [title for score, title in scored_pages]
            return titles[: self.cfg.max_pages_to_fetch]


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

    def _query(self, query: str, full_query : str) -> List[EvidenceChunk]:
        selected_pages = self.search(query)

        evidence_chunks: List[EvidenceChunk] = []

        for title in selected_pages:
            try:
                page = wikipedia.page(title, auto_suggest=False)
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue

            all_snippets = self.gather_snippets_embedded_with_titles(page.content)
            relevant_snippets = self.find_relevant_snippets(full_query, all_snippets, title)

            for snippet in relevant_snippets:
                evidence_chunks.append(
                    EvidenceChunk(
                        doc_id=str(page.pageid),
                        source_title=page.title,
                        source_url=page.url,
                        snippet=snippet,
                    )
                )

        return evidence_chunks

    def retrieve(self, span: Span, full_query: str) -> List[EvidenceChunk]:
        return self._query(span.text.strip(), full_query) 
    
    def find_relevant_snippets(self, query: str, snippets: list[str], title : str) -> list[str]:
        tokenized_snippets = [self.tokenize_snippet(s) for s in snippets]
        tokenized_query = self.tokenize_snippet(query)

        if title not in self.page_tfidf_cache:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(tokenized_snippets)
            self.page_tfidf_cache[title] = (vectorizer, tfidf_matrix)
        else:
            vectorizer, tfidf_matrix = self.page_tfidf_cache[title]

        query_vector = vectorizer.transform([tokenized_query])

        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        ranked_indices = similarities.argsort()[::-1][:self.cfg.max_snippets_per_span]
        relevant_snippets = [snippets[idx] for idx in ranked_indices if similarities[idx] > 0]
        if len(relevant_snippets)  == 0:
            relevant_snippets.append(snippets[ranked_indices[0]])
        return relevant_snippets

    def tokenize_snippet(self, snippet: str) -> str:
        snippet_obj = self.nlp(snippet)
        tokens = []
        for token in snippet_obj:
            if not token.is_stop and not token.is_punct:
                tokens.append(token.lemma_.lower())
        return " ".join(tokens)
    
    def _count_title_level(self, title_raw : str) -> int:
        return title_raw.count('=') // 2
    
    def _adjust_title_stacks(self, title_stack: list[str], title: str) -> list[str]:
        title_level = self._count_title_level(title)
        index = len(title_stack) - 1
        while index >= 0 and self._count_title_level(title_stack[index]) >= title_level:
            index -= 1
        title_place = index + 1
        if title_place < len(title_stack):
            title_stack[title_place] = title
            title_stack = title_stack[: title_place + 1]
        else:
            title_stack.append(title)
        return title_stack
    
    def _turn_stack_to_path(self, title_stack: list[str]) -> str:
        titles = [re.sub(r'=+\s*(.*?)\s*=+', r'\1', t).strip() for t in title_stack]
        return " > ".join(titles)
    
    def _filter_snippets(self, snippets: list[str]) -> list[str]:
        filtered = []
        for snip in snippets:
            title_line = snip.split('\n', 1)[0]
            if 'See also' in title_line or 'References' in title_line or 'External links' in title_line:
                continue
            filtered.append(snip)
        return filtered

    def gather_snippets_embedded_with_titles(self, page_content: str) -> list[str]:
        wikipedia_title_pattern = re.compile(r'=+\s*(.*?)\s*=+')
        paragraphs = page_content.split('\n')
        title_stack = []
        snippets = []

        for par in paragraphs:
            title_match = wikipedia_title_pattern.search(par)
            if title_match:
                title = par
                title_stack = self._adjust_title_stacks(title_stack, title)
            elif par.strip():
                title_path = self._turn_stack_to_path(title_stack) if title_stack else "Introduction"
                snippet_with_title = f"[{title_path}]\n{par.strip()}"
                snippets.append(snippet_with_title)
        
        return self._filter_snippets(snippets)
