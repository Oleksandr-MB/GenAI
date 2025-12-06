from dataclasses import dataclass

@dataclass
class KeywordModelConfig:
    max_spans: int = 32
    spacy_model_name: str = "en_core_web_sm"

@dataclass
class WikiFetcherConfig:
    language: str = "en"
    max_search_results: int = 5
    max_pages_to_fetch: int = 2
    min_title_score: float = 0.5
    max_snippet_length: int = 600
    max_snippets_per_span: int = 3
    wikipedia2vec_path: str = "TruthMeterAI/models/enwiki_20180420_100d.pkl"

@dataclass
class FactCheckerConfig:
    max_snippets_per_span: int = 3
    max_chars_per_snippet: int = 400


@dataclass
class PipelineConfig:
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    keyword: KeywordModelConfig = KeywordModelConfig()
    wiki: WikiFetcherConfig = WikiFetcherConfig()
    checker: FactCheckerConfig = FactCheckerConfig()