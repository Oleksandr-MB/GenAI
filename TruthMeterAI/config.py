from dataclasses import dataclass, field

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
    wikipedia2vec_path: str = "TruthMeterAI/models/enwiki_20180420_100d.pkl"
    # increasing will load your system more
    max_snippet_length: int = 256
    max_snippets_per_span: int = 3 

@dataclass
class FactCheckerConfig:
    # increasing will load your system more
    max_snippets_per_span: int = 3
    max_chars_per_snippet: int = 256


@dataclass
class PipelineConfig:
    llm_model_name: str = "Qwen/Qwen3-4B-Instruct-2507" # can be changed to Qwen/Qwen2.5-1.5B-Instruct for weaker hardware
    keyword: KeywordModelConfig = field(default_factory=KeywordModelConfig)
    wiki: WikiFetcherConfig = field(default_factory=WikiFetcherConfig)
    checker: FactCheckerConfig = field(default_factory=FactCheckerConfig)
