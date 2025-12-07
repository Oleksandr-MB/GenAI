# TruthMeterAI

TruthMeterAI is an end-to-end factuality pipeline that turns arbitrary prose into structured claim assessments backed by citations. It utilizes linguistic heuristics, targeted Wikipedia retrieval, and a locally hosted instruction-tuned LLM to produce verdicts you can inspect claim by claim through either a CLI or a Flask dashboard.

## How It Works
1. **Span selection** - `KeywordExtractor` loads the spaCy model from `KeywordModelConfig`, grabs high-signal named entities first, then backfills with dependency anchors (subjects, head nouns, sentence roots) until `max_spans` unique ranges are filled.
2. **Evidence retrieval** - `WikiFetcher` expands each entity/span through Wikipedia2Vec neighbors, falls back to lexical search with title scoring, threads section headings into every paragraph, drops boilerplate sections, and ranks snippets with TF-IDF similarity against the full user text before returning structured `EvidenceChunk`s.
3. **LLM judgment** - `FactChecker` splits the text into sentences, maps each one plus its evidence snippets into a strict prompt template, enforces snippet/character limits, queries the configured Hugging Face model, and parses the response (JSON first, then regex) into `ClaimAssessment` objects with graceful defaults when parsing fails.
4. **Aggregation** - `Pipeline` caches evidence per entity to avoid redundant fetches, feeds the assessments into verdict logic that weights confidences into (`true`, `false`, `mixed`, `unknown`), and emits the same structured `FactCheckResult` to the CLI renderer and the Flask dashboard.

## Repository Layout

| Path | Description |
| --- | --- |
| `keyword_extractor.py` | Selects claim-worthy spans using spaCy entities and syntactic anchors. |
| `wiki_fetcher.py` | Wraps the `wikipedia` API, embeds section headings into snippets, runs TF-IDF scoring, and returns `EvidenceChunk` objects. |
| `fact_checker.py` | Builds the LLM prompts, parses the model's structured replies, and normalizes failure cases. |
| `pipeline.py` | Orchestrates span extraction, snippet retrieval, fact checking, and aggregates an overall verdict. |
| `web_app.py`, `templates/`, `static/` | Flask application that visualizes outputs and links claims to evidence. |
| `cli.py` | Simple entry point that prints a JSON `FactCheckResult`. |
| `config.py` | Dataclasses for tuning span limits, retrieval behavior, and LLM selection. |

## Getting Started

### Installation
```bash
# Prerequisite: install Python 3.10.x on your machine
git clone https://github.com/Oleksandr-MB/GenAI
cd GenAI
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
# The requirements pin en_core_web_sm; if it fails you can install it manually:
python -m spacy download en_core_web_sm
# Download a wikipedia2vec model
cd TruthMeterAI/models
wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2
bzip2 -d enwiki_20180420_100d.pkl.bz2
cd ../..
```

## Usage

### CLI
Run the pipeline once and print structured JSON:
```bash
python -m TruthMeterAI.cli "London is the capital of France."
```
Each claim includes `label`, `confidence`, `explanation`, and the evidence snippets that were cited.

### Flask Web App
```bash
export FLASK_ENV="development"
export FLASK_APP=TruthMeterAI.web_app:app
flask run
```

on Linux/MacOS or

```bash
$env:FLASK_ENV = "development"
$env:FLASK_APP = "TruthMeterAI.web_app:app" 
flask run
```
on Windows.

Open http://127.0.0.1:5000 and type your claim text into the form. The page shows the submitted text, a verdict banner, and a claims table with label, confidence, explanation, and evidence snippet links. Under the table you will see a list of Wikipedia articles that were consulted.

> **Tip:** The first request will download the LLM weights and can take a few minutes; subsequent launches warm-start from the local cache. You will also see quite a lot of warnings, caused by the outdated joblib version that was used to pickle the Wikipedia2Vec model; you can safely ignore them.

## Configuration
Tweak behavior centrally through `config.py`.

For example, use another LLM:
```python
@dataclass
class PipelineConfig:
    llm_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"  # change here, e.g. to Qwen/Qwen2.5-1.5B-Instruct for weaker hardware
    keyword: KeywordModelConfig = KeywordModelConfig()
    wiki: WikiFetcherConfig = WikiFetcherConfig()
    checker: FactCheckerConfig = FactCheckerConfig()
```

Other options:
- **`KeywordModelConfig.max_spans`** - bounds cost on very long passages.
- **`WikiFetcherConfig.language` and `KeywordModelConfig.spacy_model_name`** - change the language.
- **`WikiFetcherConfig.max_snippets_per_span`** - trade precision vs. recall when feeding the LLM.
- **`FactCheckerConfig.max_chars_per_snippet`** - truncate evidence to stay within model limits.

Import your tweaked config wherever a `Pipeline` is created (the CLI, Flask app, or your own scripts) to apply the same behavior everywhere.

## Limitations & Future Work
- Coverage is limited to Wikipedia; niche claims without an article will often be `UNCERTAIN` or `OUT_OF_SCOPE`.
- Extremely long documents may have too many keywords leading to the context window overload with the evidence snippets.
- The default Qwen LLM is compact; plugging in a larger instruction-tuned LLM generally improves label accuracy at the cost of latency.
- Model downloads and Wikipedia queries require internet access.

# Happy fact checking!
