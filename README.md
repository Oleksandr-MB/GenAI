# TruthMeterAI

TruthMeterAI is an end-to-end factuality pipeline that turns arbitrary prose into structured claim assessments backed by citations. It utilizes linguistic heuristics, targeted Wikipedia retrieval, and a locally hosted instruction-tuned LLM to produce verdicts you can inspect claim by claim through either a CLI or a Flask dashboard.

## Highlights
- **Claim-aware span selection** - spaCy-driven named-entity and grammar heuristics limit checks to the most relevant 32 spans of the input text.
- **Wikipedia-grounded retrieval** - deterministic search + TF-IDF re-ranking produces short, section-aware snippets that include their heading path and are cached per article.
- **LLM fact judging** - the default `Qwen/Qwen2.5-1.5B-Instruct` model is prompted to emit structured labels (`SUPPORTED`, `CONTRADICTED`, `UNCERTAIN`, `OUT_OF_SCOPE`) alongside evidence indices, confidence, and rationale.
- **Transparent outputs** - every claim in the interface links directly to the snippets that justified the verdict, and overall summaries roll up claim-level labels into an intuitive "true / false / mixed / unknown" signal.
- **Two ways to run** - a zero-dependency CLI for automation and a modern Flask UI that highlights spans in the original text and lists the consulted Wikipedia articles.

## Repository Layout

| Path | Description |
| --- | --- |
| `keyword_extractor.py` | Selects claim-worthy spans using spaCy entities and syntactic anchors. |
| `wiki_fetcher.py` | Wraps the `wikipedia` API, embeds section headings into snippets, runs TF-IDF scoring, and returns `EvidenceChunk` objects. |
| `fact_checker.py` | Builds the LLM prompts, parses the model’s structured replies, and normalizes failure cases. |
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

Open http://127.0.0.1:5000 and type your claim text into the form. The interface highlights spans inline, shows a verdict banner, and lists the consulted Wikipedia articles with links.

> **Tip:** The first request will download the LLM weights and can take a few minutes; subsequent launches warm-start from the local cache.

## How It Works
1. **Span selection** - `KeywordExtractor` uses spaCy NER plus syntactic anchors to cap the workload at `max_spans` contiguous phrases.
2. **Evidence retrieval** - `WikiFetcher` searches Wikipedia per span, walks section headings to preserve context, filters boilerplate, and re-ranks snippets with TF-IDF similarity against the entire user text.
3. **LLM judgment** - `FactChecker` feeds each span, its surrounding sentence, and the selected snippets to the LLM using a strict output schema. Responses are parsed into `ClaimAssessment` objects with graceful fallbacks for malformed outputs.
4. **Aggregation** - `Pipeline` merges claim rationales and confidence-weighted labels into an overall verdict (`true`, `false`, `mixed`, or `unknown`) and exposes the structured data to both the CLI and the Flask app.

## Configuration
Tweak behavior centrally through `config.py`:

For example, use another LLM:
```python
@dataclass
class PipelineConfig:
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct" # change here
    keyword: KeywordModelConfig = KeywordModelConfig()
    wiki: WikiFetcherConfig = WikiFetcherConfig()
    checker: FactCheckerConfig = FactCheckerConfig()
```

Pass the custom config into the CLI or web app entry point (e.g., change the global `cfg` in `web_app.py`, or instantiate `Pipeline(cfg)` in your own script). Notable knobs include:
- **`KeywordModelConfig.max_spans`** - bounds cost on very long passages.
- **`WikiFetcherConfig.language` and `KeywordModelConfig.spacy_model_name`** - allow languages other than English.
- **`WikiFetcherConfig.max_snippets_per_span`** - trade precision vs. recall when feeding the LLM.
- **`FactCheckerConfig.max_chars_per_snippet`** - truncate evidence to stay within model limits.

## Limitations & Future Work
- Coverage is limited to Wikipedia; niche claims without an article will often be `UNCERTAIN` or `OUT_OF_SCOPE`.
- Extremely long documents may have important claims beyond the first `max_spans` spans.
- The default Qwen checkpoint is a compact model; plugging in a larger instruction-tuned LLM generally improves label accuracy at the cost of latency.
- Model downloads and Wikipedia queries require internet access.

# Happy fact checking!
