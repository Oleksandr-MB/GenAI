from __future__ import annotations

from typing import List

from flask import Flask, render_template, request

if __package__ in (None, ""):
    from config import PipelineConfig
    from pipeline import Pipeline
    from schemas import FactCheckResult, ClaimAssessment, EvidenceChunk
else:
    from .config import PipelineConfig
    from .pipeline import Pipeline
    from .schemas import FactCheckResult, ClaimAssessment, EvidenceChunk

app = Flask(__name__)

cfg = PipelineConfig()
pipeline = Pipeline(cfg)


def compute_overall_confidence(result: FactCheckResult) -> float:
    claims = result.claims
    if not claims:
        return 0.0

    return sum(c.confidence for c in claims) / len(claims)


def collect_checked_articles(result: FactCheckResult) -> List[EvidenceChunk]:
    seen = set()
    articles: List[EvidenceChunk] = []

    for claim in result.claims:
        for ev in (claim.evidence_used or []):
            key = ev.source_url or f"{ev.doc_id}:{ev.source_title}"
            if key in seen:
                continue
            seen.add(key)
            articles.append(ev)

    return articles


@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    result: FactCheckResult | None = None
    overall_confidence = 0.0
    articles: List[EvidenceChunk] = []
    claims: List[ClaimAssessment] = []

    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        if input_text.strip():
            result = pipeline.run(input_text)
            raw_claims = result.claims
            overall_confidence = compute_overall_confidence(result)
            articles = collect_checked_articles(result)
            claims = raw_claims

    return render_template(
        "index.html",
        input_text=input_text,
        result=result,
        overall_confidence=overall_confidence,
        articles=articles,
        claims=claims,
        result_text=result.text if result else "",
    )
