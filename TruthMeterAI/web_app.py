from flask import Flask, render_template, request
from .config import PipelineConfig
from .pipeline import Pipeline
from .schemas import FactCheckResult, ClaimAssessment

app = Flask(__name__)

cfg = PipelineConfig()
pipeline = Pipeline(cfg)


def compute_overall_confidence(result: FactCheckResult) -> float:
    
    if all(c.label == "supported" for c in result.claims) or all(c.label == "contradicted" for c in result.claims):
        return sum(c.confidence for c in result.claims) / len(result.claims)
    
    else:
        return 0.5

def collect_checked_articles(result: FactCheckResult):

    articles_by_id = {}

    for claim in result.claims:
        for ev in getattr(claim, "evidence_used", []) or []:
            if ev.doc_id not in articles_by_id:
                articles_by_id[ev.doc_id] = {
                    "doc_id": ev.doc_id,
                    "title": ev.source_title,
                    "url": ev.source_url,
                }

    return list(articles_by_id.values())


def build_highlight_segments(text: str, claims: list[ClaimAssessment]):

    if not text:
        return []

    sorted_claims = sorted(
        claims,
        key=lambda c: (c.char_start, c.char_end),
    )

    segments = []
    pos = 0
    n = len(text)

    for claim in sorted_claims:
        start = max(0, min(claim.char_start, n))
        end = max(0, min(claim.char_end, n))

        if end <= start:
            continue

        if start > pos:
            segments.append({"text": text[pos:start], "claim": None})

        segments.append({"text": text[start:end], "claim": claim})
        pos = end

    if pos < n:
        segments.append({"text": text[pos:], "claim": None})

    return segments


@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    result: FactCheckResult | None = None
    overall_confidence = None
    claims: list[ClaimAssessment] = []
    articles = []
    segments = []

    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        if input_text:
            result = pipeline.run(input_text)
            claims = result.claims
            overall_confidence = compute_overall_confidence(result)
            articles = collect_checked_articles(result)
            segments = build_highlight_segments(result.text, claims)

    return render_template(
        "index.html",
        input_text=input_text,
        result=result,
        overall_confidence=overall_confidence,
        articles=articles,
        claims=claims,
        segments=segments,
    )
