from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

from flask import Flask, render_template, request

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
            key = (ev.doc_id, ev.source_title, ev.source_url, ev.snippet)
            if key in seen:
                continue
            seen.add(key)
            articles.append(ev)

    return articles


def _sentence_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    n = len(text)

    left_candidates = [
        text.rfind(";", 0, start),
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
        text.rfind("\n", 0, start),
    ]
    left = max(left_candidates)
    sent_start = 0 if left == -1 else left + 1

    right_candidates = [
        text.find(";", end),
        text.find(".", end),
        text.find("!", end),
        text.find("?", end),
        text.find("\n", end),
    ]
    right_candidates = [x for x in right_candidates if x != -1]
    sent_end = n if not right_candidates else min(right_candidates) + 1

    return sent_start, sent_end


def group_claims_for_display(
    claims: List[ClaimAssessment], text: str
) -> List[ClaimAssessment]:
    if not claims or not text:
        return claims

    n = len(text)

    groups: dict[Tuple[int, int], List[ClaimAssessment]] = defaultdict(list)
    for claim in claims:
        start = max(0, min(claim.char_start, n))
        end = max(0, min(claim.char_end, n))
        sent_start, sent_end = _sentence_bounds(text, start, end)
        groups[(sent_start, sent_end)].append(claim)

    display_claims: List[ClaimAssessment] = []
    new_span_id = 0

    for (sent_start, sent_end), cls in sorted(groups.items(), key=lambda kv: kv[0][0]):
        if not cls:
            continue

        sent_start = max(0, min(sent_start, n))
        sent_end = max(sent_start, min(sent_end, n))
        span_text = text[sent_start:sent_end]

        labels = [c.label for c in cls]
        label = None
        for candidate in ("contradicted", "supported", "uncertain", "out_of_scope"):
            if candidate in labels:
                label = candidate
                break
        if label is None:
            label = cls[0].label

        confidences = [c.confidence for c in cls]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        best = max(cls, key=lambda c: c.confidence)
        explanation = best.explanation

        ev_seen = set()
        ev_used: List[EvidenceChunk] = []
        for c in cls:
            for ev in (c.evidence_used or []):
                key = (ev.doc_id, ev.source_title, ev.source_url, ev.snippet)
                if key in ev_seen:
                    continue
                ev_seen.add(key)
                ev_used.append(ev)

        error_type = next(
            (c.error_type for c in cls if getattr(c, "error_type", None)), None
        )

        display_claims.append(
            ClaimAssessment(
                span_id=new_span_id,
                span_text=span_text,
                char_start=sent_start,
                char_end=sent_end,
                label=label,
                confidence=confidence,
                explanation=explanation,
                evidence_used=ev_used,
                error_type=error_type,
            )
        )
        new_span_id += 1

    return display_claims


def build_highlight_segments(text: str, claims: List[ClaimAssessment]) -> List[dict]:
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

        if end <= pos:
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
    overall_confidence = 0.0
    articles: List[EvidenceChunk] = []
    claims: List[ClaimAssessment] = []
    segments = []

    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        if input_text.strip():
            result = pipeline.run(input_text)
            raw_claims = result.claims
            overall_confidence = compute_overall_confidence(result)
            articles = collect_checked_articles(result)
            segments = build_highlight_segments(result.text, raw_claims)
            claims = group_claims_for_display(raw_claims, result.text)

    return render_template(
        "index.html",
        input_text=input_text,
        result=result,
        overall_confidence=overall_confidence,
        articles=articles,
        claims=claims,
        segments=segments,
    )
