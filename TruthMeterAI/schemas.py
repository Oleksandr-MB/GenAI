from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class Span:
    span_id: int
    text: str
    char_start: int
    char_end: int


@dataclass
class EvidenceChunk:
    doc_id: str
    source_title: str
    source_url: str
    snippet: str


@dataclass
class ClaimAssessment:
    span_id: int
    span_text: str
    char_start: int
    char_end: int

    label: str 
    confidence: float

    explanation: str
    evidence_used: List[EvidenceChunk]

    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_used"] = [asdict(e) for e in self.evidence_used]
        return d


@dataclass
class FactCheckResult:
    text: str
    claims: List[ClaimAssessment]
    overall_label: str | None = None
    overall_explanation: str | None = None
    llm_overall_explanation: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "claims": [c.to_dict() for c in self.claims],
            "overall_label": self.overall_label,
            "overall_explanation": self.overall_explanation,
            "llm_overall_explanation": self.llm_overall_explanation,
        }
