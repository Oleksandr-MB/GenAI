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

@dataclass
class ClaimArgument:
    role: str
    text: str
    ent_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "ent_type": self.ent_type,
        }


@dataclass
class ClaimFrame:
    frame_id: int

    subject: str
    subject_type: str

    subject_start: int
    subject_end: int

    relation_lemma: str
    relation_text: str

    arguments: List[ClaimArgument]

    sentence: str
    sentence_start: int
    sentence_end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "subject": self.subject,
            "subject_type": self.subject_type,
            "subject_start": self.subject_start,
            "subject_end": self.subject_end,
            "relation_lemma": self.relation_lemma,
            "relation_text": self.relation_text,
            "arguments": [a.to_dict() for a in self.arguments],
            "sentence": self.sentence,
            "sentence_start": self.sentence_start,
            "sentence_end": self.sentence_end,
        }
