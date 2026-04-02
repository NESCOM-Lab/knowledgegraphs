"""
Memory tools - agent writes intermediate findings and hypotheses to a session buffer.
Designed to be instantiated once per research session.
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Finding:
    id: str
    content: str
    confidence: float  # 0.0 - 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Hypothesis:
    id: str
    statement: str
    supporting_ids: list[str]
    contradicting_ids: list[str]
    confidence: float = 0.5
    verified: bool = False
    verification_notes: str = ""


class SessionMemory:
    """
    In-memory store for findings and hypotheses within a single research session.
    Instantiate once and pass to the research agent.
    """

    def __init__(self):
        self._findings: list[Finding] = []
        self._hypotheses: list[Hypothesis] = []
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def save_finding(self, content: str, confidence: float = 0.7) -> dict:
        """
        Agent writes an intermediate finding to the buffer.

        args:
            content: the finding text (can be multi-sentence)
            confidence: 0.0 (speculation) to 1.0 (certain from graph data)

        returns:
            dict with the saved finding id
        """
        confidence = max(0.0, min(1.0, float(confidence)))
        finding = Finding(
            id=self._next_id("finding"),
            content=content,
            confidence=confidence,
        )
        self._findings.append(finding)
        return {
            "id": finding.id,
            "content": finding.content,
            "confidence": finding.confidence,
            "message": "Finding saved successfully",
        }

    def get_findings(self) -> list[dict]:
        """
        Retrieve all findings from the current session.
        Returns list sorted by confidence descending.
        """
        return [
            {
                "id": f.id,
                "content": f.content,
                "confidence": f.confidence,
            }
            for f in sorted(self._findings, key=lambda x: x.confidence, reverse=True)
        ]

    def create_hypothesis(
        self,
        statement: str,
        supporting_ids: list[str],
        contradicting_ids: list[str],
        confidence: float = 0.5,
    ) -> dict:
        """
        Formally register a hypothesis with evidence pointers.

        args:
            statement: the hypothesis in plain language
            supporting_ids: finding IDs or paper names that support it
            contradicting_ids: finding IDs or paper names that contradict it
            confidence: initial confidence 0.0-1.0

        returns:
            dict with the hypothesis id
        """
        confidence = max(0.0, min(1.0, float(confidence)))
        hyp = Hypothesis(
            id=self._next_id("hypothesis"),
            statement=statement,
            supporting_ids=supporting_ids,
            contradicting_ids=contradicting_ids,
            confidence=confidence,
        )
        self._hypotheses.append(hyp)
        return {
            "id": hyp.id,
            "statement": hyp.statement,
            "supporting_ids": hyp.supporting_ids,
            "contradicting_ids": hyp.contradicting_ids,
            "confidence": hyp.confidence,
            "message": "Hypothesis registered successfully",
        }

    def update_hypothesis(
        self,
        hypothesis_id: str,
        confidence: Optional[float] = None,
        verified: Optional[bool] = None,
        verification_notes: Optional[str] = None,
    ) -> dict:
        """Update a hypothesis after verification."""
        for hyp in self._hypotheses:
            if hyp.id == hypothesis_id:
                if confidence is not None:
                    hyp.confidence = max(0.0, min(1.0, float(confidence)))
                if verified is not None:
                    hyp.verified = verified
                if verification_notes is not None:
                    hyp.verification_notes = verification_notes
                return {
                    "id": hyp.id,
                    "statement": hyp.statement,
                    "confidence": hyp.confidence,
                    "verified": hyp.verified,
                    "message": "Hypothesis updated",
                }
        return {"error": f"Hypothesis '{hypothesis_id}' not found"}

    def get_hypotheses(self) -> list[dict]:
        """Return all hypotheses, ranked by confidence."""
        return [
            {
                "id": h.id,
                "statement": h.statement,
                "supporting_ids": h.supporting_ids,
                "contradicting_ids": h.contradicting_ids,
                "confidence": h.confidence,
                "verified": h.verified,
                "verification_notes": h.verification_notes,
            }
            for h in sorted(self._hypotheses, key=lambda x: x.confidence, reverse=True)
        ]

    def clear(self):
        """Reset the session memory."""
        self._findings.clear()
        self._hypotheses.clear()
        self._counter = 0
