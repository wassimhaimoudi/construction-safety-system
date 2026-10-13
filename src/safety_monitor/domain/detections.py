"""Core value objects for the domain layer."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import BBox


@dataclass(frozen=True)
class Detection:
    """A single model detection on a frame."""

    class_name: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class PersonAssessment:
    """Per-person PPE evaluation."""

    index: int
    bbox: BBox
    on_scaffold: bool
    helmet: str  # "present" | "missing" | "unknown"
    vest: str
    glove: str
    mask: str
    status: str  # "SAFE" | "PARTIAL" | "VIOLATION" | "UNKNOWN"
    score: int  # danger points, capped at 45
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SafetyResult:
    """Aggregate result for a frame.

    ``safety`` is the headline number: higher is safer (0-100).
    """

    safety: int
    danger: int
    level: str
    ppe_risk: int
    scaffold_risk: int
    crowd_risk: int
    persons: tuple[PersonAssessment, ...]
