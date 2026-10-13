"""Pure domain layer.

No I/O and no heavy dependencies. Everything here is deterministic and
unit-testable in isolation.
"""

from .detections import Detection, PersonAssessment, SafetyResult
from .geometry import BBox, iou, overlap_ratio
from .levels import score_to_level
from .scoring import analyze_person, compute_safety, compute_scaffolding_risk

__all__ = [
    "BBox",
    "Detection",
    "PersonAssessment",
    "SafetyResult",
    "analyze_person",
    "compute_safety",
    "compute_scaffolding_risk",
    "iou",
    "overlap_ratio",
    "score_to_level",
]
