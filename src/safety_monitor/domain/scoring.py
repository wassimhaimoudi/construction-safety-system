"""Pure safety scoring.

Given a list of :class:`Detection` objects for a single frame, produce a
:class:`SafetyResult`.

    Danger = min(100, PPE[50] + Scaffold[35] + Crowd[15])
    Safety = 100 - Danger

All functions here are deterministic and free of I/O or heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from . import taxonomy as tax
from .detections import Detection, PersonAssessment, SafetyResult
from .geometry import BBox, iou, overlap_ratio
from .levels import score_to_level

# Weight schedule: ground vs scaffold.
PPE_WEIGHTS: dict[str, dict[str, int]] = {
    "helmet": {"ground": 15, "scaffold": 25},
    "vest": {"ground": 10, "scaffold": 20},
    "glove": {"ground": 8, "scaffold": 12},
    "mask": {"ground": 4, "scaffold": 6},
}

# Minimum penalty applied when a PPE item is "unknown" (half value, floored).
UNKNOWN_FLOOR: dict[str, int] = {"helmet": 4, "vest": 0, "glove": 3, "mask": 2}

MAX_PERSON_SCORE = 45
CAP_PPE = 50
CAP_SCAFFOLD = 35
CAP_CROWD = 15

SCAFFOLD_POINTS = {"suspended": 35, "high": 20, "low": 5}
SCAFFOLD_WORKER_BONUS = 10

# Scaffolds that raise the "elevated work" context for PPE weighting.
_ON_SCAFFOLD_CLASSES = tax.SCAFFOLD_SUSPENDED_CLASSES | tax.SCAFFOLD_HIGH_CLASSES

# Association thresholds (mirrors the legacy engine).
_PPE_OVERLAP_MIN = 0.2
_PPE_IOU_MIN = 0.1
_SCAFFOLD_OVERLAP_MIN = 0.3
_SCAFFOLD_IOU_MIN = 0.1

_PPE_CLASSES = (
    tax.HELMET_PRESENT_CLASSES
    | tax.HELMET_MISSING_CLASSES
    | tax.VEST_PRESENT_CLASSES
    | tax.VEST_MISSING_CLASSES
    | tax.GLOVE_PRESENT_CLASSES
    | tax.GLOVE_MISSING_CLASSES
    | tax.MASK_PRESENT_CLASSES
    | tax.MASK_MISSING_CLASSES
)


def _dedup(dets: Iterable[Detection]) -> list[Detection]:
    """Drop exact duplicate detections (same class and bbox).

    Guards against the same object being reported by both models at identical
    coordinates. Approximate duplicates are a fusion concern handled upstream.
    """
    seen: set[tuple[str, BBox]] = set()
    out: list[Detection] = []
    for d in dets:
        key = (d.class_name, d.bbox)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _associates(det_bbox: BBox, person_bbox: BBox) -> bool:
    return (
        overlap_ratio(det_bbox, person_bbox) >= _PPE_OVERLAP_MIN
        or iou(det_bbox, person_bbox) >= _PPE_IOU_MIN
    )


def _contains(container_bbox: BBox, object_bbox: BBox) -> bool:
    """True if ``object_bbox`` sits inside ``container_bbox`` (person on scaffold)."""
    return (
        overlap_ratio(object_bbox, container_bbox) >= _SCAFFOLD_OVERLAP_MIN
        or iou(object_bbox, container_bbox) >= _SCAFFOLD_IOU_MIN
    )


def _ppe_state(
    classes: set[str],
    present: frozenset[str],
    missing: frozenset[str],
    *,
    strict_absence: bool = False,
) -> str:
    if classes & missing:
        return "missing"
    if classes & present:
        return "present"
    return "missing" if strict_absence else "unknown"


def _penalty(item: str, state: str, on_scaffold: bool) -> int:
    weights = PPE_WEIGHTS[item]
    context = "scaffold" if on_scaffold else "ground"
    if state == "present":
        return 0
    if state == "missing":
        return weights[context]
    # unknown -> half value, floored
    return max(UNKNOWN_FLOOR[item], weights[context] // 2)


def analyze_person(
    person_bbox: BBox,
    index: int,
    ppe_dets: Sequence[Detection],
    *,
    on_scaffold: bool = False,
) -> PersonAssessment:
    """Evaluate one person's PPE against the detections that belong to them."""
    associated: set[str] = {
        d.class_name for d in ppe_dets if _associates(d.bbox, person_bbox)
    }

    states = {
        "helmet": _ppe_state(
            associated, tax.HELMET_PRESENT_CLASSES, tax.HELMET_MISSING_CLASSES
        ),
        "vest": _ppe_state(
            associated,
            tax.VEST_PRESENT_CLASSES,
            tax.VEST_MISSING_CLASSES,
            strict_absence=True,
        ),
        "glove": _ppe_state(
            associated, tax.GLOVE_PRESENT_CLASSES, tax.GLOVE_MISSING_CLASSES
        ),
        "mask": _ppe_state(
            associated, tax.MASK_PRESENT_CLASSES, tax.MASK_MISSING_CLASSES
        ),
    }

    reasons: list[str] = []
    score = 0
    for item, state in states.items():
        pts = _penalty(item, state, on_scaffold)
        score += pts
        if state == "missing":
            reasons.append(f"missing {item}: +{pts}")
        elif state == "unknown":
            reasons.append(f"unknown {item}: +{pts}")

    score = min(MAX_PERSON_SCORE, score)

    has_missing = any(s == "missing" for s in states.values())
    has_present = any(s == "present" for s in states.values())
    if has_missing:
        status = "VIOLATION"
    elif has_present:
        status = "SAFE" if all(s == "present" for s in states.values()) else "PARTIAL"
    else:
        status = "UNKNOWN"

    return PersonAssessment(
        index=index,
        bbox=person_bbox,
        on_scaffold=on_scaffold,
        helmet=states["helmet"],
        vest=states["vest"],
        glove=states["glove"],
        mask=states["mask"],
        status=status,
        score=score,
        reasons=tuple(reasons),
    )


def compute_scaffolding_risk(
    scaffold_dets: Sequence[Detection],
    person_boxes: Sequence[BBox],
) -> tuple[int, list[str]]:
    """Return (score, reasons) for scaffold risk, capped at 35."""
    raw = 0
    reasons: list[str] = []
    for det in scaffold_dets:
        cls = det.class_name
        has_worker = any(_contains(det.bbox, pb) for pb in person_boxes)
        if cls in tax.SCAFFOLD_SUSPENDED_CLASSES:
            pts = SCAFFOLD_POINTS["suspended"]
            bonus = SCAFFOLD_WORKER_BONUS if has_worker else 0
            raw += pts + bonus
            reasons.append(f"suspended scaffold +{pts + bonus}")
        elif cls in tax.SCAFFOLD_HIGH_CLASSES:
            pts = SCAFFOLD_POINTS["high"]
            bonus = SCAFFOLD_WORKER_BONUS if has_worker else 0
            raw += pts + bonus
            reasons.append(f"high-risk scaffold +{pts + bonus}")
        elif cls in tax.SCAFFOLD_LOW_CLASSES:
            raw += SCAFFOLD_POINTS["low"]
            reasons.append(f"low-risk scaffold +{SCAFFOLD_POINTS['low']}")
    return min(CAP_SCAFFOLD, raw), reasons


def compute_safety(detections: Iterable[Detection]) -> SafetyResult:
    """Score a single frame from raw detections."""
    dets = _dedup(detections)

    persons = [d for d in dets if d.class_name in tax.PERSON_CLASSES]
    scaffolds = [d for d in dets if d.class_name in tax.SCAFFOLD_CLASSES]
    ppe_dets = [d for d in dets if d.class_name in _PPE_CLASSES]

    person_boxes = [p.bbox for p in persons]
    elevated_scaffolds = [d for d in scaffolds if d.class_name in _ON_SCAFFOLD_CLASSES]

    assessments: list[PersonAssessment] = []
    for i, person in enumerate(persons, start=1):
        on_scaffold = any(_contains(scaffold.bbox, person.bbox) for scaffold in elevated_scaffolds)
        assessments.append(
            analyze_person(person.bbox, i, ppe_dets, on_scaffold=on_scaffold)
        )

    if assessments:
        mean_score = sum(a.score for a in assessments) / len(assessments)
        ppe_risk = min(CAP_PPE, round(mean_score * CAP_PPE / MAX_PERSON_SCORE))
    else:
        ppe_risk = 0

    scaffold_risk, _ = compute_scaffolding_risk(scaffolds, person_boxes)

    n_violators = sum(1 for a in assessments if a.status == "VIOLATION")
    crowd_risk = min(CAP_CROWD, max(0, n_violators - 1) * 5)

    danger = min(100, ppe_risk + scaffold_risk + crowd_risk)
    safety = 100 - danger

    return SafetyResult(
        safety=safety,
        danger=danger,
        level=score_to_level(safety),
        ppe_risk=ppe_risk,
        scaffold_risk=scaffold_risk,
        crowd_risk=crowd_risk,
        persons=tuple(assessments),
    )
