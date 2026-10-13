"""Render detections and the safety verdict onto a frame (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from ..application import FrameAnalysis
from ..domain import taxonomy as tax
from ..domain.geometry import BBox

# BGR colors.
STATUS_COLORS = {
    "SAFE": (0, 200, 0),
    "PARTIAL": (0, 165, 255),
    "VIOLATION": (0, 0, 255),
    "UNKNOWN": (150, 150, 150),
}
LEVEL_COLORS = {
    "SAFE": (0, 200, 0),
    "LOW": (0, 200, 200),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 60, 255),
    "CRITICAL": (0, 0, 180),
}
_SCAFFOLD_COLOR = (0, 0, 255)
_INFRA_COLOR = (255, 200, 0)
_PPE_COLOR = (0, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(0, min(int(bbox[2]), width - 1))
    y2 = max(0, min(int(bbox[3]), height - 1))
    return x1, y1, x2, y2


def _label(canvas: np.ndarray, text: str, x: int, y: int, color) -> None:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.5, 1)
    y = max(th + 4, y)
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + baseline), color, -1)
    cv2.putText(
        canvas,
        text,
        (x + 2, y - 2),
        _FONT,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw(canvas, bbox: BBox, color, label: str, thickness: int) -> None:
    h, w = canvas.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    if label:
        _label(canvas, label, x1, y1, color)


def _banner(canvas: np.ndarray, analysis: FrameAnalysis) -> None:
    result = analysis.result
    color = LEVEL_COLORS.get(result.level, (200, 200, 200))
    text = (
        f"SAFETY {result.safety}/100  {result.level}"
        f"   danger {result.danger} (ppe {result.ppe_risk}"
        f" + scaffold {result.scaffold_risk} + crowd {result.crowd_risk})"
    )
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.6, 2)
    cv2.rectangle(canvas, (0, 0), (tw + 16, th + baseline + 12), color, -1)
    cv2.putText(
        canvas,
        text,
        (8, th + 6),
        _FONT,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def annotate(
    image: np.ndarray,
    analysis: FrameAnalysis,
    *,
    draw_ppe: bool = True,
    draw_infra: bool = True,
) -> np.ndarray:
    """Return a copy of ``image`` with boxes, labels and a verdict banner."""
    canvas = image.copy()
    assessment_by_bbox = {p.bbox: p for p in analysis.result.persons}

    for det in analysis.detections:
        cls = det.class_name
        if cls in tax.PERSON_CLASSES:
            assessment = assessment_by_bbox.get(det.bbox)
            color = STATUS_COLORS.get(
                assessment.status if assessment else "UNKNOWN", (200, 200, 200)
            )
            label = (
                f"P{assessment.index} {assessment.status}"
                if assessment
                else "person"
            )
            _draw(canvas, det.bbox, color, label, 2)
        elif cls in tax.SCAFFOLD_CLASSES:
            _draw(canvas, det.bbox, _SCAFFOLD_COLOR, cls, 2)
        elif draw_infra and cls in tax.INFRA_CLASSES:
            _draw(canvas, det.bbox, _INFRA_COLOR, cls, 1)
        elif draw_ppe and cls in tax.ANATOMY_CLASSES:
            continue
        elif draw_ppe:
            _draw(canvas, det.bbox, _PPE_COLOR, cls, 1)

    _banner(canvas, analysis)
    return canvas
