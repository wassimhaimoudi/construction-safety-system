"""Reusable test doubles and builders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from safety_monitor.domain import Detection


def det(class_name: str, bbox, confidence: float = 0.9) -> Detection:
    return Detection(class_name=class_name, confidence=confidence, bbox=tuple(bbox))


def compliant_person(x: int = 0):
    """A fully PPE-compliant worker located at horizontal offset ``x``."""
    return [
        det("person", (x + 0, 0, x + 100, 200)),
        det("helmet", (x + 25, 5, x + 75, 45)),
        det("vest", (x + 15, 60, x + 85, 150)),
        det("glove", (x + 5, 60, x + 20, 95)),
        det("mask", (x + 35, 10, x + 65, 35)),
    ]


class FakeDetector:
    """A :class:`~safety_monitor.application.Detector` returning fixed output."""

    def __init__(self, detections: Sequence[Detection] = ()):
        self._detections = list(detections)
        self.calls: list[Any] = []

    def detect(self, frame: Any) -> Sequence[Detection]:
        self.calls.append(frame)
        return list(self._detections)
