"""Combine the two models into a single :class:`Detector`."""

from __future__ import annotations

from typing import Any

from ..application.detector import Detector
from ..config import Settings
from ..domain import Detection
from .yolo import YOLODetector, load_yolo


class EnsembleDetector:
    """Runs several detectors over one frame and concatenates their output.

    Cross-model duplicate suppression happens in the domain scorer, which
    de-duplicates identical (class, bbox) pairs.
    """

    def __init__(self, *detectors: Detector) -> None:
        if not detectors:
            raise ValueError("EnsembleDetector requires at least one detector")
        self._detectors = detectors

    @property
    def detectors(self) -> tuple[Detector, ...]:
        return self._detectors

    def detect(self, frame: Any) -> list[Detection]:
        detections: list[Detection] = []
        for detector in self._detectors:
            detections.extend(detector.detect(frame))
        return detections


def build_detector(settings: Settings) -> EnsembleDetector:
    """Wire the PPE + scaffold models into an :class:`EnsembleDetector`."""
    ppe = load_yolo(
        settings.ppe_model_path,
        confidence=settings.confidence,
        image_size=settings.image_size,
        device=settings.device,
    )
    scaffold = load_yolo(
        settings.scaffold_model_path,
        confidence=settings.confidence,
        image_size=settings.image_size,
        device=settings.device,
    )
    return EnsembleDetector(ppe, scaffold)
