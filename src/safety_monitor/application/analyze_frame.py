"""Use-case: analyze a single frame.

    frame --[Detector]--> detections --[domain]--> SafetyResult

Pure orchestration: no I/O, no rendering. Callers (transport) decide how to
present the result and whether to annotate the image.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain import Detection, SafetyResult, compute_safety
from .detector import Detector, Frame


@dataclass(frozen=True)
class FrameAnalysis:
    """Everything derived from one frame, independent of presentation."""

    result: SafetyResult
    detections: tuple[Detection, ...]


def analyze_frame(detector: Detector, frame: Frame) -> FrameAnalysis:
    """Run detection then scoring for a single frame."""
    detections = tuple(detector.detect(frame))
    return FrameAnalysis(result=compute_safety(detections), detections=detections)
