"""Ultralytics-backed detector adapter.

Wraps a single ``ultralytics.YOLO`` model and maps its output to domain
:class:`Detection` objects. The heavy import is deferred to :func:`load_yolo`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..domain import Detection
from ..domain.geometry import BBox


class YOLODetector:
    """Adapts one Ultralytics model to the :class:`Detector` port."""

    def __init__(
        self,
        model: Any,
        *,
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._confidence = confidence
        self._image_size = image_size
        self._device = device

    @property
    def names(self) -> dict[int, str]:
        return dict(self._model.names)

    def detect(self, frame: Any) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=self._image_size,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        names = self._model.names
        detections: list[Detection] = []
        for box in boxes:
            class_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            bbox: BBox = (
                int(round(xyxy[0])),
                int(round(xyxy[1])),
                int(round(xyxy[2])),
                int(round(xyxy[3])),
            )
            detections.append(
                Detection(
                    class_name=names[class_id],
                    confidence=float(box.conf[0]),
                    bbox=bbox,
                )
            )
        return detections


def load_yolo(
    path: str | Path,
    *,
    confidence: float = 0.35,
    image_size: int = 640,
    device: str = "cpu",
) -> YOLODetector:
    """Load a YOLO checkpoint into a :class:`YOLODetector`."""
    from ultralytics import YOLO  # heavy import, deferred on purpose

    model = YOLO(str(path))
    return YOLODetector(
        model,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )
