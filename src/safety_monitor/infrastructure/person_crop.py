"""Crop person regions out of a frame for gallery/preview output."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..domain import Detection
from ..domain import taxonomy as tax


def crop_persons(
    image: np.ndarray,
    detections: Sequence[Detection],
    *,
    padding: float = 0.05,
) -> list[tuple[int, np.ndarray]]:
    """Return ``(person_index, crop)`` pairs in detection order.

    ``person_index`` is 1-based to match :class:`PersonAssessment.index`.
    Boxes are clamped to the image bounds; degenerate crops are skipped.
    """
    height, width = image.shape[:2]
    crops: list[tuple[int, np.ndarray]] = []
    index = 0
    for det in detections:
        if det.class_name not in tax.PERSON_CLASSES:
            continue
        index += 1
        x1, y1, x2, y2 = det.bbox
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append((index, image[y1:y2, x1:x2]))
    return crops
