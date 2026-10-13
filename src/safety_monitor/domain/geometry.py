"""Geometric helpers for associating detections.

Bounding boxes are ``(x1, y1, x2, y2)`` in pixels.

``overlap_ratio(inner, outer)`` is *asymmetric*: it returns the fraction of the
``inner`` box's area covered by its intersection with ``outer``. Callers must
pass arguments in the intended direction:

  * PPE belongs to a person  -> ``overlap_ratio(ppe_box, person_box)``
  * person stands on scaffold -> ``overlap_ratio(person_box, scaffold_box)``
"""

from __future__ import annotations

BBox = tuple[int, int, int, int]


def _area(box: BBox) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _inter_area(a: BBox, b: BBox) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two boxes."""
    inter = _inter_area(a, b)
    if inter == 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def overlap_ratio(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    area = _area(inner)
    if area <= 0:
        return 0.0
    return _inter_area(inner, outer) / area
