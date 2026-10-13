"""The ``Detector`` port.

The application depends on this Protocol, never on a concrete model runtime.
``Frame`` is intentionally opaque: the application never inspects pixels, so it
stays free of numpy/cv2 imports.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..domain import Detection

#: An opaque single frame handled by a detector adapter.
Frame = Any


@runtime_checkable
class Detector(Protocol):
    """Returns raw detections for a single frame."""

    def detect(self, frame: Frame) -> Sequence[Detection]:
        ...
