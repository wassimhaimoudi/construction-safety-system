"""Image encode/decode helpers (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a decodable image."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into a BGR ``uint8`` array.

    Raises :class:`InvalidImageError` on malformed input.
    """
    if not data:
        raise InvalidImageError("empty image payload")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("could not decode image")
    return image


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise InvalidImageError("failed to encode PNG")
    return buffer.tobytes()


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise InvalidImageError("failed to encode JPEG")
    return buffer.tobytes()
