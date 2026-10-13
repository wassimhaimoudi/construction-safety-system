"""Encode images as base64 data URLs for JSON transport."""

from __future__ import annotations

import base64

import numpy as np

from ..infrastructure.codecs import encode_jpeg


def to_data_url(image: np.ndarray, *, quality: int = 90) -> str:
    payload = encode_jpeg(image, quality=quality)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
