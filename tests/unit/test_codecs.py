"""Tests for image codecs."""

from __future__ import annotations

import numpy as np
import pytest

from safety_monitor.infrastructure.codecs import (
    InvalidImageError,
    decode_image,
    encode_jpeg,
    encode_png,
)


def _sample_image():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[5:10, 5:15] = (0, 255, 0)
    return image


def test_png_round_trip_preserves_shape():
    decoded = decode_image(encode_png(_sample_image()))
    assert decoded.shape == (20, 30, 3)
    assert decoded.dtype == np.uint8


def test_jpeg_round_trip_returns_bytes():
    data = encode_jpeg(_sample_image())
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert decode_image(data).shape == (20, 30, 3)


def test_empty_payload_is_rejected():
    with pytest.raises(InvalidImageError):
        decode_image(b"")


def test_malformed_payload_is_rejected():
    with pytest.raises(InvalidImageError):
        decode_image(b"definitely not an image")
