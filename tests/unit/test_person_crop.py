"""Tests for person cropping."""

from __future__ import annotations

import numpy as np

from safety_monitor.infrastructure.person_crop import crop_persons
from tests.support.fakes import det


def _image(height=200, width=300):
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def test_crops_persons_in_order_with_one_based_indices():
    image = _image()
    dets = [
        det("person", (0, 0, 50, 100)),
        det("helmet", (10, 10, 30, 30)),
        det("person", (100, 0, 150, 100)),
    ]

    crops = crop_persons(image, dets, padding=0.0)

    assert [index for index, _ in crops] == [1, 2]
    assert crops[0][1].shape == (100, 50, 3)
    assert crops[1][1].shape == (100, 50, 3)


def test_ignores_non_person_detections():
    image = _image()
    dets = [det("Barrier", (0, 0, 10, 10)), det("helmet", (0, 0, 5, 5))]

    assert crop_persons(image, dets) == []


def test_clamps_boxes_to_image_bounds():
    image = _image(height=100, width=100)
    dets = [det("person", (-20, -20, 500, 500))]

    crops = crop_persons(image, dets, padding=0.0)

    assert len(crops) == 1
    assert crops[0][1].shape == (100, 100, 3)


def test_padding_expands_the_crop():
    image = _image(height=200, width=200)
    dets = [det("person", (50, 50, 100, 100))]

    crop = crop_persons(image, dets, padding=0.2)[0][1]

    # 50x50 box padded by 20% (10px each side) -> 70x70.
    assert crop.shape == (70, 70, 3)
