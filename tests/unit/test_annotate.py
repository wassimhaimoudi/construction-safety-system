"""Tests for annotation rendering."""

from __future__ import annotations

import numpy as np

from safety_monitor.application import analyze_frame
from safety_monitor.infrastructure.annotate import annotate
from tests.support.fakes import FakeDetector, compliant_person, det


def _image():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_returns_same_shape_and_dtype():
    image = _image()
    analysis = analyze_frame(FakeDetector(compliant_person()), image)

    canvas = annotate(image, analysis)

    assert canvas.shape == image.shape
    assert canvas.dtype == image.dtype


def test_does_not_mutate_the_input():
    image = _image()
    original = image.copy()
    analysis = analyze_frame(FakeDetector(compliant_person()), image)

    annotate(image, analysis)

    assert np.array_equal(image, original)


def test_draws_something_for_empty_detections():
    image = _image()
    analysis = analyze_frame(FakeDetector([]), image)

    canvas = annotate(image, analysis)

    assert not np.array_equal(canvas, image)  # banner is drawn


def test_out_of_bounds_boxes_are_clamped_without_crashing():
    image = _image()
    dets = [det("person", (-50, -50, 5000, 5000)), det("Barrier", (-10, -10, 9999, 9999))]
    analysis = analyze_frame(FakeDetector(dets), image)

    canvas = annotate(image, analysis)

    assert canvas.shape == image.shape
