"""Tests for the ensemble detector (uses the pure FakeDetector)."""

from __future__ import annotations

import pytest

from safety_monitor.infrastructure.ensemble import EnsembleDetector
from tests.support.fakes import FakeDetector, det


def test_requires_at_least_one_detector():
    with pytest.raises(ValueError):
        EnsembleDetector()


def test_concatenates_outputs_in_order():
    ppe = FakeDetector([det("person", (0, 0, 10, 10))])
    scaffold = FakeDetector([det("High_Risk_Scaffolding", (0, 0, 20, 20))])

    result = EnsembleDetector(ppe, scaffold).detect("frame")

    assert [d.class_name for d in result] == ["person", "High_Risk_Scaffolding"]


def test_calls_every_detector_with_same_frame():
    ppe = FakeDetector([])
    scaffold = FakeDetector([])

    EnsembleDetector(ppe, scaffold).detect("frame")

    assert ppe.calls == ["frame"]
    assert scaffold.calls == ["frame"]
