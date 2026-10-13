"""Unit tests for geometric helpers."""

from __future__ import annotations

import pytest

from safety_monitor.domain.geometry import iou, overlap_ratio


def test_iou_identical_is_one():
    box = (0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping over a 5x10 region: inter=50, union=150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_overlap_ratio_fully_contained_is_one():
    inner = (2, 2, 8, 8)
    outer = (0, 0, 10, 10)
    assert overlap_ratio(inner, outer) == 1.0


def test_overlap_ratio_is_asymmetric():
    inner = (0, 0, 10, 10)
    outer = (5, 0, 15, 10)
    # inner half-covered by outer -> 0.5
    assert overlap_ratio(inner, outer) == pytest.approx(0.5)
    # outer is fully covered by the (0,0,10,10)? no: outer spans x=5..15, only
    # x=5..10 overlaps -> 5/10 = 0.5 as well. Use an asymmetric pair instead.
    big = (0, 0, 100, 100)
    small = (0, 0, 10, 10)
    assert overlap_ratio(small, big) == 1.0
    assert overlap_ratio(big, small) == pytest.approx(0.01)


def test_overlap_ratio_disjoint_is_zero():
    assert overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
