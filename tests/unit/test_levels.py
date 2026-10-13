"""Unit tests for safety level bands."""

from __future__ import annotations

import pytest

from safety_monitor.domain.levels import score_to_level


@pytest.mark.parametrize(
    "score, expected",
    [
        (100, "SAFE"),
        (81, "SAFE"),
        (80, "LOW"),
        (51, "LOW"),
        (50, "MEDIUM"),
        (21, "MEDIUM"),
        (20, "HIGH"),
        (1, "HIGH"),
        (0, "CRITICAL"),
    ],
)
def test_level_bands(score, expected):
    assert score_to_level(score) == expected
