"""Safety-score level bands.

The scorer produces ``safety`` (higher is safer). Level bands use the same cut
points as the legacy danger bands, applied to the safety score:

    81-100 SAFE | 51-80 LOW | 21-50 MEDIUM | 1-20 HIGH | 0 CRITICAL
"""

from __future__ import annotations

SAFE = "SAFE"
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"


def score_to_level(safety: int) -> str:
    """Map a safety score (0-100) to a level name."""
    if safety <= 0:
        return CRITICAL
    if safety <= 20:
        return HIGH
    if safety <= 50:
        return MEDIUM
    if safety <= 80:
        return LOW
    return SAFE
