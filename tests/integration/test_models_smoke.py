"""Integration smoke test against the real checkpoints.

Excluded from the default run (see pyproject ``addopts``). Run explicitly with:

    .venv/bin/python -m pytest -m integration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from safety_monitor.application import analyze_frame
from safety_monitor.infrastructure.ensemble import EnsembleDetector
from safety_monitor.infrastructure.yolo import load_yolo
from tests.support.model_classes import PPE_MODEL_CLASSES, SCAFFOLD_MODEL_CLASSES

ROOT = Path(__file__).resolve().parents[2]
PPE_PATH = ROOT / "ppe.pt"
SCAFFOLD_PATH = ROOT / "scaffold.pt"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def detector():
    if not PPE_PATH.exists() or not SCAFFOLD_PATH.exists():
        pytest.skip("model weights not present")
    ppe = load_yolo(PPE_PATH, confidence=0.35, image_size=640, device="cpu")
    scaffold = load_yolo(SCAFFOLD_PATH, confidence=0.35, image_size=640, device="cpu")
    return EnsembleDetector(ppe, scaffold)


def test_real_models_expose_expected_classes(detector):
    ppe, scaffold = detector.detectors
    assert set(ppe.names.values()) == PPE_MODEL_CLASSES
    assert set(scaffold.names.values()) == SCAFFOLD_MODEL_CLASSES


def test_blank_frame_runs_end_to_end(detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    analysis = analyze_frame(detector, frame)

    assert 0 <= analysis.result.safety <= 100
    assert analysis.result.safety == 100  # blank frame -> no detections
    assert analysis.result.persons == ()
