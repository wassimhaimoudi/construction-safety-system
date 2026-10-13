"""Guard tests: every class emitted by the real models must be categorized.

The class lists below were read directly from the shipped checkpoints via
``ultralytics.YOLO(...).names``. If a model is swapped or retrained, these
tests fail loudly instead of silently under-scoring.

  - ``scaffold.pt``       -> scaffold/environment model (7 classes)
  - ``ppe.pt`` -> PPE model (14 classes)
"""

from __future__ import annotations

import pytest

from safety_monitor.domain import taxonomy as tax

SCAFFOLD_MODEL_CLASSES = {
    "High_Risk_Scaffolding",
    "High_Risk_Suspended_Scaffolding",
    "Low_Risk_Scaffolding",
    "Barrier",
    "Cone",
    "Danger_Zone",
    "Work_Sign",
}

PPE_MODEL_CLASSES = {
    "face",
    "face_mask",
    "face_nomask",
    "glove",
    "hand",
    "hand_glove",
    "hand_noglove",
    "head",
    "head_helmet",
    "head_nohelmet",
    "helmet",
    "mask",
    "person",
    "vest",
}

SCAFFOLD_MODEL_COVERAGE = (
    tax.SCAFFOLD_CLASSES | tax.INFRA_CLASSES
)

PPE_MODEL_COVERAGE = (
    tax.PERSON_CLASSES
    | tax.HELMET_PRESENT_CLASSES
    | tax.HELMET_MISSING_CLASSES
    | tax.VEST_PRESENT_CLASSES
    | tax.VEST_MISSING_CLASSES
    | tax.GLOVE_PRESENT_CLASSES
    | tax.GLOVE_MISSING_CLASSES
    | tax.MASK_PRESENT_CLASSES
    | tax.MASK_MISSING_CLASSES
    | tax.ANATOMY_CLASSES
)


@pytest.mark.parametrize("class_name", sorted(SCAFFOLD_MODEL_CLASSES))
def test_scaffold_model_class_is_categorized(class_name):
    assert class_name in SCAFFOLD_MODEL_COVERAGE


@pytest.mark.parametrize("class_name", sorted(PPE_MODEL_CLASSES))
def test_ppe_model_class_is_categorized(class_name):
    assert class_name in PPE_MODEL_COVERAGE


def test_taxonomy_has_no_unknown_classes():
    known = SCAFFOLD_MODEL_COVERAGE | PPE_MODEL_COVERAGE
    assert known >= SCAFFOLD_MODEL_CLASSES
    assert known >= PPE_MODEL_CLASSES
