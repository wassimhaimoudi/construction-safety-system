"""Expected class lists of the shipped checkpoints.

Read directly from the weights via ``ultralytics.YOLO(...).names``. Used by both
the taxonomy guard tests and the integration smoke test.
"""

from __future__ import annotations

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
