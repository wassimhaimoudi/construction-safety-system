"""Detection class taxonomy, corrected to the real model outputs.

The two model files are swapped relative to their names (see PLAN.md):
  - ``scaffold.pt``       -> ``scaffold_risk_v3``  (scaffold + environment classes)
  - ``ppe.pt`` -> ``ppe_yolov11``       (person + PPE classes)

This module defines the canonical class sets the scorer understands. Class
names are matched exactly (case-sensitive) against
``ultralytics`` model ``names`` values.
"""

from __future__ import annotations

# Person
PERSON_CLASSES = frozenset({"person"})

# Helmet
HELMET_PRESENT_CLASSES = frozenset({"helmet", "head_helmet"})
HELMET_MISSING_CLASSES = frozenset({"head_nohelmet"})

# Vest (no explicit negative class; absence is treated as missing)
VEST_PRESENT_CLASSES = frozenset({"vest"})
VEST_MISSING_CLASSES = frozenset()

# Gloves
GLOVE_PRESENT_CLASSES = frozenset({"glove", "hand_glove"})
GLOVE_MISSING_CLASSES = frozenset({"hand_noglove"})

# Mask
MASK_PRESENT_CLASSES = frozenset({"mask", "face_mask"})
MASK_MISSING_CLASSES = frozenset({"face_nomask"})

# Anatomy-only detections (never scored)
ANATOMY_CLASSES = frozenset({"face", "hand", "head"})

# Scaffold risk
SCAFFOLD_SUSPENDED_CLASSES = frozenset({"High_Risk_Suspended_Scaffolding"})
SCAFFOLD_HIGH_CLASSES = frozenset({"High_Risk_Scaffolding"})
SCAFFOLD_LOW_CLASSES = frozenset({"Low_Risk_Scaffolding"})
SCAFFOLD_CLASSES = (
    SCAFFOLD_SUSPENDED_CLASSES | SCAFFOLD_HIGH_CLASSES | SCAFFOLD_LOW_CLASSES
)

# Environmental infrastructure (display only; does not affect the score)
DANGER_ZONE_CLASSES = frozenset({"Danger_Zone"})
BARRIER_CLASSES = frozenset({"Barrier"})
CONE_CLASSES = frozenset({"Cone"})
WORK_SIGN_CLASSES = frozenset({"Work_Sign"})
INFRA_CLASSES = (
    DANGER_ZONE_CLASSES | BARRIER_CLASSES | CONE_CLASSES | WORK_SIGN_CLASSES
)
