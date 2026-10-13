"""Golden tests for the safety scoring spec (Phase 1: frozen spec).

These encode the *contract* of :func:`safety_monitor.domain.compute_safety`
before the implementation exists. They are intentionally written first (red)
and made green in Phase 2.

Frozen formula
--------------
    Danger  = min(100, PPE[50] + Scaffold[35] + Crowd[15])
    Safety  = 100 - Danger

    Level (safety direction):
        81-100 SAFE | 51-80 LOW | 21-50 MEDIUM | 1-20 HIGH | 0 CRITICAL

Frozen per-person PPE penalties (ground / scaffold), person score capped at 45:
    helmet 15/25 | vest 10/20 | glove 8/12 | mask 4/6
    - "missing" = explicit negative class (or vest absence, strict policy)
    - "unknown" = neither present nor negative class -> half penalty, floored

Frozen taxonomy highlights (see PLAN.md):
    helmet present : helmet, head_helmet     missing : head_nohelmet
    vest present   : vest                     missing : (absence = missing)
    glove present  : glove, hand_glove        missing : hand_noglove
    mask present   : mask, face_mask          missing : face_nomask
    scaffold       : High_Risk_Scaffolding, High_Risk_Suspended_Scaffolding,
                     Low_Risk_Scaffolding
"""

from __future__ import annotations

from safety_monitor.domain import Detection, compute_safety


def det(class_name: str, bbox, confidence: float = 0.9) -> Detection:
    return Detection(class_name=class_name, confidence=confidence, bbox=tuple(bbox))


def compliant_person(x: int = 0):
    """A fully PPE-compliant worker located at horizontal offset ``x``."""
    return [
        det("person", (x + 0, 0, x + 100, 200)),
        det("helmet", (x + 25, 5, x + 75, 45)),
        det("vest", (x + 15, 60, x + 85, 150)),
        det("glove", (x + 5, 60, x + 20, 95)),
        det("mask", (x + 35, 10, x + 65, 35)),
    ]


# --------------------------------------------------------------------------
# Example 1 — compliant worker, empty environment
# --------------------------------------------------------------------------
def test_compliant_worker_is_safe():
    result = compute_safety(compliant_person())

    assert result.ppe_risk == 0
    assert result.scaffold_risk == 0
    assert result.crowd_risk == 0
    assert result.danger == 0
    assert result.safety == 100
    assert result.level == "SAFE"
    assert result.persons[0].status == "SAFE"


# --------------------------------------------------------------------------
# Example 2 — missing helmet on the ground
# --------------------------------------------------------------------------
def test_missing_helmet_ground():
    dets = compliant_person()
    dets[1] = det("head_nohelmet", (25, 5, 75, 45))  # replace helmet present

    result = compute_safety(dets)

    # person score = 15 -> ppe_risk = round(15 * 50/45) = 17
    assert result.ppe_risk == 17
    assert result.danger == 17
    assert result.safety == 83
    assert result.level == "SAFE"
    assert result.persons[0].status == "VIOLATION"


# --------------------------------------------------------------------------
# Example 3 — missing helmet + vest on a high-risk scaffold
# --------------------------------------------------------------------------
def test_missing_helmet_and_vest_on_high_scaffold():
    dets = [
        det("person", (0, 0, 100, 200)),
        det("head_nohelmet", (25, 5, 75, 45)),
        det("glove", (5, 60, 20, 95)),
        det("mask", (35, 10, 65, 35)),
        det("High_Risk_Scaffolding", (0, 0, 120, 220)),
    ]

    result = compute_safety(dets)

    # person score = helmet 25 (scaffold) + vest 20 (scaffold, strict) = 45
    # ppe_risk = round(45 * 50/45) = 50
    # scaffold_risk = high 20 + worker bonus 10 = 30
    assert result.ppe_risk == 50
    assert result.scaffold_risk == 30
    assert result.crowd_risk == 0
    assert result.danger == 80
    assert result.safety == 20
    assert result.level == "HIGH"
    assert result.persons[0].status == "VIOLATION"


# --------------------------------------------------------------------------
# Example 4 — three simultaneous helmet violations (crowd risk)
# --------------------------------------------------------------------------
def test_crowd_of_three_helmet_violations():
    dets = []
    for x in (0, 300, 600):
        dets.append(det("person", (x, 0, x + 100, 200)))
        dets.append(det("head_nohelmet", (x + 25, 5, x + 75, 45)))
        dets.append(det("vest", (x + 15, 60, x + 85, 150)))
        dets.append(det("glove", (x + 5, 60, x + 20, 95)))
        dets.append(det("mask", (x + 35, 10, x + 65, 35)))

    result = compute_safety(dets)

    assert result.ppe_risk == 17
    assert result.scaffold_risk == 0
    assert result.crowd_risk == 10  # (3 - 1) * 5
    assert result.danger == 27
    assert result.safety == 73
    assert result.level == "LOW"
    assert [p.status for p in result.persons] == ["VIOLATION"] * 3


# --------------------------------------------------------------------------
# Example 5 — compliant worker on a suspended scaffold
# --------------------------------------------------------------------------
def test_compliant_worker_on_suspended_scaffold():
    dets = compliant_person() + [
        det("High_Risk_Suspended_Scaffolding", (0, 0, 120, 220))
    ]

    result = compute_safety(dets)

    assert result.ppe_risk == 0
    assert result.scaffold_risk == 35  # suspended 35 + bonus 10, capped at 35
    assert result.crowd_risk == 0
    assert result.danger == 35
    assert result.safety == 65
    assert result.level == "LOW"
    assert result.persons[0].status == "SAFE"


# --------------------------------------------------------------------------
# Example 6 — compliant worker near a low-risk scaffold
# --------------------------------------------------------------------------
def test_compliant_worker_on_low_scaffold():
    dets = compliant_person() + [det("Low_Risk_Scaffolding", (0, 0, 120, 220))]

    result = compute_safety(dets)

    assert result.scaffold_risk == 5
    assert result.danger == 5
    assert result.safety == 95
    assert result.level == "SAFE"
    assert result.persons[0].status == "SAFE"
