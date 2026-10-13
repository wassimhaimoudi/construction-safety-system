"""Tests for the analyze_frame use-case."""

from __future__ import annotations

from safety_monitor.application import analyze_frame
from safety_monitor.application.detector import Detector
from tests.support.fakes import FakeDetector, compliant_person, det


def test_empty_detections_yield_perfect_safety():
    analysis = analyze_frame(FakeDetector([]), frame=object())

    assert analysis.detections == ()
    assert analysis.result.safety == 100
    assert analysis.result.danger == 0
    assert analysis.result.level == "SAFE"
    assert analysis.result.persons == ()


def test_applies_domain_scoring_to_detections():
    dets = compliant_person()
    dets[1] = det("head_nohelmet", (25, 5, 75, 45))

    analysis = analyze_frame(FakeDetector(dets), frame=object())

    assert analysis.result.ppe_risk == 17
    assert analysis.result.safety == 83
    assert analysis.result.persons[0].status == "VIOLATION"


def test_passes_frame_through_to_detector():
    frame = object()
    detector = FakeDetector([])

    analyze_frame(detector, frame)

    assert detector.calls == [frame]


def test_fake_detector_satisfies_port_protocol():
    assert isinstance(FakeDetector([]), Detector)
