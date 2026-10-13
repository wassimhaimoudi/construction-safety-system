"""Tests for the Ultralytics adapter (no torch: uses a fake model object)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from safety_monitor.infrastructure.yolo import YOLODetector


class FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]


class FakeModel:
    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=self._boxes)]


def test_maps_model_output_to_detections():
    model = FakeModel(
        {0: "person", 1: "helmet"},
        [FakeBox(0, 0.9, [1.2, 2.6, 10.4, 20.5]), FakeBox(1, 0.75, [3, 4, 5, 6])],
    )
    detector = YOLODetector(model)

    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert [d.class_name for d in detections] == ["person", "helmet"]
    assert detections[0].bbox == (1, 3, 10, 20)  # rounded
    assert detections[0].confidence == 0.9
    assert detections[1].bbox == (3, 4, 5, 6)


def test_forwards_inference_parameters():
    model = FakeModel({0: "person"}, [])
    detector = YOLODetector(model, confidence=0.5, image_size=320, device="cpu")

    detector.detect("frame")

    call = model.calls[0]
    assert call["conf"] == 0.5
    assert call["imgsz"] == 320
    assert call["device"] == "cpu"
    assert call["verbose"] is False


def test_empty_results_returns_empty_list():
    class NoResults:
        def predict(self, **kwargs):
            return []

    assert YOLODetector(NoResults()).detect("f") == []


def test_result_without_boxes_returns_empty_list():
    class NoBoxes:
        names = {0: "person"}

        def predict(self, **kwargs):
            return [SimpleNamespace(boxes=None)]

    assert YOLODetector(NoBoxes()).detect("f") == []


def test_names_is_a_copy():
    model = FakeModel({0: "person"}, [])
    detector = YOLODetector(model)

    names = detector.names
    names[99] = "mutated"

    assert 99 not in model.names
