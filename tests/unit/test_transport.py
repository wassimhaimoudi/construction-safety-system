"""Transport tests using FastAPI TestClient and a FakeDetector (no torch)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from safety_monitor.config import Settings
from safety_monitor.infrastructure.codecs import encode_png
from safety_monitor.transport.app import create_app
from tests.support.fakes import FakeDetector, compliant_person, det


def _png(height: int = 240, width: int = 320) -> bytes:
    return encode_png(np.zeros((height, width, 3), dtype=np.uint8))


def _client(detections=(), settings: Settings | None = None) -> TestClient:
    app = create_app(settings or Settings(), detector=FakeDetector(detections))
    return TestClient(app)


def test_health_reports_loaded_models():
    with _client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["models"] == ["ppe.pt", "scaffold.pt"]


def test_info_exposes_taxonomy_and_weights():
    with _client() as client:
        response = client.get("/api/v1/info")

    assert response.status_code == 200
    body = response.json()
    assert "head_nohelmet" in body["ppe_classes"]["helmet_missing"]
    assert body["caps"]["ppe"] == 50
    assert body["level_bands"]["SAFE"] == [81, 100]


def test_analyze_returns_safety_and_annotation():
    with _client(compliant_person()) as client:
        response = client.post(
            "/api/v1/analyze", files={"file": ("frame.png", _png(), "image/png")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["safety"] == 100
    assert body["level"] == "SAFE"
    assert body["counts"] == {"persons": 1, "violations": 0, "detections": 5}
    assert body["persons"][0]["status"] == "SAFE"
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    assert len(body["crops"]) == 1
    assert body["crops"][0]["index"] == 1


def test_analyze_reports_violation():
    dets = compliant_person()
    dets[1] = det("head_nohelmet", (25, 5, 75, 45))

    with _client(dets) as client:
        response = client.post(
            "/api/v1/analyze", files={"file": ("frame.png", _png(), "image/png")}
        )

    body = response.json()
    assert body["ppe_risk"] == 17
    assert body["safety"] == 83
    assert body["counts"]["violations"] == 1
    assert body["persons"][0]["reasons"]


def test_can_disable_annotation_and_crops():
    with _client(compliant_person()) as client:
        response = client.post(
            "/api/v1/analyze",
            params={"draw": "false", "crops": "false"},
            files={"file": ("frame.png", _png(), "image/png")},
        )

    body = response.json()
    assert body["annotated_image"] is None
    assert body["crops"] == []


def test_invalid_image_returns_400():
    with _client() as client:
        response = client.post(
            "/api/v1/analyze", files={"file": ("x.png", b"not an image", "image/png")}
        )

    assert response.status_code == 400


def test_oversized_upload_returns_413():
    settings = Settings(max_upload_bytes=16)
    with _client(detections=(), settings=settings) as client:
        response = client.post(
            "/api/v1/analyze", files={"file": ("frame.png", _png(), "image/png")}
        )

    assert response.status_code == 413


def test_missing_file_returns_422():
    with _client() as client:
        response = client.post("/api/v1/analyze")

    assert response.status_code == 422


def test_no_wildcard_cors_by_default():
    with _client() as client:
        response = client.get("/health", headers={"Origin": "http://evil.example"})

    assert "access-control-allow-origin" not in {
        k.lower() for k in response.headers
    }


def test_serves_landing_page():
    with _client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_serves_monitor_page():
    with _client() as client:
        response = client.get("/monitor")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_monitor_page_targets_new_api():
    with _client() as client:
        body = client.get("/monitor").text

    assert "/api/v1/analyze" in body
    assert "/api/analyze" not in body  # legacy endpoint must be gone
    assert "/health" in body


def test_api_routes_take_precedence_over_static():
    with _client() as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/info").status_code == 200
