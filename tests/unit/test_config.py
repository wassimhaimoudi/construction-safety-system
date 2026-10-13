"""Tests for settings loading."""

from __future__ import annotations

from pathlib import Path

from safety_monitor.config import Settings


def test_defaults():
    settings = Settings()
    assert settings.ppe_model_path == Path("ppe.pt")
    assert settings.scaffold_model_path == Path("scaffold.pt")
    assert settings.device == "cpu"
    assert settings.cors_origins == []


def test_env_override(monkeypatch):
    monkeypatch.setenv("SAFETY_CONFIDENCE", "0.5")
    monkeypatch.setenv("SAFETY_PPE_MODEL_PATH", "/models/ppe.pt")

    settings = Settings()

    assert settings.confidence == 0.5
    assert settings.ppe_model_path == Path("/models/ppe.pt")
