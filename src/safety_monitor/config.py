"""Application settings (env-driven).

All settings can be overridden with ``SAFETY_``-prefixed environment variables,
e.g. ``SAFETY_CONFIDENCE=0.5`` or ``SAFETY_PPE_MODEL_PATH=/models/ppe.pt``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAFETY_",
        env_file=".env",
        extra="ignore",
    )

    # Model weights (see PLAN.md: filenames were swapped at source).
    ppe_model_path: Path = Path("ppe.pt")
    scaffold_model_path: Path = Path("scaffold.pt")

    # Inference knobs.
    confidence: float = 0.35
    image_size: int = 640
    device: str = "cpu"

    # Transport limits.
    max_upload_bytes: int = 10 * 1024 * 1024
    cors_origins: list[str] = Field(default_factory=list)

    # Rendering.
    annotate: bool = True

    # Bundled UI (served at / and /monitor when the directory exists).
    static_dir: Path | None = Path("static")


def get_settings() -> Settings:
    return Settings()
