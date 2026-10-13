"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..application.detector import Detector
from ..config import Settings, get_settings
from .routes import router


def create_app(
    settings: Settings | None = None,
    *,
    detector: Detector | None = None,
    detector_factory: Callable[[Settings], Detector] | None = None,
) -> FastAPI:
    """Build the ASGI app.

    Models are loaded once, eagerly, during startup (lifespan) — no lazy
    singleton race. Tests can pass ``detector`` to bypass the real models.
    """
    settings = settings or get_settings()

    if detector_factory is None:
        from ..infrastructure.ensemble import build_detector

        detector_factory = build_detector

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = (
            detector if detector is not None else detector_factory(settings)
        )
        try:
            yield
        finally:
            app.state.detector = None

    app = FastAPI(
        title="Safety Monitor",
        version=__version__,
        summary="AI construction site safety analysis (stateless, image-only).",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # No wildcard CORS by default; opt in via SAFETY_CORS_ORIGINS.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    app.include_router(router)

    if settings.static_dir is not None:
        from pathlib import Path

        from .static import mount_static

        mount_static(app, Path(settings.static_dir))

    return app
