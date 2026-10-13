"""Optionally serve the bundled single-page UI from the API process."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_static(app: FastAPI, static_dir: Path) -> bool:
    """Serve ``index.html`` at ``/`` and ``monitor.html`` at ``/monitor``.

    Returns ``True`` when a static directory was found. Assets are served under
    ``/static``. API routes are registered before this, so they take precedence.
    """
    if not static_dir.is_dir():
        return False

    index = static_dir / "index.html"
    monitor = static_dir / "monitor.html"

    if index.exists():

        @app.get("/", include_in_schema=False)
        def _landing() -> FileResponse:  # pragma: no cover - trivial
            return FileResponse(index)

    if monitor.exists():

        @app.get("/monitor", include_in_schema=False)
        def _monitor() -> FileResponse:  # pragma: no cover - trivial
            return FileResponse(monitor)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return True
