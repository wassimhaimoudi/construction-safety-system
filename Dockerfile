# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────
#  Stage 1 — build the virtualenv (CPU-only torch keeps it small)
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

RUN uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-cache \
      torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --python /opt/venv/bin/python --no-cache \
      ".[service,inference]"

# ─────────────────────────────────────────────────────────────
#  Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgl1/libglib2.0-0 are required by OpenCV (ultralytics pulls both the
# full and headless opencv wheels).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAFETY_PPE_MODEL_PATH=/models/ppe.pt \
    SAFETY_SCAFFOLD_MODEL_PATH=/models/scaffold.pt

WORKDIR /app

# Model weights (override by mounting a volume at /models).
COPY ppe.pt scaffold.pt /models/

# Bundled UI (served at / and /monitor).
COPY static ./static

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "safety_monitor.main:app", "--host", "0.0.0.0", "--port", "8000"]
