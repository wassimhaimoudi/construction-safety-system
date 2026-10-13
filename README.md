# 🏗️ AI Construction Site Safety Monitor

A stateless, image-only service that scores construction-site safety from a
single photo using two YOLO models. Higher score = **safer**.

Built with a clean, test-driven layout: pure domain logic, thin adapters, and a
FastAPI edge — packaged as a Docker image.

## Scoring

```
Danger = min(100, PPE[max 50] + Scaffold[max 35] + Crowd[max 15])
Safety = 100 − Danger
```

| Level | Safety |
|---|---|
| 🟢 SAFE | 81–100 |
| 🟢 LOW | 51–80 |
| 🟡 MEDIUM | 21–50 |
| 🟠 HIGH | 1–20 |
| 🔴 CRITICAL | 0 |

Per-person PPE penalties (ground / scaffold), person total capped at 45:
helmet 15/25 · vest 10/20 · glove 8/12 · mask 4/6. Vest absence counts as
**missing** (the PPE model has no negative vest class). Crowd risk is
`min(15, (violators − 1) × 5)`.

## Architecture

```
src/safety_monitor/
├── domain/          # pure logic: taxonomy, geometry, scoring, levels (no torch)
├── application/     # analyze_frame use-case + Detector port
├── infrastructure/  # YOLO/ensemble adapters, codecs, annotate, person_crop
├── transport/       # FastAPI app, routes, schemas, static serving
├── config.py        # env-driven settings
└── main.py          # ASGI entrypoint (safety_monitor.main:app)
tests/
├── golden/          # frozen scoring spec (worked examples)
├── unit/            # domain, adapters, transport
├── integration/     # real-model smoke test (opt-in)
└── support/         # fakes + shared fixtures
```

### Models

Two Ultralytics checkpoints (the source filenames were swapped and have been
renamed to their true roles):

| File | Role | Classes |
|---|---|---|
| `ppe.pt` | person + PPE | 14 |
| `scaffold.pt` | scaffold + environment | 7 |

## Quick start (Docker)

```bash
docker compose up --build
# UI:   http://localhost:8000/
# app:  http://localhost:8000/monitor
# docs: http://localhost:8000/docs
```

Weights are baked into the image (`/models`). To mount them instead:

```yaml
volumes:
  - ./ppe.pt:/models/ppe.pt:ro
  - ./scaffold.pt:/models/scaffold.pt:ro
```

## Local development (Python 3.12)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,service,inference]" \
  --extra-index-url https://download.pytorch.org/whl/cpu

uvicorn safety_monitor.main:app --reload
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + whether models are loaded |
| `GET` | `/api/v1/info` | Taxonomy, weights, level bands, caps |
| `POST` | `/api/v1/analyze` | Multipart `file` → safety analysis JSON |

`POST /api/v1/analyze` query params: `draw` (annotated image, default true),
`crops` (per-person crops, default true).

Example response (abridged):

```json
{
  "safety": 83,
  "danger": 17,
  "level": "SAFE",
  "ppe_risk": 17,
  "scaffold_risk": 0,
  "crowd_risk": 0,
  "counts": {"persons": 1, "violations": 1, "detections": 5},
  "persons": [
    {"index": 1, "status": "VIOLATION", "helmet": "missing", "vest": "present",
     "glove": "present", "mask": "present", "score": 15, "on_scaffold": false,
     "reasons": ["missing helmet: +15"]}
  ],
  "detections": [{"class_name": "head_nohelmet", "confidence": 0.87,
                  "bbox": {"x1": 25, "y1": 5, "x2": 75, "y2": 45}}],
  "annotated_image": "data:image/jpeg;base64,...",
  "crops": [{"index": 1, "image": "data:image/jpeg;base64,..."}]
}
```

## Configuration

All settings are `SAFETY_`-prefixed environment variables (or a `.env` file):

| Variable | Default | Purpose |
|---|---|---|
| `SAFETY_PPE_MODEL_PATH` | `ppe.pt` | PPE model weights |
| `SAFETY_SCAFFOLD_MODEL_PATH` | `scaffold.pt` | scaffold model weights |
| `SAFETY_CONFIDENCE` | `0.35` | detection threshold |
| `SAFETY_IMAGE_SIZE` | `640` | inference size |
| `SAFETY_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SAFETY_MAX_UPLOAD_BYTES` | `10485760` | upload limit |
| `SAFETY_CORS_ORIGINS` | `[]` | JSON list; no wildcard by default |
| `SAFETY_STATIC_DIR` | `static` | bundled UI directory |

## Testing

```bash
pytest                      # fast suite (no torch)
pytest -m integration       # loads the real checkpoints
```
