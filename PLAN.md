# Construction Safety System — Test-Driven Reconstruction Plan

Status: **locked** (all open decisions resolved)
Project root: this directory (`/home/wassim/construction-safety-system`)
Runtime: Python, Flask (current) → FastAPI, Ultralytics YOLO, Docker

---

## 1. Locked decisions

1. **Scoring model:** `Safety = 100 − Danger`, where
   `Danger = min(100, PPE[50] + Scaffold[35] + Crowd[15])`.
   Corrected taxonomy; gloves and mask included.
2. **Service shape:** FastAPI/ASGI; deploy as a Docker image.
3. **Runtime:** keep Ultralytics (torch under the hood); install CPU-only torch wheel.
4. **Naming:** rename `vew.pt → scaffold.pt`, `scafandri.pt → ppe.pt`; align variables, UI labels, README.
5. **Weights:** no notebook available → propose defaults, freeze as golden fixtures, tunable later.
6. **Vest policy:** absence of a detected vest = missing (strict).

### Discovered facts that forced these choices
- The two model files are **swapped** vs their names:
  - `vew.pt` → training run `scaffold_risk_v3`, 7 classes (scaffold + environment).
  - `scafandri.pt` → training run `ppe_yolov11`, 14 classes (person + PPE, incl. anatomy `head`).
- `app.py`'s taxonomy constants (`app.py:38-76`) reference **invented class names**, so:
  - missing-helmet never fires (real class is `head_nohelmet`),
  - missing-vest can never fire (no negative class exists),
  - gloves and mask are entirely unhandled,
  - `Work_Sign` detections are dropped,
  - the medium-scaffold branch is dead (no such class).
- Docs/UI advertise an unimplemented sigmoid + temporal-smoothing formula
  (`static/index.html` Module 04 modal) and nonexistent classes; temporal
  smoothing is impossible in a stateless images-only service.

---

## 2. Canonical taxonomy and default weights (frozen as tests)

| Signal | Present | Missing | Neutral |
|---|---|---|---|
| person | `person` | — | — |
| helmet | `helmet`, `head_helmet` | `head_nohelmet` | — |
| vest | `vest` | *(absence = missing)* | — |
| gloves | `glove`, `hand_glove` | `hand_noglove` | — |
| mask | `mask`, `face_mask` | `face_nomask` | — |
| anatomy | — | — | `face`, `hand` |
| scaffold | `High_Risk_Scaffolding`, `High_Risk_Suspended_Scaffolding` | — | `Low_Risk_Scaffolding` |
| infra | `Danger_Zone`, `Barrier`, `Cone`, `Work_Sign` | — | — |

Default per-person penalties (ground / scaffold), person sum capped at 45:
- helmet 15 / 25
- vest 10 / 20
- glove 8 / 12
- mask 4 / 6
- unknown-only (helmet/glove/mask) = half value, floored (>= 3/4).

Scaffold risk cap = 35 (suspended 35, high 20, low 5, worker bonus +10).
Crowd = `min(15, max(0, n_violators-1) * 5)`.
PPE risk = `min(50, mean(person_scores) * 50/45)`.

**Levels (safety direction):**
- 81–100 SAFE
- 51–80 LOW
- 21–50 MEDIUM
- 1–20 HIGH
- 0 CRITICAL

---

## 3. Target tree

```
src/safety_monitor/
  main.py            # FastAPI app factory + lifespan (eager, locked model load)
  config.py          # pydantic-settings: model paths, confidences, weights (env)
  domain/            # pure, torch-free, fast unit tests
    taxonomy.py      # corrected class sets
    geometry.py      # iou, overlap_ratio (asymmetric — document direction)
    detections.py    # Detection dataclass
    scoring.py       # analyze_person_ppe, scaffold, global -> SafetyResult
    levels.py        # safety score -> level
  application/
    analyze_frame.py # detections -> SafetyResult (no I/O)
  infrastructure/
    yolo.py          # model loader + detector adapter (behind protocol)
    codecs.py        # pil<->cv2, base64
    annotate.py      # draw_annotated
    person_crop.py   # crop_person_b64
  transport/
    schemas.py       # pydantic request/response (validates conf inputs)
    routes.py        # /api/analyze, /api/status
tests/
  unit/  contract/  golden/
Dockerfile  docker-compose.yml  .dockerignore  pyproject.toml  static/
```

---

## 4. TDD sequence

1. **Freeze spec.** Write the corrected taxonomy + safety formula as a
   worked-examples table; encode as failing golden tests (pure, no torch).
2. **Extract pure domain.** Move `geometry`, `taxonomy`, `levels`, `scoring`
   out of `app.py`; behavior-preserving except taxonomy fixes.
3. **Green the golden suite.** Regression net established.
4. **Adapters.** YOLO/codecs/annotate behind protocols; add `FakeDetector`
   so application tests run without torch.
5. **FastAPI transport.** Pydantic schemas (fixes unguarded `float()` at
   `app.py:484`); eager model load in lifespan (fixes load race at
   `app.py:107`); contract tests.
6. **Dockerize.** `python:3.12-slim`, CPU-only torch index, non-root,
   `HEALTHCHECK /api/status`, expose 8000.
7. **Frontend.** Invert `levelMeta` (`static/monitor.html` JS), remove dead
   reason-filter, fix model labels/badges; pull base64 assets out of
   `static/index.html`.

---

## 5. Docker

- Base `python:3.12-slim`.
- Install torch via `--index-url https://download.pytorch.org/whl/cpu`
  to avoid the ~2GB CUDA build.
- `opencv-python-headless` retained.
- Non-root user; `HEALTHCHECK` hits `/api/status`; expose 8000.
- Model weights (`scaffold.pt`, `ppe.pt`) mounted as a volume by default;
  optional bake for reproducibility.

---

## 6. Rename / relabel checklist

- `app.py:25-26` model path constants
- `run.sh:14-21` model existence checks
- `README.md:12-13` structure description
- `static/monitor.html` model badges
- `static/index.html` module modal copy (also remove/adjust the sigmoid + 15-class claims)

---

## 7. Known defects to fix during extraction

| Severity | Finding | Ref |
|---|---|---|
| High | Docs/UI advertise a formula + classes not implemented | `static/index.html` modal 04, `app.py:81` |
| High | Cross-model fusion dedups only exact-equal bboxes → double counting | `app.py:319-331` |
| High | Unguarded `float()` input → 500; no upload size cap; CORS `*` | `app.py:484`, `:23` |
| Med | Model-load singleton race under threaded server | `app.py:107` |
| Med | `overlap_ratio` direction asymmetry undocumented/untested | `app.py:166` |
| Med | Frontend reason-filter never matches backend strings | `static/monitor.html` JS |
| Low | `worker_with_helmet` collides person/PPE; hands labelled "cone" | `app.py:38,47,75` |
| Low | Dead code, duplicated defaults, CWD-relative model paths | `app.py:7,28,33` |
| Low | 2.3MB HTML with inlined base64 assets | `static/index.html` |

---

## 8. Working agreement (for implementation)

- Work from the project root (`/home/wassim/construction-safety-system`).
- After each change (major or minor): `git add` the touched files and commit
  with a descriptive message.
- Keep the golden suite green before every commit.
