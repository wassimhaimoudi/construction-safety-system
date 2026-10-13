"""
AI Construction Site Safety Monitor — v10
Flask backend server
Dual-model pipeline: scaffold.pt (PPE/workers/infra) + ppe.pt (scaffold risk)
"""

import os, base64, io, json, tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)

MODEL_VEW_PATH       = "scaffold.pt"
MODEL_SCAFANDRI_PATH = "ppe.pt"

OUTPUT_DIR = Path("pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CONF_VEW       = 0.25
CONF_SCAFANDRI = 0.25
MAX_PERSONS    = 9

# ─────────────────────────────────────────────────────────────
# CLASS TAXONOMY
# ─────────────────────────────────────────────────────────────
PERSON_CLASSES = {
    "person", "worker", "Worker", "Person", "Human",
    "worker_with_helmet", "worker_without_helmet",
}
MISSING_PPE_CLASSES = {
    "No-Helmet", "No-Vest", "no helmet", "no vest",
    "no_helmet", "no_vest", "NoHelmet", "NoVest",
    "without_helmet", "without_vest",
}
WORN_PPE_CLASSES = {
    "helmet", "Helmet", "vest", "Vest", "safety_vest",
    "hard_hat", "HardHat", "safety_helmet", "worker_with_helmet",
}
SCAFFOLD_HIGH_RISK = {
    "High_Risk_Scaffolding", "High_Risk_Suspended_Scaffolding",
    "high_risk", "HighRisk",
}
SCAFFOLD_MEDIUM_RISK = {
    "Medium_Risk_Scaffolding", "medium_risk", "MediumRisk",
    "Scaffolding", "scaffolding",
}
SCAFFOLD_LOW_RISK = {
    "Low_Risk_Scaffolding", "low_risk", "LowRisk", "Ground_Level_Scaffolding",
}
ALL_SCAFFOLD_CLASSES = SCAFFOLD_HIGH_RISK | SCAFFOLD_MEDIUM_RISK | SCAFFOLD_LOW_RISK

DANGER_ZONE_CLASSES = {
    "Danger_Zone", "danger_zone", "DangerZone",
    "warning_sign", "WarningSign", "warning_zone",
}
BARRIER_CLASSES = {
    "barrier", "Barrier", "safety_barrier", "road_barrier",
    "caution_tape", "fence", "Fence",
}
CONE_CLASSES = {
    "cone", "Cone", "safety_cone", "traffic_cone",
}
ALL_INFRA_CLASSES = DANGER_ZONE_CLASSES | BARRIER_CLASSES | CONE_CLASSES | {"hand", "Hand"}
NEUTRAL_CLASSES = ALL_INFRA_CLASSES

# ─────────────────────────────────────────────────────────────
# SCORE WEIGHTS
# ─────────────────────────────────────────────────────────────
RISK_WEIGHT = {
    "helm_miss_ground":  15,
    "vest_miss_ground":  10,
    "helm_miss_scaffold": 25,
    "vest_miss_scaffold": 20,
    "scaffold_low":           5,
    "scaffold_high":         20,
    "scaffold_suspended":    35,
    "scaffold_worker_bonus": 10,
    "crowd_bonus_per_person": 5,
    "crowd_max":             15,
    "scaffold_env_ppe_ok": 10,
    "cap_ppe":    50,
    "cap_scaf":   35,
    "cap_crowd":  15,
    "max_person_score": 45,
}

# ─────────────────────────────────────────────────────────────
# MODEL LOADING (lazy singleton)
# ─────────────────────────────────────────────────────────────
_model_vew  = None
_model_scaf = None
VEW_NAMES   = {}
SCAF_NAMES  = {}

def load_models():
    global _model_vew, _model_scaf, VEW_NAMES, SCAF_NAMES
    if _model_vew is None:
        print("⏳ Loading models...")
        _model_vew  = YOLO(MODEL_VEW_PATH)
        _model_scaf = YOLO(MODEL_SCAFANDRI_PATH)
        VEW_NAMES   = _model_vew.names
        SCAF_NAMES  = _model_scaf.names
        print(f"✅ scaffold.pt — {_model_vew.model.nc} classes")
        print(f"✅ ppe.pt — {_model_scaf.model.nc} classes")
    return _model_vew, _model_scaf

# ─────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def score_to_level(s: int) -> str:
    if s <= 0:  return "SAFE"
    if s <= 20: return "LOW"
    if s <= 50: return "MEDIUM"
    if s <= 80: return "HIGH"
    return "CRITICAL"

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_to_b64(img_bgr) -> str:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

def results_to_list(results, names) -> List[Dict]:
    out = []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)
        name   = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if name in SCAFFOLD_HIGH_RISK or name in MISSING_PPE_CLASSES:
            risk = "HIGH"
        elif name in SCAFFOLD_MEDIUM_RISK:
            risk = "MEDIUM"
        elif name in NEUTRAL_CLASSES:
            risk = "NEUTRAL"
        else:
            risk = "LOW"
        out.append({"class": name, "confidence": round(conf, 3),
                    "risk": risk, "bbox": [x1, y1, x2, y2]})
    return out

def iou(A, B) -> float:
    xA, yA = max(A[0], B[0]), max(A[1], B[1])
    xB, yB = min(A[2], B[2]), min(A[3], B[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((A[2]-A[0])*(A[3]-A[1]) + (B[2]-B[0])*(B[3]-B[1]) - inter)

def overlap_ratio(inner, outer) -> float:
    xA, yA = max(inner[0], outer[0]), max(inner[1], outer[1])
    xB, yB = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area  = max(1, (inner[2]-inner[0]) * (inner[3]-inner[1]))
    return inter / area

def crop_person_b64(image_bgr: np.ndarray, bbox: List[int], pad: float = 0.05) -> str:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1c = max(0, x1 - px); y1c = max(0, y1 - py)
    x2c = min(w, x2 + px); y2c = min(h, y2 + py)
    crop_bgr = image_bgr[y1c:y2c, x1c:x2c]
    if crop_bgr.size == 0:
        return ""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    target_h = 120
    ratio = target_h / max(1, pil_crop.height)
    target_w = max(60, int(pil_crop.width * ratio))
    pil_crop = pil_crop.resize((target_w, target_h), Image.LANCZOS)
    buf = io.BytesIO()
    pil_crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────────────────────
# PPE ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyze_person_ppe(person_bbox, idx, all_dets, on_scaffold: bool = False) -> Dict:
    helmet_worn = helmet_missing = False
    vest_worn   = vest_missing   = False
    reasons     = []

    for det in all_dets:
        cls, bbox = det["class"], det["bbox"]
        if overlap_ratio(bbox, person_bbox) < 0.2 and iou(bbox, person_bbox) < 0.1:
            continue
        if cls in WORN_PPE_CLASSES:
            if any(k in cls.lower() for k in ("helmet", "helm", "hard_hat", "hardhat")):
                helmet_worn = True
            if "vest" in cls.lower():
                vest_worn = True
        elif cls in MISSING_PPE_CLASSES:
            if any(k in cls.lower() for k in ("helmet", "helm")):
                helmet_missing = True
                reasons.append("❌ Casque absent")
            if "vest" in cls.lower():
                vest_missing = True
                reasons.append("❌ Gilet absent")

    person_score = 0

    if helmet_missing:
        pen = RISK_WEIGHT["helm_miss_scaffold"] if on_scaffold else RISK_WEIGHT["helm_miss_ground"]
        person_score += pen
        reasons.append(f"→ Casque absent: +{pen}pts ({'échafaud' if on_scaffold else 'sol'})")
    elif not helmet_worn:
        pen = (RISK_WEIGHT["helm_miss_scaffold"] // 2) if on_scaffold else (RISK_WEIGHT["helm_miss_ground"] // 2)
        pen = max(4, pen)
        person_score += pen
        reasons.append(f"→ Casque inconnu: +{pen}pts ({'échafaud' if on_scaffold else 'sol'})")

    if vest_missing:
        pen = RISK_WEIGHT["vest_miss_scaffold"] if on_scaffold else RISK_WEIGHT["vest_miss_ground"]
        person_score += pen
        reasons.append(f"→ Gilet absent: +{pen}pts ({'échafaud' if on_scaffold else 'sol'})")
    elif not vest_worn:
        pen = (RISK_WEIGHT["vest_miss_scaffold"] // 2) if on_scaffold else (RISK_WEIGHT["vest_miss_ground"] // 2)
        pen = max(3, pen)
        person_score += pen
        reasons.append(f"→ Gilet inconnu: +{pen}pts ({'échafaud' if on_scaffold else 'sol'})")

    if helmet_missing or vest_missing:
        status = "VIOLATION"
    elif helmet_worn and vest_worn:
        if on_scaffold:
            env_pen = RISK_WEIGHT.get("scaffold_env_ppe_ok", 10)
            person_score = env_pen
            reasons += ["✅ Casque OK", "✅ Gilet OK",
                        f"⚠️ Risque échafaud environnement: +{env_pen}pts"]
            status = "PARTIAL"
        else:
            person_score = 0
            reasons += ["✅ Casque OK", "✅ Gilet OK"]
            status = "SAFE"
    elif helmet_worn or vest_worn:
        status = "PARTIAL"
        if helmet_worn: reasons.append("✅ Casque OK")
        if vest_worn:   reasons.append("✅ Gilet OK")
    else:
        status = "UNKNOWN"
        reasons.append("⚠️ PPE non évalué")

    return {
        "index":          idx,
        "helmet_status":  helmet_worn,
        "vest_status":    vest_worn,
        "helmet_missing": helmet_missing,
        "vest_missing":   vest_missing,
        "on_scaffold":    on_scaffold,
        "status":         status,
        "score":          min(45, person_score),
        "reasons":        reasons,
        "bbox":           person_bbox,
        "crop_b64":       "",
    }

# ─────────────────────────────────────────────────────────────
# SCAFFOLD RISK
# ─────────────────────────────────────────────────────────────
def compute_scaffolding_risk(scaf_dets, all_person_boxes) -> Tuple[int, List[str]]:
    raw_score = 0
    reasons   = []
    cap       = RISK_WEIGHT["cap_scaf"]

    for det in scaf_dets:
        cls, bbox = det["class"], det["bbox"]
        has_worker = any(
            overlap_ratio(pb, bbox) > 0.3 or iou(pb, bbox) > 0.1
            for pb in all_person_boxes
        )
        worker_tag = " (travailleur détecté)" if has_worker else ""
        bonus      = RISK_WEIGHT["scaffold_worker_bonus"] if has_worker else 0

        if cls in SCAFFOLD_HIGH_RISK:
            if "suspended" in cls.lower() or "suspendu" in cls.lower():
                pts = RISK_WEIGHT["scaffold_suspended"] + bonus
                reasons.append(f"🔴 Échafaud suspendu{worker_tag} +{pts}pts")
            else:
                pts = RISK_WEIGHT["scaffold_high"] + bonus
                reasons.append(f"🟠 Échafaud haute risque{worker_tag} +{pts}pts")
            raw_score += pts
        elif cls in SCAFFOLD_MEDIUM_RISK:
            pts = RISK_WEIGHT["scaffold_high"]
            raw_score += pts
            reasons.append(f"🟡 Échafaud moyen ({cls}) +{pts}pts")
        elif cls in SCAFFOLD_LOW_RISK:
            pts = RISK_WEIGHT["scaffold_low"]
            raw_score += pts
            reasons.append(f"🟢 Échafaud bas risque ({cls}) +{pts}pts")

    return min(cap, raw_score), reasons

# ─────────────────────────────────────────────────────────────
# GLOBAL RISK ENGINE
# ─────────────────────────────────────────────────────────────
def compute_global_risk(vew_dets, scaf_dets, image_bgr) -> Dict:
    all_dets   = vew_dets + scaf_dets
    infra_dets = [d for d in vew_dets if d["class"] in ALL_INFRA_CLASSES]

    scaf_raw = scaf_dets + [d for d in vew_dets if d["class"] in ALL_SCAFFOLD_CLASSES]
    seen = set(); merged_scaf = []
    for d in scaf_raw:
        k = tuple(d["bbox"])
        if k not in seen:
            seen.add(k); merged_scaf.append(d)

    persons_raw = [d for d in all_dets if d["class"] in PERSON_CLASSES]
    seen = set(); persons = []
    for p in persons_raw:
        k = tuple(p["bbox"])
        if k not in seen:
            seen.add(k); persons.append(p)

    all_person_boxes = [p["bbox"] for p in persons]
    n = len(persons)

    scaf_score, scaf_reasons = compute_scaffolding_risk(merged_scaf, all_person_boxes)

    person_results = []
    max_pp = RISK_WEIGHT["max_person_score"]
    for idx, p in enumerate(persons):
        on_scaf = any(
            overlap_ratio(p["bbox"], det["bbox"]) > 0.3 or iou(p["bbox"], det["bbox"]) > 0.1
            for det in merged_scaf if det["class"] in (SCAFFOLD_HIGH_RISK | SCAFFOLD_MEDIUM_RISK)
        )
        res = analyze_person_ppe(p["bbox"], idx + 1, all_dets, on_scaffold=on_scaf)
        res["bbox"] = p["bbox"]
        res["score_pct"] = min(100, round(res["score"] * 100 / max_pp))
        res["crop_b64"] = crop_person_b64(image_bgr, p["bbox"])
        person_results.append(res)

    if n > 0:
        mean_pp      = sum(pr["score"] for pr in person_results) / n
        ppe_risk     = min(RISK_WEIGHT["cap_ppe"], round(mean_pp * (50 / 45)))
        missing_helm = sum(1 for pr in person_results if pr["helmet_missing"])
        missing_vest = sum(1 for pr in person_results if pr["vest_missing"])
        total_missing = missing_helm + missing_vest
        score_ppe    = round(100 * (1 - total_missing / (2 * n)), 2) if n else 100.0
    else:
        ppe_risk = missing_helm = missing_vest = total_missing = 0
        score_ppe = 100.0

    n_violators = sum(1 for pr in person_results if pr["status"] == "VIOLATION")
    crowd_raw   = max(0, n_violators - 1) * RISK_WEIGHT["crowd_bonus_per_person"]
    crowd_bonus = min(RISK_WEIGHT["cap_crowd"], crowd_raw)
    crowd_reason = (
        f"👥 {n_violators} violations simultanées → +{crowd_bonus}pts"
        if crowd_bonus > 0 else None
    )

    total_score = min(100, int(ppe_risk + scaf_score + crowd_bonus))
    level = score_to_level(total_score)

    return {
        "score":          total_score,
        "level":          level,
        "ppe_risk":       ppe_risk,
        "scaf_score":     scaf_score,
        "crowd_bonus":    crowd_bonus,
        "scaf_reasons":   scaf_reasons,
        "crowd_reason":   crowd_reason,
        "person_results": person_results,
        "n_persons":      n,
        "n_violators":    n_violators,
        "missing_helm":   missing_helm,
        "missing_vest":   missing_vest,
        "merged_scaf":    merged_scaf,
        "score_ppe":      score_ppe,
        "infra_dets":     infra_dets,
        "n_infra":        len(infra_dets),
    }

# ─────────────────────────────────────────────────────────────
# IMAGE ANNOTATION
# ─────────────────────────────────────────────────────────────
COLOR_DANGER  = (  0,   0, 255)
COLOR_SAFE    = (  0, 220,   0)
COLOR_WARN    = (  0, 165, 255)
COLOR_NEUTRAL = (160, 160, 160)

def draw_annotated(image_bgr, risk) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]

    for det in risk.get("infra_dets", []):
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        color = (0, 50, 220) if cls in DANGER_ZONE_CLASSES else \
                (255, 180, 0) if cls in BARRIER_CLASSES else (0, 220, 220)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, f"{cls} {det['confidence']:.0%}", (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    for det in risk["merged_scaf"]:
        x1, y1, x2, y2 = det["bbox"]
        color = COLOR_DANGER if det["class"] in SCAFFOLD_HIGH_RISK else \
                COLOR_WARN   if det["class"] in SCAFFOLD_MEDIUM_RISK else COLOR_NEUTRAL
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"SCAF:{det['class']} {det['confidence']:.0%}", (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for pr in risk["person_results"]:
        x1, y1, x2, y2 = pr["bbox"]
        color = {"SAFE": COLOR_SAFE, "VIOLATION": COLOR_DANGER, "PARTIAL": COLOR_WARN}.get(pr["status"], (180,100,0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cl = max(10, (x2 - x1) // 6)
        for (px, py) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            dx = cl if px == x1 else -cl
            dy = cl if py == y1 else -cl
            cv2.line(img, (px, py), (px+dx, py), color, 5)
            cv2.line(img, (px, py), (px, py+dy), color, 5)
        id_text = f" P{pr['index']} "
        (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1-th-6)), (x1+tw+4, y1), color, -1)
        cv2.putText(img, id_text, (x1+2, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        person_score_pct = pr["score_pct"]
        h_icon = "H:OK" if pr["helmet_status"] else ("H:XX" if pr["helmet_missing"] else "H: ?")
        v_icon = "V:OK" if pr["vest_status"]   else ("V:XX" if pr["vest_missing"]   else "V: ?")
        badge  = f" {h_icon}  {v_icon}  Risk:{person_score_pct}% "
        (bw, bh2), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        by1 = min(y2 + 2, h - bh2 - 8)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, by1), (min(x1+bw+6, w-2), by1+bh2+6), color, -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.putText(img, badge, (x1+3, by1+bh2+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 1, cv2.LINE_AA)

    banner_color = {
        "SAFE":     (20, 160, 20),
        "LOW":      (20, 180, 20),
        "MEDIUM":   (0, 150, 200),
        "HIGH":     (0, 120, 230),
        "CRITICAL": (0, 30, 200),
    }.get(risk["level"], (100,100,100))
    bh_size = 40
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (w, bh_size), banner_color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    banner_txt = (
        f"  RISK: {risk['level']}  |  Score: {risk['score']}/100  |  "
        f"Personnes: {risk['n_persons']}  |  Violations: {risk['n_violators']}  |  "
        f"Infra: {risk['n_infra']} obj  |  PPE: +{risk['ppe_risk']}  Scaffold: +{risk['scaf_score']}  Crowd: +{risk['crowd_bonus']}"
    )
    cv2.putText(img, banner_txt, (6, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
    return img

# ─────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/monitor")
def monitor():
    return send_from_directory("static", "monitor.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    conf_vew  = float(request.form.get("conf_vew",  CONF_VEW))
    conf_scaf = float(request.form.get("conf_scaf", CONF_SCAFANDRI))

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    try:
        model_vew, model_scaf = load_models()
    except Exception as e:
        return jsonify({"error": f"Model loading failed: {e}. Make sure scaffold.pt and ppe.pt are in the same folder as app.py"}), 500

    image_bgr = pil_to_cv2(pil_img)

    res_vew  = model_vew.predict(source=image_bgr,  conf=conf_vew,  verbose=False)[0]
    res_scaf = model_scaf.predict(source=image_bgr, conf=conf_scaf, verbose=False)[0]
    vew_dets  = results_to_list(res_vew,  VEW_NAMES)
    scaf_dets = results_to_list(res_scaf, SCAF_NAMES)

    risk = compute_global_risk(vew_dets, scaf_dets, image_bgr)
    annotated_bgr = draw_annotated(image_bgr, risk)
    annotated_b64 = cv2_to_b64(annotated_bgr)

    n_viol      = risk["n_violators"]
    crowd_share = round(risk["crowd_bonus"] / max(1, n_viol)) if n_viol > 0 else 0

    persons_out = []
    for pr in risk["person_results"]:
        persons_out.append({
            "index":          pr["index"],
            "status":         pr["status"],
            "score":          pr["score"],
            "score_pct":      pr["score_pct"],
            "helmet_status":  pr["helmet_status"],
            "helmet_missing": pr["helmet_missing"],
            "vest_status":    pr["vest_status"],
            "vest_missing":   pr["vest_missing"],
            "on_scaffold":    pr["on_scaffold"],
            "reasons":        pr["reasons"],
            "crop_b64":       pr["crop_b64"],
            "crowd_share":    crowd_share if pr["status"] == "VIOLATION" else 0,
        })

    infra_out = []
    for det in risk["infra_dets"]:
        cls = det["class"]
        cat = "danger" if cls in DANGER_ZONE_CLASSES else \
              "barrier" if cls in BARRIER_CLASSES else "cone"
        infra_out.append({"class": cls, "confidence": det["confidence"],
                           "category": cat, "bbox": det["bbox"]})

    return jsonify({
        "annotated_b64": annotated_b64,
        "score":         risk["score"],
        "level":         risk["level"],
        "ppe_risk":      risk["ppe_risk"],
        "scaf_score":    risk["scaf_score"],
        "crowd_bonus":   risk["crowd_bonus"],
        "n_persons":     risk["n_persons"],
        "n_violators":   risk["n_violators"],
        "missing_helm":  risk["missing_helm"],
        "missing_vest":  risk["missing_vest"],
        "score_ppe":     risk["score_ppe"],
        "scaf_reasons":  risk["scaf_reasons"],
        "crowd_reason":  risk["crowd_reason"],
        "n_infra":       risk["n_infra"],
        "infra_dets":    infra_out,
        "persons":       persons_out,
    })

@app.route("/api/status")
def status():
    models_ok = os.path.exists(MODEL_VEW_PATH) and os.path.exists(MODEL_SCAFANDRI_PATH)
    return jsonify({
        "status": "ok",
        "models": {
            "vew":      os.path.exists(MODEL_VEW_PATH),
            "scafandri": os.path.exists(MODEL_SCAFANDRI_PATH),
        },
        "ready": models_ok,
    })

if __name__ == "__main__":
    print("🏗️  AI Safety Monitor — Web Server")
    print(f"   scaffold.pt      : {'✅' if os.path.exists(MODEL_VEW_PATH) else '❌ MISSING'}")
    print(f"   ppe.pt: {'✅' if os.path.exists(MODEL_SCAFANDRI_PATH) else '❌ MISSING'}")
    print("   → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
