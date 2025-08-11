# backend/api.py
import os
import uuid
import json
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# -----------------------------
# Paths & constants
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
META_PATH = MODELS_DIR / "training_metadata.json"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# -----------------------------
# App & globals
# -----------------------------
app = Flask(__name__)
CORS(app)

MODEL = None
META: dict = {}
THRESHOLD = 0.5
TRAIN_FEATURES: list[str] = []

# in-memory recent transactions (drives dashboard/analytics)
_recent: list[dict] = []


# -----------------------------
# Helpers
# -----------------------------
def _load_model_from_meta():
    """
    Load model & metadata produced by scripts/train_model.py.
    Returns (model, meta, threshold, train_features).
    """
    if not META_PATH.exists():
        return None, {}, 0.5, []

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_path = meta.get("model_path")
    if not model_path:
        return None, meta, float(meta.get("threshold", 0.5)), meta.get("train_features", [])

    # support absolute or relative paths
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = ROOT / model_path
    if not model_file.exists():
        return None, meta, float(meta.get("threshold", 0.5)), meta.get("train_features", [])

    model = joblib.load(model_file)
    thr = float(meta.get("threshold", meta.get("best_threshold", 0.5)))
    feats = meta.get("train_features", [])
    return model, meta, thr, feats


def _ensure_model_loaded():
    global MODEL, META, THRESHOLD, TRAIN_FEATURES
    if MODEL is None:
        MODEL, META, THRESHOLD, TRAIN_FEATURES = _load_model_from_meta()
        if MODEL:
            app.logger.info(f"Loaded model: {META.get('model_path')}")
        else:
            app.logger.warning("No model loaded. Train a model first.")


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _score_txn(payload: dict) -> dict:
    """
    Score a single transaction dict using the trained pipeline.
    Returns a result dict with risk_score, prediction, reasons, and echoed fields.
    """
    _ensure_model_loaded()
    if MODEL is None:
        return {"error": "model_not_loaded"}

    # Build 1-row DataFrame with the training features (pipeline handles preprocessing)
    row = {}
    for col in TRAIN_FEATURES:
        if col in payload:
            row[col] = payload[col]
        else:
            # numeric defaults for common numeric features
            if col.lower() in {"amount", "hour", "latenighttxn"}:
                row[col] = 0
            else:
                row[col] = "Unknown"

    X = pd.DataFrame([row], columns=TRAIN_FEATURES)
    proba = float(MODEL.predict_proba(X)[:, 1][0])
    pred = int(proba >= THRESHOLD)

    # human-friendly reasons (simple heuristics)
    reasons = []
    amt = _safe_float(payload.get("Amount", payload.get("amount")), 0)
    hour = int(_safe_float(payload.get("Hour", payload.get("hour")), 0) or 0)
    ch = str(payload.get("Channel", payload.get("channel", "")) or "").title()
    rr = str(payload.get("RiskRating", payload.get("risk_rating", "")) or "").title()

    if amt is not None and amt >= 1_000_000:
        reasons.append("High amount ≥ NPR 1,000,000")
    if rr in {"High", "Very High"}:
        reasons.append(f"Customer risk rating: {rr}")
    if ch == "Online" and amt and amt >= 500_000:
        reasons.append("Large online transaction")
    if hour >= 22 or hour <= 5:
        reasons.append("Late night transaction")

    res = {
        "risk_score": round(proba, 3),
        "prediction": pred,
        "threshold": THRESHOLD,
        "reasons": reasons,
    }
    # echo back selected input fields for UI tables
    for k in ["Amount", "Country", "Channel", "RiskRating", "TransactionType", "Hour", "Currency"]:
        if k in payload:
            res[k.lower()] = payload[k]
    return res


# -----------------------------
# Routes
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    _ensure_model_loaded()
    return jsonify({
        "status": "ok",
        "time": _now_iso(),
        "ml_loaded": MODEL is not None,
    })


@app.route("/api/model/info", methods=["GET"])
def model_info():
    _ensure_model_loaded()
    if MODEL is None:
        return jsonify({"ml_loaded": False, "message": "Train a model first."}), 200

    # Flatten a few fields for easy consumption by the UI
    train_samples = int(META.get("train_samples", 0))
    test_samples = int(META.get("test_samples", 0))
    samples_total = train_samples + test_samples

    trained_at = META.get("trained_at") or META.get("time")  # either human or timestamp
    model_type = META.get("model_name") or (type(MODEL.named_steps["model"]).__name__ if hasattr(MODEL, "named_steps") else "Model")

    payload = {
        "ml_loaded": True,
        "threshold": THRESHOLD,
        "train_features": TRAIN_FEATURES,
        "features": len(TRAIN_FEATURES),
        "model_path": META.get("model_path"),
        "model_type": model_type,
        "samples": samples_total,
        "trained_at": trained_at,
        "model_version": 1,

        # flatten metrics
        "roc_auc": META.get("roc_auc"),
        "pr_auc": META.get("pr_auc"),
        "accuracy": META.get("accuracy"),

        # keep legacy "metrics" shape too (backward compat)
        "metrics": {
            "roc_auc": META.get("roc_auc"),
            "pr_auc": META.get("pr_auc"),
            "accuracy": META.get("accuracy"),
            "train_samples": train_samples,
            "test_samples": test_samples,
        },

        # for the Model page
        "classification_report": META.get("classification_report"),
        "confusion_matrix": META.get("confusion_matrix"),
    }
    return jsonify(payload), 200


@app.route("/api/model/reload", methods=["POST"])
def model_reload():
    global MODEL, META, THRESHOLD, TRAIN_FEATURES
    MODEL = None
    META = {}
    THRESHOLD = 0.5
    TRAIN_FEATURES = []
    _ensure_model_loaded()
    return jsonify({"ok": MODEL is not None, "ml_loaded": MODEL is not None})


@app.route("/api/predict", methods=["POST"])
@app.route("/api/transactions/score", methods=["POST"])
@app.route("/api/score", methods=["POST"])  # extra alias for the UI
def predict_one():
    payload = request.get_json(silent=True) or {}
    res = _score_txn(payload)
    if "error" in res:
        return jsonify(res), 400

    # Append to recent store for dashboard/analytics
    tx = {
        "transaction_id": f"UI_{str(uuid.uuid4())[:8]}",
        "timestamp": _now_iso(),
        "created_at": _now_iso(),
        "amount": _safe_float(payload.get("Amount", payload.get("amount")), 0),
        "country": payload.get("Country") or payload.get("country") or "Unknown",
        "channel": (payload.get("Channel") or payload.get("channel") or "Unknown").title(),
        "risk_rating": (payload.get("RiskRating") or payload.get("risk_rating") or "Unknown").title(),
        "transaction_type": (payload.get("TransactionType") or payload.get("transaction_type") or "Unknown").title(),
        "prediction": int(res["prediction"]),
        "risk_score": float(res["risk_score"]),
    }
    _recent.insert(0, tx)
    if len(_recent) > 200:
        _recent.pop()

    return jsonify(res), 200


@app.route("/api/transactions/recent", methods=["GET"])
def recent_transactions():
    limit = int(request.args.get("limit", "25"))
    return jsonify(_recent[:limit])


@app.route("/api/dashboard/metrics", methods=["GET"])
def dashboard_metrics():
    total = len(_recent)
    today_str = date.today().isoformat()
    suspicious_today = sum(1 for r in _recent if r.get("prediction") == 1 and (r.get("timestamp", "")).startswith(today_str))
    avg_risk = round(float(np.mean([r["risk_score"] for r in _recent])) if _recent else 0.0, 3)
    active_alerts = sum(1 for r in _recent if r.get("prediction") == 1)  # simplistic

    return jsonify({
        "total_transactions": total,
        "suspicious_today": suspicious_today,
        "avg_risk_score": avg_risk,
        "active_alerts": active_alerts,
    })


@app.route("/api/analytics/country_volume", methods=["GET"])
def analytics_country_volume():
    if not _recent:
        return jsonify([])
    counts = pd.DataFrame(_recent).groupby("country").size().sort_values(ascending=False)
    return jsonify([{"country": k, "count": int(v)} for k, v in counts.items()])


@app.route("/api/analytics/risk_rating", methods=["GET"])
def analytics_risk_rating():
    """Risk rating distribution (High/Medium/Low/Unknown)."""
    if not _recent:
        return jsonify([])
    df = pd.DataFrame(_recent)
    if "risk_rating" not in df:
        return jsonify([])
    counts = (
        df["risk_rating"]
        .fillna("Unknown")
        .astype(str)
        .str.title()
        .value_counts()
    )
    order = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    out = [{"risk": k, "count": int(v)} for k, v in counts.items()]
    out.sort(key=lambda d: order.get(d["risk"], 99))
    return jsonify(out)


@app.route("/api/analytics/prediction_breakdown", methods=["GET"])
def analytics_prediction_breakdown():
    """Prediction breakdown (Normal vs Suspicious)."""
    if not _recent:
        return jsonify([])
    df = pd.DataFrame(_recent)
    if "prediction" not in df:
        return jsonify([])
    counts = df["prediction"].map({1: "Suspicious", 0: "Normal"}).value_counts()
    return jsonify([{"label": k, "count": int(v)} for k, v in counts.items()])


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """
    Accept a CSV with columns similar to your scoring form and pre-populate recent store.
    Body JSON: {"rows":[{...}, {...}] }  OR multipart file 'file'
    """
    added = 0
    rows = []

    if request.files:
        f = request.files.get("file")
        if f:
            df = pd.read_csv(f)
            rows = df.to_dict(orient="records")
    else:
        data = request.get_json(silent=True) or {}
        rows = data.get("rows", [])

    for row in rows:
        res = _score_txn(row)
        if "error" in res:
            continue
        tx = {
            "transaction_id": f"UP_{str(uuid.uuid4())[:8]}",
            "timestamp": _now_iso(),
            "created_at": _now_iso(),
            "amount": _safe_float(row.get("Amount", row.get("amount")), 0),
            "country": row.get("Country") or row.get("country") or "Unknown",
            "channel": (row.get("Channel") or row.get("channel") or "Unknown").title(),
            "risk_rating": (row.get("RiskRating") or row.get("risk_rating") or "Unknown").title(),
            "transaction_type": (row.get("TransactionType") or row.get("transaction_type") or "Unknown").title(),
            "prediction": int(res["prediction"]),
            "risk_score": float(res["risk_score"]),
        }
        _recent.insert(0, tx)
        added += 1

    if len(_recent) > 500:
        del _recent[500:]

    return jsonify({"added": added})


if __name__ == "__main__":
    _ensure_model_loaded()
    app.run(host=API_HOST, port=API_PORT, debug=False)
