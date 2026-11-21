import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone

import joblib
from flask import Flask, jsonify, request

from src.forecast_core import MIN_HISTORY, recursive_forecast

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("btc-api")
app = Flask(__name__)
START_TIME = time.time()
_state = {"artifact": None, "mtime": None}


def model_dir():
    return os.environ.get("MODEL_DIR", "/app/models")


def get_artifact():
    """Load the model file; auto-reload when the file changes on disk."""
    path = os.path.join(model_dir(), "model.joblib")
    if not os.path.exists(path):
        return _state["artifact"]
    mtime = os.path.getmtime(path)
    if _state["mtime"] != mtime:
        try:
            _state["artifact"] = joblib.load(path)
            log.info("Loaded model %s", _state["artifact"]["version"])
        except Exception:  # noqa: BLE001
            log.exception("Could not load model file - keeping previous model")
        _state["mtime"] = mtime
    return _state["artifact"]


def error(message, code):
    return jsonify({"error": message}), code


@app.get("/health")
def health():
    art = get_artifact()
    return jsonify(
        {
            "status": "ok",
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_version": art["version"] if art else "none",
            "model_stage": os.environ.get("MODEL_STAGE", "dev"),
        }
    )


def _valid_history(h):
    return (
        isinstance(h, list)
        and len(h) >= MIN_HISTORY
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
            for v in h
        )
    )


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        if request.get_data() and payload is None:
            return error("Request body must be valid JSON", 400)
        payload = payload or {}
        if not isinstance(payload, dict):
            return error("JSON body must be an object", 400)

        days = payload.get("forecast_days", 7)
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 30:
            return error("forecast_days must be an integer between 1 and 30", 400)

        art = get_artifact()
        if art is None:
            return error("No model available yet", 503)

        history = payload.get("history")
        if history is not None:
            if not _valid_history(history):
                return error(f"history must be a list of >= {MIN_HISTORY} positive numbers", 400)
            last_date = datetime.now(timezone.utc).date()
        else:
            history = art["history"]
            last_date = datetime.strptime(art["last_date"], "%Y-%m-%d").date()

        preds = recursive_forecast(art["model"], history, days)
        out = [
            {"date": str(last_date + timedelta(days=i + 1)), "predicted_close": round(p, 2)}
            for i, p in enumerate(preds)
        ]
        return jsonify({"model_version": art["version"], "forecast_days": days, "predictions": out})
    except Exception:  # noqa: BLE001
        log.exception("Prediction failed")
        return error("Internal server error", 500)


@app.get("/metrics")
def metrics():
    try:
        path = os.path.join(model_dir(), "metrics.json")
        if not os.path.exists(path):
            return error("No metrics available yet", 404)
        with open(path) as fh:
            return jsonify(json.load(fh))
    except Exception:  # noqa: BLE001
        log.exception("Metrics failed")
        return error("Internal server error", 500)


@app.post("/reload")
def reload_model():
    key = os.environ.get("RELOAD_API_KEY")
    if key and request.headers.get("X-API-Key") != key:
        return error("Unauthorized", 401)
    art = get_artifact()
    if art is None:
        return error("No model file found", 404)
    return jsonify({"status": "reloaded", "model_version": art["version"]})


@app.errorhandler(404)
def not_found(_):
    return error("Not found", 404)


@app.errorhandler(405)
def bad_method(_):
    return error("Method not allowed", 405)
