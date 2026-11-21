import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.forecast_core import (
    HORIZON,
    build_training_set,
    compute_metrics,
    recursive_forecast,
)

MAPE_THRESHOLD = float(os.environ.get("MAPE_THRESHOLD", "15"))


def make_model():
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]
    return df[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)


def _atomic_write(path, writer):
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)


def train_and_register(df, model_dir, threshold=MAPE_THRESHOLD, force=False):
    """Evaluate on the last 7 days; save the model only if it passes."""
    close = df["close"].astype(float).values
    if len(close) < 200:
        raise ValueError("Need at least 200 rows of data to train")
    train, test = close[:-HORIZON], close[-HORIZON:]
    model = make_model().fit(*build_training_set(train))
    metrics = compute_metrics(test, recursive_forecast(model, train, HORIZON))
    passed = metrics["mape"] <= threshold
    result = {"registered": False, "passed": passed, "threshold": threshold, **metrics}
    if not (passed or force):
        return result
    final = make_model().fit(*build_training_set(close))  # refit on all data
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    artifact = {
        "model": final,
        "history": [float(v) for v in close[-120:]],
        "last_date": pd.Timestamp(df["date"].iloc[-1]).strftime("%Y-%m-%d"),
        "version": version,
    }
    os.makedirs(model_dir, exist_ok=True)
    _atomic_write(os.path.join(model_dir, "model.joblib"), lambda p: joblib.dump(artifact, p))
    meta = {**metrics, "model_version": version, "trained_at": version[1:]}

    def write_json(p):
        with open(p, "w") as fh:
            json.dump(meta, fh)

    _atomic_write(os.path.join(model_dir, "metrics.json"), write_json)
    result.update(registered=True, model_version=version)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="clean CSV (date, close)")
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models"))
    ap.add_argument("--threshold", type=float, default=MAPE_THRESHOLD)
    ap.add_argument("--if-missing", action="store_true", help="skip if a model exists")
    a = ap.parse_args()
    if a.if_missing and os.path.exists(os.path.join(a.model_dir, "model.joblib")):
        print("Model already exists - nothing to do.")
        raise SystemExit(0)
    if a.input:
        data = load_csv(a.input)
    else:
        from src.etl_fetch import fetch_btc

        data = fetch_btc()
    print(json.dumps(train_and_register(data, a.model_dir, a.threshold, a.if_missing)))
