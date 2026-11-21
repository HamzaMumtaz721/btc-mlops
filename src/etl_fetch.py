import argparse
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
START_DATE = "2018-01-01"
BINANCE_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
DAY_MS = 86_400_000


def synthetic_btc(n_days=1500, seed=42):
    """Fake random-walk prices. Used for tests / when internet is down."""
    rng = np.random.default_rng(seed)
    close = 20000 * np.exp(np.cumsum(rng.normal(0.0008, 0.03, n_days)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.uniform(1e9, 5e10, n_days),
        }
    )


def fetch_yfinance(start):
    import yfinance as yf

    df = yf.download("BTC-USD", start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns=str.lower)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"])
    return df


def _binance_klines(url, start_ms):
    rows = []
    while True:
        params = {"symbol": "BTCUSDT", "interval": "1d", "startTime": start_ms, "limit": 1000}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + DAY_MS
        if len(batch) < 1000:
            break
    return rows


def fetch_binance(start):
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    last_error = None
    for url in BINANCE_URLS:
        try:
            rows = _binance_klines(url, start_ms)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        raise RuntimeError(f"Binance failed: {last_error}")
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    df = df[df["date"] < today]  # drop today's unfinished candle
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_btc(start=START_DATE):
    allow = os.environ.get("ALLOW_SYNTHETIC_FALLBACK", "true").lower() == "true"
    for name, func in [("yfinance", fetch_yfinance), ("Binance", fetch_binance)]:
        try:
            df = func(start)
            if len(df) > 0:
                log.info("Fetched %d rows from %s", len(df), name)
                return df
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed: %s", name, exc)
    if not allow:
        raise RuntimeError("All data sources failed")
    log.warning("All real sources failed. Using SYNTHETIC data.")
    return synthetic_btc()


def save_raw(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/btc_raw.csv")
    args = ap.parse_args()
    print("rows saved:", save_raw(fetch_btc(), args.out))
