import argparse
import os

import numpy as np
import pandas as pd


def preprocess(raw_path, out_path):
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df = df[df["close"] > 0].set_index("date")
    df = df.asfreq("D").ffill().reset_index()
    df["daily_return"] = np.log(df["close"]).diff()
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_30"] = df["close"].rolling(30).mean()
    df["volatility_7"] = df["daily_return"].rolling(7).std()
    df["volatility_30"] = df["daily_return"].rolling(30).std()
    df["target_next_close"] = df["close"].shift(-1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/btc_raw.csv")
    ap.add_argument("--out", default="data/processed/btc_clean.csv")
    a = ap.parse_args()
    print("rows:", preprocess(a.raw, a.out))
