import numpy as np
import pandas as pd

LAGS = 7
HORIZON = 7
MIN_HISTORY = 31


def feature_frame(close):
    """Build features (lagged returns, moving averages, volatility) per day."""
    s = pd.Series(close, dtype="float64").reset_index(drop=True)
    r = np.log(s).diff()
    feats = {f"ret_lag{i}": r.shift(i - 1) for i in range(1, LAGS + 1)}
    feats["ma7_ratio"] = s.rolling(7).mean() / s
    feats["ma30_ratio"] = s.rolling(30).mean() / s
    feats["vol7"] = r.rolling(7).std()
    feats["vol30"] = r.rolling(30).std()
    return pd.DataFrame(feats)


def build_training_set(close):
    """Target = next day's log return."""
    s = pd.Series(close, dtype="float64").reset_index(drop=True)
    x = feature_frame(s)
    y = np.log(s).diff().shift(-1)
    mask = x.notna().all(axis=1) & y.notna()
    return x[mask], y[mask]


def recursive_forecast(model, close, days=HORIZON):
    """Predict day by day, feeding each prediction back in as history."""
    history = [float(v) for v in close]
    out = []
    for _ in range(days):
        row = feature_frame(history[-120:]).iloc[[-1]]
        ret = float(model.predict(row)[0])
        nxt = history[-1] * float(np.exp(ret))
        history.append(nxt)
        out.append(nxt)
    return out


def compute_metrics(actual, predicted):
    a = np.asarray(actual, dtype="float64")
    p = np.asarray(predicted, dtype="float64")
    err = a - p
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err / a)) * 100),
    }
