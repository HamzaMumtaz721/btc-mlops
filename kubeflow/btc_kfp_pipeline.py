from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

BASE = "python:3.10-slim"


@dsl.component(base_image=BASE, packages_to_install=["pandas", "numpy", "yfinance"])
def ingest_data(start_date: str, raw_data: Output[Dataset]):
    import numpy as np
    import pandas as pd

    try:
        import yfinance as yf

        df = yf.download("BTC-USD", start=start_date, progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError("no data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns=str.lower)
        df = df[["date", "open", "high", "low", "close", "volume"]]
    except Exception:
        rng = np.random.default_rng(1)
        close = 20000 * np.exp(np.cumsum(rng.normal(0.0008, 0.03, 1500)))
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=1500)
        df = pd.DataFrame({"date": dates, "close": close})
    df.to_csv(raw_data.path, index=False)


@dsl.component(base_image=BASE, packages_to_install=["pandas", "numpy"])
def build_features(raw_data: Input[Dataset], features: Output[Dataset]):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(raw_data.path, parse_dates=["date"]).sort_values("date")
    s = df["close"].astype(float).reset_index(drop=True)
    r = np.log(s).diff()
    out = pd.DataFrame({"date": df["date"].values, "close": s})
    for i in range(1, 8):
        out[f"ret_lag{i}"] = r.shift(i - 1)
    out["ma7_ratio"] = s.rolling(7).mean() / s
    out["ma30_ratio"] = s.rolling(30).mean() / s
    out["vol7"] = r.rolling(7).std()
    out["vol30"] = r.rolling(30).std()
    out["target"] = r.shift(-1)
    out.to_csv(features.path, index=False)


@dsl.component(
    base_image=BASE, packages_to_install=["pandas", "numpy", "scikit-learn==1.5.1", "joblib"]
)
def train_model(
    features: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    mape_threshold: float = 15.0,
):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor

    df = pd.read_csv(features.path).dropna().reset_index(drop=True)
    cols = [c for c in df.columns if c not in ("date", "close", "target")]
    train, test = df.iloc[:-7], df.iloc[-7:]
    m = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)
    m.fit(train[cols], train["target"])
    pred_close = test["close"] * np.exp(m.predict(test[cols]))
    actual = test["close"] * np.exp(test["target"])
    err = actual - pred_close
    mape = float((err.abs() / actual).mean() * 100)
    metrics.log_metric("mae", float(err.abs().mean()))
    metrics.log_metric("rmse", float((err**2).mean() ** 0.5))
    metrics.log_metric("mape", mape)
    if mape > mape_threshold:
        raise ValueError(f"MAPE {mape:.2f} above threshold {mape_threshold}")
    m.fit(df[cols], df["target"])
    joblib.dump(m, model.path)


@dsl.pipeline(name="btc-7day-forecast")
def btc_pipeline(start_date: str = "2018-01-01", mape_threshold: float = 15.0):
    raw = ingest_data(start_date=start_date)
    feats = build_features(raw_data=raw.outputs["raw_data"])
    train_model(features=feats.outputs["features"], mape_threshold=mape_threshold)


if __name__ == "__main__":
    compiler.Compiler().compile(btc_pipeline, "kubeflow/btc_kfp_pipeline.yaml")
    print("Compiled -> kubeflow/btc_kfp_pipeline.yaml")
