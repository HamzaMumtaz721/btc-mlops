import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/airflow/models")
API_URL = os.environ.get("API_URL", "http://api:5000")
RAW = f"{DATA_DIR}/raw/btc_raw.csv"
CLEAN = f"{DATA_DIR}/processed/btc_clean.csv"


def fetch_data():
    from src.etl_fetch import fetch_btc, save_raw

    rows = save_raw(fetch_btc(), RAW)
    print(f"Saved {rows} rows to {RAW}")
    return rows


def preprocess_data():
    from src.preprocess import preprocess

    rows = preprocess(RAW, CLEAN)
    print(f"Saved {rows} processed rows to {CLEAN}")
    return rows


def train_and_evaluate():
    from src.train_forecast import load_csv, train_and_register

    result = train_and_register(load_csv(CLEAN), MODEL_DIR)
    print("Training result:", result)
    return result  # stored in XCom for the next task


def reload_api(ti):
    result = ti.xcom_pull(task_ids="train_and_evaluate")
    if not result or not result.get("registered"):
        raise AirflowSkipException("Model did not pass threshold - not reloading")
    headers = {"X-API-Key": os.environ.get("RELOAD_API_KEY", "")}
    resp = requests.post(f"{API_URL}/reload", headers=headers, timeout=30)
    resp.raise_for_status()
    print("API reloaded:", resp.json())


with DAG(
    dag_id="btc_etl_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["btc", "mlops"],
) as dag:
    t1 = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t2 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t3 = PythonOperator(task_id="train_and_evaluate", python_callable=train_and_evaluate)
    t4 = PythonOperator(task_id="reload_api", python_callable=reload_api)
    t1 >> t2 >> t3 >> t4
