"""
DAG 1: dag_ingest — Daily ingestion from multiple data sources.

Demonstrates:
    - EmptyOperator:      start marker (replace with FileSensor in production)
    - PythonOperator:     six parallel fetch tasks
    - XCom:               each fetch pushes its output path; validate pulls them
    - TriggerRule:        validate_raw_data runs only if ALL upstream succeed
    - TriggerDagRunOperator: chains to dag_transform after success
    - Retries:            each fetch retries 3x with 5-min delay
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import TriggerRule

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
default_args = {
    "owner": "oil-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------
def _fetch_oil_prices(**context):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_oil_prices",
        str(PROJECT_ROOT / "scripts" / "ingest" / "fetch_oil_prices.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output_path = mod.fetch(data_dir=str(DATA_RAW / "prices"))
    context["ti"].xcom_push(key="output_path", value=output_path)
    return output_path


def _fetch_reddit_sentiment(**context):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_reddit_sentiment",
        str(PROJECT_ROOT / "scripts" / "ingest" / "fetch_reddit_sentiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output_path = mod.fetch(data_dir=str(DATA_RAW / "reddit"))
    context["ti"].xcom_push(key="output_path", value=output_path)
    return output_path


def _fetch_rss_news(**context):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_rss_news",
        str(PROJECT_ROOT / "scripts" / "ingest" / "fetch_rss_news.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output_path = mod.fetch(data_dir=str(DATA_RAW / "news"))
    context["ti"].xcom_push(key="output_path", value=output_path)
    return output_path


def _fetch_trump_statements(**context):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_trump_statements",
        str(PROJECT_ROOT / "scripts" / "ingest" / "fetch_trump_statements.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output_path = mod.fetch(data_dir=str(DATA_RAW / "trump"))
    context["ti"].xcom_push(key="output_path", value=output_path)
    return output_path


def _fetch_war_news(**context):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_war_news",
        str(PROJECT_ROOT / "scripts" / "ingest" / "fetch_war_news.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output_path = mod.fetch(data_dir=str(DATA_RAW / "war_news"))
    context["ti"].xcom_push(key="output_path", value=output_path)
    return output_path


def _fetch_taiwan_tensions(**context):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_taiwan_tensions",
        str(PROJECT_ROOT / "scripts" / "ingest" / "fetch_taiwan_tensions.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output_path = mod.fetch(data_dir=str(DATA_RAW / "taiwan"))
    context["ti"].xcom_push(key="output_path", value=output_path)
    return output_path


def _validate_raw_data(**context):
    ti = context["ti"]
    paths = {
        "prices": ti.xcom_pull(task_ids="fetch_oil_prices", key="output_path"),
        "reddit": ti.xcom_pull(task_ids="fetch_reddit_sentiment", key="output_path"),
        "news": ti.xcom_pull(task_ids="fetch_rss_news", key="output_path"),
        "trump": ti.xcom_pull(task_ids="fetch_trump_statements", key="output_path"),
        "war_news": ti.xcom_pull(task_ids="fetch_war_news", key="output_path"),
        "taiwan": ti.xcom_pull(task_ids="fetch_taiwan_tensions", key="output_path"),
    }
    for source, path in paths.items():
        if path and Path(path).exists():
            size = Path(path).stat().st_size
            logger.info("OK %s — %s (%d bytes)", source, path, size)
        else:
            logger.warning("MISSING %s — file missing or path is None: %s", source, path)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_ingest",
    default_args=default_args,
    description="Daily ingestion: oil prices, Reddit sentiment, RSS news, geopolitics",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingest", "oil-pulse"],
) as dag:

    start_sensor = EmptyOperator(task_id="start_sensor")

    fetch_prices = PythonOperator(task_id="fetch_oil_prices", python_callable=_fetch_oil_prices)
    fetch_reddit = PythonOperator(task_id="fetch_reddit_sentiment", python_callable=_fetch_reddit_sentiment)
    fetch_news = PythonOperator(task_id="fetch_rss_news", python_callable=_fetch_rss_news)
    fetch_trump = PythonOperator(task_id="fetch_trump_statements", python_callable=_fetch_trump_statements)
    fetch_war = PythonOperator(task_id="fetch_war_news", python_callable=_fetch_war_news)
    fetch_taiwan = PythonOperator(task_id="fetch_taiwan_tensions", python_callable=_fetch_taiwan_tensions)

    validate = PythonOperator(
        task_id="validate_raw_data",
        python_callable=_validate_raw_data,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    trigger_transform = TriggerDagRunOperator(
        task_id="trigger_transform",
        trigger_dag_id="dag_transform",
        wait_for_completion=False,
    )

    start_sensor >> [fetch_prices, fetch_reddit, fetch_news, fetch_trump, fetch_war, fetch_taiwan] >> validate >> trigger_transform
