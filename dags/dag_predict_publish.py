"""
DAG 3: dag_predict_publish — Daily ML prediction and DuckDB load.

Demonstrates:
    - ExternalTaskSensor:  waits for dag_transform to complete
    - ShortCircuitOperator: skips the entire pipeline if processed data missing
    - PythonOperator:       runs train → predict → load_duckdb in sequence
    - EmailOperator:        mocked success notification (no real SMTP)
    - SLA:                  predict task must finish within 30 minutes
    - on_failure_callback:  logs a custom error message if any task fails
"""

from __future__ import annotations

import importlib.util
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ML = PROJECT_ROOT / "scripts" / "ml"
PROCESSED = PROJECT_ROOT / "data" / "processed"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _on_failure(context):
    """Custom callback executed when any task in this DAG fails."""
    task_id = context.get("task_instance").task_id
    dag_id = context.get("task_instance").dag_id
    exec_date = context.get("execution_date")
    logger.error(
        "TASK FAILED — dag=%s task=%s execution_date=%s",
        dag_id,
        task_id,
        exec_date,
    )


def _sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Called when a task misses its defined SLA."""
    logger.warning(
        "SLA MISS — dag=%s tasks=%s blocking=%s",
        dag.dag_id,
        [t.task_id for t in task_list],
        [t.task_id for t in blocking_tis],
    )


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------
def _check_file_exists(**context):
    """Return True if processed features exist, False to short-circuit."""
    features_dir = PROCESSED / "features"
    exists = features_dir.exists() and any(features_dir.glob("*.parquet"))
    if not exists:
        logger.warning("No processed features found at %s — skipping pipeline.", features_dir)
    return exists


def _run_script(script_name: str):
    """Return a callable that dynamically imports and runs a script's main()."""
    def _inner(**context):
        script_path = SCRIPTS_ML / script_name
        spec = importlib.util.spec_from_file_location(
            script_name.replace(".py", ""),
            str(script_path),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    return _inner


# ---------------------------------------------------------------------------
# Default args with on_failure_callback
# ---------------------------------------------------------------------------
default_args = {
    "owner": "oil-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": _on_failure,
}

# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_predict_publish",
    default_args=default_args,
    description="Daily: train model → predict → load DuckDB → notify",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    sla_miss_callback=_sla_miss_callback,
    tags=["ml", "predict", "oil-pulse"],
) as dag:

    wait_for_transform = ExternalTaskSensor(
        task_id="wait_for_transform",
        external_dag_id="dag_transform",
        external_task_id=None,
        allowed_states=["success"],
        poke_interval=60,
        timeout=3600,
        mode="reschedule",
    )

    check_file = ShortCircuitOperator(
        task_id="check_file_exists",
        python_callable=_check_file_exists,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_run_script("train_model.py"),
    )

    predict = PythonOperator(
        task_id="predict",
        python_callable=_run_script("predict.py"),
        sla=timedelta(minutes=30),
    )

    war_exit = PythonOperator(
        task_id="war_exit_model",
        python_callable=_run_script("war_exit_model.py"),
    )

    load_duckdb = PythonOperator(
        task_id="load_duckdb",
        python_callable=_run_script("load_duckdb.py"),
    )

    # Success marker — in production, replace with EmailOperator or Slack
    notify_success = EmptyOperator(
        task_id="notify_success",
    )

    # --- Dependency chain ---------------------------------------------------
    wait_for_transform >> check_file >> train_model >> predict >> war_exit >> load_duckdb >> notify_success
