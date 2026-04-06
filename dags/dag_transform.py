"""
DAG 2: dag_transform — Daily Spark transformations.

Demonstrates:
    - ExternalTaskSensor: waits for dag_ingest to complete before starting
    - BashOperator:       calls each PySpark script in sequence
    - BranchPythonOperator: routes to features or notify_low_data based on volume
    - XCom:               passes record counts between tasks
    - TaskGroup:          groups Spark tasks visually as 'spark_processing'
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.sdk import TaskGroup

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts" / "transform"
PROCESSED = PROJECT_ROOT / "data" / "processed"

logger = logging.getLogger(__name__)

MIN_RECORD_THRESHOLD = 10  # minimum rows to proceed with feature engineering

# ---------------------------------------------------------------------------
default_args = {
    "owner": "oil-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ---------------------------------------------------------------------------
# Branch logic
# ---------------------------------------------------------------------------
def _check_data_volume(**context):
    """Check if spark_clean produced enough rows. Return the next task id."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="spark_processing.spark_clean", key="record_count")

    # If XCom is missing, try reading from the clean output directory
    if record_count is None:
        clean_dir = PROCESSED / "clean"
        if clean_dir.exists() and any(clean_dir.glob("*.parquet")):
            record_count = MIN_RECORD_THRESHOLD + 1  # assume enough
            logger.info("XCom missing; clean parquet exists — assuming sufficient data.")
        else:
            record_count = 0

    record_count = int(record_count)
    logger.info("Record count from spark_clean: %d (threshold: %d)", record_count, MIN_RECORD_THRESHOLD)

    if record_count >= MIN_RECORD_THRESHOLD:
        return "spark_processing.spark_features"
    return "notify_low_data"


def _parse_spark_output(**context):
    """Parse record count printed by spark_clean.py from BashOperator output."""
    ti = context["ti"]
    raw_output = ti.xcom_pull(task_ids="spark_processing.spark_clean")
    if raw_output:
        for line in str(raw_output).strip().split("\n"):
            if line.strip().isdigit():
                ti.xcom_push(key="record_count", value=int(line.strip()))
                return
    ti.xcom_push(key="record_count", value=0)


def _notify_low_data(**context):
    """Log a warning when data volume is below threshold."""
    logger.warning(
        "Data volume below threshold (%d). "
        "Skipping feature engineering and aggregation.",
        MIN_RECORD_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_transform",
    default_args=default_args,
    description="Daily Spark transforms: clean → features → aggregate",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["transform", "spark", "oil-pulse"],
) as dag:

    # Wait for the ingest DAG's most recent successful run
    wait_for_ingest = ExternalTaskSensor(
        task_id="wait_for_ingest",
        external_dag_id="dag_ingest",
        external_task_id=None,  # wait for entire DAG
        allowed_states=["success"],
        poke_interval=60,
        timeout=3600,
        mode="reschedule",
    )

    # --- Spark processing task group ----------------------------------------
    with TaskGroup("spark_processing") as spark_group:

        spark_clean = BashOperator(
            task_id="spark_clean",
            bash_command=f"python {SCRIPTS / 'spark_clean.py'}",
            do_xcom_push=True,  # captures last line of stdout
        )

        # Parse the stdout output to extract record count into XCom
        parse_output = PythonOperator(
            task_id="parse_clean_output",
            python_callable=_parse_spark_output,
        )

        spark_features = BashOperator(
            task_id="spark_features",
            bash_command=f"python {SCRIPTS / 'spark_features.py'}",
        )

        spark_aggregate = BashOperator(
            task_id="spark_aggregate",
            bash_command=f"python {SCRIPTS / 'spark_aggregate.py'}",
        )

        spark_clean >> parse_output
        spark_features >> spark_aggregate

    # Branch: enough data → features, else → notify
    check_volume = BranchPythonOperator(
        task_id="check_data_volume",
        python_callable=_check_data_volume,
    )

    notify_low = PythonOperator(
        task_id="notify_low_data",
        python_callable=_notify_low_data,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # --- Dependencies -------------------------------------------------------
    # wait → clean → parse → branch → [features → aggregate] OR [notify]
    wait_for_ingest >> spark_clean >> parse_output >> check_volume
    check_volume >> spark_features >> spark_aggregate >> end
    check_volume >> notify_low >> end
