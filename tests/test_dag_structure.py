"""
Test DAG structure: load DAGs via DagBag, check imports, task counts, dependencies.

Run:
    pytest tests/test_dag_structure.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so DAGs can resolve relative imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Point Airflow at the dags folder
DAGS_DIR = str(PROJECT_ROOT / "dags")


@pytest.fixture(scope="module")
def dagbag():
    """Load all DAGs from the dags directory."""
    # Avoid Airflow DB initialization overhead
    os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "True")
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
    # Disable import timeout — signal.SIGALRM is unavailable on Windows
    os.environ["AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT"] = "0"

    from airflow.models import DagBag

    bag = DagBag(dag_folder=DAGS_DIR, include_examples=False)
    return bag


class TestDagIngest:
    """Tests for dag_ingest."""

    def test_no_import_errors(self, dagbag):
        """dag_ingest must load without import errors."""
        assert "dag_ingest" in dagbag.dags, f"dag_ingest not found. Errors: {dagbag.import_errors}"
        assert dagbag.import_errors.get(os.path.join(DAGS_DIR, "dag_ingest.py")) is None

    def test_task_count(self, dagbag):
        """dag_ingest should have exactly 9 tasks (6 fetchers + sensor + validate + trigger)."""
        dag = dagbag.dags["dag_ingest"]
        assert len(dag.tasks) == 9

    def test_dependencies(self, dagbag):
        """Verify fan-out/fan-in pattern: sensor → 5 fetchers → validate."""
        dag = dagbag.dags["dag_ingest"]
        sensor = dag.get_task("start_sensor")
        validate = dag.get_task("validate_raw_data")

        # Sensor has 6 downstream tasks
        downstream_ids = {t.task_id for t in sensor.downstream_list}
        assert downstream_ids == {
            "fetch_oil_prices", "fetch_reddit_sentiment", "fetch_rss_news",
            "fetch_trump_statements", "fetch_war_news", "fetch_taiwan_tensions",
        }

        # Validate has 6 upstream tasks
        upstream_ids = {t.task_id for t in validate.upstream_list}
        assert upstream_ids == {
            "fetch_oil_prices", "fetch_reddit_sentiment", "fetch_rss_news",
            "fetch_trump_statements", "fetch_war_news", "fetch_taiwan_tensions",
        }


class TestDagTransform:
    """Tests for dag_transform."""

    def test_no_import_errors(self, dagbag):
        """dag_transform must load without import errors."""
        assert "dag_transform" in dagbag.dags, f"dag_transform not found. Errors: {dagbag.import_errors}"

    def test_task_count(self, dagbag):
        """dag_transform should have the expected number of tasks."""
        dag = dagbag.dags["dag_transform"]
        # spark_clean, parse_clean_output, check_data_volume,
        # spark_features, spark_aggregate, notify_low_data, end, trigger_predict
        assert len(dag.tasks) == 8

    def test_triggered_schedule(self, dagbag):
        """dag_transform has no schedule (triggered by dag_ingest)."""
        dag = dagbag.dags["dag_transform"]
        assert dag.schedule is None


class TestDagPredictPublish:
    """Tests for dag_predict_publish."""

    def test_no_import_errors(self, dagbag):
        """dag_predict_publish must load without import errors."""
        assert "dag_predict_publish" in dagbag.dags

    def test_task_count(self, dagbag):
        """dag_predict_publish should have exactly 6 tasks."""
        dag = dagbag.dags["dag_predict_publish"]
        assert len(dag.tasks) == 6

    def test_linear_chain(self, dagbag):
        """Tasks should form a linear chain."""
        dag = dagbag.dags["dag_predict_publish"]
        expected_chain = [
            "check_file_exists",
            "train_model",
            "predict",
            "war_exit_model",
            "load_duckdb",
            "notify_success",
        ]
        for i in range(len(expected_chain) - 1):
            task = dag.get_task(expected_chain[i])
            downstream_ids = {t.task_id for t in task.downstream_list}
            assert expected_chain[i + 1] in downstream_ids
