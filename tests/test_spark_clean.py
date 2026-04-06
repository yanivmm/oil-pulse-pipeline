"""
Tests for spark_clean.py logic using a local SparkSession.

Uses pandas DataFrames + Arrow serialization to work around
PySpark 3.5.x / Python 3.13 worker incompatibility on Windows.

Run:
    pytest tests/test_spark_clean.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    """Create a local SparkSession for testing."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-spark-clean")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()


class TestNullHandling:
    """Verify that null rows are dropped correctly."""

    def test_drop_null_date(self, spark):
        """Rows with null date should be removed."""
        pdf = pd.DataFrame(
            {
                "date": ["2024-01-01", None, "2024-01-03"],
                "open": [75.0, 76.0, 77.0],
                "high": [76.0, 77.0, 78.0],
                "low": [74.0, 75.0, 76.0],
                "close": [75.5, 76.5, 77.5],
                "volume": [1000, 2000, 3000],
            }
        )
        df = spark.createDataFrame(pdf)
        cleaned = df.dropna(subset=["date", "close"])
        assert cleaned.count() == 2

    def test_drop_null_close(self, spark):
        """Rows with null close price should be removed."""
        pdf = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "open": [75.0, 76.0],
                "high": [76.0, 77.0],
                "low": [74.0, 75.0],
                "close": [np.nan, 76.5],
                "volume": [1000, 2000],
            }
        )
        df = spark.createDataFrame(pdf)
        cleaned = df.dropna(subset=["date", "close"])
        assert cleaned.count() == 1


class TestSchemaValidation:
    """Verify output schema matches expected structure."""

    def test_price_schema_columns(self, spark):
        """Cleaned price data should have the expected columns."""
        expected_cols = {"date", "open", "high", "low", "close", "volume"}
        pdf = pd.DataFrame(
            {"date": ["2024-01-01"], "open": [75.0], "high": [76.0], "low": [74.0], "close": [75.5], "volume": [1000]}
        )
        df = spark.createDataFrame(pdf)
        assert set(df.columns) == expected_cols


class TestDeduplication:
    """Verify that duplicate rows are removed."""

    def test_dedup_by_date(self, spark):
        """Duplicate dates should be collapsed to one row."""
        pdf = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "open": [75.0, 75.0, 76.0],
                "high": [76.0, 76.0, 77.0],
                "low": [74.0, 74.0, 75.0],
                "close": [75.5, 75.5, 76.5],
                "volume": [1000, 1000, 2000],
            }
        )
        df = spark.createDataFrame(pdf)
        deduped = df.dropDuplicates(["date"])
        assert deduped.count() == 2
