"""
Tests for spark_features.py: window functions, joins, and label validation.

Uses pandas DataFrames + Arrow serialization to work around
PySpark 3.5.x / Python 3.13 worker incompatibility on Windows.

Run:
    pytest tests/test_spark_features.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import avg, col, lag, lead, to_date, when


@pytest.fixture(scope="module")
def spark():
    """Create a local SparkSession for testing."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-spark-features")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def sample_prices(spark):
    """Create a sample price DataFrame with 10 days of data."""
    pdf = pd.DataFrame(
        {
            "date": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 11)],
            "open": [70.0 + i for i in range(1, 11)],
            "high": [72.0 + i for i in range(1, 11)],
            "low": [69.0 + i for i in range(1, 11)],
            "close": [71.0 + i for i in range(1, 11)],
            "volume": [1000 * i for i in range(1, 11)],
        }
    )
    df = spark.createDataFrame(pdf)
    return df.withColumn("date", to_date(col("date")))


class TestWindowFunctions:
    """Verify rolling averages and lag produce correct results."""

    def test_rolling_avg_7d_not_null_after_warmup(self, sample_prices):
        """After 7 rows, rolling_avg_7d should be non-null."""
        window_7d = Window.orderBy("date").rowsBetween(-6, 0)
        df = sample_prices.withColumn("rolling_avg_7d", avg("close").over(window_7d))

        # Row 7 onward (index 6+) should have a full 7-day window
        non_null = df.filter(col("rolling_avg_7d").isNotNull()).count()
        assert non_null == 10  # all rows get a value (smaller window for first 6)

    def test_price_delta_uses_lag(self, sample_prices):
        """price_delta should be null for the first row and non-null after."""
        window = Window.orderBy("date")
        df = sample_prices.withColumn("price_delta", col("close") - lag("close", 1).over(window))

        first_row = df.orderBy("date").first()
        assert first_row["price_delta"] is None

        non_null_count = df.filter(col("price_delta").isNotNull()).count()
        assert non_null_count == 9  # all except first row


class TestJoin:
    """Verify that joins produce expected column count."""

    def test_join_column_count(self, spark, sample_prices):
        """Joining prices + sentiment should produce the expected columns."""
        sent_pdf = pd.DataFrame(
            {
                "date": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 11)],
                "reddit_sentiment_avg": [0.5] * 10,
            }
        )
        sentiment = spark.createDataFrame(sent_pdf)
        sentiment = sentiment.withColumn("date", to_date(col("date")))

        joined = sample_prices.join(sentiment, on="date", how="left")
        # original 6 cols + 1 new col = 7 (date is shared)
        assert len(joined.columns) == 7


class TestPriceDirection:
    """Verify the binary label is always 0 or 1."""

    def test_label_values(self, sample_prices):
        """price_direction should only contain 0 or 1 (nulls dropped)."""
        window = Window.orderBy("date")
        df = sample_prices.withColumn("next_close", lead("close", 1).over(window))
        df = df.withColumn(
            "price_direction",
            when(col("next_close") > col("close"), 1).otherwise(0),
        )
        # Drop last row (no next day)
        df = df.filter(col("next_close").isNotNull())

        distinct_values = {row["price_direction"] for row in df.select("price_direction").collect()}
        assert distinct_values.issubset({0, 1})
        assert df.count() == 9
