"""
Spark aggregation: create daily summary for dashboard consumption.

Reads from:
    data/processed/features/features.parquet

Writes to:
    data/processed/aggregated/daily_summary.parquet  (full history)
    data/processed/aggregated/daily_summary_30d.csv   (last 30 days, for dashboard)

Can be run standalone:
    python scripts/transform/spark_aggregate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("HADOOP_HOME", "C:\\hadoop")
    os.environ.setdefault("SPARK_LOCAL_DIRS", str(Path.home() / "spark-temp"))

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, max as spark_max, min as spark_min, round as spark_round, sum as spark_sum

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES = PROJECT_ROOT / "data" / "processed" / "features"
AGGREGATED = PROJECT_ROOT / "data" / "processed" / "aggregated"


def get_spark() -> SparkSession:
    """Create a local SparkSession."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("oil-pulse-aggregate")
        .config("spark.driver.host", "localhost")
        .config("spark.sql.warehouse.dir", str(Path.home() / "spark-warehouse"))
        .getOrCreate()
    )


def aggregate(spark: SparkSession) -> int:
    """Build daily summary and return row count."""
    features = spark.read.parquet(str(FEATURES / "features.parquet"))

    daily = (
        features.groupBy("date")
        .agg(
            spark_round(avg("close"), 2).alias("avg_price"),
            spark_round(spark_min("low"), 2).alias("min_price"),
            spark_round(spark_max("high"), 2).alias("max_price"),
            spark_round(avg("reddit_sentiment_avg"), 4).alias("avg_reddit_sentiment"),
            spark_round(avg("news_sentiment"), 4).alias("avg_news_sentiment"),
            spark_round(avg("rolling_avg_7d"), 2).alias("rolling_avg_7d"),
            spark_round(avg("price_delta"), 4).alias("avg_price_delta"),
            # Trump & war features
            spark_round(avg("trump_sentiment_avg"), 4).alias("avg_trump_sentiment"),
            spark_round(avg("trump_sentiment_rolling_3d"), 4).alias("trump_sentiment_rolling_3d"),
            spark_round(avg("war_sentiment_avg"), 4).alias("avg_war_sentiment"),
            spark_sum("favor_count").alias("favor_count"),
            spark_sum("against_count").alias("against_count"),
            spark_round(avg("favor_ratio"), 4).alias("favor_ratio"),
            # Base sentiment for War Exit model
            spark_round(avg("base_war_sentiment"), 4).alias("base_war_sentiment"),
            spark_round(avg("base_sentiment_rolling_7d"), 4).alias("base_sentiment_rolling_7d"),
            # Label
            spark_max("price_direction").alias("price_direction"),
        )
        .orderBy("date")
    )

    AGGREGATED.mkdir(parents=True, exist_ok=True)

    # Full history as Parquet
    daily.write.mode("overwrite").parquet(str(AGGREGATED / "daily_summary.parquet"))

    # Last 30 days as CSV for the Streamlit dashboard
    row_count = daily.count()
    daily_pd = daily.toPandas()
    last_30 = daily_pd.tail(30)
    last_30.to_csv(str(AGGREGATED / "daily_summary_30d.csv"), index=False)

    print(f"Aggregated: {row_count} total days, {len(last_30)} in CSV")
    return row_count


def main():
    """Run aggregation."""
    spark = get_spark()
    try:
        aggregate(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
