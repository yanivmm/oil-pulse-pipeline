"""
Spark feature engineering: window functions, joins, and label creation.

Window functions used:
    1. 7-day rolling average of oil close price
    2. 3-day rolling average of Reddit sentiment
    3. Price delta from previous day (lag)
    4. 3-day rolling average of Trump statement sentiment
    5. 7-day rolling average of base (conservative/politics) war sentiment

Joins price data with Reddit, news, Trump statements, and war news on date.
Adds binary label: price_direction (1 if next day price > today, 0 otherwise).

Reads from:
    data/processed/clean/prices.parquet
    data/processed/clean/reddit.parquet
    data/processed/clean/news.parquet
    data/processed/clean/trump.parquet
    data/processed/clean/war_news.parquet

Writes to:
    data/processed/features/features.parquet

Can be run standalone:
    python scripts/transform/spark_features.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("HADOOP_HOME", "C:\\hadoop")
    os.environ.setdefault("SPARK_LOCAL_DIRS", str(Path.home() / "spark-temp"))

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    avg,
    col,
    count as spark_count,
    lag,
    lead,
    lit,
    round as spark_round,
    sum as spark_sum,
    when,
)

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN = PROJECT_ROOT / "data" / "processed" / "clean"
FEATURES = PROJECT_ROOT / "data" / "processed" / "features"


def get_spark() -> SparkSession:
    """Create a local SparkSession."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("oil-pulse-features")
        .config("spark.driver.host", "localhost")
        .config("spark.sql.warehouse.dir", str(Path.home() / "spark-warehouse"))
        .getOrCreate()
    )


def _parquet_exists(path: Path) -> bool:
    """Check if a parquet directory exists and contains data files."""
    return path.exists() and any(path.glob("*.parquet"))


def build_features(spark: SparkSession) -> int:
    """Engineer features and return row count."""
    # --- Load cleaned parquet -----------------------------------------------
    prices = spark.read.parquet(str(CLEAN / "prices.parquet"))

    # Optional sources — may not exist if no API keys / feeds down
    has_reddit = _parquet_exists(CLEAN / "reddit.parquet")
    has_news = _parquet_exists(CLEAN / "news.parquet")
    has_trump = _parquet_exists(CLEAN / "trump.parquet")
    has_war = _parquet_exists(CLEAN / "war_news.parquet")

    if has_reddit:
        reddit = spark.read.parquet(str(CLEAN / "reddit.parquet"))
    if has_news:
        news = spark.read.parquet(str(CLEAN / "news.parquet"))
    if has_trump:
        trump = spark.read.parquet(str(CLEAN / "trump.parquet"))
    if has_war:
        war_news = spark.read.parquet(str(CLEAN / "war_news.parquet"))

    # --- Window definitions -------------------------------------------------
    price_window_7d = Window.orderBy("date").rowsBetween(-6, 0)
    price_window_3d = Window.orderBy("date").rowsBetween(-2, 0)
    price_window_prev = Window.orderBy("date")

    # --- Price features -----------------------------------------------------
    prices = (
        prices.withColumn(
            "rolling_avg_7d",
            spark_round(avg("close").over(price_window_7d), 4),
        )
        .withColumn(
            "price_delta",
            spark_round(col("close") - lag("close", 1).over(price_window_prev), 4),
        )
        .withColumn(
            "next_day_close",
            lead("close", 1).over(price_window_prev),
        )
        .withColumn(
            "price_direction",
            when(col("next_day_close") > col("close"), 1).otherwise(0),
        )
    )

    # --- Reddit sentiment aggregation per day --------------------------------
    if has_reddit:
        reddit_daily = reddit.groupBy("date").agg(
            avg("sentiment_compound").alias("reddit_sentiment_avg")
        )
        reddit_daily = reddit_daily.withColumn(
            "sentiment_rolling_3d",
            spark_round(avg("reddit_sentiment_avg").over(price_window_3d), 4),
        )

        # Base subreddit sentiment (r/conservative + r/politics) for War Exit
        reddit_base_war = reddit.filter(
            (col("subreddit_group") == "base")
            & (col("topic").contains("war"))
        )
        if reddit_base_war.count() > 0:
            base_daily = reddit_base_war.groupBy("date").agg(
                avg("sentiment_compound").alias("base_war_sentiment")
            )
            base_window_7d = Window.orderBy("date").rowsBetween(-6, 0)
            base_daily = base_daily.withColumn(
                "base_sentiment_rolling_7d",
                spark_round(avg("base_war_sentiment").over(base_window_7d), 4),
            )
        else:
            base_daily = None
    else:
        base_daily = None

    # --- News sentiment aggregation per day ----------------------------------
    if has_news:
        news_daily = news.groupBy("date").agg(
            avg("sentiment_compound").alias("news_sentiment")
        )

    # --- Trump sentiment aggregation per day ---------------------------------
    if has_trump:
        trump_daily = trump.groupBy("date").agg(
            avg("sentiment_compound").alias("trump_sentiment_avg")
        )
        trump_daily = trump_daily.withColumn(
            "trump_sentiment_rolling_3d",
            spark_round(avg("trump_sentiment_avg").over(price_window_3d), 4),
        )

    # --- War news: sentiment + favor/against counts per day ------------------
    if has_war:
        war_daily = war_news.groupBy("date").agg(
            avg("sentiment_compound").alias("war_sentiment_avg"),
            spark_count("*").alias("war_article_count"),
            spark_sum(when(col("stance") == "favor", 1).otherwise(0)).alias("favor_count"),
            spark_sum(when(col("stance") == "against", 1).otherwise(0)).alias("against_count"),
        )
        # Favor ratio: favor / (favor + against), avoid division by zero
        war_daily = war_daily.withColumn(
            "favor_ratio",
            spark_round(
                when(
                    (col("favor_count") + col("against_count")) > 0,
                    col("favor_count") / (col("favor_count") + col("against_count")),
                ).otherwise(0.5),
                4,
            ),
        )

    # --- Join everything on date ---------------------------------------------
    features = prices
    if has_reddit:
        features = features.join(reddit_daily, on="date", how="left")
    if has_reddit and base_daily is not None:
        features = features.join(base_daily, on="date", how="left")
    if has_news:
        features = features.join(news_daily, on="date", how="left")
    if has_trump:
        features = features.join(trump_daily, on="date", how="left")
    if has_war:
        features = features.join(war_daily, on="date", how="left")

    # Ensure all expected columns exist (even if sources are missing)
    default_cols = {
        "reddit_sentiment_avg": 0.0,
        "sentiment_rolling_3d": 0.0,
        "news_sentiment": 0.0,
        "trump_sentiment_avg": 0.0,
        "trump_sentiment_rolling_3d": 0.0,
        "war_sentiment_avg": 0.0,
        "war_article_count": 0,
        "favor_count": 0,
        "against_count": 0,
        "favor_ratio": 0.5,
        "base_war_sentiment": 0.0,
        "base_sentiment_rolling_7d": 0.0,
    }
    for col_name, default_val in default_cols.items():
        if col_name not in features.columns:
            features = features.withColumn(col_name, lit(default_val))

    features = (
        features.select(
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rolling_avg_7d",
            "price_delta",
            "reddit_sentiment_avg",
            "sentiment_rolling_3d",
            "news_sentiment",
            "trump_sentiment_avg",
            "trump_sentiment_rolling_3d",
            "war_sentiment_avg",
            "war_article_count",
            "favor_count",
            "against_count",
            "favor_ratio",
            "base_war_sentiment",
            "base_sentiment_rolling_7d",
            "price_direction",
        )
        .orderBy("date")
    )

    # Fill nulls
    features = features.fillna(
        {
            "reddit_sentiment_avg": 0.0,
            "sentiment_rolling_3d": 0.0,
            "news_sentiment": 0.0,
            "price_delta": 0.0,
            "trump_sentiment_avg": 0.0,
            "trump_sentiment_rolling_3d": 0.0,
            "war_sentiment_avg": 0.0,
            "war_article_count": 0,
            "favor_count": 0,
            "against_count": 0,
            "favor_ratio": 0.5,
            "base_war_sentiment": 0.0,
            "base_sentiment_rolling_7d": 0.0,
        }
    )

    FEATURES.mkdir(parents=True, exist_ok=True)
    features.write.mode("overwrite").parquet(str(FEATURES / "features.parquet"))
    count = features.count()
    print(f"Features saved: {count} rows, {len(features.columns)} columns")
    return count


def main():
    """Run feature engineering."""
    spark = get_spark()
    try:
        build_features(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
