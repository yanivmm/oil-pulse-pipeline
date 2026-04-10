"""
Spark cleaning step: load raw CSVs, enforce schemas, handle nulls, deduplicate.

Reads from:
    data/raw/prices/*.csv
    data/raw/reddit/*.csv
    data/raw/news/*.csv
    data/raw/trump/*.csv
    data/raw/war_news/*.csv
    data/raw/taiwan/*.csv

Writes to:
    data/processed/clean/prices.parquet
    data/processed/clean/reddit.parquet
    data/processed/clean/news.parquet
    data/processed/clean/trump.parquet
    data/processed/clean/war_news.parquet
    data/processed/clean/taiwan.parquet

Prints the total record count to stdout (captured by Airflow XCom).

Can be run standalone:
    python scripts/transform/spark_clean.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows: set HADOOP_HOME if not set, suppress native IO warnings
if sys.platform == "win32":
    os.environ.setdefault("HADOOP_HOME", "C:\\hadoop")
    os.environ.setdefault("SPARK_LOCAL_DIRS", str(Path.home() / "spark-temp"))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, trim
from pyspark.sql.types import (
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
CLEAN = PROJECT_ROOT / "data" / "processed" / "clean"

# ---------------------------------------------------------------------------
# Explicit schemas — demonstrates StructType/StructField usage
# ---------------------------------------------------------------------------
PRICE_SCHEMA = StructType(
    [
        StructField("date", StringType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", LongType(), True),
    ]
)

REDDIT_SCHEMA = StructType(
    [
        StructField("date", StringType(), True),
        StructField("subreddit", StringType(), True),
        StructField("subreddit_group", StringType(), True),
        StructField("title", StringType(), True),
        StructField("score", IntegerType(), True),
        StructField("sentiment_compound", FloatType(), True),
        StructField("topic", StringType(), True),
    ]
)

NEWS_SCHEMA = StructType(
    [
        StructField("date", StringType(), True),
        StructField("source", StringType(), True),
        StructField("title", StringType(), True),
        StructField("published_date", StringType(), True),
        StructField("summary", StringType(), True),
        StructField("sentiment_compound", FloatType(), True),
    ]
)

TRUMP_SCHEMA = StructType(
    [
        StructField("date", StringType(), True),
        StructField("source", StringType(), True),
        StructField("query", StringType(), True),
        StructField("title", StringType(), True),
        StructField("published_date", StringType(), True),
        StructField("summary", StringType(), True),
        StructField("sentiment_compound", FloatType(), True),
        StructField("topic_tags", StringType(), True),
    ]
)

WAR_NEWS_SCHEMA = StructType(
    [
        StructField("date", StringType(), True),
        StructField("source", StringType(), True),
        StructField("title", StringType(), True),
        StructField("published_date", StringType(), True),
        StructField("summary", StringType(), True),
        StructField("sentiment_compound", FloatType(), True),
        StructField("stance", StringType(), True),
    ]
)


TAIWAN_SCHEMA = StructType(
    [
        StructField("date", StringType(), True),
        StructField("source", StringType(), True),
        StructField("title", StringType(), True),
        StructField("published_date", StringType(), True),
        StructField("summary", StringType(), True),
        StructField("sentiment_compound", FloatType(), True),
        StructField("escalation", StringType(), True),
        StructField("iran_spillover", StringType(), True),
    ]
)


def get_spark() -> SparkSession:
    """Create a local SparkSession."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("oil-pulse-clean")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.driver.host", "localhost")
        .config("spark.sql.warehouse.dir", str(Path.home() / "spark-warehouse"))
        .getOrCreate()
    )


def clean_prices(spark: SparkSession) -> int:
    """Clean oil price data. Returns row count."""
    path = str(RAW / "prices")
    if not Path(path).exists() or not list(Path(path).glob("*.csv")):
        return 0

    df = spark.read.csv(path, header=True, schema=PRICE_SCHEMA)
    df = (
        df.withColumn("date", to_date(col("date")))
        .dropna(subset=["date", "close"])
        .dropDuplicates(["date"])
        .orderBy("date")
    )
    df.write.mode("overwrite").parquet(str(CLEAN / "prices.parquet"))
    return df.count()


def clean_reddit(spark: SparkSession) -> int:
    """Clean Reddit sentiment data. Returns row count."""
    path = str(RAW / "reddit")
    if not Path(path).exists() or not list(Path(path).glob("*.csv")):
        return 0

    df = spark.read.csv(path, header=True, schema=REDDIT_SCHEMA)
    df = (
        df.withColumn("date", to_date(col("date")))
        .withColumn("title", trim(col("title")))
        .dropna(subset=["date", "sentiment_compound"])
        .dropDuplicates(["date", "subreddit", "title"])
    )
    df.write.mode("overwrite").parquet(str(CLEAN / "reddit.parquet"))
    return df.count()


def clean_news(spark: SparkSession) -> int:
    """Clean RSS news data. Returns row count."""
    path = str(RAW / "news")
    if not Path(path).exists() or not list(Path(path).glob("*.csv")):
        return 0

    df = spark.read.csv(path, header=True, schema=NEWS_SCHEMA)
    df = (
        df.withColumn("date", to_date(col("date")))
        .withColumn("title", trim(col("title")))
        .dropna(subset=["date", "sentiment_compound"])
        .dropDuplicates(["date", "source", "title"])
    )
    df.write.mode("overwrite").parquet(str(CLEAN / "news.parquet"))
    return df.count()


def clean_trump(spark: SparkSession) -> int:
    """Clean Trump statement news data. Returns row count."""
    path = str(RAW / "trump")
    if not Path(path).exists() or not list(Path(path).glob("*.csv")):
        return 0

    df = spark.read.csv(path, header=True, schema=TRUMP_SCHEMA)
    df = (
        df.withColumn("date", to_date(col("date")))
        .withColumn("title", trim(col("title")))
        .dropna(subset=["date", "sentiment_compound"])
        .dropDuplicates(["date", "source", "title"])
    )
    df.write.mode("overwrite").parquet(str(CLEAN / "trump.parquet"))
    return df.count()


def clean_war_news(spark: SparkSession) -> int:
    """Clean war/conflict news data. Returns row count."""
    path = str(RAW / "war_news")
    if not Path(path).exists() or not list(Path(path).glob("*.csv")):
        return 0

    df = spark.read.csv(path, header=True, schema=WAR_NEWS_SCHEMA)
    df = (
        df.withColumn("date", to_date(col("date")))
        .withColumn("title", trim(col("title")))
        .dropna(subset=["date", "sentiment_compound"])
        .dropDuplicates(["date", "source", "title"])
    )
    df.write.mode("overwrite").parquet(str(CLEAN / "war_news.parquet"))
    return df.count()


def clean_taiwan(spark: SparkSession) -> int:
    """Clean Taiwan/China tension data. Returns row count."""
    path = str(RAW / "taiwan")
    if not Path(path).exists() or not list(Path(path).glob("*.csv")):
        return 0

    df = spark.read.csv(path, header=True, schema=TAIWAN_SCHEMA)
    df = (
        df.withColumn("date", to_date(col("date")))
        .withColumn("title", trim(col("title")))
        .dropna(subset=["date", "sentiment_compound"])
        .dropDuplicates(["date", "source", "title"])
    )
    df.write.mode("overwrite").parquet(str(CLEAN / "taiwan.parquet"))
    return df.count()


def main():
    """Run all cleaning steps and print total record count."""
    CLEAN.mkdir(parents=True, exist_ok=True)
    spark = get_spark()

    try:
        price_count = clean_prices(spark)
        reddit_count = clean_reddit(spark)
        news_count = clean_news(spark)
        trump_count = clean_trump(spark)
        war_count = clean_war_news(spark)
        taiwan_count = clean_taiwan(spark)

        total = price_count + reddit_count + news_count + trump_count + war_count + taiwan_count
        print(
            f"Cleaned: prices={price_count}, reddit={reddit_count}, "
            f"news={news_count}, trump={trump_count}, war_news={war_count}, "
            f"taiwan={taiwan_count}"
        )
        # Last line is an integer — Airflow BashOperator captures it via XCom
        print(total)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
