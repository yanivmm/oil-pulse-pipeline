"""
Load aggregated data, predictions, and raw articles into DuckDB.

Tables created/updated:
    daily_summary      — from aggregated CSV
    predictions        — from prediction JSON files
    trump_statements   — from cleaned trump parquet (for dashboard quotes)
    war_news           — from cleaned war_news parquet (for dashboard quotes)
    war_exit_index     — from war exit prediction JSON files

Reads from:
    data/processed/aggregated/daily_summary_30d.csv
    data/processed/predictions/*.json
    data/processed/clean/trump.parquet
    data/processed/clean/war_news.parquet
    data/processed/predictions/war_exit_*.json

Writes to:
    data/oil_pulse.duckdb

Can be run standalone:
    python scripts/ml/load_duckdb.py
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AGGREGATED_CSV = PROJECT_ROOT / "data" / "processed" / "aggregated" / "daily_summary_30d.csv"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"
CLEAN = PROJECT_ROOT / "data" / "processed" / "clean"
DB_PATH = PROJECT_ROOT / "data" / "oil_pulse.duckdb"


def _load_predictions() -> pd.DataFrame:
    """Load all prediction JSON files into a DataFrame."""
    pred_files = sorted(
        f for f in PREDICTIONS_DIR.glob("*.json") if not f.name.startswith("war_exit_")
    )
    rows = []
    for f in pred_files:
        with open(f) as fh:
            data = json.load(fh)
            rows.append(
                {
                    "date": data["date"],
                    "prediction": data["prediction"],
                    "prediction_numeric": data["prediction_numeric"],
                    "confidence": data["confidence"],
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "prediction", "prediction_numeric", "confidence"]
    )


def _load_war_exit_predictions() -> pd.DataFrame:
    """Load war exit prediction JSON files into a DataFrame."""
    war_exit_files = sorted(PREDICTIONS_DIR.glob("war_exit_*.json"))
    rows = []
    for f in war_exit_files:
        with open(f) as fh:
            data = json.load(fh)
            rows.append(data)
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "exit_probability", "pressure_index", "base_sentiment_trend"]
    )


def main():
    """Load data into DuckDB with upsert logic."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))

    try:
        # --- daily_summary table ------------------------------------------------
        if AGGREGATED_CSV.exists():
            df_summary = pd.read_csv(AGGREGATED_CSV)
            if not df_summary.empty:
                df_summary["date"] = pd.to_datetime(df_summary["date"]).dt.date
                con.execute("DROP TABLE IF EXISTS daily_summary")
                con.execute("CREATE TABLE daily_summary AS SELECT * FROM df_summary")
                print(f"Loaded {len(df_summary)} rows into daily_summary")
        else:
            print("Warning: aggregated CSV not found, skipping daily_summary load")

        # --- predictions table --------------------------------------------------
        df_preds = _load_predictions()
        if not df_preds.empty:
            df_preds["date"] = pd.to_datetime(df_preds["date"]).dt.date
            con.execute("DROP TABLE IF EXISTS predictions")
            con.execute("CREATE TABLE predictions AS SELECT * FROM df_preds")
            print(f"Loaded {len(df_preds)} rows into predictions")
        else:
            print("No prediction files found")

        # --- trump_statements table (for dashboard quotes) ----------------------
        trump_parquet = CLEAN / "trump.parquet"
        if trump_parquet.exists() and any(trump_parquet.glob("*.parquet")):
            df_trump = pd.read_parquet(str(trump_parquet))
            if not df_trump.empty:
                con.execute("DROP TABLE IF EXISTS trump_statements")
                con.execute("CREATE TABLE trump_statements AS SELECT * FROM df_trump")
                print(f"Loaded {len(df_trump)} rows into trump_statements")
        else:
            print("No trump parquet found, skipping")

        # --- war_news table (for dashboard quotes) ------------------------------
        war_parquet = CLEAN / "war_news.parquet"
        if war_parquet.exists() and any(war_parquet.glob("*.parquet")):
            df_war = pd.read_parquet(str(war_parquet))
            if not df_war.empty:
                con.execute("DROP TABLE IF EXISTS war_news")
                con.execute("CREATE TABLE war_news AS SELECT * FROM df_war")
                print(f"Loaded {len(df_war)} rows into war_news")
        else:
            print("No war_news parquet found, skipping")

        # --- war_exit_index table -----------------------------------------------
        df_exit = _load_war_exit_predictions()
        if not df_exit.empty:
            df_exit["date"] = pd.to_datetime(df_exit["date"]).dt.date
            con.execute("DROP TABLE IF EXISTS war_exit_index")
            con.execute("CREATE TABLE war_exit_index AS SELECT * FROM df_exit")
            print(f"Loaded {len(df_exit)} rows into war_exit_index")
        else:
            print("No war exit predictions found")

        # --- taiwan_tensions table ----------------------------------------------
        taiwan_parquet = CLEAN / "taiwan.parquet"
        if taiwan_parquet.exists() and any(taiwan_parquet.glob("*.parquet")):
            df_taiwan = pd.read_parquet(str(taiwan_parquet))
            if not df_taiwan.empty:
                con.execute("DROP TABLE IF EXISTS taiwan_tensions")
                con.execute("CREATE TABLE taiwan_tensions AS SELECT * FROM df_taiwan")
                print(f"Loaded {len(df_taiwan)} rows into taiwan_tensions")
        else:
            print("No taiwan parquet found, skipping")

    finally:
        con.close()

    print(f"DuckDB updated at {DB_PATH}")


if __name__ == "__main__":
    main()
