"""
Fetch crude oil futures (CL=F) from Yahoo Finance.

- First run (no existing data):  fetches the last 90 days.
- Subsequent runs:               fetches the last 2 days (overlap for safety).
- Saves a CSV per day to ``data/raw/prices/YYYY-MM-DD.csv``.

Can be run standalone:
    python scripts/ingest/fetch_oil_prices.py
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch(data_dir: str | None = None) -> str:
    """Download crude oil prices and save to CSV.

    Args:
        data_dir: Directory for output CSVs. Defaults to ``data/raw/prices``.

    Returns:
        Path to the CSV file written.
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "data", "raw", "prices"
        )
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide lookback: 90 days if no prior data, else 2 days
    existing_files = sorted(out_dir.glob("*.csv"))
    lookback_days = 2 if existing_files else 90

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)

    ticker = yf.Ticker("CL=F")
    df = ticker.history(start=str(start_date), end=str(end_date))

    if df.empty:
        # Return empty file path so downstream knows nothing was fetched
        empty_path = str(out_dir / f"{end_date}.csv")
        pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        ).to_csv(empty_path, index=False)
        return empty_path

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Ensure a 'date' column exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        df["date"] = end_date

    keep_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    output_path = out_dir / f"{end_date}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return str(output_path)


def main():
    """Entry point for standalone execution."""
    path = fetch()
    print(f"Output: {path}")


if __name__ == "__main__":
    main()
