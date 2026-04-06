"""
Evaluate past predictions against actual oil price movements.

Compares predictions stored in data/processed/predictions/ against actual
price direction from features.parquet. Writes results to
data/processed/evaluation/evaluation_log.csv.

Tracks:
    - Daily accuracy (1-day lookback)
    - Weekly accuracy (5-day lookback)
    - Running cumulative accuracy

This file is NOT displayed on the dashboard — it's an offline monitoring tool.

Run:
    python scripts/ml/evaluate_predictions.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "features.parquet"
EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "evaluation"
EVAL_LOG = EVAL_DIR / "evaluation_log.csv"


def _load_predictions() -> pd.DataFrame:
    """Load all prediction JSON files into a DataFrame."""
    rows = []
    for f in sorted(PREDICTIONS_DIR.glob("*.json")):
        if f.name.startswith("war_exit_"):
            continue
        with open(f) as fh:
            data = json.load(fh)
            rows.append({
                "pred_date": data["date"],
                "prediction": data["prediction"],
                "prediction_numeric": data["prediction_numeric"],
                "confidence": data["confidence"],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_actuals() -> pd.DataFrame:
    """Load actual price direction from features parquet."""
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(FEATURES_PATH)
    if "price_direction" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = df["date"].astype(str)
    return df[["date", "price_direction", "close"]].copy()


def evaluate() -> str:
    """Compare predictions vs actuals and write evaluation log.

    Returns:
        Path to the evaluation log CSV.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions()
    actuals = _load_actuals()

    if preds.empty:
        print("No predictions found.")
        return str(EVAL_LOG)

    if actuals.empty:
        print("No actual data found.")
        return str(EVAL_LOG)

    # Merge predictions with actuals
    merged = preds.merge(
        actuals, left_on="pred_date", right_on="date", how="left",
    )

    rows = []
    for _, row in merged.iterrows():
        pred_date = row["pred_date"]
        predicted = row["prediction_numeric"]
        actual = row.get("price_direction")
        confidence = row["confidence"]

        if pd.isna(actual):
            correct = None
            status = "pending"
        else:
            actual = int(actual)
            correct = int(predicted == actual)
            status = "correct" if correct else "incorrect"

        rows.append({
            "pred_date": pred_date,
            "predicted": "Up" if predicted == 1 else "Down",
            "actual": "Up" if actual == 1 else ("Down" if actual == 0 else "pending"),
            "correct": correct,
            "confidence": round(confidence, 4),
            "status": status,
            "evaluated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })

    eval_df = pd.DataFrame(rows)

    # Calculate running stats
    evaluated = eval_df[eval_df["correct"].notna()]
    if not evaluated.empty:
        total = len(evaluated)
        correct_count = int(evaluated["correct"].sum())
        accuracy = correct_count / total

        # Weekly accuracy (last 5 evaluated predictions)
        recent = evaluated.tail(5)
        weekly_accuracy = recent["correct"].mean() if len(recent) > 0 else 0

        print(f"Evaluation Summary:")
        print(f"  Total evaluated:    {total}")
        print(f"  Correct:            {correct_count}")
        print(f"  Cumulative accuracy: {accuracy:.1%}")
        print(f"  Last-5 accuracy:    {weekly_accuracy:.1%}")
        print(f"  Pending:            {len(eval_df) - total}")
    else:
        print("No predictions could be evaluated yet (all pending).")

    # Write log
    eval_df.to_csv(EVAL_LOG, index=False)
    print(f"Evaluation log saved to {EVAL_LOG}")
    return str(EVAL_LOG)


def main():
    evaluate()


if __name__ == "__main__":
    main()
