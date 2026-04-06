"""
War Exit Pressure Model — forecasts likelihood of Trump leaving the war
based on his base's opinion sentiment.

Uses logistic regression on base_war_sentiment (rolling 7-day avg from
r/conservative + r/politics war-tagged posts) and war news stance ratios.

When base sentiment on war drops below threshold for sustained periods,
the "exit probability" rises — indicating political pressure to disengage.

NOTE: This is a sentiment-based analytical tool, NOT a political forecast.
It measures public opinion pressure, not actual policy decisions.

Reads from:
    data/processed/features/features.parquet

Saves to:
    data/processed/predictions/war_exit_YYYY-MM-DD.json

Can be run standalone:
    python scripts/ml/war_exit_model.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "features.parquet"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"

# Features used for the pressure index
PRESSURE_FEATURES = [
    "base_war_sentiment",
    "base_sentiment_rolling_7d",
    "favor_ratio",
    "war_sentiment_avg",
]

# Thresholds for the pressure scoring
# When base sentiment is negative (below 0), there's pressure to exit
SENTIMENT_NEUTRAL = 0.0
# How much weight each component gets in the pressure index
WEIGHTS = {
    "base_sentiment": 0.35,   # base subreddit war sentiment
    "favor_ratio": 0.30,      # proportion of war news that's favorable
    "war_sentiment": 0.20,    # overall war news sentiment
    "trend": 0.15,            # direction of sentiment change
}


def _compute_pressure_index(row: pd.Series, prev_base: float) -> float:
    """Compute a 0-1 pressure index from sentiment features.

    Higher = more pressure to exit the war.
    """
    # Base sentiment: negative = more exit pressure
    base_sent = row.get("base_sentiment_rolling_7d", 0.0)
    # Transform to 0-1: sentiment of -1 → pressure 1.0, sentiment of +1 → pressure 0.0
    base_pressure = max(0.0, min(1.0, (SENTIMENT_NEUTRAL - base_sent + 0.5) / 1.0))

    # Favor ratio: low favor ratio = more bad news = more pressure
    favor = row.get("favor_ratio", 0.5)
    favor_pressure = max(0.0, min(1.0, 1.0 - favor))

    # War sentiment: negative war news = more pressure
    war_sent = row.get("war_sentiment_avg", 0.0)
    war_pressure = max(0.0, min(1.0, (SENTIMENT_NEUTRAL - war_sent + 0.5) / 1.0))

    # Trend: if base sentiment is declining, pressure is rising
    trend = base_sent - prev_base if prev_base != 0.0 else 0.0
    trend_pressure = max(0.0, min(1.0, (-trend + 0.1) / 0.2))

    # Weighted combination
    pressure = (
        WEIGHTS["base_sentiment"] * base_pressure
        + WEIGHTS["favor_ratio"] * favor_pressure
        + WEIGHTS["war_sentiment"] * war_pressure
        + WEIGHTS["trend"] * trend_pressure
    )
    return round(max(0.0, min(1.0, pressure)), 4)


def _sigmoid(x: float) -> float:
    """Sigmoid function for converting pressure index to probability."""
    return 1.0 / (1.0 + np.exp(-x))


def main():
    """Compute war exit pressure index and save prediction."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        print("Not enough data for war exit model.")
        return

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Compute pressure index
    pressure_index = _compute_pressure_index(
        latest, prev_base=previous.get("base_sentiment_rolling_7d", 0.0)
    )

    # Convert pressure index to probability using adjusted sigmoid
    # Center at 0.5 pressure = 50% probability, steeper curve above 0.6
    logit_input = (pressure_index - 0.5) * 6  # scale for sigmoid sensitivity
    exit_probability = round(float(_sigmoid(logit_input)), 4)

    # Determine trend direction
    base_current = latest.get("base_sentiment_rolling_7d", 0.0)
    base_prev = previous.get("base_sentiment_rolling_7d", 0.0)
    if base_current < base_prev - 0.01:
        trend = "declining"
    elif base_current > base_prev + 0.01:
        trend = "improving"
    else:
        trend = "stable"

    pred_date = str(latest.get("date", datetime.utcnow().strftime("%Y-%m-%d")))

    result = {
        "date": pred_date,
        "exit_probability": exit_probability,
        "pressure_index": pressure_index,
        "base_sentiment_trend": trend,
        "base_sentiment_current": round(float(base_current), 4),
        "favor_ratio": round(float(latest.get("favor_ratio", 0.5)), 4),
        "war_sentiment_avg": round(float(latest.get("war_sentiment_avg", 0.0)), 4),
    }

    output_path = PREDICTIONS_DIR / f"war_exit_{pred_date}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"War Exit Pressure for {pred_date}:")
    print(f"  Pressure Index: {pressure_index:.1%}")
    print(f"  Exit Probability: {exit_probability:.1%}")
    print(f"  Base Sentiment Trend: {trend}")
    print(f"  Favor Ratio: {latest.get('favor_ratio', 0.5):.1%}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
