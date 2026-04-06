"""
Run daily prediction using the trained RandomForest model.

Loads today's features row, applies the model, and saves prediction as JSON.

Reads from:
    data/processed/features/features.parquet
    models/rf_model.pkl

Saves to:
    data/processed/predictions/YYYY-MM-DD.json

Can be run standalone:
    python scripts/ml/predict.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"

FEATURE_COLS = [
    "price_delta",
    "rolling_avg_7d",
    "sentiment_rolling_3d",
    "news_sentiment",
    "trump_sentiment_avg",
    "war_sentiment_avg",
    "favor_ratio",
]


def main():
    """Load model and predict today's direction."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load features — use the most recent row
    df = pd.read_parquet(FEATURES_PATH)
    df = df.dropna(subset=FEATURE_COLS)

    if df.empty:
        print("No feature rows available for prediction.")
        return

    latest = df.iloc[-1]
    X = latest[FEATURE_COLS].values.reshape(1, -1)

    # Load model
    model = joblib.load(MODEL_PATH)
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    confidence = float(np.max(probabilities))

    direction = "Up" if prediction == 1 else "Down"
    pred_date = str(latest.get("date", datetime.utcnow().strftime("%Y-%m-%d")))

    result = {
        "date": pred_date,
        "prediction": direction,
        "prediction_numeric": prediction,
        "confidence": round(confidence, 4),
        "features": {col: round(float(latest[col]), 4) for col in FEATURE_COLS},
    }

    output_path = PREDICTIONS_DIR / f"{pred_date}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Prediction for {pred_date}: {direction} (confidence: {confidence:.2%})")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
