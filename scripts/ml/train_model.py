"""
Train a RandomForestClassifier to predict next-day oil price direction.

Features: price_delta, rolling_avg_7d, sentiment_rolling_3d, news_sentiment
Target:   price_direction (1 = up, 0 = down)

Reads from:
    data/processed/features/features.parquet

Saves to:
    models/rf_model.pkl

Can be run standalone:
    python scripts/ml/train_model.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLS = [
    "price_delta",
    "rolling_avg_7d",
    "sentiment_rolling_3d",
    "news_sentiment",
    "trump_sentiment_avg",
    "war_sentiment_avg",
    "favor_ratio",
]
TARGET_COL = "price_direction"


def main():
    """Train model and print evaluation metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    df = pd.read_parquet(FEATURES_PATH)

    # Drop rows where target or features are null (warmup period)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    if len(df) < 20:
        print(f"Only {len(df)} rows available — not enough to train. Skipping.")
        return

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # time-series: no shuffle
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Save model
    model_path = MODELS_DIR / "rf_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save accuracy for dashboard
    metrics = {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4)}
    metrics_path = MODELS_DIR / "metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
