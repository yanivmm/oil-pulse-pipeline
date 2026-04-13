"""
Oil Buy/Sell Signal Model — generates trading signals and backtests accuracy.

Combines sentiment features, price momentum, and geopolitical indicators
to produce BUY / SELL / HOLD signals for crude oil.

Strategy:
    1. Composite score = weighted sum of normalised signals
    2. BUY when composite > +threshold, SELL when < -threshold, else HOLD
    3. Backtest: compare next-day price_direction to signal issued today

Reads from:
    data/oil_pulse.duckdb  (daily_summary table)

Saves to:
    data/processed/predictions/buy_signals.parquet

Can be run standalone:
    python scripts/ml/buy_signal_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "oil_pulse.duckdb"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "predictions" / "buy_signals.parquet"

# Signal weights (sum to 1.0)
WEIGHTS = {
    "price_momentum": 0.25,    # rolling price delta direction
    "news_sentiment": 0.20,    # avg_news_sentiment
    "reddit_sentiment": 0.15,  # avg_reddit_sentiment
    "trump_sentiment": 0.15,   # avg_trump_sentiment
    "war_sentiment": 0.15,     # avg_war_sentiment (inverted: negative war = oil up)
    "favor_ratio": 0.10,       # favor_ratio (high = war likely to end = oil down)
}

# Composite score thresholds
BUY_THRESHOLD = 0.12
SELL_THRESHOLD = -0.12


def _normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise to [-1, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return 2 * (series - mn) / (mx - mn) - 1


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Build buy/sell/hold signals from daily_summary data."""
    df = df.sort_values("date").reset_index(drop=True)

    # --- Normalise each signal ---
    # Price momentum: positive delta = bullish
    df["sig_momentum"] = _normalise(df["avg_price_delta"].fillna(0))

    # News sentiment: positive = bullish (market optimism)
    df["sig_news"] = _normalise(df["avg_news_sentiment"].fillna(0))

    # Reddit sentiment: positive = bullish
    df["sig_reddit"] = _normalise(df["avg_reddit_sentiment"].fillna(0))

    # Trump sentiment: positive trump = geopolitical stability = slightly bearish for oil
    # But we keep it simple: positive trump → positive signal
    df["sig_trump"] = _normalise(df["avg_trump_sentiment"].fillna(0))

    # War sentiment: INVERTED — negative war sentiment = escalation = oil UP (buy)
    df["sig_war"] = -_normalise(df["avg_war_sentiment"].fillna(0))

    # Favor ratio: high favor = war likely ending = oil DOWN (sell)
    df["sig_favor"] = -_normalise(df["favor_ratio"].fillna(0.5))

    # --- Composite score ---
    df["composite_score"] = (
        WEIGHTS["price_momentum"] * df["sig_momentum"]
        + WEIGHTS["news_sentiment"] * df["sig_news"]
        + WEIGHTS["reddit_sentiment"] * df["sig_reddit"]
        + WEIGHTS["trump_sentiment"] * df["sig_trump"]
        + WEIGHTS["war_sentiment"] * df["sig_war"]
        + WEIGHTS["favor_ratio"] * df["sig_favor"]
    )

    # --- Generate signal ---
    df["signal"] = "HOLD"
    df.loc[df["composite_score"] > BUY_THRESHOLD, "signal"] = "BUY"
    df.loc[df["composite_score"] < SELL_THRESHOLD, "signal"] = "SELL"

    # --- Signal strength (confidence) ---
    df["signal_strength"] = df["composite_score"].abs().clip(0, 1)

    # --- Backtest: compare signal to NEXT day's actual price direction ---
    # price_direction: 1 = up, 0 = down
    df["next_day_direction"] = df["price_direction"].shift(-1)
    df["signal_numeric"] = df["signal"].map({"BUY": 1, "SELL": 0, "HOLD": np.nan})

    # Accuracy: only for BUY/SELL signals (HOLD is neutral)
    mask_traded = df["signal"].isin(["BUY", "SELL"]) & df["next_day_direction"].notna()
    df["correct"] = np.nan
    df.loc[mask_traded, "correct"] = (
        df.loc[mask_traded, "signal_numeric"] == df.loc[mask_traded, "next_day_direction"]
    ).astype(float)

    # --- Top reasoning for each signal ---
    signal_cols = {
        "sig_momentum": "Price Momentum",
        "sig_news": "News Sentiment",
        "sig_reddit": "Reddit Sentiment",
        "sig_trump": "Trump Sentiment",
        "sig_war": "War Escalation",
        "sig_favor": "War Stance",
    }

    reasons = []
    for _, row in df.iterrows():
        # Find the top 2 contributors
        contribs = {}
        for col, label in signal_cols.items():
            w = WEIGHTS[col.replace("sig_", "").replace("momentum", "price_momentum")
                        .replace("news", "news_sentiment")
                        .replace("reddit", "reddit_sentiment")
                        .replace("trump", "trump_sentiment")
                        .replace("war", "war_sentiment")
                        .replace("favor", "favor_ratio")]
            contribs[label] = row[col] * w

        sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        top2 = sorted_c[:2]
        parts = []
        for name, val in top2:
            direction = "↑" if val > 0 else "↓"
            parts.append(f"{name} {direction}")
        reasons.append(" + ".join(parts))
    df["reason"] = reasons

    return df


def main():
    """Build signals, backtest, and save."""
    import duckdb

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM daily_summary ORDER BY date").df()
    con.close()

    if df.empty:
        print("No data in daily_summary.")
        return

    df["date"] = pd.to_datetime(df["date"]).dt.date
    result = build_signals(df)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(str(OUT_PATH), index=False)

    # Print summary
    n_buy = (result["signal"] == "BUY").sum()
    n_sell = (result["signal"] == "SELL").sum()
    n_hold = (result["signal"] == "HOLD").sum()
    traded = result["correct"].dropna()
    accuracy = traded.mean() if len(traded) > 0 else 0.0

    print(f"Buy signals: {n_buy} | Sell: {n_sell} | Hold: {n_hold}")
    print(f"Backtest accuracy: {accuracy:.1%} ({len(traded)} trades evaluated)")
    print(f"Latest signal: {result['signal'].iloc[-1]} "
          f"(score: {result['composite_score'].iloc[-1]:+.3f})")
    print(f"Reason: {result['reason'].iloc[-1]}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
