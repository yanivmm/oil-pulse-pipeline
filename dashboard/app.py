"""
Oil Pulse Dashboard — Enhanced Streamlit app reading from DuckDB.

Features:
    - Dark theme with vibrant, colorful charts
    - 5 metric cards: prediction, accuracy, price, favor ratio, war exit pressure
    - Oil price chart with Trump statement sentiment markers
    - Volatility bar chart (daily price delta)
    - 3-column sentiment grid: Reddit, War news favor/against, Trump trend
    - News & quotes ticker with sentiment badges
    - Favor vs Against gauges and area chart
    - War Exit Pressure Forecaster with gauge and trend
    - Prediction history table

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "oil_pulse.duckdb"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

# Color palette
C_OIL = "#4da3ff"
C_OIL_LIGHT = "#80bfff"
C_WAR = "#ff6b6b"
C_TRUMP = "#ffd43b"
C_GREEN = "#51cf66"
C_RED = "#ff6b6b"
C_NEUTRAL = "#adb5bd"
C_BG = "#0e1117"
C_CARD = "#1a1d23"
C_SURFACE = "#262730"
C_TEXT = "#fafafa"
C_TEXT_DIM = "#a0a4ab"
PLOTLY_TEMPLATE = "plotly_dark"


def _connect_duckdb():
    import duckdb
    if not DB_PATH.exists():
        return None
    return duckdb.connect(str(DB_PATH), read_only=True)


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def _safe_query(con, query: str) -> pd.DataFrame:
    """Run a query, return empty DataFrame if table doesn't exist."""
    try:
        return con.execute(query).fetchdf()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Oil Pulse Pipeline", page_icon="🛢️", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {C_BG};
        color: {C_TEXT};
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    .stMetric {{
        background: {C_CARD};
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid {C_SURFACE};
    }}
    div[data-testid="stExpander"] {{
        background: {C_CARD};
        border-radius: 8px;
        border: 1px solid {C_SURFACE};
    }}
    .stDivider {{
        border-color: {C_SURFACE};
    }}
    .sentiment-badge-favor {{
        background: rgba(81,207,102,0.15); color: {C_GREEN}; padding: 2px 8px;
        border-radius: 12px; font-size: 0.8em; font-weight: 600;
    }}
    .sentiment-badge-against {{
        background: rgba(255,107,107,0.15); color: {C_RED}; padding: 2px 8px;
        border-radius: 12px; font-size: 0.8em; font-weight: 600;
    }}
    .sentiment-badge-neutral {{
        background: rgba(173,181,189,0.15); color: {C_NEUTRAL}; padding: 2px 8px;
        border-radius: 12px; font-size: 0.8em; font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛢️ Oil Pulse Pipeline")
st.markdown(
    "**Tracking crude oil prices, geopolitical sentiment, Trump's war stance, "
    "and ML predictions** — all in real time."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
con = _connect_duckdb()
if con is None:
    st.warning("⚠️ Database not found. Run the pipeline first.")
    st.stop()

df_summary = _safe_query(con, "SELECT * FROM daily_summary ORDER BY date")
df_preds = _safe_query(con, "SELECT * FROM predictions ORDER BY date")
df_trump = _safe_query(con, "SELECT * FROM trump_statements ORDER BY date DESC LIMIT 50")
df_war = _safe_query(con, "SELECT * FROM war_news ORDER BY date DESC LIMIT 50")
df_exit = _safe_query(con, "SELECT * FROM war_exit_index ORDER BY date DESC LIMIT 30")
con.close()

if df_summary.empty:
    st.info("No data in daily_summary table yet. Run the full pipeline first.")
    st.stop()

df_summary["date"] = pd.to_datetime(df_summary["date"]).dt.date

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Filters")
min_date = df_summary["date"].min()
max_date = df_summary["date"].max()

date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df_summary["date"] >= start_date) & (df_summary["date"] <= end_date)
df_filtered = df_summary[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
st.sidebar.markdown("• Yahoo Finance (oil prices)")
st.sidebar.markdown("• BBC, CNBC, MarketWatch (news)")
st.sidebar.markdown("• Google News (Trump, war)")
st.sidebar.markdown("• Al Jazeera (Middle East)")
st.sidebar.markdown("• Reddit (7 subreddits)")

# ---------------------------------------------------------------------------
# ROW 1 — 5 Metric Cards
# ---------------------------------------------------------------------------
metrics = _load_metrics()
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if not df_preds.empty:
        latest_pred = df_preds.iloc[-1]
        direction = latest_pred.get("prediction", "N/A")
        arrow = "↑" if direction == "Up" else "↓"
        st.metric("Today's Prediction", f"{arrow} {direction}")
    else:
        st.metric("Today's Prediction", "N/A")

with c2:
    acc = metrics.get("accuracy", "N/A")
    st.metric("Model Accuracy", f"{acc:.1%}" if isinstance(acc, float) else acc)

with c3:
    price = df_filtered["avg_price"].iloc[-1] if not df_filtered.empty else "N/A"
    st.metric("Oil Price", f"${price:.2f}" if isinstance(price, (int, float)) else price)

with c4:
    fr = df_filtered["favor_ratio"].iloc[-1] if not df_filtered.empty and "favor_ratio" in df_filtered.columns else None
    if fr is not None and pd.notna(fr):
        st.metric("Favor Ratio", f"{fr:.0%}")
    else:
        st.metric("Favor Ratio", "N/A")

with c5:
    if not df_exit.empty:
        exit_prob = df_exit.iloc[0].get("exit_probability", None)
        if exit_prob is not None:
            st.metric("War Exit Pressure", f"{exit_prob:.0%}")
        else:
            st.metric("War Exit Pressure", "N/A")
    else:
        st.metric("War Exit Pressure", "N/A")

st.divider()

# ---------------------------------------------------------------------------
# ROW 2 — Price + Trump overlay / Volatility (2 columns)
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"#### 📈 Oil Price & Trump Sentiment")
    if not df_filtered.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered["date"], y=df_filtered["avg_price"],
            mode="lines", name="Oil Price", line=dict(color=C_OIL, width=2),
        ))
        if "rolling_avg_7d" in df_filtered.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered["date"], y=df_filtered["rolling_avg_7d"],
                mode="lines", name="7-Day Avg",
                line=dict(color=C_OIL_LIGHT, width=1.5, dash="dash"),
            ))
        # Trump sentiment markers
        if not df_trump.empty and "avg_trump_sentiment" in df_filtered.columns:
            trump_days = df_filtered[df_filtered["avg_trump_sentiment"].abs() > 0.01]
            if not trump_days.empty:
                colors = [C_GREEN if s > 0 else C_RED for s in trump_days["avg_trump_sentiment"]]
                fig.add_trace(go.Scatter(
                    x=trump_days["date"], y=trump_days["avg_price"],
                    mode="markers", name="Trump Statement",
                    marker=dict(size=8, color=colors, symbol="diamond", line=dict(width=1, color="white")),
                    text=[f"Sentiment: {s:.2f}" for s in trump_days["avg_trump_sentiment"]],
                    hovertemplate="%{text}<extra></extra>",
                ))
        fig.update_layout(
            height=260, template=PLOTLY_TEMPLATE, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="", yaxis_title="USD",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key="price_trump")

with col_right:
    st.markdown("#### 📊 Daily Price Volatility")
    if not df_filtered.empty and "avg_price_delta" in df_filtered.columns:
        df_vol = df_filtered[["date", "avg_price_delta"]].dropna()
        colors = [C_GREEN if d >= 0 else C_RED for d in df_vol["avg_price_delta"]]
        fig_vol = go.Figure(go.Bar(
            x=df_vol["date"], y=df_vol["avg_price_delta"],
            marker_color=colors, name="Price Delta",
        ))
        fig_vol.update_layout(
            height=260, template=PLOTLY_TEMPLATE, margin=dict(t=10, b=30, l=40, r=10),
            xaxis_title="", yaxis_title="$ Change",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_vol, use_container_width=True, key="volatility")

st.divider()

# ---------------------------------------------------------------------------
# ROW 3 — Sentiment Grid (3 columns)
# ---------------------------------------------------------------------------
sc1, sc2, sc3 = st.columns(3)

with sc1:
    st.markdown("#### 💬 Reddit Sentiment")
    if not df_filtered.empty:
        fig_r = go.Figure()
        if "avg_reddit_sentiment" in df_filtered.columns:
            fig_r.add_trace(go.Scatter(
                x=df_filtered["date"], y=df_filtered["avg_reddit_sentiment"],
                mode="lines+markers", name="Reddit", line=dict(color=C_OIL, width=1.5),
                marker=dict(size=3),
            ))
        if "avg_news_sentiment" in df_filtered.columns:
            fig_r.add_trace(go.Scatter(
                x=df_filtered["date"], y=df_filtered["avg_news_sentiment"],
                mode="lines+markers", name="News", line=dict(color=C_NEUTRAL, width=1.5),
                marker=dict(size=3),
            ))
        fig_r.update_layout(
            height=220, template=PLOTLY_TEMPLATE, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="", yaxis_title="Sentiment",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_r, use_container_width=True, key="reddit_sent")

with sc2:
    st.markdown("#### ⚔️ War News Favor vs Against")
    if not df_filtered.empty and "favor_count" in df_filtered.columns:
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(
            x=df_filtered["date"], y=df_filtered["favor_count"],
            name="Favor", marker_color=C_GREEN,
        ))
        fig_w.add_trace(go.Bar(
            x=df_filtered["date"], y=df_filtered["against_count"],
            name="Against", marker_color=C_RED,
        ))
        fig_w.update_layout(
            barmode="stack", height=220, template=PLOTLY_TEMPLATE,
            margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="", yaxis_title="Articles",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_w, use_container_width=True, key="war_fav")

with sc3:
    st.markdown(f"#### 🏛️ Trump Sentiment Trend")
    if not df_filtered.empty and "avg_trump_sentiment" in df_filtered.columns:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=df_filtered["date"], y=df_filtered["avg_trump_sentiment"],
            mode="lines+markers", name="Trump Sentiment",
            line=dict(color=C_TRUMP, width=2), marker=dict(size=4),
        ))
        if "trump_sentiment_rolling_3d" in df_filtered.columns:
            fig_t.add_trace(go.Scatter(
                x=df_filtered["date"], y=df_filtered["trump_sentiment_rolling_3d"],
                mode="lines", name="3-Day Avg",
                line=dict(color=C_TRUMP, width=1, dash="dash"),
            ))
        fig_t.add_hline(y=0, line_dash="dot", line_color=C_NEUTRAL, opacity=0.5)
        fig_t.update_layout(
            height=220, template=PLOTLY_TEMPLATE, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="", yaxis_title="Sentiment",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_t, use_container_width=True, key="trump_trend")

st.divider()

# ---------------------------------------------------------------------------
# ROW 4 — News Quotes Ticker
# ---------------------------------------------------------------------------
with st.expander("📰 **Latest Headlines — Trump Statements & War News**", expanded=False):
    q1, q2 = st.columns(2)

    with q1:
        st.markdown("**🏛️ Trump / Oil / Middle East**")
        if not df_trump.empty:
            for _, row in df_trump.head(8).iterrows():
                sent = row.get("sentiment_compound", 0)
                if sent > 0.05:
                    badge = f'<span class="sentiment-badge-favor">+{sent:.2f}</span>'
                elif sent < -0.05:
                    badge = f'<span class="sentiment-badge-against">{sent:.2f}</span>'
                else:
                    badge = f'<span class="sentiment-badge-neutral">{sent:.2f}</span>'
                source = row.get("source", "")
                title = row.get("title", "")
                st.markdown(f"{badge} **{source}** — {title}", unsafe_allow_html=True)
        else:
            st.info("No Trump-related articles yet.")

    with q2:
        st.markdown("**⚔️ War / Conflict / Oil**")
        if not df_war.empty:
            for _, row in df_war.head(8).iterrows():
                sent = row.get("sentiment_compound", 0)
                stance = row.get("stance", "neutral")
                if stance == "favor":
                    badge = f'<span class="sentiment-badge-favor">{stance} ({sent:+.2f})</span>'
                elif stance == "against":
                    badge = f'<span class="sentiment-badge-against">{stance} ({sent:+.2f})</span>'
                else:
                    badge = f'<span class="sentiment-badge-neutral">{stance} ({sent:+.2f})</span>'
                source = row.get("source", "")
                title = row.get("title", "")
                st.markdown(f"{badge} **{source}** — {title}", unsafe_allow_html=True)
        else:
            st.info("No war news articles yet.")

st.divider()

# ---------------------------------------------------------------------------
# ROW 5 — Favor vs Against Gauge + Area Chart
# ---------------------------------------------------------------------------
g1, g2 = st.columns([1, 2])

with g1:
    st.markdown("#### Favor/Against Gauge")
    if not df_filtered.empty and "favor_ratio" in df_filtered.columns:
        current_ratio = df_filtered["favor_ratio"].iloc[-1]
        if pd.notna(current_ratio):
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_ratio * 100,
                number={"suffix": "%"},
                title={"text": "Favorable Outlook"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": C_OIL},
                    "steps": [
                        {"range": [0, 30], "color": "#3d1f1f"},
                        {"range": [30, 60], "color": "#3d3520"},
                        {"range": [60, 100], "color": "#1f3d25"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.75,
                        "value": current_ratio * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=250, margin=dict(t=40, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT),
            )
            st.plotly_chart(fig_gauge, use_container_width=True, key="favor_gauge")

with g2:
    st.markdown("#### Favor vs Against Over Time")
    if not df_filtered.empty and "favor_count" in df_filtered.columns:
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=df_filtered["date"], y=df_filtered["favor_count"],
            fill="tozeroy", name="Favor", line=dict(color=C_GREEN), fillcolor="rgba(40,167,69,0.2)",
        ))
        fig_area.add_trace(go.Scatter(
            x=df_filtered["date"], y=df_filtered["against_count"],
            fill="tozeroy", name="Against", line=dict(color=C_RED), fillcolor="rgba(220,53,69,0.2)",
        ))
        fig_area.update_layout(
            height=250, template=PLOTLY_TEMPLATE, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="", yaxis_title="Article Count",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_area, use_container_width=True, key="favor_area")

st.divider()

# ---------------------------------------------------------------------------
# ROW 6 — War Exit Pressure Forecaster
# ---------------------------------------------------------------------------
st.markdown("### 🔮 War Exit Pressure Forecaster")
st.caption(
    "Based on sentiment analysis of r/conservative + r/politics and news coverage "
    "of Trump's base opinion on the Middle East conflict. "
    "This is an analytical tool measuring public opinion pressure, not a political forecast."
)

we1, we2 = st.columns([1, 2])

with we1:
    if not df_exit.empty:
        exit_row = df_exit.iloc[0]
        ep = exit_row.get("exit_probability", 0.5)
        fig_we = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ep * 100,
            number={"suffix": "%"},
            title={"text": "Exit Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": C_TRUMP},
                "steps": [
                    {"range": [0, 30], "color": "#1f3d25"},
                    {"range": [30, 60], "color": "#3d3520"},
                    {"range": [60, 100], "color": "#3d1f1f"},
                ],
            },
        ))
        fig_we.update_layout(
            height=250, margin=dict(t=40, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT),
        )
        st.plotly_chart(fig_we, use_container_width=True, key="war_exit_gauge")

        trend = exit_row.get("base_sentiment_trend", "stable")
        trend_emoji = {"declining": "📉", "improving": "📈", "stable": "➡️"}.get(trend, "➡️")
        st.markdown(f"**Trend:** {trend_emoji} {trend.title()}")
        st.markdown(f"**Pressure Index:** {exit_row.get('pressure_index', 0):.1%}")
    else:
        st.info("Run the war exit model to see forecasts.")

with we2:
    if not df_filtered.empty and "base_war_sentiment" in df_filtered.columns:
        fig_base = go.Figure()
        fig_base.add_trace(go.Scatter(
            x=df_filtered["date"], y=df_filtered["base_war_sentiment"],
            mode="lines+markers", name="Base War Sentiment",
            line=dict(color=C_WAR, width=2), marker=dict(size=3),
        ))
        if "base_sentiment_rolling_7d" in df_filtered.columns:
            fig_base.add_trace(go.Scatter(
                x=df_filtered["date"], y=df_filtered["base_sentiment_rolling_7d"],
                mode="lines", name="7-Day Rolling Avg",
                line=dict(color=C_TRUMP, width=2, dash="dash"),
            ))
        fig_base.add_hline(y=0, line_dash="dot", line_color=C_NEUTRAL, opacity=0.5)
        fig_base.update_layout(
            height=250, template=PLOTLY_TEMPLATE, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="", yaxis_title="Sentiment",
            title="Base (Conservative/Politics) War Sentiment",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_base, use_container_width=True, key="base_sent")

st.divider()

# ---------------------------------------------------------------------------
# ROW 7 — Prediction History (compact)
# ---------------------------------------------------------------------------
st.markdown("#### 📋 Prediction History")

if not df_preds.empty:
    df_preds_display = df_preds.copy()
    df_preds_display["date"] = pd.to_datetime(df_preds_display["date"]).dt.date

    if not df_summary.empty:
        df_actual = df_summary[["date", "price_direction"]].copy()
        df_actual.rename(columns={"price_direction": "actual_numeric"}, inplace=True)
        df_merged = df_preds_display.merge(df_actual, on="date", how="left")
        df_merged["actual"] = df_merged["actual_numeric"].map({1: "Up", 0: "Down"})
        df_merged["correct"] = (
            df_merged["prediction_numeric"] == df_merged["actual_numeric"]
        ).map({True: "✓", False: "✗"})
        display_cols = ["date", "prediction", "actual", "confidence", "correct"]
        st.dataframe(
            df_merged[display_cols].sort_values("date", ascending=False),
            use_container_width=True, hide_index=True, height=200, key="pred_table",
        )
    else:
        st.dataframe(df_preds_display, use_container_width=True, hide_index=True, key="pred_raw")
else:
    st.info("No predictions yet. Run the predict pipeline first.")

# ---------------------------------------------------------------------------
st.caption(
    "Oil Pulse Pipeline — Portfolio Project | "
    "Data: Yahoo Finance, Reddit, BBC, CNBC, Google News, Al Jazeera | "
    "ML: RandomForest + Sentiment Analysis"
)
