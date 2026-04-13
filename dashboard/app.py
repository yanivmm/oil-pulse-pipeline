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
    /* Hide deploy button and accessibility button */
    .stDeployButton, [data-testid="stStatusWidget"],
    button[kind="header"], #MainMenu,
    .stAppDeployButton,
    div[data-testid="stDecoration"] {{
        display: none !important;
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
df_taiwan = _safe_query(con, "SELECT * FROM taiwan_tensions ORDER BY date")
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
st.sidebar.markdown("• Google News (Trump, war, Taiwan)")
st.sidebar.markdown("• Al Jazeera (Middle East, Asia)")
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
# ROW 6 — Round 3: Iran Strike Probability
# ---------------------------------------------------------------------------

# --- Ceasefire auto-detection ---
# If war sentiment turns positive & favor_ratio dominated for 5+ days → ceasefire likely
_ceasefire_detected = False
if not df_filtered.empty and "avg_war_sentiment" in df_filtered.columns:
    _recent_war = df_filtered.tail(5)
    _recent_sent = _recent_war["avg_war_sentiment"].mean() if len(_recent_war) > 0 else -0.1
    _recent_favor = _recent_war["favor_ratio"].mean() if "favor_ratio" in _recent_war.columns else 0.0
    if _recent_sent > 0.15 and _recent_favor > 0.70:
        _ceasefire_detected = True

if _ceasefire_detected:
    st.markdown("### 🕊️ Ceasefire Detected — Round 3 Monitoring Paused")
    st.success(
        "War sentiment has turned **positive** and favorable coverage exceeds **70%** "
        "over the last 5 days. Round 3 (Iran) strike probability section is paused. "
        "If tensions resume, this section will reactivate automatically."
    )
else:
    C_IRAN = "#ff8787"  # red-ish for Iran threat
    st.markdown("### 🚀 Round 3 Probability — Iran Strike")
    st.caption(
        "Israel struck Iran twice previously — probability of a 3rd round, based on "
        "base opinion sentiment (r/conservative + r/politics), war news stance, "
        "and pressure index trajectory. Section auto-hides upon ceasefire detection."
    )

    we1, we2 = st.columns([1, 2])

    with we1:
        if not df_exit.empty:
            exit_row = df_exit.iloc[0]
            ep = exit_row.get("exit_probability", 0.5)
            # Invert: original was "exit probability" — now it's "strike probability"
            # High exit pressure = low strike chance, so: strike_prob = 1 - exit_prob
            strike_prob = 1.0 - ep
            fig_we = go.Figure(go.Indicator(
                mode="gauge+number",
                value=strike_prob * 100,
                number={"suffix": "%"},
                title={"text": "Round 3 Strike Index"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": C_IRAN},
                    "steps": [
                        {"range": [0, 25], "color": "#1f3d25"},
                        {"range": [25, 50], "color": "#3d3520"},
                        {"range": [50, 75], "color": "#3d2a1f"},
                        {"range": [75, 100], "color": "#3d1f1f"},
                    ],
                },
            ))
            fig_we.update_layout(
                height=250, margin=dict(t=40, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT),
            )
            st.plotly_chart(fig_we, use_container_width=True, key="war_exit_gauge")

            # --- Time estimation for Round 3 ---
            pressure = exit_row.get("pressure_index", 0.5)
            if strike_prob >= 0.65:
                _iran_months = 3
            elif strike_prob >= 0.45:
                _iran_months = 6
            elif strike_prob >= 0.25:
                _iran_months = 12
            else:
                _iran_months = 24
            _iran_est = pd.Timestamp.today() + pd.DateOffset(months=_iran_months)
            _iran_est_str = _iran_est.strftime("%B %Y")
            _iran_risk = ("HIGH" if strike_prob >= 0.55 else
                          ("MODERATE" if strike_prob >= 0.30 else "LOW"))

            st.markdown(
                f"<div style='text-align:center;padding:8px;border:1px solid {C_IRAN};"
                f"border-radius:8px;margin-bottom:8px'>"
                f"<span style='font-size:0.8em;color:#aaa'>ESTIMATED WINDOW</span><br>"
                f"<span style='font-size:1.4em;font-weight:bold;color:{C_IRAN}'>{_iran_est_str}</span><br>"
                f"<span style='font-size:0.75em;color:#aaa'>Risk: {_iran_risk} | Pressure: {pressure:.0%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            trend = exit_row.get("base_sentiment_trend", "stable")
            trend_emoji = {"declining": "📉", "improving": "📈", "stable": "➡️"}.get(trend, "➡️")
            st.markdown(f"**Base Trend:** {trend_emoji} {trend.title()}")
            st.markdown(f"**Favor Ratio:** {exit_row.get('favor_ratio', 0.5):.0%}")

            # Intelligence bullets
            _iran_bullets = []
            _fav = exit_row.get("favor_ratio", 0.5)
            _war_s = exit_row.get("war_sentiment_avg", 0.0)
            _iran_bullets.append(
                f"Favor ratio **{_fav:.0%}** — "
                + ("public opinion strongly against continued ops → low strike motivation" if _fav > 0.5
                   else "negative war stance in base → political cover for escalation")
            )
            _iran_bullets.append(
                f"War sentiment **{_war_s:.3f}** — "
                + ("hawkish media framing supports military action" if _war_s < -0.1
                   else "neutral-to-positive framing; less urgency for strikes")
            )
            _iran_bullets.append(
                f"Base trend **{trend}** — "
                + ("declining support = pressure to act decisively before losing base" if trend == "declining"
                   else ("improving sentiment = reduced urgency" if trend == "improving"
                         else "no clear directional pressure"))
            )
            st.markdown(
                f"<div style='background:rgba(255,135,135,0.08);border-left:3px solid {C_IRAN};"
                f"padding:10px 14px;border-radius:4px;margin-top:6px'>"
                f"<b style='color:{C_IRAN}'>🔍 Round 3 Assessment</b></div>",
                unsafe_allow_html=True,
            )
            for _b in _iran_bullets:
                st.markdown(f"- {_b}")
        else:
            st.info("Run the war exit model to see Round 3 forecasts.")

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
                title="Base Sentiment — Iran Strike Pressure",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_base, use_container_width=True, key="base_sent")

st.divider()

# ---------------------------------------------------------------------------
# ROW 7 — Taiwan Invasion Probability
# ---------------------------------------------------------------------------
C_TAIWAN = "#e599f7"  # light purple for Taiwan theme

st.markdown("### 🇹🇼 Taiwan Invasion Probability")
st.caption(
    "Estimated probability and approximate date of a Chinese invasion of Taiwan. "
    "Driven by news volume, escalation signals, Iran-spillover correlation, "
    "and sentiment trajectory from monitored OSINT sources."
)

if not df_taiwan.empty:
    df_taiwan["date"] = pd.to_datetime(df_taiwan["date"]).dt.date
    # Ensure iran_spillover is numeric (stored as string from Spark/parquet)
    df_taiwan["iran_spillover"] = df_taiwan["iran_spillover"].map(
        {"True": 1, "False": 0, True: 1, False: 0}
    ).fillna(0).astype(int)

    # Compute daily aggregates
    tw_daily = df_taiwan.groupby("date").agg(
        article_count=("title", "count"),
        avg_sentiment=("sentiment_compound", "mean"),
        escalation_pct=("escalation", lambda x: (x == "escalation").mean()),
        iran_pct=("iran_spillover", "mean"),
    ).reset_index().sort_values("date")

    # Compute probability index per day
    tw_daily["war_probability"] = (
        0.20
        + tw_daily["escalation_pct"] * 0.30
        - tw_daily["avg_sentiment"] * 0.15
        + tw_daily["iran_pct"] * 0.10
        + (tw_daily["article_count"] / 200).clip(upper=0.15)
    ).clip(0.05, 0.95)

    latest_prob = tw_daily["war_probability"].iloc[-1] if len(tw_daily) > 0 else 0.5

    # --- Approximate date prediction ---
    # Heuristic: higher probability → sooner window.
    # Map probability to months-from-now, then pick a symbolic start-of-quarter date.
    _today = pd.Timestamp.today()
    if latest_prob >= 0.75:
        _months_out = 6
    elif latest_prob >= 0.55:
        _months_out = 12
    elif latest_prob >= 0.35:
        _months_out = 24
    else:
        _months_out = 36
    _est_date = _today + pd.DateOffset(months=_months_out)
    # Snap to Q1 start for symbolic weight (new-year / new-cycle)
    _est_date = pd.Timestamp(year=_est_date.year, month=(((_est_date.month - 1) // 3) * 3 + 1), day=1)
    est_date_str = _est_date.strftime("%B %Y")

    # --- Build intelligence reasoning bullets ---
    n_esc = len(df_taiwan[df_taiwan["escalation"] == "escalation"])
    n_total = len(df_taiwan)
    esc_ratio = n_esc / n_total if n_total else 0
    iran_n = int(df_taiwan["iran_spillover"].sum())
    avg_sent = df_taiwan["sentiment_compound"].mean()

    # Trend direction (last 3 days vs previous 3)
    if len(tw_daily) >= 6:
        _recent = tw_daily["war_probability"].iloc[-3:].mean()
        _prev = tw_daily["war_probability"].iloc[-6:-3].mean()
        trend_delta = _recent - _prev
    elif len(tw_daily) >= 2:
        trend_delta = tw_daily["war_probability"].iloc[-1] - tw_daily["war_probability"].iloc[0]
    else:
        trend_delta = 0.0

    trend_word = "rising" if trend_delta > 0.02 else ("falling" if trend_delta < -0.02 else "stable")

    # Latest headlines for context
    _latest_articles = df_taiwan.sort_values("date", ascending=False).head(5)
    _headlines = _latest_articles["title"].tolist()

    intel_bullets = []
    intel_bullets.append(
        f"Escalation ratio at **{esc_ratio:.0%}** — "
        + ("majority of coverage frames military posturing or threat" if esc_ratio > 0.5
           else "balanced coverage; escalation signals below majority threshold")
    )
    intel_bullets.append(
        f"Iran spillover detected in **{iran_n}** articles — "
        + ("regional multi-front pressure elevates timeline" if iran_n > 5
           else "limited cross-theater correlation so far")
    )
    intel_bullets.append(
        f"Sentiment trajectory **{trend_word}** (Δ {trend_delta:+.1%}) — "
        + ("negative momentum suggests growing hawkish rhetoric" if trend_delta > 0.02
           else ("diplomatic signals may be cooling tensions" if trend_delta < -0.02
                 else "no clear shift in tone detected"))
    )
    intel_bullets.append(
        f"Average sentiment **{avg_sent:.3f}** — "
        + ("deeply negative; media framing skews confrontational" if avg_sent < -0.15
           else ("mildly negative; cautious but not alarmist" if avg_sent < 0
                 else "neutral-to-positive; low urgency in reporting"))
    )
    if latest_prob >= 0.55:
        intel_bullets.append(
            f"Volume signal: **{n_total}** articles tracked — high coverage density "
            "correlates with elevated geopolitical attention"
        )

    # --- Layout ---
    tw1, tw2 = st.columns([1, 2])

    with tw1:
        fig_tw_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_prob * 100,
            number={"suffix": "%"},
            title={"text": "Taiwan Invasion Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": C_TAIWAN},
                "steps": [
                    {"range": [0, 25], "color": "#1f3d25"},
                    {"range": [25, 50], "color": "#3d3520"},
                    {"range": [50, 75], "color": "#3d2a1f"},
                    {"range": [75, 100], "color": "#3d1f1f"},
                ],
            },
        ))
        fig_tw_gauge.update_layout(
            height=250, margin=dict(t=40, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT),
        )
        st.plotly_chart(fig_tw_gauge, use_container_width=True, key="taiwan_gauge")

        # Estimated date callout
        _risk_label = ("HIGH" if latest_prob >= 0.6 else
                       ("MODERATE" if latest_prob >= 0.35 else "LOW"))
        st.markdown(
            f"<div style='text-align:center;padding:8px;border:1px solid {C_TAIWAN};"
            f"border-radius:8px;margin-bottom:8px'>"
            f"<span style='font-size:0.8em;color:#aaa'>ESTIMATED WINDOW</span><br>"
            f"<span style='font-size:1.4em;font-weight:bold;color:{C_TAIWAN}'>{est_date_str}</span><br>"
            f"<span style='font-size:0.75em;color:#aaa'>Risk Level: {_risk_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"**Articles:** {n_total} | **Escalation:** {n_esc} ({esc_ratio:.0%})")
        st.markdown(f"**Iran Spillover:** {iran_n} | **Trend:** {trend_word}")

    with tw2:
        # Escalation breakdown stacked bar + probability line overlay
        fig_tw = go.Figure()
        if len(tw_daily) > 0:
            fig_tw.add_trace(go.Bar(
                x=tw_daily["date"], y=tw_daily["escalation_pct"] * 100,
                name="Escalation %", marker_color="#ff6b6b", opacity=0.6,
            ))
            fig_tw.add_trace(go.Bar(
                x=tw_daily["date"],
                y=(1 - tw_daily["escalation_pct"]) * 100,
                name="Neutral/De-esc %", marker_color="#51cf66", opacity=0.6,
            ))
            fig_tw.add_trace(go.Scatter(
                x=tw_daily["date"], y=tw_daily["war_probability"] * 100,
                mode="lines+markers", name="Invasion Probability %",
                line=dict(color=C_TAIWAN, width=3), marker=dict(size=6),
                yaxis="y2",
            ))
            fig_tw.update_layout(
                barmode="stack",
                height=280, template=PLOTLY_TEMPLATE,
                margin=dict(t=30, b=30, l=40, r=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis_title="", yaxis_title="Sentiment Split %",
                yaxis2=dict(
                    title="Probability %", overlaying="y", side="right",
                    range=[0, 100], showgrid=False,
                ),
                title="Taiwan Invasion — Escalation Trend & Probability",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_tw, use_container_width=True, key="taiwan_trend")

    # --- Intelligence Assessment ---
    st.markdown(
        f"<div style='background:rgba(229,153,247,0.08);border-left:3px solid {C_TAIWAN};"
        f"padding:12px 16px;border-radius:4px;margin-top:4px'>"
        f"<b style='color:{C_TAIWAN}'>🔍 Intelligence Assessment — Taiwan Invasion</b></div>",
        unsafe_allow_html=True,
    )
    for bullet in intel_bullets:
        st.markdown(f"- {bullet}")

    # Latest headlines
    if _headlines:
        with st.expander("📰 Latest monitored headlines", expanded=False):
            for h in _headlines:
                st.markdown(f"- {h}")
else:
    st.info("No Taiwan tension data yet. Run the ingest pipeline first.")

st.divider()

# ---------------------------------------------------------------------------
# ROW 8 — Oil Buy/Sell Signal & Accuracy Tracker
# ---------------------------------------------------------------------------
C_BUY = "#51cf66"
C_SELL = "#ff6b6b"
C_HOLD = "#868e96"

st.markdown("### 📈 Oil Trading Signal — Buy / Sell / Hold")
st.caption(
    "Composite model combining price momentum, news sentiment, Reddit sentiment, "
    "Trump sentiment, war escalation, and war stance signals. Backtested against "
    "next-day price direction. The model learns from its mistakes via accuracy tracking."
)

# Load buy signals parquet
_buy_sig_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "predictions" / "buy_signals.parquet"
df_signals = pd.DataFrame()
if _buy_sig_path.exists():
    df_signals = pd.read_parquet(str(_buy_sig_path))
    df_signals["date"] = pd.to_datetime(df_signals["date"]).dt.date

if not df_signals.empty:
    # --- Accuracy metrics ---
    _traded = df_signals[df_signals["correct"].notna()]
    _accuracy = _traded["correct"].mean() if len(_traded) > 0 else 0.0
    _n_trades = len(_traded)
    _n_correct = int(_traded["correct"].sum()) if len(_traded) > 0 else 0
    _n_buy = (df_signals["signal"] == "BUY").sum()
    _n_sell = (df_signals["signal"] == "SELL").sum()
    _n_hold = (df_signals["signal"] == "HOLD").sum()
    _latest_sig = df_signals.iloc[-1]

    bs1, bs2 = st.columns([1, 2])

    with bs1:
        # Latest signal callout
        _sig_color = {
            "BUY": C_BUY, "SELL": C_SELL, "HOLD": C_HOLD
        }.get(_latest_sig["signal"], C_HOLD)
        _sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(_latest_sig["signal"], "⚪")

        st.markdown(
            f"<div style='text-align:center;padding:12px;border:2px solid {_sig_color};"
            f"border-radius:10px;margin-bottom:10px'>"
            f"<span style='font-size:0.8em;color:#aaa'>LATEST SIGNAL</span><br>"
            f"<span style='font-size:2em;font-weight:bold;color:{_sig_color}'>"
            f"{_sig_emoji} {_latest_sig['signal']}</span><br>"
            f"<span style='font-size:0.85em;color:#ccc'>Score: {_latest_sig['composite_score']:+.3f}"
            f" | Strength: {_latest_sig['signal_strength']:.0%}</span><br>"
            f"<span style='font-size:0.75em;color:#999'>{_latest_sig['reason']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Accuracy gauge
        fig_acc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=_accuracy * 100,
            number={"suffix": "%"},
            title={"text": "Backtest Accuracy"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": C_BUY if _accuracy >= 0.55 else C_SELL},
                "steps": [
                    {"range": [0, 40], "color": "#3d1f1f"},
                    {"range": [40, 55], "color": "#3d3520"},
                    {"range": [55, 100], "color": "#1f3d25"},
                ],
            },
        ))
        fig_acc.update_layout(
            height=200, margin=dict(t=40, b=0, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT),
        )
        st.plotly_chart(fig_acc, use_container_width=True, key="accuracy_gauge")

        st.markdown(
            f"**Trades:** {_n_trades} evaluated | **Correct:** {_n_correct} "
            f"({_accuracy:.0%})<br>"
            f"**Signals:** {_n_buy} BUY · {_n_sell} SELL · {_n_hold} HOLD",
            unsafe_allow_html=True,
        )

    with bs2:
        # Price chart with buy/sell markers
        fig_bs = go.Figure()

        # Price line
        fig_bs.add_trace(go.Scatter(
            x=df_signals["date"], y=df_signals["avg_price"],
            mode="lines", name="Oil Price",
            line=dict(color="#ffd43b", width=2),
        ))

        # BUY markers
        _buys = df_signals[df_signals["signal"] == "BUY"]
        if not _buys.empty:
            fig_bs.add_trace(go.Scatter(
                x=_buys["date"], y=_buys["avg_price"],
                mode="markers", name="BUY",
                marker=dict(color=C_BUY, size=10, symbol="triangle-up"),
                text=_buys["reason"], hovertemplate="%{x}<br>$%{y:.2f}<br>%{text}",
            ))

        # SELL markers
        _sells = df_signals[df_signals["signal"] == "SELL"]
        if not _sells.empty:
            fig_bs.add_trace(go.Scatter(
                x=_sells["date"], y=_sells["avg_price"],
                mode="markers", name="SELL",
                marker=dict(color=C_SELL, size=10, symbol="triangle-down"),
                text=_sells["reason"], hovertemplate="%{x}<br>$%{y:.2f}<br>%{text}",
            ))

        # Composite score on secondary axis
        fig_bs.add_trace(go.Bar(
            x=df_signals["date"], y=df_signals["composite_score"],
            name="Composite Score", opacity=0.3,
            marker_color=df_signals["composite_score"].apply(
                lambda v: C_BUY if v > 0 else C_SELL
            ),
            yaxis="y2",
        ))

        fig_bs.update_layout(
            height=350, template=PLOTLY_TEMPLATE,
            margin=dict(t=30, b=30, l=50, r=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="", yaxis_title="Oil Price ($)",
            yaxis2=dict(
                title="Score", overlaying="y", side="right",
                range=[-1, 1], showgrid=False,
            ),
            title="Oil Price with Buy/Sell Signals",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bs, use_container_width=True, key="buy_signal_chart")

    # --- Backtest detail table ---
    with st.expander("📊 Signal Backtest Detail", expanded=False):
        _detail = df_signals[["date", "signal", "avg_price", "composite_score",
                              "signal_strength", "reason", "next_day_direction", "correct"]].copy()
        _detail["next_day"] = _detail["next_day_direction"].map({1.0: "Up ↑", 0.0: "Down ↓"})
        _detail["result"] = _detail["correct"].map({1.0: "✓ Correct", 0.0: "✗ Wrong"})
        _detail = _detail.drop(columns=["next_day_direction", "correct"])
        st.dataframe(
            _detail.sort_values("date", ascending=False),
            use_container_width=True, hide_index=True, height=250, key="backtest_table",
        )

    # --- Model self-assessment ---
    if len(_traded) >= 3:
        # Check recent accuracy (last 5 trades)
        _recent_trades = _traded.tail(5)
        _recent_acc = _recent_trades["correct"].mean()
        st.markdown(
            f"<div style='background:rgba(81,207,102,0.08);border-left:3px solid {C_BUY};"
            f"padding:10px 14px;border-radius:4px;margin-top:6px'>"
            f"<b style='color:{C_BUY}'>🧠 Model Self-Assessment</b></div>",
            unsafe_allow_html=True,
        )
        if _recent_acc >= 0.7:
            st.markdown(
                f"- Recent accuracy **{_recent_acc:.0%}** (last 5 trades) — "
                "model is performing well, signals are reliable"
            )
        elif _recent_acc >= 0.4:
            st.markdown(
                f"- Recent accuracy **{_recent_acc:.0%}** (last 5 trades) — "
                "mixed results, use signals as one input among many"
            )
        else:
            st.markdown(
                f"- Recent accuracy **{_recent_acc:.0%}** (last 5 trades) — "
                "⚠️ model is underperforming, signals may be contrarian indicators"
            )
        st.markdown(
            f"- Overall accuracy: **{_accuracy:.0%}** across {_n_trades} trades"
        )
else:
    st.info("No buy signals yet. Run: `python scripts/ml/buy_signal_model.py`")

st.divider()

# ---------------------------------------------------------------------------
# ROW 9 — Prediction History (compact)
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
    "ML: RandomForest + Sentiment Analysis | "
    "Taiwan-China: Google News + Al Jazeera conflict monitoring"
)
