"""
Fetch Taiwan–China conflict news and compute a Taiwan war probability index.

Sources:
    - Google News RSS: Taiwan, China military, South China Sea, TSMC
    - Al Jazeera RSS: filtered for Taiwan/China keywords

The probability index is a heuristic based on:
    1. Volume of conflict-related articles
    2. Average sentiment (negative = escalation)
    3. Iran-tension spillover proxy (if Iran tensions high, Taiwan risk rises)

Saves to ``data/raw/taiwan/YYYY-MM-DD.csv``.

Can be run standalone:
    python scripts/ingest/fetch_taiwan_tensions.py
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import feedparser
import pandas as pd
from email.utils import parsedate_to_datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _parse_published_date(published_str: str) -> str:
    """Parse RSS date string to YYYY-MM-DD, fall back to today."""
    if published_str:
        try:
            return parsedate_to_datetime(published_str).strftime("%Y-%m-%d")
        except Exception:
            pass
    return datetime.utcnow().strftime("%Y-%m-%d")


RSS_FEEDS = {
    "google_taiwan_china": (
        "https://news.google.com/rss/search?"
        "q=taiwan+china+military+strait&hl=en-US&gl=US&ceid=US:en"
    ),
    "google_south_china_sea": (
        "https://news.google.com/rss/search?"
        "q=south+china+sea+military+conflict&hl=en-US&gl=US&ceid=US:en"
    ),
    "google_tsmc_geopolitics": (
        "https://news.google.com/rss/search?"
        "q=TSMC+taiwan+geopolitics+semiconductor+war&hl=en-US&gl=US&ceid=US:en"
    ),
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}

TAIWAN_FILTER = re.compile(
    r"\b(taiwan|taipei|tsmc|semiconductor|china\s+military|chinese\s+navy|"
    r"pla|south\s+china\s+sea|strait|xi\s+jinping|reunification|"
    r"blockade|invasion|drills?|exercises?|warship|fighter\s+jet|"
    r"pacific\s+fleet|aukus|indo-?pacific)\b",
    re.I,
)

# Iran/ME tension keywords that correlate with Taiwan risk
IRAN_SPILLOVER = re.compile(
    r"\b(iran|tehran|hezbollah|houthi|hormuz|gulf|missile|nuclear)\b",
    re.I,
)

analyzer = SentimentIntensityAnalyzer()


def _classify_escalation(compound: float) -> str:
    """Classify sentiment into escalation/de-escalation/neutral."""
    if compound < -0.1:
        return "escalation"
    elif compound > 0.1:
        return "de-escalation"
    return "neutral"


def _extract_source(title: str, default: str) -> str:
    """Extract news source from Google News title format."""
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return default


def _parse_google_feed(url: str, feed_name: str) -> list[dict]:
    """Parse Google News RSS feed."""
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries:
        raw_title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))
        published = entry.get("published", "")

        source = _extract_source(raw_title, feed_name)
        title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title

        text = f"{title} {summary}"
        compound = analyzer.polarity_scores(text)["compound"]

        # Track if article mentions Iran (spillover signal)
        iran_mention = bool(IRAN_SPILLOVER.search(text))

        rows.append(
            {
                "date": _parse_published_date(published),
                "source": source,
                "title": title[:500],
                "published_date": published,
                "summary": summary[:1000],
                "sentiment_compound": round(compound, 4),
                "escalation": _classify_escalation(compound),
                "iran_spillover": iran_mention,
            }
        )
    return rows


def _parse_aljazeera(url: str) -> list[dict]:
    """Parse Al Jazeera RSS, filtering for Taiwan/China keywords."""
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))
        published = entry.get("published", "")

        text = f"{title} {summary}"
        if not TAIWAN_FILTER.search(text):
            continue

        compound = analyzer.polarity_scores(text)["compound"]
        iran_mention = bool(IRAN_SPILLOVER.search(text))

        rows.append(
            {
                "date": _parse_published_date(published),
                "source": "aljazeera",
                "title": title[:500],
                "published_date": published,
                "summary": summary[:1000],
                "sentiment_compound": round(compound, 4),
                "escalation": _classify_escalation(compound),
                "iran_spillover": iran_mention,
            }
        )
    return rows


def fetch(data_dir: str | None = None) -> str:
    """Fetch Taiwan–China tension news and save to CSV.

    Args:
        data_dir: Output directory. Defaults to ``data/raw/taiwan``.

    Returns:
        Path to the CSV file written.
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "data", "raw", "taiwan"
        )
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    seen_titles: set[str] = set()

    for feed_name, url in RSS_FEEDS.items():
        try:
            if feed_name == "aljazeera":
                rows = _parse_aljazeera(url)
            else:
                rows = _parse_google_feed(url, feed_name)

            new_rows = []
            for r in rows:
                key = r["title"].lower().strip()
                if key not in seen_titles:
                    seen_titles.add(key)
                    new_rows.append(r)
            all_rows.extend(new_rows)
            print(f"Fetched {len(new_rows)} Taiwan/China articles from {feed_name}")
        except Exception as exc:
            print(f"Warning: failed to fetch {feed_name}: {exc}")

    df = pd.DataFrame(all_rows)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    output_path = out_dir / f"{today}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} Taiwan/China articles to {output_path}")

    # Compute a naive probability index
    if not df.empty:
        n_articles = len(df)
        avg_sent = df["sentiment_compound"].mean()
        escalation_ratio = len(df[df["escalation"] == "escalation"]) / n_articles
        iran_pct = df["iran_spillover"].mean()

        # Heuristic formula: baseline 20%, boosted by negative sentiment,
        # escalation percentage, volume, and Iran spillover
        prob = min(0.95, max(0.05,
            0.20
            + escalation_ratio * 0.30       # up to +30% from escalation
            - avg_sent * 0.15               # negative sentiment adds risk
            + iran_pct * 0.10               # Iran spillover adds +10%
            + min(n_articles / 200, 0.15)   # volume signal, capped at +15%
        ))
        print(f"Taiwan war probability index: {prob:.1%}")
        print(f"  articles={n_articles}, avg_sentiment={avg_sent:.3f}, "
              f"escalation_ratio={escalation_ratio:.1%}, iran_spillover={iran_pct:.1%}")
    return str(output_path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fetch()
