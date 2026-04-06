"""
Fetch Middle East war/conflict news and classify stance (favor/against/neutral).

Sources:
    - Google News RSS: Iran, Israel, oil, Strait of Hormuz
    - Al Jazeera RSS: filtered for oil/war keywords

Extracts: title, published_date, summary, source, sentiment_compound, stance.
Saves to ``data/raw/war_news/YYYY-MM-DD.csv``.

Can be run standalone:
    python scripts/ingest/fetch_war_news.py
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
    "google_mideast_oil": (
        "https://news.google.com/rss/search?"
        "q=iran+israel+oil+strait+hormuz&hl=en-US&gl=US&ceid=US:en"
    ),
    "google_opec_war": (
        "https://news.google.com/rss/search?"
        "q=opec+oil+war+middle+east&hl=en-US&gl=US&ceid=US:en"
    ),
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}

# Only keep Al Jazeera articles matching these keywords
WAR_OIL_FILTER = re.compile(
    r"\b(oil|crude|opec|iran|israel|houthi|hormuz|yemen|war|conflict|"
    r"sanction|missile|attack|military|barrel|brent|wti|energy|petroleum|"
    r"hezbollah|hamas|gaza|idf|naval)\b",
    re.I,
)

analyzer = SentimentIntensityAnalyzer()

# Stance thresholds on VADER compound score
FAVOR_THRESHOLD = 0.05    # positive outlook / de-escalation
AGAINST_THRESHOLD = -0.05  # negative / escalation


def _classify_stance(compound: float) -> str:
    """Classify sentiment into favor/against/neutral."""
    if compound > FAVOR_THRESHOLD:
        return "favor"
    elif compound < AGAINST_THRESHOLD:
        return "against"
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

        rows.append(
            {
                "date": _parse_published_date(published),
                "source": source,
                "title": title[:500],
                "published_date": published,
                "summary": summary[:1000],
                "sentiment_compound": round(compound, 4),
                "stance": _classify_stance(compound),
            }
        )
    return rows


def _parse_aljazeera(url: str) -> list[dict]:
    """Parse Al Jazeera RSS, filtering for war/oil keywords."""
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))
        published = entry.get("published", "")

        text = f"{title} {summary}"
        if not WAR_OIL_FILTER.search(text):
            continue  # skip non-relevant articles

        compound = analyzer.polarity_scores(text)["compound"]

        rows.append(
            {
                "date": _parse_published_date(published),
                "source": "aljazeera",
                "title": title[:500],
                "published_date": published,
                "summary": summary[:1000],
                "sentiment_compound": round(compound, 4),
                "stance": _classify_stance(compound),
            }
        )
    return rows


def fetch(data_dir: str | None = None) -> str:
    """Fetch war/conflict news and save to CSV.

    Args:
        data_dir: Output directory. Defaults to ``data/raw/war_news``.

    Returns:
        Path to the CSV file written.
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "data", "raw", "war_news"
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

            # De-duplicate
            new_rows = []
            for r in rows:
                key = r["title"].lower().strip()
                if key not in seen_titles:
                    seen_titles.add(key)
                    new_rows.append(r)
            all_rows.extend(new_rows)
            print(f"Fetched {len(new_rows)} articles from {feed_name}")
        except Exception as exc:
            print(f"Warning: failed to fetch {feed_name}: {exc}")

    df = pd.DataFrame(all_rows)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    output_path = out_dir / f"{today}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} war/conflict articles to {output_path}")

    # Print stance summary
    if not df.empty:
        stance_counts = df["stance"].value_counts().to_dict()
        print(f"Stance breakdown: {stance_counts}")

    return str(output_path)


def main():
    """Entry point for standalone execution."""
    path = fetch()
    print(f"Output: {path}")


if __name__ == "__main__":
    main()
