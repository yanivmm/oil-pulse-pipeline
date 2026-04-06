"""
Fetch business news from RSS feeds and score sentiment with VADER.

Sources:
    - Reuters Business News
    - BBC Business News

Extracts: title, published_date, summary, source, sentiment_compound.
Saves to ``data/raw/news/YYYY-MM-DD.csv``.

Can be run standalone:
    python scripts/ingest/fetch_rss_news.py
"""

from __future__ import annotations

import os
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

# RSS feed URLs — public, no authentication needed
# Note: Reuters discontinued public RSS feeds; we use CNBC + MarketWatch instead
RSS_FEEDS = {
    "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "bbc": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "marketwatch": "http://feeds.marketwatch.com/marketwatch/topstories/",
}

analyzer = SentimentIntensityAnalyzer()


def _parse_feed(url: str, source: str) -> list[dict]:
    """Parse a single RSS feed and return list of row dicts."""
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        published = entry.get("published", "")
        text = f"{title} {summary}"
        sentiment = analyzer.polarity_scores(text)["compound"]

        rows.append(
            {
                "date": _parse_published_date(published),
                "source": source,
                "title": title[:500],
                "published_date": published,
                "summary": summary[:1000],
                "sentiment_compound": round(sentiment, 4),
            }
        )
    return rows


def fetch(data_dir: str | None = None) -> str:
    """Fetch news from all RSS feeds and save to CSV.

    Args:
        data_dir: Output directory. Defaults to ``data/raw/news``.

    Returns:
        Path to the CSV file written.
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "data", "raw", "news"
        )
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for source, url in RSS_FEEDS.items():
        try:
            rows = _parse_feed(url, source)
            all_rows.extend(rows)
            print(f"Fetched {len(rows)} articles from {source}")
        except Exception as exc:
            print(f"Warning: failed to fetch {source}: {exc}")

    df = pd.DataFrame(all_rows)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    output_path = out_dir / f"{today}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} articles to {output_path}")
    return str(output_path)


def main():
    """Entry point for standalone execution."""
    path = fetch()
    print(f"Output: {path}")


if __name__ == "__main__":
    main()
