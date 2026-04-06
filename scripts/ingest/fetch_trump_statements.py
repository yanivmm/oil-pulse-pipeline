"""
Fetch news articles quoting Trump's statements on oil, war, and Middle East.

Sources: Google News RSS with targeted queries capturing Truth Social posts
quoted by media, press conferences, and policy statements.

Extracts: title, published_date, summary, source, sentiment_compound, topic_tags.
Saves to ``data/raw/trump/YYYY-MM-DD.csv``.

Can be run standalone:
    python scripts/ingest/fetch_trump_statements.py
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

# Google News RSS — captures Truth Social via news coverage, no API needed
GOOGLE_NEWS_QUERIES = {
    "trump_oil_mideast": (
        "https://news.google.com/rss/search?"
        "q=trump+oil+middle+east&hl=en-US&gl=US&ceid=US:en"
    ),
    "trump_truth_war": (
        "https://news.google.com/rss/search?"
        "q=trump+truth+social+war+oil&hl=en-US&gl=US&ceid=US:en"
    ),
    "trump_iran_sanctions": (
        "https://news.google.com/rss/search?"
        "q=trump+iran+sanctions+oil&hl=en-US&gl=US&ceid=US:en"
    ),
}

# Keywords for topic tagging
TOPIC_KEYWORDS = {
    "oil": re.compile(r"\b(oil|crude|barrel|opec|brent|wti|petroleum|energy)\b", re.I),
    "iran": re.compile(r"\b(iran|tehran|khamenei|rouhani|persian)\b", re.I),
    "israel": re.compile(r"\b(israel|netanyahu|idf|gaza|hamas|hezbollah)\b", re.I),
    "hormuz": re.compile(r"\b(hormuz|strait|naval|blockade)\b", re.I),
    "war": re.compile(r"\b(war|conflict|military|strike|attack|bomb|missile)\b", re.I),
    "sanctions": re.compile(r"\b(sanction|tariff|embargo|trade.?war|ban)\b", re.I),
    "truth_social": re.compile(r"\b(truth.?social|tweet|post|truth)\b", re.I),
}

analyzer = SentimentIntensityAnalyzer()


def _tag_topics(text: str) -> str:
    """Return comma-separated topic tags based on keyword matching."""
    tags = [topic for topic, pattern in TOPIC_KEYWORDS.items() if pattern.search(text)]
    return ",".join(tags) if tags else "general"


def _extract_source(title: str) -> str:
    """Extract news source from Google News title format 'Headline - Source'."""
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return "unknown"


def _parse_feed(url: str, query_name: str) -> list[dict]:
    """Parse a single Google News RSS feed and return list of row dicts."""
    feed = feedparser.parse(url)
    rows = []
    seen_titles = set()

    for entry in feed.entries:
        raw_title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))
        published = entry.get("published", "")

        # De-duplicate by title
        title_key = raw_title.lower().strip()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        source = _extract_source(raw_title)
        # Clean title: remove " - Source" suffix
        title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title

        text = f"{title} {summary}"
        sentiment = analyzer.polarity_scores(text)["compound"]
        topics = _tag_topics(text)

        rows.append(
            {
                "date": _parse_published_date(published),
                "source": source,
                "query": query_name,
                "title": title[:500],
                "published_date": published,
                "summary": summary[:1000],
                "sentiment_compound": round(sentiment, 4),
                "topic_tags": topics,
            }
        )
    return rows


def fetch(data_dir: str | None = None) -> str:
    """Fetch Trump-related news from Google News RSS and save to CSV.

    Args:
        data_dir: Output directory. Defaults to ``data/raw/trump``.

    Returns:
        Path to the CSV file written.
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "data", "raw", "trump"
        )
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    global_seen: set[str] = set()

    for query_name, url in GOOGLE_NEWS_QUERIES.items():
        try:
            rows = _parse_feed(url, query_name)
            # Cross-query dedup
            new_rows = []
            for r in rows:
                key = r["title"].lower().strip()
                if key not in global_seen:
                    global_seen.add(key)
                    new_rows.append(r)
            all_rows.extend(new_rows)
            print(f"Fetched {len(new_rows)} articles from query '{query_name}'")
        except Exception as exc:
            print(f"Warning: failed to fetch '{query_name}': {exc}")

    df = pd.DataFrame(all_rows)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    output_path = out_dir / f"{today}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} Trump-related articles to {output_path}")
    return str(output_path)


def main():
    """Entry point for standalone execution."""
    path = fetch()
    print(f"Output: {path}")


if __name__ == "__main__":
    main()
