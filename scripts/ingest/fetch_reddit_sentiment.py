"""
Fetch top posts from Reddit and score sentiment with VADER.

Subreddits:
    General:  r/geopolitics, r/economy, r/worldnews, r/energy, r/oil
    Base:     r/politics, r/conservative  (tracks Trump's base opinion — War Exit model)

Uses Reddit's **public JSON endpoints** (no API key needed).
Fetches top ~100 posts per subreddit, scores title+body with VADER.
Tags each post with topic (trump/oil/war) and subreddit_group (base/general).
Saves to ``data/raw/reddit/YYYY-MM-DD.csv``.

Can be run standalone:
    python scripts/ingest/fetch_reddit_sentiment.py
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Subreddit configuration: name → group
SUBREDDITS = {
    "geopolitics": "general",
    "economy": "general",
    "worldnews": "general",
    "energy": "general",
    "oil": "general",
    "politics": "base",
    "conservative": "base",
}

POSTS_PER_SUB = 100

# Topic classification keywords
TOPIC_PATTERNS = {
    "trump": re.compile(r"\b(trump|maga|truth.?social|potus)\b", re.I),
    "oil": re.compile(r"\b(oil|crude|opec|barrel|brent|wti|petroleum|energy)\b", re.I),
    "war": re.compile(
        r"\b(iran|israel|war|conflict|houthi|hormuz|yemen|military|"
        r"hezbollah|hamas|gaza|missile|attack|sanction)\b",
        re.I,
    ),
}

HEADERS = {
    "User-Agent": "oil-pulse-pipeline/0.1 (portfolio project; non-commercial)",
}


def _fetch_subreddit_json(sub_name: str, limit: int = 100) -> list[dict]:
    """Fetch posts from Reddit's public JSON endpoint (no auth needed)."""
    posts: list[dict] = []
    after = None
    per_page = min(limit, 100)

    while len(posts) < limit:
        url = f"https://www.reddit.com/r/{sub_name}/hot.json"
        params = {"limit": per_page, "raw_json": 1}
        if after:
            params["after"] = after

        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 429:
            # Rate limited — wait and retry once
            time.sleep(3)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()

        data = resp.json().get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            p = child.get("data", {})
            posts.append(p)

        after = data.get("after")
        if not after:
            break

        # Be polite — Reddit rate-limits unauthenticated to ~10 req/min
        time.sleep(2)

    return posts[:limit]


def _score_sentiment(text: str) -> float:
    """Return VADER compound sentiment score for a text string."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]


def _classify_topics(text: str) -> str:
    """Return comma-separated topic tags based on keyword matching."""
    tags = [topic for topic, pattern in TOPIC_PATTERNS.items() if pattern.search(text)]
    return ",".join(tags) if tags else "general"


def fetch(data_dir: str | None = None) -> str:
    """Fetch Reddit posts and compute sentiment.

    Args:
        data_dir: Output directory. Defaults to ``data/raw/reddit``.

    Returns:
        Path to the CSV file written.
    """
    if data_dir is None:
        data_dir = str(
            Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "reddit"
        )
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for sub_name, group in SUBREDDITS.items():
        try:
            posts = _fetch_subreddit_json(sub_name, limit=POSTS_PER_SUB)
            for p in posts:
                title = (p.get("title") or "")[:500]
                body = (p.get("selftext") or "")[:1000]
                text = f"{title} {body}"
                # Use post's actual creation timestamp for the date
                created = p.get("created_utc", 0)
                if created:
                    post_date = datetime.utcfromtimestamp(created).strftime("%Y-%m-%d")
                else:
                    post_date = datetime.utcnow().strftime("%Y-%m-%d")
                rows.append(
                    {
                        "date": post_date,
                        "subreddit": sub_name,
                        "subreddit_group": group,
                        "title": title,
                        "score": p.get("score", 0),
                        "sentiment_compound": round(_score_sentiment(text), 4),
                        "topic": _classify_topics(text),
                    }
                )
            print(f"Fetched {len(posts)} posts from r/{sub_name} ({group})")
        except Exception as exc:
            print(f"Warning: failed to fetch r/{sub_name}: {exc}")

    df = pd.DataFrame(rows)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    output_path = out_dir / f"{today}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} total posts to {output_path}")

    # Print topic summary
    if not df.empty:
        print(f"Topics: {df['topic'].value_counts().to_dict()}")
        print(f"Groups: {df['subreddit_group'].value_counts().to_dict()}")

    return str(output_path)


def main():
    """Entry point for standalone execution."""
    path = fetch()
    print(f"Output: {path}")


if __name__ == "__main__":
    main()
