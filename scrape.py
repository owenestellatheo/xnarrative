"""X/Twitter scraper using twitter-api-client.

NOTE: This library's API changes occasionally as X updates. If something here breaks,
the likely fix is in `_search_one_query`. Verify by reading the current README at
https://github.com/trevorhobenshield/twitter-api-client

USE A BURNER ACCOUNT. X aggressively rate-limits and sometimes suspends accounts
that hit unofficial endpoints.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    X_USERNAME, X_EMAIL, X_PASSWORD, SESSION_FILE,
    MAX_TWEETS_PER_LANG, SCRAPE_SLEEP_SECONDS,
)


def _get_search_client():
    """Lazy import so the rest of the pipeline works even if twitter-api-client breaks."""
    from twitter.search import Search
    # twitter-api-client persists session to ~/.twitter by default; we just pass creds
    # and let it manage cookies. If you want to share session, point its cookie cache
    # at SESSION_FILE (advanced).
    if not (X_USERNAME and X_EMAIL and X_PASSWORD):
        raise RuntimeError(
            "X credentials not set. Add X_USERNAME / X_EMAIL / X_PASSWORD to .env. "
            "Use a BURNER account, not your real one."
        )
    return Search(X_EMAIL, X_USERNAME, X_PASSWORD, save=False, debug=0)


def _flatten_tweet(raw: dict, query_lang: str) -> Optional[dict]:
    """Extract the fields we care about from twitter-api-client's nested response.

    Schemas vary; this defends against missing fields. If a tweet has no text or ID, skip.
    """
    try:
        # twitter-api-client returns the raw GraphQL tree; we have to dig.
        # The structure differs by endpoint version. We try a few paths.
        legacy = raw.get("legacy") or raw.get("tweet", {}).get("legacy") or raw
        user_legacy = (
            raw.get("core", {}).get("user_results", {}).get("result", {}).get("legacy")
            or raw.get("user", {}).get("legacy")
            or {}
        )

        tweet_id = legacy.get("id_str") or raw.get("rest_id") or raw.get("id_str")
        text = legacy.get("full_text") or legacy.get("text")
        if not tweet_id or not text:
            return None

        created_at_raw = legacy.get("created_at")
        try:
            created_at = datetime.strptime(
                created_at_raw, "%a %b %d %H:%M:%S %z %Y"
            ).astimezone(timezone.utc)
        except (TypeError, ValueError):
            created_at = datetime.now(timezone.utc)

        return {
            "id": str(tweet_id),
            "text": text,
            "lang": legacy.get("lang") or query_lang,
            "created_at": created_at,
            "like_count": legacy.get("favorite_count", 0),
            "retweet_count": legacy.get("retweet_count", 0),
            "reply_count": legacy.get("reply_count", 0),
            "quote_count": legacy.get("quote_count", 0),
            "author_id": user_legacy.get("screen_name", "unknown"),
            "author_name": user_legacy.get("name", ""),
            "author_followers": user_legacy.get("followers_count", 0),
            "author_following": user_legacy.get("friends_count", 0),
            "author_verified": user_legacy.get("verified", False)
            or raw.get("core", {}).get("user_results", {}).get("result", {}).get("is_blue_verified", False),
            "query_lang": query_lang,
            "raw_url": f"https://x.com/{user_legacy.get('screen_name', 'i')}/status/{tweet_id}",
        }
    except Exception as e:
        print(f"  ⚠️  skipped malformed tweet: {e}")
        return None


def _search_one_query(search, query: str, lang: str, limit: int) -> list[dict]:
    """Run one search query and return flattened tweet dicts.

    twitter-api-client's Search.run() returns a list-of-lists structure (one per query).
    """
    print(f"  → query: {query!r} (lang={lang}, limit={limit})")
    try:
        results = search.run(
            limit=limit,
            retries=2,
            queries=[{"category": "Latest", "query": query}],
        )
    except Exception as e:
        print(f"  ⚠️  search failed: {e}")
        return []

    # results is typically [[tweet_dict, ...]] for one query
    flat_results = []
    for query_results in (results or []):
        for raw in (query_results or []):
            tweet = _flatten_tweet(raw, lang)
            if tweet:
                flat_results.append(tweet)
    print(f"  ✓ got {len(flat_results)} tweets")
    return flat_results


def scrape_multilingual(
    queries_by_language: dict[str, list[dict]],
    per_lang_limit: int = MAX_TWEETS_PER_LANG,
) -> pd.DataFrame:
    """Run all queries from query localization, dedup by tweet ID, return one DataFrame.

    queries_by_language: {"en": [{"query": "...", "rationale": "..."}], "fa": [...], ...}
    """
    search = _get_search_client()
    all_tweets: dict[str, dict] = {}  # id -> tweet, for dedup

    for lang, queries in queries_by_language.items():
        per_query_limit = max(50, per_lang_limit // max(len(queries), 1))
        for q in queries:
            tweets = _search_one_query(search, q["query"], lang, per_query_limit)
            for t in tweets:
                if t["id"] not in all_tweets:
                    all_tweets[t["id"]] = t
            time.sleep(SCRAPE_SLEEP_SECONDS)

    df = pd.DataFrame(list(all_tweets.values()))
    if df.empty:
        print("⚠️  No tweets scraped. Check credentials and that search queries return results on x.com directly.")
        return df

    df = df.sort_values("created_at").reset_index(drop=True)
    print(f"\n✓ Total deduped: {len(df)} tweets across {df['lang'].nunique()} languages")
    return df


# --- Optional: read scraped tweets from a JSONL file for offline testing ---
def load_from_jsonl(path: str) -> pd.DataFrame:
    """For offline testing — load tweets from a JSONL file.

    Schema: one JSON object per line with the same fields as _flatten_tweet returns.
    Useful when X scraping is broken and you have cached data, or to test on a fixture.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df
