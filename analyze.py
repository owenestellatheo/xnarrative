"""Main analysis pipeline.

Stages run sequentially. Each stage caches its output to cache/ so you can iterate
on later stages without re-running earlier ones. Use --force-stage <name> to invalidate
from that stage forward.

CLI:
    python analyze.py "tehran sentiment on US and new admin" --languages en,fa,ar --hours 24
"""
import argparse
import json
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import prompts
from config import (
    CACHE_DIR, STAGES, MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS,
    TRANSLATION_BATCH, PER_TWEET_ANALYSIS_BATCH,
)
from llm import call_claude_json


# ============================================================
# Cache helpers
# ============================================================
def _cache_path(stage: str) -> Path:
    return CACHE_DIR / f"{stage}.pkl"


def _save_cache(stage: str, obj: Any) -> None:
    with open(_cache_path(stage), "wb") as f:
        pickle.dump(obj, f)


def _load_cache(stage: str) -> Any | None:
    p = _cache_path(stage)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _invalidate_from(stage: str) -> None:
    """Delete cache for `stage` and all stages after it."""
    idx = STAGES.index(stage)
    for s in STAGES[idx:]:
        p = _cache_path(s)
        if p.exists():
            p.unlink()
            print(f"  invalidated cache: {s}")


# ============================================================
# Stage 1: Query localization
# ============================================================
def stage_localize(query: str, languages: list[str], hours: int) -> dict:
    cached = _load_cache("localize")
    if cached is not None:
        print("✓ localize (cached)")
        return cached

    print("→ localize: generating per-language X queries...")
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")
    result = call_claude_json(
        prompts.LOCALIZE_QUERY_SYSTEM,
        prompts.LOCALIZE_QUERY_USER.format(
            query=query, languages=languages, hours=hours, since_date=since,
        ),
        model=MODEL_HAIKU,
        max_tokens=2048,
    )
    _save_cache("localize", result)
    print(f"  → entities: {[e['name'] for e in result.get('target_entities', [])]}")
    return result


# ============================================================
# Stage 2: Scrape
# ============================================================
def stage_scrape(localized: dict, source: str | None) -> pd.DataFrame:
    cached = _load_cache("scrape")
    if cached is not None:
        print(f"✓ scrape (cached, {len(cached)} tweets)")
        return cached

    print("→ scrape: querying X...")
    if source:
        # Offline mode: read from a JSONL fixture
        from scrape import load_from_jsonl
        df = load_from_jsonl(source)
    else:
        from scrape import scrape_multilingual
        df = scrape_multilingual(localized["queries_by_language"])

    _save_cache("scrape", df)
    return df


# ============================================================
# Stage 3: Translate non-pivot-language tweets
# ============================================================
def stage_translate(df: pd.DataFrame, pivot_lang: str = "en") -> pd.DataFrame:
    cached = _load_cache("translate")
    if cached is not None:
        print(f"✓ translate (cached)")
        return cached

    print(f"→ translate: translating non-{pivot_lang} tweets to {pivot_lang}...")
    df = df.copy()
    df["translated_text"] = df["text"]  # default: identity
    df["translation_notes"] = ""

    to_translate = df[df["lang"] != pivot_lang]
    if to_translate.empty:
        print(f"  no tweets need translation")
        _save_cache("translate", df)
        return df

    translations: dict[str, dict] = {}
    rows = to_translate[["id", "text", "lang"]].to_dict(orient="records")
    for i in range(0, len(rows), TRANSLATION_BATCH):
        batch = rows[i : i + TRANSLATION_BATCH]
        tweets_json = json.dumps(batch, ensure_ascii=False)
        result = call_claude_json(
            prompts.TRANSLATE_SYSTEM,
            prompts.TRANSLATE_USER.format(tweets_json=tweets_json),
            model=MODEL_SONNET,
            max_tokens=8192,
        )
        for t in result.get("translations", []):
            translations[t["id"]] = t
        print(f"  translated {min(i + TRANSLATION_BATCH, len(rows))}/{len(rows)}")

    # Apply
    df["translated_text"] = df.apply(
        lambda r: translations.get(r["id"], {}).get("translated", r["text"]) if r["lang"] != pivot_lang else r["text"],
        axis=1,
    )
    df["translation_notes"] = df["id"].map(
        lambda i: translations.get(i, {}).get("translation_notes", "")
    ).fillna("")

    _save_cache("translate", df)
    return df


# ============================================================
# Stage 4 + 5: Embed + cluster
# ============================================================
def stage_embed(df: pd.DataFrame) -> np.ndarray:
    cached = _load_cache("embed")
    if cached is not None:
        print(f"✓ embed (cached, shape={cached.shape})")
        return cached
    print("→ embed: generating multilingual embeddings...")
    from embed_cluster import embed_tweets
    emb = embed_tweets(df)
    _save_cache("embed", emb)
    return emb


def stage_cluster(embeddings: np.ndarray) -> np.ndarray:
    cached = _load_cache("cluster")
    if cached is not None:
        print(f"✓ cluster (cached)")
        return cached
    print("→ cluster: HDBSCAN...")
    from embed_cluster import cluster_embeddings
    labels = cluster_embeddings(embeddings)
    _save_cache("cluster", labels)
    return labels


# ============================================================
# Stage 6: Per-tweet structured analysis
# ============================================================
def stage_per_tweet(df: pd.DataFrame, target_entities: list[dict]) -> pd.DataFrame:
    cached = _load_cache("per_tweet")
    if cached is not None:
        print(f"✓ per_tweet (cached)")
        return cached

    print("→ per_tweet: stance + sentiment + themes per tweet...")
    df = df.copy()
    analyses: dict[str, dict] = {}
    entity_names = [e["name"] for e in target_entities]

    rows = df[["id", "translated_text", "lang"]].to_dict(orient="records")
    for i in range(0, len(rows), PER_TWEET_ANALYSIS_BATCH):
        batch = rows[i : i + PER_TWEET_ANALYSIS_BATCH]
        tweets_json = json.dumps(
            [{"id": r["id"], "text": r["translated_text"], "lang": r["lang"]} for r in batch],
            ensure_ascii=False,
        )
        result = call_claude_json(
            prompts.PER_TWEET_SYSTEM,
            prompts.PER_TWEET_USER.format(
                target_entities=json.dumps(entity_names),
                tweets_json=tweets_json,
            ),
            model=MODEL_SONNET,
            max_tokens=8192,
        )
        for a in result.get("analyses", []):
            analyses[a["id"]] = a
        print(f"  analyzed {min(i + PER_TWEET_ANALYSIS_BATCH, len(rows))}/{len(rows)}")

    # Merge per-tweet analysis into df
    df["stance_by_entity"] = df["id"].map(lambda i: analyses.get(i, {}).get("stance_by_entity", {}))
    df["sentiment_by_entity"] = df["id"].map(lambda i: analyses.get(i, {}).get("sentiment_by_entity", {}))
    df["themes"] = df["id"].map(lambda i: analyses.get(i, {}).get("themes", []))
    df["rhetorical_mode"] = df["id"].map(lambda i: analyses.get(i, {}).get("rhetorical_mode", "other"))

    # Average sentiment across target entities (for outlier scoring)
    def _avg_sentiment(d: dict) -> float:
        vals = [v for v in d.values() if isinstance(v, (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    df["sentiment_score_avg"] = df["sentiment_by_entity"].apply(_avg_sentiment)

    _save_cache("per_tweet", df)
    return df


# ============================================================
# Stage 7: Outlier detection (deterministic, no LLM)
# ============================================================
def stage_outliers(df: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray) -> dict:
    cached = _load_cache("outliers")
    if cached is not None:
        print(f"✓ outliers (cached)")
        return cached

    print("→ outliers: engagement, sentiment, temporal...")
    from outliers import engagement_outliers, sentiment_outliers
    from embed_cluster import dominant_cluster

    dom = dominant_cluster(labels)
    eng = engagement_outliers(df)
    sent = sentiment_outliers(df, labels, embeddings, dom)

    result = {
        "engagement_outliers": eng.head(10).to_dict(orient="records"),
        "sentiment_outliers": sent.to_dict(orient="records"),
        "dominant_cluster": dom,
    }
    _save_cache("outliers", result)
    print(f"  engagement outliers: {len(eng)}, sentiment outliers: {len(sent)}")
    return result


# ============================================================
# Stage 8: Temporal analysis
# ============================================================
def stage_temporal(df: pd.DataFrame) -> dict:
    cached = _load_cache("temporal")
    if cached is not None:
        print(f"✓ temporal (cached)")
        return cached

    print("→ temporal: binning, spikes, cadence...")
    from outliers import temporal_bin, detect_spikes, detect_cadence

    binned = temporal_bin(df)
    spikes = detect_spikes(binned)
    cadence = detect_cadence(df)

    result = {
        "binned": binned,
        "spikes": spikes,
        "cadence": cadence,
    }
    _save_cache("temporal", result)
    print(f"  spikes detected: {len(spikes)}, suspicious authors: {len(cadence['suspicious_authors'])}")
    return result


# ============================================================
# Stage 9: Spike summarization
# ============================================================
def stage_spike_summary(df: pd.DataFrame, temporal: dict) -> list[dict]:
    cached = _load_cache("spike_summary")
    if cached is not None:
        print(f"✓ spike_summary (cached)")
        return cached

    spikes = temporal["spikes"]
    if spikes.empty:
        _save_cache("spike_summary", [])
        return []

    print(f"→ spike_summary: summarizing {len(spikes)} spike(s)...")
    bin_width = pd.Timedelta(minutes=30)
    summaries = []
    for _, spike in spikes.head(5).iterrows():  # cap at 5 to keep cost down
        start = spike["bucket_start"]
        end = start + bin_width
        in_window = df[(df["created_at"] >= start) & (df["created_at"] < end)]
        if in_window.empty:
            continue
        sample = in_window.head(15)[["translated_text", "author_id"]].to_dict(orient="records")
        sample_text = "\n".join(f"- @{t['author_id']}: {t['translated_text'][:300]}" for t in sample)
        try:
            summary = call_claude_json(
                prompts.SPIKE_SUMMARY_SYSTEM,
                prompts.SPIKE_SUMMARY_USER.format(
                    start=start.isoformat(),
                    end=end.isoformat(),
                    tweet_count=int(spike["count"]),
                    baseline=int(spike["baseline"]),
                    tweets_text=sample_text,
                ),
                model=MODEL_SONNET,
                max_tokens=512,
            ) if False else None  # spike summary is plain text, not JSON
        except Exception:
            summary = None

        # Use plain-text call instead
        from llm import call_claude
        summary_text = call_claude(
            prompts.SPIKE_SUMMARY_SYSTEM,
            prompts.SPIKE_SUMMARY_USER.format(
                start=start.isoformat(),
                end=end.isoformat(),
                tweet_count=int(spike["count"]),
                baseline=int(spike["baseline"]),
                tweets_text=sample_text,
            ),
            model=MODEL_SONNET,
            max_tokens=300,
        )
        summaries.append({
            "start": start.isoformat(),
            "tweet_count": int(spike["count"]),
            "baseline": int(spike["baseline"]),
            "zscore": float(spike["zscore"]),
            "summary": summary_text.strip(),
        })

    _save_cache("spike_summary", summaries)
    return summaries


# ============================================================
# Stage 10: Narrative synthesis (Opus)
# ============================================================
def stage_narrative(
    query: str,
    target_entities: list[dict],
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_reps: dict[int, list[int]],
    outliers: dict,
    temporal: dict,
    spike_summaries: list[dict],
) -> dict:
    cached = _load_cache("narrative")
    if cached is not None:
        print(f"✓ narrative (cached)")
        return cached

    print("→ narrative: synthesizing assessment (Opus)...")

    # Sentiment table per entity
    entity_names = [e["name"] for e in target_entities]
    sentiment_table_lines = []
    for ent in entity_names:
        stances = [d.get(ent) for d in df["stance_by_entity"] if isinstance(d, dict)]
        stances = [s for s in stances if s and s != "off-topic"]
        if not stances:
            sentiment_table_lines.append(f"  {ent}: no on-topic tweets")
            continue
        total = len(stances)
        support = stances.count("support")
        oppose = stances.count("oppose")
        mixed = stances.count("mixed")
        sentiment_table_lines.append(
            f"  {ent} (n={total}): support {100*support/total:.0f}%, oppose {100*oppose/total:.0f}%, mixed {100*mixed/total:.0f}%"
        )
    sentiment_table = "\n".join(sentiment_table_lines)

    # Top themes
    from collections import Counter
    all_themes = []
    for themes in df["themes"]:
        if isinstance(themes, list):
            all_themes.extend(themes)
    top_themes = Counter(all_themes).most_common(15)
    top_themes_str = "\n".join(f"  {t} ({c})" for t, c in top_themes)

    # Cluster summaries
    cluster_lines = []
    for cid, idxs in list(cluster_reps.items())[:8]:
        size = int((labels == cid).sum())
        tweets = df.iloc[idxs]
        for _, row in tweets.iterrows():
            cluster_lines.append(f"  [cluster {cid}, size={size}] {row['translated_text'][:200]}")
    cluster_summaries = "\n".join(cluster_lines)

    # Outlier summaries
    eng_o = "\n".join(
        f"  @{o['author_id']} ({o['author_followers']} followers, z={o.get('engagement_zscore', 0):.1f}): {o['translated_text'][:200]}"
        for o in outliers["engagement_outliers"][:5]
    ) or "  (none)"
    sent_o = "\n".join(
        f"  @{o['author_id']} (score={o.get('outlier_score', 0):.2f}): {o['translated_text'][:200]}"
        for o in outliers["sentiment_outliers"][:5]
    ) or "  (none)"
    spike_o = "\n".join(
        f"  {s['start']}: {s['summary']}" for s in spike_summaries
    ) or "  (none detected)"

    # Per-language theme distribution
    per_lang_lines = []
    for lang, group in df.groupby("lang"):
        themes = Counter()
        for t in group["themes"]:
            if isinstance(t, list):
                themes.update(t)
        top = themes.most_common(5)
        per_lang_lines.append(f"  {lang} (n={len(group)}): {', '.join(f'{t}({c})' for t, c in top)}")
    per_language_themes = "\n".join(per_lang_lines)

    languages = sorted(df["lang"].unique().tolist())

    result = call_claude_json(
        prompts.NARRATIVE_SYNTHESIS_SYSTEM,
        prompts.NARRATIVE_SYNTHESIS_USER.format(
            query=query,
            target_entities=json.dumps(entity_names),
            languages=languages,
            total_tweets=len(df),
            sentiment_table=sentiment_table,
            top_themes=top_themes_str,
            cluster_summaries=cluster_summaries,
            engagement_outliers=eng_o,
            sentiment_outliers=sent_o,
            temporal_spikes=spike_o,
            per_language_themes=per_language_themes,
        ),
        model=MODEL_OPUS,
        max_tokens=4096,
        temperature=0.4,
    )
    _save_cache("narrative", result)
    return result


# ============================================================
# Stage 11: Knowledge graph
# ============================================================
def stage_knowledge_graph(
    df: pd.DataFrame,
    target_entities: list[dict],
    cluster_reps: dict[int, list[int]],
    outliers: dict,
) -> dict:
    cached = _load_cache("knowledge_graph")
    if cached is not None:
        print(f"✓ knowledge_graph (cached)")
        return cached

    print("→ knowledge_graph: extracting entities and relations...")

    # Curate input: cluster reps + outlier tweets, dedup
    curated_ids: set[str] = set()
    for idxs in cluster_reps.values():
        for i in idxs:
            curated_ids.add(df.iloc[i]["id"])
    for o in outliers["engagement_outliers"][:5] + outliers["sentiment_outliers"][:5]:
        curated_ids.add(o["id"])

    curated = df[df["id"].isin(curated_ids)]
    if len(curated) > 80:
        curated = curated.head(80)

    tweets_json = json.dumps(
        [
            {"id": r["id"], "text": r["translated_text"][:400]}
            for _, r in curated.iterrows()
        ],
        ensure_ascii=False,
    )

    entity_names = [e["name"] for e in target_entities]
    result = call_claude_json(
        prompts.KG_EXTRACTION_SYSTEM,
        prompts.KG_EXTRACTION_USER.format(
            target_entities=json.dumps(entity_names),
            tweets_json=tweets_json,
        ),
        model=MODEL_SONNET,
        max_tokens=4096,
    )
    print(f"  entities: {len(result.get('entities', []))}, relations: {len(result.get('relations', []))}")
    _save_cache("knowledge_graph", result)
    return result


# ============================================================
# Driver
# ============================================================
def run_pipeline(query: str, languages: list[str], hours: int, source: str | None = None) -> dict:
    localized = stage_localize(query, languages, hours)
    df = stage_scrape(localized, source)
    if df.empty:
        print("No tweets — exiting.")
        sys.exit(1)

    df = stage_translate(df)
    embeddings = stage_embed(df)
    labels = stage_cluster(embeddings)

    df = stage_per_tweet(df, localized["target_entities"])

    outliers_out = stage_outliers(df, embeddings, labels)
    temporal_out = stage_temporal(df)
    spike_summaries = stage_spike_summary(df, temporal_out)

    from embed_cluster import cluster_representatives
    cluster_reps = cluster_representatives(df, embeddings, labels, n_per_cluster=3)

    narrative = stage_narrative(
        query, localized["target_entities"], df, labels, cluster_reps,
        outliers_out, temporal_out, spike_summaries,
    )

    kg = stage_knowledge_graph(df, localized["target_entities"], cluster_reps, outliers_out)

    bundle = {
        "query": query,
        "languages": languages,
        "hours": hours,
        "localized": localized,
        "df": df,
        "embeddings": embeddings,
        "labels": labels,
        "cluster_reps": cluster_reps,
        "outliers": outliers_out,
        "temporal": temporal_out,
        "spike_summaries": spike_summaries,
        "narrative": narrative,
        "knowledge_graph": kg,
    }
    _save_cache("final_bundle", bundle)
    return bundle


def main():
    ap = argparse.ArgumentParser(description="X narrative analysis pipeline")
    ap.add_argument("query", help="Natural language analyst query")
    ap.add_argument("--languages", default="en", help="Comma-separated language codes (e.g. en,fa,ar)")
    ap.add_argument("--hours", type=int, default=24, help="Time window in hours")
    ap.add_argument("--source", default=None, help="Optional JSONL file of pre-scraped tweets (offline mode)")
    ap.add_argument("--force-stage", default=None, help=f"Invalidate cache from this stage forward. One of: {STAGES}")
    args = ap.parse_args()

    if args.force_stage:
        if args.force_stage not in STAGES:
            print(f"Unknown stage: {args.force_stage}. Options: {STAGES}")
            sys.exit(1)
        _invalidate_from(args.force_stage)

    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    run_pipeline(args.query, languages, args.hours, args.source)
    print("\n✓ Pipeline complete. Run `streamlit run render.py` to view results.")


if __name__ == "__main__":
    main()
