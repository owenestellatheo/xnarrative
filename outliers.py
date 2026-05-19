"""Outlier detection: engagement, sentiment, temporal.

Pure numpy/pandas/scipy. No LLM calls here — these are deterministic.
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import ENGAGEMENT_OUTLIER_ZSCORE, SPIKE_ZSCORE_THRESHOLD, TEMPORAL_BIN_MINUTES


# ============================================================
# Engagement outliers
# ============================================================
def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """engagement / max(followers, 100). Retweets and replies weighted more than likes."""
    weighted = (
        df["like_count"].fillna(0)
        + 2 * df["retweet_count"].fillna(0)
        + 3 * df["reply_count"].fillna(0)
        + 2 * df["quote_count"].fillna(0)
    )
    floor = df["author_followers"].fillna(0).clip(lower=100)
    return weighted / floor


def engagement_outliers(df: pd.DataFrame, threshold: float = ENGAGEMENT_OUTLIER_ZSCORE) -> pd.DataFrame:
    """Tweets with disproportionate engagement relative to follower count.

    Returns a DataFrame slice with added 'engagement_score' and 'engagement_zscore' columns,
    sorted descending by score.
    """
    out = df.copy()
    out["engagement_score"] = compute_engagement_score(out)

    if len(out) < 5 or out["engagement_score"].std() == 0:
        out["engagement_zscore"] = 0.0
        return out.iloc[0:0]

    # Log-transform because engagement scores are heavy-tailed
    log_score = np.log1p(out["engagement_score"])
    out["engagement_zscore"] = stats.zscore(log_score)
    flagged = out[out["engagement_zscore"] > threshold].copy()
    flagged = flagged.sort_values("engagement_zscore", ascending=False)
    return flagged


# ============================================================
# Sentiment outliers
# ============================================================
def sentiment_outliers(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    embeddings: np.ndarray,
    dominant_label: int | None,
    n_outliers: int = 10,
) -> pd.DataFrame:
    """Tweets that cut against the dominant narrative.

    Score = semantic_distance_from_dominant_centroid × |sentiment_delta_from_corpus_mean|

    Requires `sentiment_score_avg` column (mean across target entities).
    """
    if dominant_label is None or "sentiment_score_avg" not in df.columns:
        return df.iloc[0:0]

    dominant_mask = cluster_labels == dominant_label
    if dominant_mask.sum() == 0:
        return df.iloc[0:0]

    dominant_emb = embeddings[dominant_mask]
    centroid = dominant_emb.mean(axis=0)
    centroid_n = centroid / (np.linalg.norm(centroid) + 1e-9)
    emb_n = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    sims = emb_n @ centroid_n
    distance = 1 - sims  # cosine distance

    sentiment_mean = df["sentiment_score_avg"].mean()
    sentiment_delta = (df["sentiment_score_avg"] - sentiment_mean).abs()

    out = df.copy()
    out["distance_from_dominant"] = distance
    out["sentiment_delta"] = sentiment_delta.values
    out["outlier_score"] = distance * sentiment_delta.values

    # Restrict to tweets NOT in the dominant cluster (they're definitionally less aligned)
    candidates = out[~dominant_mask].copy()
    return candidates.sort_values("outlier_score", ascending=False).head(n_outliers)


# ============================================================
# Temporal: binning, spikes, cadence
# ============================================================
def temporal_bin(df: pd.DataFrame, bin_minutes: int = TEMPORAL_BIN_MINUTES) -> pd.DataFrame:
    """Return a DataFrame with columns [bucket_start, count] over the corpus time range."""
    if df.empty:
        return pd.DataFrame(columns=["bucket_start", "count"])
    freq = f"{bin_minutes}min"
    binned = (
        df.set_index("created_at")
        .resample(freq)
        .size()
        .reset_index(name="count")
        .rename(columns={"created_at": "bucket_start"})
    )
    return binned


def detect_spikes(binned: pd.DataFrame, zscore_threshold: float = SPIKE_ZSCORE_THRESHOLD) -> pd.DataFrame:
    """Spike = bucket where count is > threshold std-devs above rolling median.

    Returns DataFrame of spike buckets with columns: bucket_start, count, baseline, zscore.
    """
    if len(binned) < 5:
        return pd.DataFrame(columns=["bucket_start", "count", "baseline", "zscore"])

    # Rolling baseline (median over a window, robust to spikes)
    window = max(5, len(binned) // 6)
    baseline = binned["count"].rolling(window=window, min_periods=1, center=True).median()
    mad = (binned["count"] - baseline).abs().rolling(window=window, min_periods=1, center=True).median()
    # Modified z-score using MAD (robust to outliers)
    scale = mad.replace(0, np.nan).fillna(binned["count"].std() or 1.0)
    zscore = 0.6745 * (binned["count"] - baseline) / scale

    spikes = binned.copy()
    spikes["baseline"] = baseline
    spikes["zscore"] = zscore.fillna(0)
    flagged = spikes[spikes["zscore"] > zscore_threshold].copy()
    return flagged.sort_values("zscore", ascending=False)


def detect_cadence(df: pd.DataFrame, per_author: bool = True) -> dict:
    """Look for suspicious periodic posting patterns.

    Returns {"corpus_autocorr_peak_minutes": float | None,
             "suspicious_authors": [{author_id, interval_minutes, n_posts}, ...]}
    """
    result = {"corpus_autocorr_peak_minutes": None, "suspicious_authors": []}
    if df.empty:
        return result

    # Corpus-level: autocorrelation of binned volume
    binned = temporal_bin(df, bin_minutes=15)
    if len(binned) > 20:
        counts = binned["count"].values.astype(float)
        counts = counts - counts.mean()
        # autocorrelation
        if counts.std() > 0:
            acf = np.correlate(counts, counts, mode="full")
            acf = acf[acf.size // 2 :]
            acf = acf / acf[0]
            # Look for peak in lags 2..len/2 (skip lag-0, lag-1)
            if len(acf) > 4:
                lags = np.arange(len(acf))
                peak_lag = int(np.argmax(acf[2 : len(acf) // 2]) + 2)
                if acf[peak_lag] > 0.4:
                    result["corpus_autocorr_peak_minutes"] = peak_lag * 15

    # Per-author: detect users who post at suspiciously regular intervals
    if per_author and "author_id" in df.columns:
        for author, group in df.groupby("author_id"):
            if len(group) < 5:
                continue
            times = group["created_at"].sort_values()
            intervals_min = times.diff().dt.total_seconds().dropna() / 60
            if len(intervals_min) < 4:
                continue
            cv = intervals_min.std() / (intervals_min.mean() + 1e-9)  # coefficient of variation
            # Low CV = very regular = suspicious
            if cv < 0.15 and intervals_min.mean() < 120:
                result["suspicious_authors"].append({
                    "author_id": author,
                    "interval_minutes": float(intervals_min.mean()),
                    "n_posts": len(group),
                    "cv": float(cv),
                })

    return result
