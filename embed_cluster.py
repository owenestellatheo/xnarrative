"""Embedding (Voyage multilingual) + clustering (HDBSCAN) + cluster representatives."""
from typing import Optional
import numpy as np
import pandas as pd

from config import VOYAGE_API_KEY, MIN_CLUSTER_SIZE


def embed_tweets(df: pd.DataFrame, text_col: str = "translated_text") -> np.ndarray:
    """Get embeddings for all tweets. Uses translated_text if available, else text.

    Voyage's voyage-3 model handles multilingual well, but we embed the translated
    text where available because clustering benefits from a common semantic space.
    """
    import voyageai
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY not set. Add it to .env")

    vo = voyageai.Client(api_key=VOYAGE_API_KEY)

    # Fall back to original text where translation is missing
    texts = df[text_col].fillna(df["text"]).tolist() if text_col in df.columns else df["text"].tolist()

    embeddings: list[list[float]] = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate very long tweets (Voyage has token limits per item)
        batch = [t[:2000] for t in batch]
        result = vo.embed(batch, model="voyage-3", input_type="document")
        embeddings.extend(result.embeddings)
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(embeddings, dtype=np.float32)


def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int = MIN_CLUSTER_SIZE) -> np.ndarray:
    """HDBSCAN clustering. Returns cluster label per tweet (-1 = noise).

    Uses cosine distance via L2-normalized vectors + euclidean.
    """
    import hdbscan

    # Normalize for cosine-like behavior with euclidean metric
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms

    n = len(normed)
    effective_min = min(min_cluster_size, max(2, n // 10))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(normed)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/n:.1f}%)")
    return labels


def cluster_representatives(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_per_cluster: int = 3,
) -> dict[int, list[int]]:
    """For each cluster, return the indices of the N tweets closest to the centroid.

    Returns: {cluster_label: [row_idx, row_idx, ...]}
    """
    reps: dict[int, list[int]] = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        cluster_idx = np.where(mask)[0]
        cluster_emb = embeddings[mask]
        centroid = cluster_emb.mean(axis=0)
        # cosine similarity = dot product on normalized vectors
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        e_norm = cluster_emb / (np.linalg.norm(cluster_emb, axis=1, keepdims=True) + 1e-9)
        sims = e_norm @ c_norm
        top_in_cluster = np.argsort(-sims)[:n_per_cluster]
        reps[int(cluster_id)] = [int(cluster_idx[i]) for i in top_in_cluster]
    return reps


def dominant_cluster(labels: np.ndarray) -> Optional[int]:
    """Return the label of the largest non-noise cluster, or None if none exist."""
    from collections import Counter
    counts = Counter(int(l) for l in labels if l != -1)
    if not counts:
        return None
    return counts.most_common(1)[0][0]
