"""
Step4 (Target/Related "surprise"): novelty score (= dedup + surprise proxy in one
number) per embedding model, aggregated into a cross-model consensus count so a
single embedding space's geometric artifacts can't manufacture a false surprise
signal. Applies identically to the `target` and `related` text levels.
"""
import math

import numpy as np
import pandas as pd

from .embedding_store import EmbeddingRegistry, EmbeddingStore
from .levels import get_level_id_cols
from .text_pooling import pool_level


def _rolling_novelty_for_group(emb: np.ndarray, window: int) -> np.ndarray:
    """1 - cos_sim(today's emb, mean of the PRECEDING `window` days' emb), causal (no leakage)."""
    n = len(emb)
    novelty = np.zeros(n, dtype=np.float32)
    csum = np.cumsum(emb, axis=0)
    for t in range(n):
        lo, hi = max(0, t - window), t  # window = [lo, hi), excludes t itself
        if hi <= lo:
            continue
        roll_mean = (csum[hi - 1] - (csum[lo - 1] if lo > 0 else 0)) / (hi - lo)
        denom = np.linalg.norm(emb[t]) * np.linalg.norm(roll_mean)
        if denom < 1e-8:
            continue
        novelty[t] = 1.0 - float(np.dot(emb[t], roll_mean) / denom)
    return novelty


def _novelty_single_model(df: pd.DataFrame, id_cols, store: EmbeddingStore, window: int) -> np.ndarray:
    """`df` must be reset_index(drop=True) and sorted by (ticker, date)."""
    n = len(df)
    pooled = pool_level(df, id_cols, store)
    novelty = np.zeros(n, dtype=np.float32)
    for _, g in df.groupby("ticker", sort=False):
        pos = g.index.to_numpy()
        novelty[pos] = _rolling_novelty_for_group(pooled[pos], window)
    return novelty


def build_novelty_consensus_factors(df: pd.DataFrame, level_name: str, registry: EmbeddingRegistry,
                                     train_mask: np.ndarray, mom_5d: pd.Series, window: int = 20,
                                     quantile: float = 0.9, consensus_frac: float = 2.0 / 3.0) -> pd.DataFrame:
    """
    `df` must be reset_index(drop=True) and sorted by (ticker, date); `train_mask`/`mom_5d`
    aligned to df's positional index.

    Returns columns: {level}_novelty_consensus, {level}_already_priced, {level}_drift_flag.
    With only 1 embedding model available (today), novelty_consensus degenerates to a
    plain 0/1 novelty-above-threshold flag; as more `*_textemb.parquet` files land under
    data/, EmbeddingRegistry picks them up automatically and this becomes a real k-of-N vote.
    """
    id_cols = dict(get_level_id_cols(list(df.columns)))[level_name]
    n = len(df)
    n_models = len(registry)
    if n_models == 0:
        raise ValueError("No *_textemb.parquet files found by EmbeddingRegistry.")

    vote_count = np.zeros(n, dtype=np.float32)
    for model_name in registry.model_names:
        path = registry.path(model_name)
        store = EmbeddingStore.get(path)
        novelty = _novelty_single_model(df, id_cols, store, window)
        threshold = float(np.quantile(novelty[train_mask], quantile))
        vote_count += (novelty > threshold).astype(np.float32)
        print(f"[NoveltyConsensus] level={level_name} model={model_name} "
              f"train_threshold={threshold:.4f} hit_rate={(novelty > threshold).mean():.3f}")
        EmbeddingStore.release(path)  # free this model's matrix before loading the next one

    k = max(1, math.ceil(consensus_frac * n_models))
    is_surprise = (vote_count >= k).astype(np.float32)
    cum_ret_5d = mom_5d.to_numpy() if isinstance(mom_5d, pd.Series) else np.asarray(mom_5d)
    recent_move = np.clip(np.abs(cum_ret_5d), 0.0, 1.0)

    out = pd.DataFrame({
        f"{level_name}_novelty_consensus": vote_count / n_models,
        f"{level_name}_already_priced": is_surprise * recent_move,
        f"{level_name}_drift_flag": is_surprise * (1.0 - recent_move),
    }, index=df.index)
    return out
