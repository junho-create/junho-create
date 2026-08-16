"""
Orchestrates price + macro/sector/target/related factor construction into the
Alpha360-style flattened-window panel that DoubleAdapt consumes. Each Step2-4
block is optional (`use_macro`/`use_sector`/`use_target_novelty`/
`use_related_novelty`) so they can be ablated independently against the
Step1-only baseline (`forecasting_task/run_DoubleAdapt.py`'s original
behaviour == all four flags off).
"""
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .embedding_store import EmbeddingRegistry
from .levels import PRICE_FACTOR_NAMES
from .macro_factors import build_macro_factors
from .price_factors import build_price_factors
from .sector_factors import build_sector_factors
from .target_factors import build_novelty_consensus_factors
from .text_pooling import build_text_pca_factors
from .text_signal import build_text_signal_factors

# New scalar features that need train-period standardization before windowing
# (unlike the text_pc* columns, which the original pipeline left as raw PCA
# output, and unlike the already-bounded [0,1] novelty/align/flag features).
GLOBAL_STANDARDIZE_COLS = [
    "macro_streak", "macro_regime_shift", "macro_direction",
    "sector_peer_ret", "sector_rel_strength", "sector_cluster_size",
]


@dataclass
class PanelConfig:
    seq_len: int = 60
    horizon: int = 20
    n_text_components: int = 6
    train_start: str = "2019-01-01"
    train_end: str = "2021-12-31"
    primary_emb_model: Optional[str] = None  # None => first model EmbeddingRegistry finds
    use_macro: bool = False
    use_sector: bool = False
    use_target_novelty: bool = False
    use_related_novelty: bool = False
    use_sector_novelty: bool = False
    novelty_window: int = 20
    novelty_quantile: float = 0.9
    novelty_consensus_frac: float = 2.0 / 3.0
    sector_agreement_threshold: float = 0.8
    macro_similarity_threshold: float = 0.9
    macro_drift_window: int = 20
    use_text_signal: bool = False
    text_signal_hidden: int = 128
    text_signal_layers: int = 2
    text_signal_heads: int = 4
    text_signal_epochs: int = 15


def _standardize(raw: pd.DataFrame, cols: List[str], train_mask: np.ndarray, scope: str) -> None:
    """In-place train-period standardization. scope='ticker' matches each ticker's own
    mean/std (price factors); scope='global' uses one pooled mean/std (cross-sectional
    features like macro_* that are identical across tickers, or peer/relative-strength
    features that are meant to be compared across tickers)."""
    if scope == "ticker":
        stats = raw.loc[train_mask].groupby("ticker")[cols].agg(["mean", "std"])
        for col in cols:
            mean_map = raw["ticker"].map(stats[(col, "mean")])
            std_map = raw["ticker"].map(stats[(col, "std")]).replace(0, 1.0)
            raw[col] = (raw[col] - mean_map) / std_map
    else:
        mean = raw.loc[train_mask, cols].mean()
        std = raw.loc[train_mask, cols].std().replace(0, 1.0)
        raw[cols] = (raw[cols] - mean) / std


def _cache_key(df: pd.DataFrame, cfg: PanelConfig, registry: "EmbeddingRegistry") -> str:
    """Cache is invalidated by anything that could change build_panel's output: the
    exact ticker/date universe in `df`, every PanelConfig field, and each embedding
    model's mtime (so a re-downloaded/replaced parquet busts the cache)."""
    key_obj = {
        "tickers": sorted(df["ticker"].unique().tolist()),
        "n_rows": len(df),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "cfg": asdict(cfg),
        "models": registry.model_names,
        "emb_mtimes": {m: os.path.getmtime(registry.path(m)) for m in registry.model_names},
    }
    blob = json.dumps(key_obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def build_panel(df: pd.DataFrame, data_dir: str, cfg: PanelConfig, cache_dir: Optional[str] = None):
    """
    `df` must already be filtered to the desired tickers and sorted by (ticker, date)
    with a fresh RangeIndex (reset_index(drop=True)).

    `cache_dir`, if given, memoizes the full return value on disk keyed by every input
    that affects it (ticker/date universe, PanelConfig, embedding-model mtimes) --
    panel construction is backbone-independent, so repeated runs that only vary
    `--backbone` (e.g. comparing GRU/PatchTST/.../iTransformer on the same feature
    set) can skip the ~40min rebuild entirely.

    Returns: panel_df, raw_close, label_mean, label_std, factor_num, seq_len, factor_cols
    (same contract as the original run_DoubleAdapt.build_panel).
    """
    registry = EmbeddingRegistry(data_dir)
    if len(registry) == 0:
        raise ValueError(f"No *_textemb.parquet files found under {data_dir}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(df, cfg, registry)
        cache_path = os.path.join(cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            print(f"[Panel] cache hit: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        print(f"[Panel] cache miss ({key}), building from scratch")

    train_start = pd.Timestamp(cfg.train_start)
    train_end = pd.Timestamp(cfg.train_end)
    train_mask = ((df["date"] >= train_start) & (df["date"] <= train_end)).values

    primary_model = cfg.primary_emb_model or registry.default_primary
    primary_path = registry.path(primary_model)
    print(f"[Panel] embedding models available: {registry.model_names} (primary={primary_model})")

    # fillna(0): first few rows of each ticker lack history for shift-based
    # factors (e.g. mom_5d needs 5 prior days); neutral rather than NaN so it
    # doesn't poison whole batches/gradients later.
    price_factors = build_price_factors(df).fillna(0.0)
    text_factors = build_text_pca_factors(df, primary_path, cfg.n_text_components, train_mask)

    extra_blocks = []
    if cfg.use_macro:
        extra_blocks.append(build_macro_factors(
            df, primary_path, cfg.horizon, train_mask, price_factors["mom_5d"],
            similarity_threshold=cfg.macro_similarity_threshold, drift_window=cfg.macro_drift_window,
        ))
    if cfg.use_sector:
        extra_blocks.append(build_sector_factors(
            df, price_factors, primary_path, train_mask, n_components=cfg.n_text_components,
            agreement_threshold=cfg.sector_agreement_threshold,
        ))
    if cfg.use_target_novelty:
        extra_blocks.append(build_novelty_consensus_factors(
            df, "target", registry, train_mask, price_factors["mom_5d"],
            window=cfg.novelty_window, quantile=cfg.novelty_quantile,
            consensus_frac=cfg.novelty_consensus_frac,
        ))
    if cfg.use_related_novelty:
        extra_blocks.append(build_novelty_consensus_factors(
            df, "related", registry, train_mask, price_factors["mom_5d"],
            window=cfg.novelty_window, quantile=cfg.novelty_quantile,
            consensus_frac=cfg.novelty_consensus_frac,
        ))
    if cfg.use_sector_novelty:
        # Unlike target/related, sector_category text_ids are shared by every ticker in the
        # same cluster (~9-10 tickers), so the novelty signal is far less per-sample-differentiated
        # -- worth testing, but expected to be a weaker signal than target/related.
        extra_blocks.append(build_novelty_consensus_factors(
            df, "sector", registry, train_mask, price_factors["mom_5d"],
            window=cfg.novelty_window, quantile=cfg.novelty_quantile,
            consensus_frac=cfg.novelty_consensus_frac,
        ))
    if cfg.use_text_signal:
        extra_blocks.append(build_text_signal_factors(
            df, primary_path, cfg.horizon, train_mask,
            hidden=cfg.text_signal_hidden, num_layers=cfg.text_signal_layers,
            nhead=cfg.text_signal_heads, epochs=cfg.text_signal_epochs,
        ))

    factor_cols = list(PRICE_FACTOR_NAMES) + list(text_factors.columns)
    concat_parts = [df[["date", "ticker", "close"]], price_factors, text_factors]
    for block in extra_blocks:
        factor_cols += list(block.columns)
        concat_parts.append(block)
    raw = pd.concat(concat_parts, axis=1)

    _standardize(raw, PRICE_FACTOR_NAMES, train_mask, scope="ticker")
    global_cols = [c for c in GLOBAL_STANDARDIZE_COLS if c in raw.columns]
    if global_cols:
        _standardize(raw, global_cols, train_mask, scope="global")

    factor_num = len(factor_cols)
    seq_len = cfg.seq_len
    horizon = cfg.horizon

    raw["fwd_logret"] = raw.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s.shift(-horizon) / s)
    )
    label_mean = float(raw.loc[train_mask, "fwd_logret"].mean())
    label_std = float(raw.loc[train_mask, "fwd_logret"].std())
    raw["label_std"] = (raw["fwd_logret"] - label_mean) / label_std

    feature_rows, label_rows, close_rows, idx_tuples = [], [], [], []
    for ticker, g in raw.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        fmat = g[factor_cols].to_numpy(dtype=np.float32)
        if len(g) < seq_len:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(fmat, window_shape=seq_len, axis=0)
        flat = windows.reshape(windows.shape[0], -1)  # factor-major flatten, Alpha360 convention
        cutoff_idx = np.arange(seq_len - 1, len(g))
        feature_rows.append(flat)
        label_rows.append(g["label_std"].to_numpy(dtype=np.float32)[cutoff_idx])
        close_rows.append(g["close"].to_numpy(dtype=np.float32)[cutoff_idx])
        dates = g["date"].to_numpy()[cutoff_idx]
        idx_tuples.extend((d, ticker) for d in dates)

    feature_mat = np.concatenate(feature_rows, axis=0)
    label_vec = np.concatenate(label_rows, axis=0)
    close_vec = np.concatenate(close_rows, axis=0)

    index = pd.MultiIndex.from_tuples(idx_tuples, names=["datetime", "instrument"])
    feature_df = pd.DataFrame(feature_mat, index=index, columns=range(feature_mat.shape[1]))
    label_df = pd.DataFrame({0: label_vec}, index=index)
    panel_df = pd.concat({"feature": feature_df, "label": label_df}, axis=1)
    panel_df = panel_df.sort_index(level=["datetime", "instrument"])

    raw_close = pd.Series(close_vec, index=index, name="close").sort_index(level=["datetime", "instrument"])

    print(f"[Panel] factor_cols ({factor_num}): {factor_cols}")
    result = (panel_df, raw_close, label_mean, label_std, factor_num, seq_len, factor_cols)
    if cache_dir:
        t0 = time.time()
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Panel] wrote cache {cache_path} in {time.time() - t0:.1f}s")
    return result
