"""
Step2 (Macro): dedup-as-feature + embedding-drift regime detection + price-macro
alignment gating, built from the market-wide `macro_category*` text-id columns.

Macro news is broadcast: every ticker shares the same `macro_category*` ids on
a given date (verified empirically -- one text_id per date, not per ticker).
So all of this is computed once per date, then broadcast back onto the full
(date, ticker) panel; only the final alignment-gating feature is ticker-specific.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .embedding_store import EmbeddingStore
from .levels import get_level_id_cols
from .text_pooling import pool_level


def _market_fwd_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Mean across tickers of each ticker's horizon-ahead log return of close, indexed by date."""
    close_wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    fwd = np.log(close_wide.shift(-horizon) / close_wide)
    return fwd.mean(axis=1)


def build_macro_factors(df: pd.DataFrame, emb_path: str, horizon: int, train_mask: np.ndarray,
                         mom_5d: pd.Series, similarity_threshold: float = 0.9,
                         drift_window: int = 20, ridge_alpha: float = 1.0) -> pd.DataFrame:
    """
    Returns a DataFrame aligned to df.index with columns:
      macro_streak, macro_regime_shift, macro_direction, macro_price_align
    `mom_5d` must be aligned to df.index (i.e. price_factors["mom_5d"]).
    """
    store = EmbeddingStore.get(emb_path)
    macro_cols = dict(get_level_id_cols(list(df.columns)))["macro"]

    day_df = df[["date"] + macro_cols].drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    n = len(day_df)
    pooled = pool_level(day_df, macro_cols, store)  # [n_days, dim]

    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    unit = pooled / np.clip(norm, 1e-8, None)

    # 1) cross-day repetition streak (dedup-as-feature, stand-in for raw article coverage
    #    intensity which isn't reconstructable from the already-LLM-summarized category ids).
    sim_prev = np.zeros(n)
    sim_prev[1:] = (unit[1:] * unit[:-1]).sum(axis=1)
    streak = np.zeros(n)
    for t in range(1, n):
        streak[t] = streak[t - 1] + 1 if sim_prev[t] > similarity_threshold else 0

    # 2) rolling-mean embedding drift = regime-change indicator.
    roll_mean = np.zeros_like(pooled)
    csum = np.cumsum(pooled, axis=0)
    for t in range(n):
        lo = max(0, t - drift_window + 1)
        span = t - lo + 1
        roll_mean[t] = (csum[t] - (csum[lo - 1] if lo > 0 else 0)) / span
    roll_unit = roll_mean / np.clip(np.linalg.norm(roll_mean, axis=1, keepdims=True), 1e-8, None)
    regime_shift = np.zeros(n)
    regime_shift[drift_window:] = 1.0 - (roll_unit[drift_window:] * roll_unit[:-drift_window]).sum(axis=1)

    day_df["macro_streak"] = streak
    day_df["macro_regime_shift"] = regime_shift

    # 3) macro-direction probe: Ridge(macro_emb -> horizon-ahead market-avg return),
    #    fit on the training period only (same train/apply-everywhere convention as the
    #    text PCA in text_pooling.build_text_pca_factors), then applied to all dates.
    #    PCA down to a handful of dims first: raw dim (4096) >> n_train_dates (~750),
    #    so fitting Ridge directly on the raw embedding is underdetermined enough that
    #    it just memorizes noise (R^2 ~= 1.0 on train, meaningless out of sample).
    train_dates = set(df.loc[train_mask, "date"].unique())
    day_train_mask = day_df["date"].isin(train_dates).values
    mkt_fwd_ret = _market_fwd_return(df, horizon).reindex(day_df["date"]).to_numpy()
    fit_mask = day_train_mask & ~np.isnan(mkt_fwd_ret)

    probe_pca = PCA(n_components=min(20, fit_mask.sum() - 1), random_state=0)
    probe_pca.fit(pooled[fit_mask])
    pooled_reduced = probe_pca.transform(pooled)

    ridge = Ridge(alpha=ridge_alpha, random_state=0)
    ridge.fit(pooled_reduced[fit_mask], mkt_fwd_ret[fit_mask])
    day_df["macro_direction"] = ridge.predict(pooled_reduced)
    print(f"[MacroFactors] direction probe train R^2="
          f"{ridge.score(pooled_reduced[fit_mask], mkt_fwd_ret[fit_mask]):.4f}")

    out = df[["date"]].merge(
        day_df[["date", "macro_streak", "macro_regime_shift", "macro_direction"]], on="date", how="left"
    )
    out.index = df.index

    # 4) ticker-specific gating: does the macro-implied direction agree with the
    #    ticker's own recent momentum?
    out["macro_price_align"] = (np.sign(out["macro_direction"]) == np.sign(mom_5d.reindex(df.index))).astype(float)

    return out[["macro_streak", "macro_regime_shift", "macro_direction", "macro_price_align"]]
