"""
Step3 (Sector): sector membership is NOT reconstructed via statistical
consensus clustering. Empirically, tickers that share the same
`sector_category1` text_id on a given date form a partition that's stable
across the whole 2019-2023 span (pairwise co-membership agreement ~100%
between dates 5 years apart) -- `make_dataset/tag_company_news.py` already
GICS-tags company news upstream, and that grouping survives into
`sector_category*` as "which tickers got the same day's sector summary."
So we recover that fixed partition directly (thresholded co-membership graph
-> connected components) instead of clustering price correlation / embedding
similarity from scratch.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA

from .embedding_store import EmbeddingStore
from .levels import get_level_id_cols
from .text_pooling import pool_level


def build_sector_clusters(df: pd.DataFrame, train_mask: np.ndarray,
                           agreement_threshold: float = 0.8) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Two tickers are in the same cluster if they share `sector_category1` on
    at least `agreement_threshold` of the training-period dates where BOTH
    have a non-null tag. Returns {ticker: cluster_id}, cluster_sizes.
    """
    train_df = df.loc[train_mask, ["date", "ticker", "sector_category1"]].dropna()
    tickers = sorted(df["ticker"].unique())
    tidx = {t: i for i, t in enumerate(tickers)}
    m = len(tickers)
    co = np.zeros((m, m), dtype=np.int32)
    cnt = np.zeros(m, dtype=np.int32)

    for _, day_g in train_df.groupby("date", sort=False):
        day_idx = day_g["ticker"].map(tidx).to_numpy()
        cnt[day_idx] += 1
        for _, same_tag_g in day_g.groupby("sector_category1", sort=False):
            if len(same_tag_g) < 2:
                continue
            ii = same_tag_g["ticker"].map(tidx).to_numpy()
            co[np.ix_(ii, ii)] += 1

    denom = np.minimum.outer(cnt, cnt)
    denom[denom == 0] = 1
    agree = co / denom
    np.fill_diagonal(agree, 1.0)
    adj = (agree >= agreement_threshold) & (np.minimum.outer(cnt, cnt) > 0)

    n_components, labels = connected_components(csr_matrix(adj), directed=False)
    ticker_to_cluster = {t: int(labels[tidx[t]]) for t in tickers}
    cluster_sizes = np.bincount(labels, minlength=n_components)
    n_singletons = int((cluster_sizes == 1).sum())
    print(f"[SectorClusters] {n_components} clusters from {m} tickers "
          f"(sizes={sorted(cluster_sizes.tolist(), reverse=True)}, singletons={n_singletons})")
    return ticker_to_cluster, cluster_sizes


def build_sector_factors(df: pd.DataFrame, price_factors: pd.DataFrame, emb_path: str,
                          train_mask: np.ndarray, n_components: int = 6,
                          agreement_threshold: float = 0.8) -> pd.DataFrame:
    """
    `df` and `price_factors` share df.index (price_factors must have `log_ret_close`).
    Returns columns: sector_peer_ret, sector_rel_strength, sector_cluster_size,
    sector_pooled_pc0..pc{n_components-1}.
    """
    ticker_to_cluster, cluster_sizes = build_sector_clusters(df, train_mask, agreement_threshold)
    cluster = df["ticker"].map(ticker_to_cluster).to_numpy()

    # --- peer return / relative strength ---
    ret = price_factors["log_ret_close"].fillna(0.0).to_numpy()
    tmp = pd.DataFrame({"date": df["date"].to_numpy(), "cluster": cluster, "ret": ret})
    grp = tmp.groupby(["date", "cluster"])["ret"]
    cluster_sum = grp.transform("sum").to_numpy()
    cluster_cnt = grp.transform("count").to_numpy()
    peer_sum = cluster_sum - ret
    peer_cnt = cluster_cnt - 1
    peer_ret = np.divide(peer_sum, peer_cnt, out=np.zeros_like(peer_sum), where=peer_cnt > 0)
    rel_strength = ret - peer_ret

    # --- pooled cluster-level sector text embedding (PCA, fit on train only) ---
    store = EmbeddingStore.get(emb_path)
    sector_cols = dict(get_level_id_cols(list(df.columns)))["sector"]
    pooled = pool_level(df, sector_cols, store)
    pooled_df = pd.DataFrame(pooled, index=df.index)
    pooled_df["date"] = df["date"].to_numpy()
    pooled_df["cluster"] = cluster
    dim_cols = list(range(pooled.shape[1]))
    cluster_mean = pooled_df.groupby(["date", "cluster"])[dim_cols].transform("mean").to_numpy()

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(cluster_mean[train_mask])
    reduced = pca.transform(cluster_mean)
    print(f"[SectorFactors] pooled cluster embedding explained_var={pca.explained_variance_ratio_.sum():.3f}")

    out = {
        "sector_peer_ret": peer_ret,
        "sector_rel_strength": rel_strength,
        "sector_cluster_size": cluster_sizes[cluster].astype(np.float32),
    }
    for k in range(n_components):
        out[f"sector_pooled_pc{k}"] = reduced[:, k]
    return pd.DataFrame(out, index=df.index)
