"""Shared helpers for pooling per-row text-id columns into embedding vectors."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .embedding_store import EmbeddingStore
from .levels import get_level_id_cols


def pool_level(df: pd.DataFrame, id_cols, store: EmbeddingStore) -> np.ndarray:
    """Mean-pool `store` embeddings across `id_cols` for every row of `df`. -> [n, store.emb_dim]."""
    n = len(df)
    pooled = np.zeros((n, store.emb_dim), dtype=np.float32)
    if not id_cols:
        return pooled
    id_arr = df[id_cols].values  # [n, len(id_cols)] object array
    for i in range(n):
        pooled[i] = store.lookup_mean(id_arr[i].tolist())
    return pooled


def build_text_pca_factors(df: pd.DataFrame, emb_path: str, n_components: int,
                            train_mask: np.ndarray) -> pd.DataFrame:
    """Mean-pool per level per row, then PCA (fit on train_mask rows only). Step1 baseline text features."""
    store = EmbeddingStore.get(emb_path)
    level_cols = get_level_id_cols(list(df.columns))
    out_cols = {}
    for level_name, cols in level_cols:
        pooled = pool_level(df, cols, store)
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(pooled[train_mask])
        reduced = pca.transform(pooled)
        for k in range(n_components):
            out_cols[f"{level_name}_pc{k}"] = reduced[:, k]
        print(f"[TextFactors] level={level_name}: pooled {len(cols)} id-cols, "
              f"explained_var={pca.explained_variance_ratio_.sum():.3f}")
    return pd.DataFrame(out_cols, index=df.index)
