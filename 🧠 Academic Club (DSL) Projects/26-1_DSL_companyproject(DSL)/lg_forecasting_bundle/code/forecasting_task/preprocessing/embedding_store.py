"""
Multi-model text-embedding loading.

`EmbeddingStore` wraps a single `<model>_textemb.parquet` file (columns:
`text_id`, plus embedding dims) as a float16 matrix + text_id -> row-index
dict. `EmbeddingRegistry` auto-discovers every `*_textemb.parquet` file under
a data directory so the rest of the pipeline (Step2/3/4 consensus code) can
iterate "however many embedding models happen to be available" without any
hardcoded count.
"""
import ctypes
import gc
import glob
import os
from typing import Dict, Iterable, List, Optional

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

import numpy as np
import pandas as pd


class EmbeddingStore:
    """Loads one `<model>_textemb.parquet` file; float16 matrix + text_id -> row dict.

    Cached per `emb_path` (NOT a true singleton) so several models can be
    loaded concurrently, e.g. `EmbeddingStore.get(linq_path)` and
    `EmbeddingStore.get(gemini_path)` return distinct instances.
    """

    _instances: Dict[str, "EmbeddingStore"] = {}

    def __init__(self, emb_path: str):
        print(f"[EmbeddingStore] Loading {emb_path} ...")
        df = pd.read_parquet(emb_path)
        emb_cols = [c for c in df.columns if c != "text_id"]
        self.emb_dim = len(emb_cols)
        ids = df["text_id"].values
        mat = df[emb_cols].values.astype(np.float16)
        self._idx = {tid: i for i, tid in enumerate(ids)}
        self._mat = mat
        del df
        print(f"[EmbeddingStore] Loaded {len(ids):,} embeddings, dim={self.emb_dim} from {emb_path}")

    @classmethod
    def get(cls, emb_path: str) -> "EmbeddingStore":
        emb_path = os.path.abspath(emb_path)
        if emb_path not in cls._instances:
            cls._instances[emb_path] = cls(emb_path)
        return cls._instances[emb_path]

    @classmethod
    def release(cls, emb_path: str) -> None:
        """Drop the cached matrix for `emb_path` to free RAM before loading the next model.

        Explicit gc.collect() + malloc_trim(0) because the matrix is ~1-16GB (float16, up to
        858k x 4096); gc.collect() alone frees the Python objects but glibc malloc keeps the
        underlying pages in the process's arena rather than returning them to the OS, so
        multi-level runs (e.g. target+related+sector novelty back to back in one process)
        kept accumulating unreturned heap across each 6-model loop until the process got
        OOM-killed by the 128.8GB cgroup limit, even though nothing was still reachable from
        Python. malloc_trim(0) forces glibc to actually give the freed pages back."""
        emb_path = os.path.abspath(emb_path)
        cls._instances.pop(emb_path, None)
        gc.collect()
        if _libc is not None:
            _libc.malloc_trim(0)

    def coverage(self, ids: Iterable) -> float:
        """Fraction of non-null ids in `ids` that are actually present in this store."""
        total, found = 0, 0
        for tid in ids:
            if tid is None or (isinstance(tid, float) and np.isnan(tid)):
                continue
            total += 1
            if tid in self._idx:
                found += 1
        return found / total if total else 0.0

    def lookup_mean(self, ids) -> np.ndarray:
        valid_rows = []
        for tid in ids:
            if tid is None or (isinstance(tid, float) and np.isnan(tid)):
                continue
            idx = self._idx.get(tid, None)
            if idx is not None:
                valid_rows.append(idx)
        if not valid_rows:
            return np.zeros(self.emb_dim, dtype=np.float32)
        return self._mat[valid_rows].mean(axis=0).astype(np.float32)

    def lookup_single(self, tid) -> Optional[np.ndarray]:
        if tid is None or (isinstance(tid, float) and np.isnan(tid)):
            return None
        idx = self._idx.get(tid, None)
        if idx is None:
            return None
        return self._mat[idx].astype(np.float32)



# Preference order for picking a default "primary" model (used for the single-model
# features: Step1 text PCA, Step2 macro, Step3 sector pooled embedding) when the
# caller doesn't specify one. Roughly strongest/most-complete first; `bert` (384d,
# the weakest embedder in the set per README) goes last so it doesn't win by
# alphabetical accident. Anything not in this list falls back to alphabetical order.
DEFAULT_MODEL_PREFERENCE = ["linq", "lgai", "qwen", "nvda", "gemini", "bert"]


class EmbeddingRegistry:
    """Auto-discovers `<model>_textemb.parquet` files under `data_dir`.

    Model count is whatever's on disk right now (1 today = linq only, up to 6
    once bert/gemini/lgai/nvda/qwen land) -- nothing downstream should
    hardcode "6 models".
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._paths: Dict[str, str] = {}
        for path in sorted(glob.glob(os.path.join(data_dir, "*_textemb.parquet"))):
            model_name = os.path.basename(path)[: -len("_textemb.parquet")]
            self._paths[model_name] = path

    @property
    def model_names(self) -> List[str]:
        return list(self._paths.keys())

    @property
    def default_primary(self) -> str:
        """Preferred model for single-model features; see DEFAULT_MODEL_PREFERENCE."""
        available = self.model_names
        for name in DEFAULT_MODEL_PREFERENCE:
            if name in available:
                return name
        return available[0]

    def path(self, model_name: str) -> str:
        return self._paths[model_name]

    def __len__(self) -> int:
        return len(self._paths)

    def __repr__(self) -> str:
        return f"EmbeddingRegistry({self._paths})"
