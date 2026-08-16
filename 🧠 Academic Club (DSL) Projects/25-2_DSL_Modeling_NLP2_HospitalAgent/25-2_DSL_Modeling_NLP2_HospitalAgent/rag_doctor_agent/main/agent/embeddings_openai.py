from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

class OpenAIEmbeddingClient:
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("EMBED_MODEL", "text-embedding-3-large")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            raise RuntimeError("OPENAI_API_KEY is missing or openai package not available.")
        self.client = OpenAI()

    def embed(self, texts: List[str]) -> np.ndarray:
        # Batching is handled implicitly by OpenAI client; keep small batches if needed
        res = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [np.array(d.embedding, dtype="float32") for d in res.data]
        # normalize
        vecs = [v / (np.linalg.norm(v) + 1e-8) for v in vecs]
        return np.vstack(vecs)
