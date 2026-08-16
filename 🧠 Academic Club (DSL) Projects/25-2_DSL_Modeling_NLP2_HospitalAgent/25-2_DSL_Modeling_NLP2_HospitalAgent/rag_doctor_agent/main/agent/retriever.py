from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os, json, math, glob, re
import numpy as np
from dotenv import load_dotenv

from .utils import normalize_text, tokenize_ko_en, uniq_keep_order
from .augmentation import expand_symptoms

load_dotenv()

RAG_DATA_DIR = os.getenv("RAG_DATA_DIR",
                         os.path.join(os.path.dirname(__file__), "..", "data"))
DB_DIR    = os.path.join(RAG_DATA_DIR, "db_data")
INDEX_DIR = os.path.join(DB_DIR, "index")
PREPROC_DIR = os.path.join(DB_DIR, "preprocessed")

# --------------------------------------------------------------------------- #
# Embeddings (OpenAI)
# --------------------------------------------------------------------------- #
from .embeddings_openai import OpenAIEmbeddingClient

@dataclass
class Doc:
    id: str
    text: str
    meta: Dict[str, Any]
    source: str = ""
    type: str   = ""

# --------------------------------------------------------------------------- #
# Hybrid Vector + BM25-like Index
# --------------------------------------------------------------------------- #
class HybridIndex:
    def __init__(self):
        self.docs: List[Doc]         = []
        self.emb_matrix: Optional[np.ndarray] = None
        self.doc_toks: List[List[str]] = []
        self.df: Dict[str, int]      = {}
        self.N: int                  = 0

    # ------------------- utilities ------------------- #
    def _tokenize(self, text: str) -> List[str]:
        return tokenize_ko_en(text)

    def _update_df(self, toks: List[str]):
        for t in set(toks):
            self.df[t] = self.df.get(t, 0) + 1

    # ------------------- add docs ------------------- #
    def add_docs(self, docs: List[Doc], embedder: OpenAIEmbeddingClient):
        if not docs: return
        self.docs.extend(docs)
        self.N = len(self.docs)

        new_toks = [self._tokenize(d.text) for d in docs]
        for ts in new_toks:
            self._update_df(ts)
        self.doc_toks.extend(new_toks)

        embs = embedder.embed([d.text for d in docs])
        self.emb_matrix = embs if self.emb_matrix is None \
            else np.vstack([self.emb_matrix, embs])

    # ------------------- scoring ------------------- #
    def _cosine_sim(self, q: np.ndarray) -> np.ndarray:
        if self.emb_matrix is None:
            return np.zeros(0, dtype="float32")
        qn = q / (np.linalg.norm(q) + 1e-8)
        Mn = self.emb_matrix / (
              np.linalg.norm(self.emb_matrix, axis=1, keepdims=True) + 1e-8)
        return (Mn @ qn.reshape(-1, 1)).ravel()

    def _bm25_like(self, query: str, k1=1.2, b=0.75) -> np.ndarray:
        if self.N == 0: return np.zeros(0, dtype="float32")
        q_toks = self._tokenize(query)
        avgdl  = np.mean([len(t) for t in self.doc_toks]) + 1e-8
        idf = {t: math.log((self.N - self.df.get(t,0) + 0.5) /
                           (self.df.get(t,0) + 0.5) + 1)
               for t in set(q_toks)}

        scores = np.zeros(self.N, dtype="float32")
        for i, toks in enumerate(self.doc_toks):
            dl = len(toks) + 1e-8
            tf = {}
            for t in toks: tf[t] = tf.get(t,0) + 1
            s = 0.0
            for t in q_toks:
                if t not in idf: continue
                ft = tf.get(t,0)
                if ft == 0:     continue
                denom = ft + k1 * (1 - b + b * dl / avgdl)
                s += idf[t] * (ft*(k1+1)) / denom
            scores[i] = s
        if scores.max() > 0:
            scores = scores / (scores.max() + 1e-8)
        return scores

    # ------------------- search ------------------- #
    def search(self, query: str, embedder: OpenAIEmbeddingClient,
               alpha=0.65, top_k=8) -> List[Tuple[Doc, float]]:
        if self.N == 0: return []
        qv = embedder.embed([query])[0]
        hybrid = alpha * self._cosine_sim(qv) \
               + (1 - alpha) * self._bm25_like(query)

        idx = np.argsort(-hybrid)[:max(top_k*3, top_k)]

        def overlap(a,b):
            at,bt = set(tokenize_ko_en(a)), set(tokenize_ko_en(b))
            return len(at & bt)/(len(at|bt)+1e-8) if at and bt else 0.0

        rescored = [(self.docs[i],
                     float(hybrid[i] + 0.05*overlap(query, self.docs[i].text)))
                    for i in idx]
        rescored.sort(key=lambda x: -x[1])
        return rescored[:top_k]

# --------------------------------------------------------------------------- #
# Retriever facade
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self):
        self.index    = HybridIndex()
        self.embedder = OpenAIEmbeddingClient()

    # ------------- load / ingest ------------- #
    def load_index(self) -> bool:
        meta = os.path.join(INDEX_DIR, "index_meta.json")
        vec  = os.path.join(INDEX_DIR, "vectors.npy")
        docs = os.path.join(INDEX_DIR, "docs.jsonl")
        toks = os.path.join(INDEX_DIR, "doc_toks.jsonl")
        df   = os.path.join(INDEX_DIR, "df.json")

        if not all(os.path.exists(p) for p in [meta,vec,docs,toks,df]):
            return False

        with open(docs, "r", encoding="utf-8") as f:
            self.index.docs = [Doc(**json.loads(l)) for l in f]
        self.index.emb_matrix = np.load(vec)
        with open(toks, "r", encoding="utf-8") as f:
            self.index.doc_toks = [json.loads(l) for l in f]
        self.index.N = len(self.index.docs)
        self.index.df = json.load(open(df, "r", encoding="utf-8"))
        return True

    def ingest_from_db_data(self) -> Dict[str, Any]:
        # db_data/** 내 *.json/JSONL
        files = sorted(glob.glob(os.path.join(PREPROC_DIR, "*.jsonl"))) + \
                sorted(glob.glob(os.path.join(DB_DIR, "*.json")))      + \
                sorted(glob.glob(os.path.join(DB_DIR, "*.jsonl")))

        docs: List[Doc] = []
        for fp in files:
            # ▶▶ 절대/상대 경로 혼합 문제 해결 ◀◀
            if os.path.commonpath(
                    [os.path.abspath(DB_DIR), os.path.abspath(fp)]
                ) != os.path.abspath(DB_DIR):
                continue

            with open(fp, "r", encoding="utf-8") as f:
                if fp.endswith(".jsonl"):
                    for line in f:
                        docs.append(Doc(**json.loads(line)))
                else:
                    data = json.load(f)
                    items = data if isinstance(data, list) else data.get("docs", [])
                    for d in items:
                        docs.append(Doc(**d))

        if not docs:
            return {"message": "No db_data docs found.", "counts": {"docs": 0}}

        self.index = HybridIndex()
        self.index.add_docs(docs, self.embedder)

        # persist
        os.makedirs(INDEX_DIR, exist_ok=True)
        np.save(os.path.join(INDEX_DIR,"vectors.npy"), self.index.emb_matrix)
        with open(os.path.join(INDEX_DIR,"docs.jsonl"),"w",encoding="utf-8") as f:
            for d in self.index.docs:
                f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")
        with open(os.path.join(INDEX_DIR,"doc_toks.jsonl"),"w",encoding="utf-8") as f:
            for t in self.index.doc_toks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with open(os.path.join(INDEX_DIR,"df.json"),"w",encoding="utf-8") as f:
            json.dump(self.index.df, f, ensure_ascii=False)
        with open(os.path.join(INDEX_DIR,"index_meta.json"),"w",encoding="utf-8") as f:
            json.dump({"N": self.index.N}, f, ensure_ascii=False)

        return {"message": f"Indexed {len(docs)} docs", "counts": {"docs": len(docs)}}

    # ------------- retrieve ------------- #
    def retrieve(self, symptoms: List[str], top_k=8, alpha=None):
        if not self.load_index():
            raise RuntimeError("Index not found – run pipeline index/build first")
        aug = expand_symptoms(symptoms or [])
        
        # 증상 + 진료과 조합으로 검색하여 의료진 정보도 포함
        # 허리 통증 -> 허리 통증 + 정형외과 + 척추센터
        dept_keywords = []
        for symptom in aug:
            if any(keyword in symptom.lower() for keyword in ['허리', '척추', '디스크', '관절']):
                dept_keywords.extend(['정형외과', '척추센터', '척추비수술클리닉'])
            elif any(keyword in symptom.lower() for keyword in ['두통', '머리', '어지러움']):
                dept_keywords.extend(['신경과', '신경외과'])
            elif any(keyword in symptom.lower() for keyword in ['소화', '배', '명치', '위']):
                dept_keywords.extend(['내과', '소화기내과'])
            elif any(keyword in symptom.lower() for keyword in ['무릎', '어깨', '팔', '다리']):
                dept_keywords.extend(['정형외과', '관절센터'])
        
        # 증상 + 진료과 조합으로 검색
        combined_terms = aug + dept_keywords
        query = " ; ".join(combined_terms)
        
        alpha = alpha if alpha is not None else \
                float(os.getenv("HYBRID_ALPHA", "0.65"))
        hits = self.index.search(query, self.embedder, alpha=alpha, top_k=top_k)
        return [{"id": d.id, "text": d.text, "meta": d.meta, "score": s}
                for d,s in hits]

# --------------------------------------------------------------------------- #
# Shared Retriever (optional singleton)
# --------------------------------------------------------------------------- #
SHARED_RETRIEVER: Optional[Retriever] = None

def get_shared_retriever() -> Retriever:
    global SHARED_RETRIEVER
    if SHARED_RETRIEVER is None:
        SHARED_RETRIEVER = Retriever()
    return SHARED_RETRIEVER
