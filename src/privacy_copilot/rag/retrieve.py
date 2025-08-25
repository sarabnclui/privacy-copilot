# src/privacy_copilot/rag/retrieve.py
from pathlib import Path
from typing import List, Dict
import json
import math

# Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

DB_DIR = "data/public/chroma_db"
COLLECTION = "gdpr_corpus"
MODEL_NAME = "intfloat/multilingual-e5-base"

def _load_chunks(path: str = "data/public/chunks.jsonl"):
    texts, metas, ids = [], [], []
    p = Path(path)
    if not p.exists():
        return texts, metas, ids
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            texts.append(obj["text"])
            ids.append(obj.get("id") or str(i))
            metas.append({k: obj[k] for k in obj if k != "text"})
    return texts, metas, ids

class Retriever:
    """
    Hybrid retriever:
      - Try Chroma (persistent) first.
      - If Chroma import/init fails (e.g., SQLite not available on Streamlit Cloud),
        fall back to a simple in-memory cosine retriever.
    """
    def __init__(self, k_default: int = 6):
        self.k_default = k_default
        self._mode = "fallback"  # default until Chroma proves available
        self._chroma = None
        self._collection = None

        # Try Chroma first
        try:
            # Optional SQLite shim (some clouds lack JSON/FTS in sqlite3)
            try:
                import pysqlite3  # type: ignore
                import sys
                sys.modules["sqlite3"] = pysqlite3
            except Exception:
                pass

            import chromadb
            from chromadb.utils import embedding_functions

            class E5QueryEF(embedding_functions.EmbeddingFunction):
                def __init__(self, model_name: str):
                    self.model = SentenceTransformer(model_name)
                def __call__(self, inputs: List[str]):
                    texts = [f"query: {t}" for t in inputs]
                    return self.model.encode(texts, normalize_embeddings=True).tolist()

            self._chroma = chromadb.PersistentClient(path=DB_DIR)
            self._collection = self._chroma.get_collection(
                name=COLLECTION, embedding_function=E5QueryEF(MODEL_NAME)
            )
            # Probe query to confirm it works
            _ = self._collection.query(query_texts=["ping"], n_results=1)
            self._mode = "chroma"
        except Exception:
            # Fall back to in-memory retriever
            self._setup_fallback()

    # ---------- Fallback (in-memory) ----------
    def _setup_fallback(self):
        texts, metas, ids = _load_chunks()
        self._texts = texts
        self._metas = metas
        self._ids = ids
        self._model = SentenceTransformer(MODEL_NAME)
        if texts:
            vecs = self._model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True)
            self._vecs = np.asarray(vecs, dtype=np.float32)
        else:
            self._vecs = np.zeros((0, 384), dtype=np.float32)  # shape won't be used if empty
        self._mode = "fallback"

    def _search_fallback(self, query: str, k: int):
        if len(self._texts) == 0:
            return []
        q = self._model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        sims = (self._vecs @ q)  # cosine since unit-normalized
        idx = np.argsort(-sims)[:k]
        hits = []
        for i in idx:
            hits.append({
                "text": self._texts[i],
                "similarity": float(sims[i]),
                **(self._metas[i] or {})
            })
        return hits

    # ---------- Public API ----------
    def search(self, query: str, k: int = None) -> List[Dict]:
        k = k or self.k_default
        if self._mode == "chroma":
            res = self._collection.query(
                query_texts=[query], n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            hits = []
            for i in range(len(res["ids"][0])):
                dist = float(res["distances"][0][i]) if "distances" in res else None
                sim = (1.0 - dist) if dist is not None else None
                hits.append({
                    "text": res["documents"][0][i],
                    "similarity": sim,
                    **(res["metadatas"][0][i] or {})
                })
            return hits
        else:
            return self._search_fallback(query, k)
