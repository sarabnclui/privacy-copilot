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




    import os
import logging
import traceback
from typing import List, Optional

# --- Debug toggles (set via env or st.secrets)
DEBUG_RETRIEVER = os.getenv("DEBUG_RETRIEVER", "0") in ("1", "true", "True", "yes")
DEBUG_PDB       = os.getenv("DEBUG_PDB", "0")       in ("1", "true", "True", "yes")
DEBUG_DEBUGPY   = os.getenv("DEBUG_DEBUGPY", "0")   in ("1", "true", "True", "yes")
DEBUGPY_PORT    = int(os.getenv("DEBUGPY_PORT", "5678"))

DB_DIR      = os.getenv("CHROMA_DB_DIR", "./.chroma")
COLLECTION  = os.getenv("CHROMA_COLLECTION", "my_collection")
MODEL_NAME  = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")

# --- Optional Streamlit-friendly logging
try:
    import streamlit as st
    _IN_ST = True
except Exception:
    st = None
    _IN_ST = False

logging.basicConfig(level=logging.DEBUG if DEBUG_RETRIEVER else logging.INFO)
log = logging.getLogger("retriever")

def _maybe_break(label: str):
    """Conditional breakpoint with a label so you know where you are."""
    if DEBUG_RETRIEVER:
        log.debug(f"[BRK] {label}")
        if DEBUG_PDB:
            import pdb; pdb.set_trace()
        elif DEBUG_DEBUGPY:
            # Start debugpy listener once; subsequent calls just hit a break
            import debugpy
            if not debugpy.is_client_connected():
                try:
                    debugpy.listen(("0.0.0.0", DEBUGPY_PORT))
                    log.info(f"debugpy listening on 0.0.0.0:{DEBUGPY_PORT} (attach from VS Code)")
                except Exception as e:
                    log.exception(f"debugpy.listen failed: {e}")
            try:
                debugpy.breakpoint()
            except Exception:
                log.exception("debugpy.breakpoint failed")

def _show_exception(title: str, err: BaseException):
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    log.error(f"{title}: {err}\n{tb}")
    if _IN_ST:
        with st.expander(f"⚠️ {title}", expanded=False):
            st.code(tb, language="text")

# ------------------------------------------------------------------

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

        log.info("Initializing Retriever...")
        if _IN_ST:
            st.caption(f"Retriever init: DB_DIR='{DB_DIR}', COLLECTION='{COLLECTION}', MODEL='{MODEL_NAME}'")

        # ---- BREAKPOINT #1: entering __init__
        _maybe_break("enter __init__")

        # Try Chroma first
        try:
            # Optional SQLite shim (some clouds lack JSON/FTS in sqlite3)
            try:
                _maybe_break("before pysqlite3 shim")
                import pysqlite3  # type: ignore
                import sys
                sys.modules["sqlite3"] = pysqlite3
                log.debug("pysqlite3 shim applied")
            except Exception as shim_err:
                log.debug(f"pysqlite3 shim not used: {shim_err}")

            # ---- BREAKPOINT #2: before importing chromadb
            _maybe_break("before chromadb import")

            import chromadb
            from chromadb.utils import embedding_functions
            from sentence_transformers import SentenceTransformer  # ensure import error shows here

            log.debug("chromadb and sentence_transformers imported")

            class E5QueryEF(embedding_functions.EmbeddingFunction):
                def __init__(self, model_name: str):
                    # ---- BREAKPOINT #3: before loading model
                    _maybe_break("before SentenceTransformer(model)")
                    self.model = SentenceTransformer(model_name)
                    log.debug(f"SentenceTransformer loaded: {model_name}")
                def __call__(self, inputs: List[str]):
                    texts = [f"query: {t}" for t in inputs]
                    return self.model.encode(texts, normalize_embeddings=True).tolist()

            # ---- BREAKPOINT #4: before PersistentClient
            _maybe_break("before chromadb.PersistentClient")
            self._chroma = chromadb.PersistentClient(path=DB_DIR)

            # ---- BREAKPOINT #5: before get_collection
            _maybe_break("before get_collection")
            self._collection = self._chroma.get_collection(
                name=COLLECTION, embedding_function=E5QueryEF(MODEL_NAME)
            )

            # Probe query to confirm it works
            # ---- BREAKPOINT #6: before probe query
            _maybe_break("before probe query")
            _ = self._collection.query(query_texts=["ping"], n_results=1)
            log.info("Chroma probe succeeded")

            self._mode = "chroma"
            # ---- BREAKPOINT #7: after successful chroma setup
            _maybe_break("after chroma setup")

        except Exception as e:
            _show_exception("Chroma init failed; falling back", e)
            # ---- BREAKPOINT #8: in exception before fallback
            _maybe_break("in except before fallback")
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
