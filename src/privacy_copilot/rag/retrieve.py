# src/privacy_copilot/rag/retrieve.py
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

DB_DIR = "data/public/chroma_db"
COLLECTION = "gdpr_corpus"
MODEL_NAME = "intfloat/multilingual-e5-base"

class SBertQueryEF(embedding_functions.EmbeddingFunction):
    """Embedding function for queries (E5 uses 'query:' prefix)."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, inputs: List[str]) -> List[List[float]]:
        texts = [f"query: {x}" for x in inputs]
        return self.model.encode(texts, normalize_embeddings=True).tolist()

class Retriever:
    def __init__(self, db_dir: str = DB_DIR, collection: str = COLLECTION):
        self.client = chromadb.PersistentClient(path=db_dir)
        # Use a query embedding function here (indexing used 'passage:' in the builder)
        self.coll = self.client.get_collection(
            name=collection,
            embedding_function=SBertQueryEF(MODEL_NAME)
        )

    def search(self, query: str, k: int = 6) -> List[Dict]:
        res = self.coll.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "text": res["documents"][0][i],
                "score": float(res["distances"][0][i]) if "distances" in res else None,
                **(res["metadatas"][0][i] or {})
            })
        return hits
