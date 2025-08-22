# src/privacy_copilot/rag/index/build_index_chroma.py
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

DB_DIR = "data/public/chroma_db"
COLLECTION = "gdpr_corpus"
MODEL_NAME = "intfloat/multilingual-e5-base"

def load_chunks(path="data/public/chunks.jsonl"):
    texts, metas, ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            texts.append(obj["text"])
            ids.append(obj.get("id") or str(i))
            metas.append({k: obj[k] for k in obj if k != "text"})
    return texts, metas, ids

class SBertEF(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, inputs):
        texts = [f"passage: {x}" for x in inputs]
        return self.model.encode(texts, normalize_embeddings=True).tolist()

def main():
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)
    # recreate collection each run (simple for dev)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    ef = SBertEF(MODEL_NAME)
    coll = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    texts, metas, ids = load_chunks()
    coll.add(ids=ids, documents=texts, metadatas=metas)
    print(f"Indexed {len(ids)} chunks → {DB_DIR}\\{COLLECTION}")

if __name__ == "__main__":
    main()
