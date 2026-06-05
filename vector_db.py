import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np

from embedding_utils import embed_texts

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CHUNKS_FILE = DATA_DIR / "knowledge_chunks.json"


class VectorDB:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []
        self.sources: List[str] = []

    def add_texts(self, texts: List[str], vectors: np.ndarray, source_name: str = "") -> None:
        if len(texts) == 0:
            return
        vectors = np.array(vectors, dtype="float32")
        self.texts.extend(texts)
        self.sources.extend([source_name] * len(texts))
        self.index.add(vectors)

    def query_by_vector(self, vector: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        if len(self.texts) == 0:
            return []
        top_k = min(top_k, len(self.texts))
        vector = np.array([vector], dtype="float32")
        distances, indices = self.index.search(vector, top_k)
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "source": self.sources[idx],
                    "score": float(distance),
                })
        return results


_vector_db: VectorDB | None = None


def _save_chunks(records: List[Dict[str, str]]) -> None:
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _load_chunk_records() -> List[Dict[str, str]]:
    if not CHUNKS_FILE.exists():
        return []
    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def build_vector_db(chunks: List[str], source_name: str = "") -> Dict[str, Any]:
    """Build and persist a vector database from text chunks."""
    global _vector_db

    chunks = [c.strip() for c in chunks if c and c.strip()]
    if not chunks:
        raise ValueError("No valid text chunks to index.")

    vectors = embed_texts(chunks)
    _vector_db = VectorDB(dim=vectors.shape[1])
    _vector_db.add_texts(chunks, vectors, source_name=source_name)

    records = [{"text": chunk, "source": source_name} for chunk in chunks]
    _save_chunks(records)

    return {"chunks": len(chunks), "source": source_name}


def load_vector_db() -> bool:
    """Load persisted chunks and rebuild the FAISS index at startup."""
    global _vector_db

    records = _load_chunk_records()
    texts = [r.get("text", "") for r in records if r.get("text")]
    if not texts:
        _vector_db = None
        return False

    vectors = embed_texts(texts)
    _vector_db = VectorDB(dim=vectors.shape[1])
    sources = [r.get("source", "") for r in records]
    _vector_db.texts.extend(texts)
    _vector_db.sources.extend(sources)
    _vector_db.index.add(np.array(vectors, dtype="float32"))
    return True


def search(question: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Search the current vector database by question text."""
    global _vector_db

    if _vector_db is None:
        load_vector_db()

    if _vector_db is None:
        return []

    query_vector = embed_texts([question])[0]
    return _vector_db.query_by_vector(query_vector, top_k=top_k)


def has_knowledge_base() -> bool:
    if _vector_db is not None and len(_vector_db.texts) > 0:
        return True
    return CHUNKS_FILE.exists() and len(_load_chunk_records()) > 0


def clear_vector_db() -> None:
    global _vector_db
    _vector_db = None
    if CHUNKS_FILE.exists():
        CHUNKS_FILE.unlink()
