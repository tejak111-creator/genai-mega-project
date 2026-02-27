from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss is not installed. Run: pip install faiss-cpu") from e

from app.rag.models import DocumentChunk


@dataclass
class PersistedIndex:
    index: faiss.Index
    chunks: List[DocumentChunk]
    embedding_dim: int


def save_faiss_index(
    index: faiss.Index,
    chunks: List[DocumentChunk],
    embedding_dim: int,
    dir_path: str,
    index_filename: str = "vectors.faiss",
    meta_filename: str = "chunks.pkl",
) -> None:
    """
    Saves:
    - FAISS index to vectors.faiss
    - chunks metadata list to chunks.pkl
    """
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    index_path = d / index_filename
    meta_path = d / meta_filename

    faiss.write_index(index, str(index_path))

    payload = {
        "embedding_dim": embedding_dim,
        "chunks": chunks,
    }
    meta_path.write_bytes(pickle.dumps(payload))


def load_faiss_index(
    dir_path: str,
    index_filename: str = "vectors.faiss",
    meta_filename: str = "chunks.pkl",
) -> Optional[PersistedIndex]:
    """
    Returns PersistedIndex if both files exist, else None.
    """
    d = Path(dir_path)
    index_path = d / index_filename
    meta_path = d / meta_filename

    if not index_path.exists() or not meta_path.exists():
        return None

    index = faiss.read_index(str(index_path))

    payload = pickle.loads(meta_path.read_bytes())
    embedding_dim = int(payload["embedding_dim"])
    chunks = payload["chunks"]

    return PersistedIndex(index=index, chunks=chunks, embedding_dim=embedding_dim)