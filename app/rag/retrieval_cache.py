from __future__ import annotations

from typing import List, Tuple

from app.core.cache import SimpleCache
from app.rag.vector_store import SearchResult


_retrieval_cache = SimpleCache()


def get_cached_results(query: str, top_k: int) -> List[SearchResult] | None:
    return _retrieval_cache.get("retrieval", query, top_k)


def set_cached_results(query: str, top_k: int, results: List[SearchResult]) -> None:
    _retrieval_cache.set(results, "retrieval", query, top_k)