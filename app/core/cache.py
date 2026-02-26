from __future__ import annotations

from typing import Any, Dict, Tuple
import hashlib
import json

class SimpleCache:
    def __init__(self):
        self._store: Dict[str, Any] = {}
    
    def _key(self, *parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    
    def get(self, *parts: Any) -> Any | None:
        return self._store.get(self._key(*parts))
    
    def set(self, value: Any, *parts: Any) -> None:
        self._store[self._key(*parts)] = value
        