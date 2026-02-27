from __future__ import annotations

from app.core.cache import SimpleCache

_tool_cache = SimpleCache()


def get_tool_cache(tool_name: str, args: dict):
    return _tool_cache.get("tool", tool_name, str(args))


def set_tool_cache(tool_name: str, args: dict, result):
    _tool_cache.set(result, "tool", tool_name, str(args))