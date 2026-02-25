from __future__ import annotations

from typing import List

class AgentMemory:
    """
    Simple in-memory conversation memory.
    """
    def __init__(self):
        self.messages: List[str] = []
    def add(self, message: str):
        self.messages.append(message)
    def get_context(self) -> str:
        return "\n".join(self.messages)
    def clear(self):
        self.messages = []