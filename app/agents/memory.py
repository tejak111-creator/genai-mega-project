from __future__ import annotations

from typing import List


class AgentMemory:
    """
    Simple conversation memory.
    Stores messages in order.
    """

    def __init__(self):
        self.messages: List[str] = []

    def add(self, text: str) -> None:
        self.messages.append(text)

    def get_context(self) -> str:
        return "\n".join(self.messages)

    def clear(self) -> None:
        self.messages.clear()