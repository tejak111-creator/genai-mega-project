from __future__ import annotations

from app.agents.tools import ToolRegistry
from app.agents.memory import AgentMemory
from app.agents.tool_calling import parse_tool_call, ToolCall
from app.core.llm import get_provider
from app.core.cache import SimpleCache

_tool_cache = SimpleCache()

class FunctionCallingAgent:
    """
    Produces a strict tool call JSON, executes tool, then finalizes answer.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.memory = AgentMemory()
        self.llm = get_provider()

    def decide_tool_call(self, question: str) -> ToolCall:
        prompt = (
            "You are an agent that MUST return STRICT JSON ONLY.\n"
            "Choose the best tool and provide arguments.\n\n"
            "Available tools:\n"
            f"{self.registry.list_specs()}\n\n"
            "Return JSON exactly like:\n"
            '{"tool_name":"calculator","arguments":{"expression":"2+2"}}\n\n'
            f"Question: {question}\n"
        )
        raw = self.llm.generate(prompt).strip()

        # Stub fallback: if not JSON, pick based on simple heuristic
        if not raw.startswith("{"):
            if any(ch.isdigit() for ch in question) and any(op in question for op in ["+", "-", "*", "/"]):
                return ToolCall(tool_name="calculator", arguments={"expression": question})
            return ToolCall(tool_name="text_length", arguments={"text": question})

        return parse_tool_call(raw)

    def run(self, question: str) -> str:
        self.memory.clear()
        self.memory.add(f"Q: {question}")

        call = self.decide_tool_call(question)
        self.memory.add(f"ToolCall: {call.tool_name} args={call.arguments}")

        tool = self.registry.get(call.tool_name)
        cached = _tool_cache.get("tool", call.tool_name, call.arguments)
        if cached is None:
            result = tool.run(**call.arguments)
            _tool_cache.set(result,"tool", call.tool_name, call.arguments)
        else:
            result = cached
        self.memory.add(f"ToolResult: {result}")

        final_prompt = (
            "You are a helpful assistant.\n"
            f"Memory:\n{self.memory.get_context()}\n\n"
            "Answer the user in 1-3 sentences.\n"
        )
        return self.llm.generate(final_prompt)