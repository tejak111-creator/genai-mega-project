from __future__ import annotations

from app.agents.tools import ToolRegistry
from app.agents.memory import AgentMemory
from app.agents.tool_calling import parse_tool_call, ToolCall
from app.agents.tool_cache import get_tool_cache, set_tool_cache
from app.core.llm import get_provider
from app.core.prompts import SYSTEM_AGENT_PROMPT


class FunctionCallingAgent:
    """
    Multi-step function-calling agent.
    """

    def __init__(self, registry: ToolRegistry, max_steps: int = 3):
        self.registry = registry
        self.memory = AgentMemory()
        self.llm = get_provider()
        self.max_steps = max_steps

    def decide_tool_call(self, question: str) -> ToolCall:
        prompt = (
            SYSTEM_AGENT_PROMPT
            + "\n\n"
            + "You are an agent that can use tools.\n"
            + "Return STRICT JSON ONLY:\n"
            + '{"tool_name":"calculator","arguments":{"expression":"2+2"}}\n\n'
            + "Available tools:\n"
            + f"{self.registry.list_specs()}\n\n"
            + f"Question: {question}\n"
        )

        raw = self.llm.generate(prompt).strip()

        # Fallback if model didn't return JSON
        if not raw.startswith("{"):
            q = question.lower()

            if any(op in q for op in ["+", "-", "*", "/"]) or any(c.isdigit() for c in q):
                return ToolCall(
                    tool_name="calculator",
                    arguments={"expression": question},
                )

            return ToolCall(
                tool_name="text_length",
                arguments={"text": question},
            )

        return parse_tool_call(raw)

    def run(self, question: str) -> str:
        self.memory.clear()
        self.memory.add(f"User: {question}")

        step = 0

        while step < self.max_steps:
            step += 1

            call = self.decide_tool_call(question)

            tool = self.registry.get(call.tool_name)
            if tool is None:
                self.memory.add(f"Tool: {call.tool_name}")
                self.memory.add("Result: Tool not found")
                return f"Unknown tool: {call.tool_name}"

            # -------------------------
            # Tool cache integration
            # -------------------------
            cached = get_tool_cache(call.tool_name, call.arguments)
            if cached is not None:
                result = cached
            else:
                result = tool.run(**call.arguments)
                set_tool_cache(call.tool_name, call.arguments, result)

            self.memory.add(f"Tool: {call.tool_name}")
            self.memory.add(f"Result: {result}")

            # Final answer prompt
            final_prompt = (
                SYSTEM_AGENT_PROMPT
                + "\n\n"
                + "Conversation:\n"
                + self.memory.get_context()
                + "\n\nProvide final answer:"
            )

            answer = self.llm.generate(final_prompt)

            if answer:
                return answer

        return "Unable to complete task."