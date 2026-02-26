from __future__ import annotations

from app.agents.tools import ToolRegistry
from app.agents.memory import AgentMemory
from app.agents.tool_calling import parse_tool_call, ToolCall
from app.core.llm import get_provider
from app.core.prompts import SYSTEM_AGENT_PROMPT


class FunctionCallingAgent:
    """
    Function-calling style agent:
    - LLM decides a tool call in STRICT JSON
    - We parse the JSON
    - Execute tool safely
    - LLM produces final response using memory
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.memory = AgentMemory()
        self.llm = get_provider()

    def decide_tool_call(self, question: str) -> ToolCall:
        # ReAct style guidance + strict JSON requirement
        prompt = (
            SYSTEM_AGENT_PROMPT.strip()
            + "\n\n"
            + "You are an agent. Follow this process internally:\n"
              "Thought: What should I do?\n"
              "Action: Choose a tool\n"
              "Action Input: Provide arguments\n\n"
              "Return STRICT JSON ONLY in this exact format:\n"
              '{"tool_name":"calculator","arguments":{"expression":"2+2"}}\n\n'
            + "Available tools:\n"
            + f"{self.registry.list_specs()}\n\n"
            + f"Question: {question}\n"
        )

        raw = self.llm.generate(prompt).strip()

        # If LLM didn't return JSON (common with stub), use deterministic fallback
        if not raw.startswith("{"):
            q = question.lower()

            # Heuristic: if it looks like math or contains digits/operators, use calculator
            if any(op in q for op in ["+", "-", "*", "/"]) or any(c.isdigit() for c in q):
                return ToolCall(tool_name="calculator", arguments={"expression": question})

            # Otherwise use text_length
            return ToolCall(tool_name="text_length", arguments={"text": question})

        return parse_tool_call(raw)

    def run(self, question: str) -> str:
        self.memory.clear()
        self.memory.add(f"Q: {question}")

        call = self.decide_tool_call(question)
        self.memory.add(f"ToolCall: {call.tool_name} args={call.arguments}")

        tool = self.registry.get(call.tool_name)
        result = tool.run(**call.arguments)
        self.memory.add(f"ToolResult: {result}")

        final_prompt = (
            SYSTEM_AGENT_PROMPT.strip()
            + "\n\n"
            + "Use the memory below to answer. Keep it short.\n\n"
            + f"Memory:\n{self.memory.get_context()}\n\n"
            + "Final answer (1-3 sentences):\n"
        )

        return self.llm.generate(final_prompt)