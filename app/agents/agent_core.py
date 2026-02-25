from __future__ import annotations

import inspect
import re

from app.agents.tools import ToolRegistry
from app.core.llm import get_provider
from app.agents.memory import AgentMemory


class MultiStepAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.memory = AgentMemory()
        self.llm = get_provider()

    def plan(self, question: str) -> str:
        prompt = (
            "You are an intelligent agent.\n"
            "Create a step-by-step plan to solve the problem.\n"
            f"Question: {question}\n"
        )
        return self.llm.generate(prompt)

    def choose_tool(self, step: str) -> str:
        prompt = (
            "Select the best tool for this step.\n"
            "Available tools:\n"
            f"{self.registry.list_descriptions()}\n\n"
            f"Step: {step}\n"
            "Return tool name only."
        )

        tool_name = self.llm.generate(prompt).strip()

        if tool_name not in self.registry.tools:
            tool_name = list(self.registry.tools.keys())[0]

        return tool_name

    def _extract_math(self, text: str) -> str:
        # crude extraction: pull a math-looking substring
        # e.g. "What is 10 + 5?" -> "10 + 5"
        m = re.search(r"([0-9][0-9\s\+\-\*\/\%\(\)\.]+)", text)
        return m.group(1).strip() if m else text.strip()

    def _run_tool(self, tool, tool_name: str, question: str):
        sig = inspect.signature(tool.run)
        params = sig.parameters

        # Prefer tool-specific clean inputs
        expression_val = self._extract_math(question) if tool_name == "calculator" else question
        text_val = question

        kwargs = {}
        if "expression" in params:
            kwargs["expression"] = expression_val
        if "text" in params:
            kwargs["text"] = text_val
        if "query" in params:
            kwargs["query"] = question
        if "input" in params:
            kwargs["input"] = question

        if kwargs:
            return tool.run(**kwargs)

        return tool.run(input=question)

    def run(self, question: str) -> str:
        self.memory.clear()

        plan = self.plan(question)
        self.memory.add(f"Plan: {plan}")

        tool_name = self.choose_tool(plan)
        tool = self.registry.get(tool_name)

        result = self._run_tool(tool, tool_name, question)
        self.memory.add(f"Tool result: {result}")

        final_prompt = (
            "You are an assistant.\n"
            f"Memory:\n{self.memory.get_context()}\n\n"
            f"Question: {question}\n"
            "Provide final answer."
        )

        return self.llm.generate(final_prompt)