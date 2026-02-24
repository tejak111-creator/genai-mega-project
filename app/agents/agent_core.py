from __future__ import annotations

from app.agents.tools import ToolRegistry
from app.core.llm import get_provider

class SimpleAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.llm = get_provider()

    def run(self, question: str) -> str:
        #Step 1 : Ask LLM what tool to use
        tool_prompt = (
             "You are an agent. Decide which tool to use.\n"
            "Available tools:\n"
            f"{self.registry.list_descriptions()}\n\n"
            f"Question: {question}\n"
            "Return tool name only."
        )

        tool_name = self.llm.generate(tool_prompt).strip()
        
        #Fallback if stub
        if tool_name not in self.registry.tools:
            tool_name = list(self.registry.tools.keys())[0]
        
        tool = self.registry.get(tool_name)

        # Step 2 - Execute tool
        if tool_name == "calculator":
            result = tool.run(expression=question)
        elif tool_name == "text_length":
            result = tool.run(text=question)
        else:
            result = tool.run()

        # Step 3 - Final Answer
        final_prompt = (
             f"Question: {question}\n"
            f"Tool used: {tool_name}\n"
            f"Tool result: {result}\n"
            "Provide final answer."
        )

        answer = self.llm.generate(final_prompt)

        return answer