from typing import TypedDict
from app.agents.sample_tools import calculator, text_length
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    question: str
    tool_result: str
    answer: str


def tool_node(state: AgentState):
    question = state["question"]

    # simple logic for demo
    if any(op in question for op in ["+","-","*","/"]):
        result = calculator(question)
    else:
        result = text_length(question)

    return {"tool_result": result}


def answer_node(state: AgentState):
    tool_result = state["tool_result"]

    answer = f"The result is {tool_result}"

    return {"answer": answer}


def create_langgraph_agent():

    workflow = StateGraph(AgentState)

    workflow.add_node("tool", tool_node)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("tool")

    workflow.add_edge("tool", "answer")
    workflow.add_edge("answer", END)

    app = workflow.compile()

    return app