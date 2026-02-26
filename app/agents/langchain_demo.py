from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    return str(eval(expression))
