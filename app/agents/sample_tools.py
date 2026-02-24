from __future__ import annotations

def calculator(expression: str) -> str:
    """
    Very simple calculator tool
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"error: {e}"
    
def text_length(text: str) -> str:
    """
    Returns length of text.
    """
    return str(len(text))