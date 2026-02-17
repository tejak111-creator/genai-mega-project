#GuardRail/Safety Layer
from dataclasses import dataclass

#automatically generates common methods for a class, helps avoid boiler plate and _init_
@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None

def basic_input_guard(prompt: str) -> GuardrailResult:
    """
    Day 17 we will expand
    Prompt injection detection, policy filters,tool allowlists
    """
    if not prompt.strip():
        return GuardrailResult(False, "empty_prompt")
    
    banned= ["passwords dump","credit card dump"]
    lower = prompt.lower()
    #below we see generator expression
    if any(b in lower for b in banned):
        return GuardrailResult(False, "disallowed_content")
    
    return GuardrailResult(True)