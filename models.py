from __future__ import annotations
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

class AvailableActions(BaseModel):
    clickables: List[str] = Field(default_factory=list)
    has_search_bar: bool


class TakeStepResponse(BaseModel):
    action_taken: str
    available_actions: AvailableActions
    done: bool
    instruction_text: str
    message: str
    observation: str
    url: str

class InitiateGoalResponse(BaseModel):
    available_actions: AvailableActions
    instruction_text: str
    message: str
    observation: str
    observation_mode: str
    session_id: str
    url: str



# Optional: to carry around context for validation/logging
class Context(BaseModel):
    instruction: str
    observation: str
    available_actions: AvailableActions
    history: List[str] = Field(default_factory=list)

# ---------- Output-side models ----------
class ActionSuggestion(BaseModel):
    plan: str
    thought: str
    env: str  # "search[...]" OR "click[<ID>]"

class Explanation(BaseModel):
    expl: str

# ---------- Post-parse environment validations ----------
def validate_env_constraints(sug: ActionSuggestion, aa: AvailableActions) -> None:
    """
    Enforces:
      - env == search[query] allowed only if has_search_bar is True
      - env == click[id] must be one of aa.clickables (exact match)
    Raises ValueError on violation.
    """
    e = sug.env.strip()
    if e.startswith("search[") and e.endswith("]"):
        if not aa.has_search_bar:
            raise ValueError("search[...] not allowed: has_search_bar is False.")
        # query can be empty/any; you can add stricter checks if needed
        return

    if e.startswith("click[") and e.endswith("]"):
        target = e[len("click["):-1]
        if target not in aa.clickables:
            raise ValueError(f"click target '{target}' not in clickables.")
        return

    raise ValueError("env must be either search[...] or click[...].")
