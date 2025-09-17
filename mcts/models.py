
from __future__ import annotations
from typing import Dict, List, Tuple
from pydantic import BaseModel, RootModel, ValidationError
import re

_ENV_RE = re.compile(r"^(search|click)\[([^\[\]]+)\]$")

class Candidate(BaseModel):
    plan: str
    thought: str
    env: str

class ActionSuggestion(RootModel):
    root: Dict[str, Candidate]

    def items(self):
        return self.root.items()

    def values(self):
        return self.root.values()

    def __getitem__(self, k: str) -> Candidate:
        return self.root[k]

class Explanation(BaseModel):
    expl: str

class AvailableActionsInner(BaseModel):
    clickables: List[str]
    has_search_bar: bool

class AvailableActions(BaseModel):
    available_actions: AvailableActionsInner

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


def parse_env(env: str) -> Tuple[str, str]:
    m = _ENV_RE.match(env.strip())
    if not m:
        raise ValueError(f"Invalid env format: {env!r}. Expected 'search[...]' or 'click[ID]'.")
    kind, payload = m.group(1), m.group(2)
    return kind, payload

def validate_env_constraints(suggestions: ActionSuggestion, actions: AvailableActions) -> None:
    """
    Validate *all* candidates respect the environment constraints.
    - search[...] allowed only if has_search_bar is True
    - click[ID] must use an ID from clickables exactly
    """
    clickables = actions.available_actions.clickables
    has_search = actions.available_actions.has_search_bar

    for key, cand in suggestions.items():
        kind, payload = parse_env(cand.env)
        if kind == "search":
            if not has_search:
                raise ValueError(f"Candidate {key} uses search[...] but has_search_bar is false.")
            # allow any free-text query for search
        elif kind == "click":
            if payload not in clickables:
                raise ValueError(f"Candidate {key} clicks unknown ID {payload!r}; allowed: {clickables}.")
        else:
            raise ValueError(f"Candidate {key} has unsupported env kind: {kind}")
