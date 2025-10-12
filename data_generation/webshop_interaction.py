from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------
# HTTP client configuration
# -------------------------
_API_BASE = "http://127.0.0.1:5000/api"

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json", "Connection": "keep-alive"})

# conservative retry for transient 50x
_retries = Retry(
    total=3,
    backoff_factor=0.2,
    status_forcelist=[502, 503, 504],
    allowed_methods=frozenset(["POST"]),
    raise_on_status=False,
)
_session.mount("http://", HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=_retries))
_session.mount("https://", HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=_retries))


def _post(path: str, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    url = f"{_API_BASE}/{path.lstrip('/')}"
    r = _session.post(url, data=json.dumps(payload), timeout=timeout)
    # If the server returned JSON error with non-200, try to surface it cleanly
    try:
        data = r.json()
    except Exception:
        r.raise_for_status()
        # if still here, it's unexpected non-JSON 200; return empty dict
        return {}
    if not r.ok:
        # raise a readable exception with server message if present
        msg = data.get("error") or data.get("message") or r.text
        raise RuntimeError(f"POST {url} failed ({r.status_code}): {msg}")
    return data


# -------------------------
# Public API (live episode)
# -------------------------
def initialize_goal(goal_idx: Optional[int] = None, observation_mode: str = "text") -> Dict[str, Any]:
    """
    Start a *live* episode.

    Returns a snapshot:
      {
        "session_id": str,
        "instruction_text": str,
        "url": str,
        "observation": str,
        "available_actions": {"has_search_bar": bool, "clickables": [...]},
        "goal_idx": int | None,   # echoed if provided
        ...
      }
    """
    payload: Dict[str, Any] = {"observation_mode": observation_mode}
    if goal_idx is not None:
        payload["goal_idx"] = int(goal_idx)
    return _post("start", payload)


def take_step(session_id: str, env: str) -> Dict[str, Any]:
    """
    Execute one *live* step on an existing session.

    Args:
      session_id: live session id from initialize_goal()
      env: environment string, e.g., "search[...]" or "click[ID]"

    Returns a snapshot with fields:
      - "reward": float
      - "done": bool
      - "observation", "available_actions", etc.
    """
    payload = {"session_id": session_id, "action": env}
    return _post("step", payload)


def reset_session(session_id: str, goal_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Reset an existing *live* session (keeps session_id). If goal_text is provided,
    the server will reset with that instruction text.
    """
    payload: Dict[str, Any] = {"session_id": session_id}
    if goal_text:
        payload["goal"] = goal_text
    return _post("reset", payload)


def end_session(session_id: str) -> Dict[str, Any]:
    """Delete/close a *live* session on the server."""
    return _post("end", {"session_id": session_id})


def is_done(resp: Dict[str, Any]) -> bool:
    """Convenience helper for termination checks."""
    try:
        if bool(resp.get("done")):
            return True
        # Some servers only set reward; treat any positive reward as terminal in WebShop
        return float(resp.get("reward", 0.0)) > 0.0
    except Exception:
        return False


# --------------------------------
# Stateless reconstruction (replay)
# --------------------------------
def replay(goal_idx: int, actions: List[str], observation_mode: str = "text") -> Dict[str, Any]:
    """
    Stateless reconstruction of a page by deterministically starting at goal_idx
    and replaying `actions` in order. The server cleans up its internal per-session
    state, so this does not grow memory.

    Returns a snapshot:
      {
        "message": "replay_ok",
        "session_id": "<temp_session_id>",   # informational only
        "goal_idx": int,
        "steps_replayed": int,
        "stopped_early": bool,
        "reward": float,
        "done": bool,
        "instruction_text": str,
        "url": str,
        "observation": str,
        "available_actions": {...}
      }
    """
    payload = {
        "goal_idx": int(goal_idx),
        "actions": list(actions),
        "observation_mode": observation_mode,
    }
    return _post("replay", payload)


# -------------------------
# Legacy compatibility shim
# -------------------------
# If your older code imports `webshop_interaction` with the functions below, you can:
#   - either rename this file to `webshop_interaction.py`, or
#   - add `from webshop_interactor import *` inside your old module.
# We expose the same function names for convenience.

def start(goal_idx: Optional[int] = None, observation_mode: str = "text") -> Dict[str, Any]:
    """Alias for initialize_goal (kept for readability in some callers)."""
    return initialize_goal(goal_idx=goal_idx, observation_mode=observation_mode)


def step(session_id: str, action: str) -> Dict[str, Any]:
    """Alias for take_step."""
    return take_step(session_id, action)
