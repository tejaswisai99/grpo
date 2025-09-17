# mcts.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List, Tuple
import hashlib
import math
import time

# Types for callbacks (kept minimal to avoid circular deps)
ProposeFn = Callable[[str, str, dict, List[dict], int], Dict[str, dict]]  # returns 4 candidates dict[str]->{"plan","thought","env"}
ExplainFn = Callable[[str, str, str, str, str, str], dict]  # optional, can be no-op
UCB_C = 1.4

def _state_key(instruction: str, observation: str, step_counter: int) -> str:
    # Compact hash for state identity; observation can be long, but sha256 is stable.
    h = hashlib.sha256()
    # Use only a slice of observation to keep identical pages mapping; tweak if needed.
    h.update(instruction.encode("utf-8"))
    h.update(str(step_counter).encode("utf-8"))
    # Using whole observation is fine, sha handles long strings.
    h.update(observation.encode("utf-8"))
    return h.hexdigest()

@dataclass
class ActionStat:
    plan: str
    thought: str
    env: str
    prior_rank: int  # 1..4 (1 is top)
    N: int = 0
    Q: float = 0.0  # average return

@dataclass
class Node:
    instruction: str
    observation: str
    available_actions: dict
    step_counter: int
    history: List[dict]
    key: str
    actions: Dict[str, ActionStat] = field(default_factory=dict)  # action_key -> ActionStat
    visits: int = 0

class MCTS:
    """
    Lightweight bandit-style MCTS:
      - No environment cloning; we expand on-demand with 4 proposals.
      - We select one action to *actually* execute in the real env.
      - At episode end, we backpropagate the final reward over the executed path.
    This gives you delayed-reward credit assignment without circular imports or env forking.
    """
    def __init__(self, propose_fn: ProposeFn, explain_fn: Optional[ExplainFn] = None, ucb_c: float = UCB_C):
        self.propose_fn = propose_fn
        self.explain_fn = explain_fn
        self.ucb_c = ucb_c
        self.nodes: Dict[str, Node] = {}           # key -> Node
        self.episode_path: List[Tuple[str, str]] = []  # [(node_key, env), ...] for current episode
        self._last_node_key: Optional[str] = None

    def _get_or_expand(
        self,
        instruction: str,
        observation: str,
        available_actions: dict,
        history: List[dict],
        step_counter: int
    ) -> Node:
        key = _state_key(instruction, observation, step_counter)
        node = self.nodes.get(key)
        if node is not None:
            return node
        # Expand: get 4 candidates from LLM
        proposals = self.propose_fn(instruction, observation, available_actions, history, step_counter)
        # proposals is dict like {"1":{"plan":...,"thought":...,"env":"click[...]"}, ...}
        # Rank prior by key order 1..4
        actions: Dict[str, ActionStat] = {}
        for rank_key in ["1", "2", "3", "4"]:
            cand = proposals[rank_key]
            a = ActionStat(plan=cand["plan"], thought=cand["thought"], env=cand["env"], prior_rank=int(rank_key))
            actions[a.env] = a  # use env string as unique action id at node
        node = Node(
            instruction=instruction,
            observation=observation,
            available_actions=available_actions,
            step_counter=step_counter,
            history=list(history),
            key=key,
            actions=actions,
            visits=0,
        )
        self.nodes[key] = node
        return node

    def select(self, node: Node) -> ActionStat:
        """
        UCB1 over already-known actions.
        Tie-break by better prior_rank.
        """
        node.visits += 1
        total_N = sum(max(1, a.N) for a in node.actions.values())  # avoid log(0)
        best_score = -1e9
        best = None
        for a in node.actions.values():
            # UCB: Q + c * sqrt(log(total) / (1+N))
            exploit = a.Q
            explore = self.ucb_c * math.sqrt(math.log(total_N + 1) / (1 + a.N))
            score = exploit + explore
            # gentle bias toward top-ranked prior when tied
            score += 1e-6 * (5 - a.prior_rank)  # rank 1 gets +4e-6 etc
            if score > best_score:
                best_score = score
                best = a
        assert best is not None
        return best

    def choose_action(
        self,
        instruction: str,
        observation: str,
        available_actions: dict,
        history: List[dict],
        step_counter: int
    ) -> Dict[str, str]:
        """
        Returns {"plan":..., "thought":..., "env":...} for the action to execute.
        Also remembers the node/action to backprop at episode end.
        """
        node = self._get_or_expand(instruction, observation, available_actions, history, step_counter)
        self._last_node_key = node.key
        act = self.select(node)
        # record in episode path; final reward will be backpropped in end_episode()
        self.episode_path.append((node.key, act.env))
        return {"plan": act.plan, "thought": act.thought, "env": act.env}

    def end_episode(self, final_reward: float) -> None:
        """
        Backprop the terminal reward to each (node, action) visited in this episode.
        """
        for node_key, env in self.episode_path:
            node = self.nodes.get(node_key)
            if not node:
                continue
            a = node.actions.get(env)
            if not a:
                continue
            a.N += 1
            # incremental average
            a.Q += (final_reward - a.Q) / float(a.N)
        # reset episode buffer
        self.episode_path.clear()
        self._last_node_key = None
