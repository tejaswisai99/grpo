# mcts_replay.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
import hashlib, math

import webshop_interaction as ws

# Proposer callback: (instr, obs, available_actions_dict, history_json, step_t) -> {"1":{plan,thought,env},...}
ProposeFn = Callable[[str, str, dict, List[dict], int], Dict[str, dict]]

def _hash_key(goal_idx: int, url: str, step_t: int) -> str:
    h = hashlib.sha256()
    h.update(str(goal_idx).encode()); h.update(b"|"); h.update(url.encode()); h.update(b"|"); h.update(str(step_t).encode())
    return h.hexdigest()

@dataclass
class EdgeStat:
    N: int = 0
    W: float = 0.0
    P: float = 0.25  # uniform prior by default

    @property
    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N

@dataclass
class Node:
    key: str
    goal_idx: int
    url: str
    instr: str
    obs: str
    avail: dict
    step_t: int
    expanded: bool = False
    actions: List[dict] = field(default_factory=list)  # [{"plan","thought","env"}...]
    edges: Dict[str, EdgeStat] = field(default_factory=dict)  # env -> stats

class MCTSReplay:
    """
    PUCT-style MCTS using *stateless* replays.
    At a root (goal_idx, prefix), we run K simulations. Each simulation reconstructs
    states via ws.replay(goal_idx, prefix_at_depth) and stops at terminal or depth cap.
    After K sims, we execute argmax_N at the root *once* on the live session.
    """
    def __init__(self, propose_fn: ProposeFn, c_puct: float = 1.2, sims_per_root: int = 32, max_depth: int = 12):
        self.propose_fn = propose_fn
        self.c_puct = float(c_puct)
        self.sims = int(sims_per_root)
        self.max_depth = int(max_depth)
        self.nodes: Dict[str, Node] = {}
        # per-root cache of replays: tuple(prefix) -> snapshot
        self._cache: Dict[Tuple[int, Tuple[str, ...]], dict] = {}

    # ---------- Core helpers ----------
    def _get_state(self, goal_idx: int, prefix: List[str]) -> dict:
        key = (goal_idx, tuple(prefix))
        if key in self._cache:
            return self._cache[key]
        snap = ws.replay(goal_idx=goal_idx, actions=prefix, observation_mode="text")
        self._cache[key] = snap
        return snap

    def _expand_if_needed(self, node: Node, history_envs: List[str]):
        if node.expanded:
            return
        # Build a lightweight history for the proposer (just env strings)
        hist = [{"env": e} for e in history_envs]
        cands = self.propose_fn(node.instr, node.obs, node.avail, hist, node.step_t)
        # ensure fixed order "1".."4"
        arr = []
        for k in ("1", "2", "3", "4"):
            if k in cands:
                arr.append(cands[k])
        # Filter invalid env strings against available_actions (safety)
        valid = []
        has_search = bool(node.avail.get("has_search_bar"))
        clickables = set(map(str, node.avail.get("clickables", [])))
        for c in arr:
            env = c.get("env", "")
            if env.startswith("search["):
                if has_search: valid.append(c)
            elif env.startswith("click[") and env.endswith("]"):
                cid = env[6:-1]
                if cid in clickables: valid.append(c)
        # Fallback if LLM gave nothing valid: pick up to 4 from clickables
        if not valid:
            picks = []
            # prefer useful affordances if present
            priority = ["buy now", "next >", "back to search"]
            for p in priority:
                if p in clickables: picks.append({"plan":"fallback","thought":"fallback","env":f"click[{p}]"})
            # fill with first few remaining clickables
            for cid in clickables:
                if len(picks) >= 4: break
                if cid not in [e["env"][6:-1] for e in picks if e["env"].startswith("click[")]:
                    picks.append({"plan":"fallback","thought":"fallback","env":f"click[{cid}]"})
            valid = picks[:4]
        node.actions = valid[:4]
        # uniform priors
        node.edges = {a["env"]: EdgeStat(P=1.0/len(node.actions)) for a in node.actions}
        node.expanded = True

    def _select_env(self, node: Node) -> str:
        total_N = sum(s.N for s in node.edges.values())
        best_env, best_score = None, -1e18
        for env, s in node.edges.items():
            u = self.c_puct * s.P * math.sqrt(max(1, total_N)) / (1 + s.N)
            score = s.Q + u
            if score > best_score:
                best_score, best_env = score, env
        assert best_env is not None
        return best_env

    def _node_from_snap(self, goal_idx: int, snap: dict, step_t: int) -> Node:
        key = _hash_key(goal_idx, snap["url"], step_t)
        node = self.nodes.get(key)
        if node is None:
            node = Node(
                key=key,
                goal_idx=goal_idx,
                url=snap["url"],
                instr=snap["instruction_text"],
                obs=snap["observation"],
                avail=snap["available_actions"],
                step_t=step_t
            )
            self.nodes[key] = node
        return node

    def _backprop(self, visited: List[Tuple[Node, str]], R: float):
        for node, env in visited:
            s = node.edges[env]
            s.N += 1
            s.W += R  # no discount (end-only)
            # Q is derived as W/N

    # ---------- Public API ----------
    def plan_at_root(self, goal_idx: int, prefix_envs: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Run K simulations from root defined by (goal_idx, prefix_envs).
        Returns:
          chosen_env (str), debug_info (dict with root edges {env:{N,Q}} and candidate list).
        """
        self._cache.clear()  # per-root cache
        # reconstruct current root state
        root_snap = self._get_state(goal_idx, prefix_envs)
        root_node = self._node_from_snap(goal_idx, root_snap, len(prefix_envs))
        self._expand_if_needed(root_node, prefix_envs)

        # run simulations
        for _ in range(self.sims):
            # local path (node, env_taken) from this root
            visited: List[Tuple[Node, str]] = []
            # local prefix grows as we go deeper in this simulation
            local_prefix = list(prefix_envs)
            depth = 0
            snap = root_snap
            node = root_node

            while True:
                # terminal?
                if ws.is_done({"done": snap.get("done", False), "reward": snap.get("reward", 0.0)}):
                    R = float(snap.get("reward", 0.0))
                    self._backprop(visited, R)
                    break
                if depth >= self.max_depth:
                    self._backprop(visited, 0.0)
                    break

                # ensure node is expanded
                self._expand_if_needed(node, local_prefix)

                # select an edge and apply
                env = self._select_env(node)
                visited.append((node, env))
                local_prefix.append(env)
                # get next state by stateless replay
                snap = self._get_state(goal_idx, local_prefix)
                # next node
                node = self._node_from_snap(goal_idx, snap, len(local_prefix))
                depth += 1

        # pick argmax-N at root
        best_env = max(root_node.edges.items(), key=lambda kv: kv[1].N)[0]
        debug = {
            "root_candidates": root_node.actions,
            "root_edges": {env: {"N": st.N, "Q": (0.0 if st.N == 0 else st.W / st.N)} for env, st in root_node.edges.items()},
            "root_url": root_node.url,
        }
        return best_env, debug
