# mcts_replay.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Any, Optional
import hashlib, math

import webshop_interaction as ws

ProposeFn = Callable[[str, str, dict, List[dict], int], Dict[str, dict]]

ExplainFn = Callable[[str, str, dict, List[dict], str, str, str, str], str]

def _hash_key(goal_idx: int, url: str, step_t: int) -> str:
    h = hashlib.sha256()
    h.update(str(goal_idx).encode()); h.update(b"|"); h.update(url.encode()); h.update(b"|"); h.update(str(step_t).encode())
    return h.hexdigest()

@dataclass
class EdgeStat:
    N: int = 0
    W: float = 0.0
    P: float = 0.25  # prior
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
    # Each action is a FULL 4-tuple: {"plan","thought","env","expl"}
    actions: List[dict] = field(default_factory=list)
    # Edges keyed by env (env strings are unique affordances at a state)
    edges: Dict[str, EdgeStat] = field(default_factory=dict)

class MCTSReplay:
    """
    PUCT-style MCTS using stateless replays.
    At expansion, candidates are completed to 4-tuples by generating 'expl'
    using the next state's observation.
    """
    def __init__(
        self,
        propose_fn: ProposeFn,
        explain_fn: ExplainFn,
        c_puct: float = 1.2,
        sims_per_root: int = 32,
        max_depth: int = 12
    ):
        self.propose_fn = propose_fn
        self.explain_fn = explain_fn
        self.c_puct = float(c_puct)
        self.sims = int(sims_per_root)
        self.max_depth = int(max_depth)
        self.nodes: Dict[str, Node] = {}
        # per-root cache of stateless snapshots: (goal_idx, tuple(prefix_envs)) -> snap
        self._cache: Dict[Tuple[int, Tuple[str, ...]], dict] = {}

    # ---------- Stateless environment helpers ----------
    def _get_state(self, goal_idx: int, prefix_envs: List[str]) -> dict:
        key = (goal_idx, tuple(prefix_envs))
        if key in self._cache:
            return self._cache[key]
        snap = ws.replay(goal_idx=goal_idx, actions=prefix_envs, observation_mode="text")
        self._cache[key] = snap
        return snap

    # ---------- Proposer/explainer plumbing ----------
    def _expand_if_needed(
        self,
        node: Node,
        goal_idx: int,
        local_prefix_envs: List[str],
        local_hist_full: List[dict],
    ):
        """
        Expand a node by:
          1) proposing 3-tuples using FULL previous history (composite),
          2) for each candidate env, simulate one step to get curr_obs,
          3) call explain_fn and attach 'expl', so actions are 4-tuples.
        """
        if node.expanded:
            return

        # (1) propose 3-tuples conditioned on full history
        cands_map = self.propose_fn(
            node.instr, node.obs, node.avail, local_hist_full, node.step_t
        )

        # Keep fixed order and validate
        ordered: List[dict] = []
        for k in ("1", "2", "3", "4"):
            if k in cands_map:
                ordered.append(cands_map[k])

        has_search = bool(node.avail.get("has_search_bar"))
        clickables = set(map(str, node.avail.get("clickables", [])))

        valid_triplets: List[dict] = []
        for c in ordered:
            env = c.get("env", "")
            if env.startswith("search["):
                if has_search:
                    valid_triplets.append(c)
            elif env.startswith("click[") and env.endswith("]"):
                cid = env[6:-1]
                if cid in clickables:
                    valid_triplets.append(c)

        # Fallback: construct up to 4 clickable fallbacks as 3-tuples
        if not valid_triplets:
            picks = []
            for p in ["buy now", "next >", "back to search"]:
                if p in clickables:
                    picks.append({"plan": "fallback", "thought": "fallback", "env": f"click[{p}]"})
            for cid in clickables:
                if len(picks) >= 4: break
                if cid not in [e["env"][6:-1] for e in picks if e["env"].startswith("click[")]:
                    picks.append({"plan": "fallback", "thought": "fallback", "env": f"click[{cid}]"})
            valid_triplets = picks[:4]

        # (2) and (3) complete to 4-tuples by simulating next state and explaining
        full_actions: List[dict] = []
        edges: Dict[str, EdgeStat] = {}
        for c in valid_triplets:
            plan = c.get("plan", "n/a")
            thought = c.get("thought", "n/a")
            env = c.get("env", "")

            # simulate one step for this candidate to obtain current observation
            next_prefix = local_prefix_envs + [env]
            next_snap = self._get_state(goal_idx, next_prefix)
            curr_obs = next_snap.get("observation", "")

            # compute expl using the provided explain_fn (short, post-hoc)
            try:
                expl = self.explain_fn(
                    node.instr,          # instruction
                    node.obs,            # full composite history so far
                    plan, thought, env,  # chosen triple
                    curr_obs             # current obs after sim step
                ) or ""
            except Exception:
                expl = ""

            full_actions.append({"plan": plan, "thought": thought, "env": env, "expl": expl})
            edges[env] = EdgeStat(P=1.0 / max(1, len(valid_triplets)))

        node.actions = full_actions
        node.edges = edges
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
            s.W += R

    # ---------- Public API ----------
    def plan_at_root(
        self,
        goal_idx: int,
        prefix_envs: List[str],
        exec_history_records: List[dict],  # full 4-tuples for executed steps
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Runs K simulations from the root. Simulations maintain a *composite* local history.
        Returns chosen_env and debug info containing root candidates (all 4 fields).
        """
        self._cache.clear()  # reset stateless-cache per root

        # root snapshot and node
        root_snap = self._get_state(goal_idx, prefix_envs)
        root_node = self._node_from_snap(goal_idx, root_snap, len(prefix_envs))

        # Expand root using full executed history so far
        self._expand_if_needed(root_node, goal_idx, prefix_envs, exec_history_records)

        # K simulations
        for _ in range(self.sims):
            visited: List[Tuple[Node, str]] = []
            local_prefix_envs = list(prefix_envs)
            # full composite history for this simulation path
            local_hist_full = list(exec_history_records)

            snap = root_snap
            node = root_node
            depth = 0

            while True:
                # terminal?
                if ws.is_done({"done": snap.get("done", False), "reward": snap.get("reward", 0.0)}):
                    self._backprop(visited, float(snap.get("reward", 0.0)))
                    break
                if depth >= self.max_depth:
                    self._backprop(visited, 0.0)
                    break

                # Ensure expanded with the full composite history so far
                self._expand_if_needed(node, goal_idx, local_prefix_envs, local_hist_full)

                # Select & advance
                env = self._select_env(node)
                visited.append((node, env))
                local_prefix_envs.append(env)

                # Append the *chosen* full action (4-tuple) to local composite history
                chosen_full = next((a for a in node.actions if a.get("env") == env), None)
                if chosen_full:
                    local_hist_full.append(chosen_full)
                else:
                    # Should not happen; keep history consistent
                    local_hist_full.append({"plan": "n/a", "thought": "n/a", "env": env, "expl": ""})

                # Move to next state and node
                snap = self._get_state(goal_idx, local_prefix_envs)
                node = self._node_from_snap(goal_idx, snap, len(local_prefix_envs))
                depth += 1

        # Act with argmax visit count
        best_env = max(root_node.edges.items(), key=lambda kv: kv[1].N)[0]
        debug = {
            "root_candidates": root_node.actions,  # each has plan/thought/env/expl
            "root_edges": {e: {"N": st.N, "Q": (0.0 if st.N == 0 else st.W / st.N)} for e, st in root_node.edges.items()},
            "root_url": root_node.url,
        }
        return best_env, debug