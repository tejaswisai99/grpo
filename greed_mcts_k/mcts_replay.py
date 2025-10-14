# mcts_replay.py

##Q is the emprirical
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Any, Optional
import hashlib, math

import webshop_interaction as ws

# Proposer: instruction, obs, avail, FULL composite history, step_t -> {"1": {"plan","thought","env"}, ...}
ProposeFn = Callable[[str, str, dict, List[dict], int], Dict[str, dict]]

# Explainer: (instruction, prev_obs, plan, thought, env, curr_obs) -> expl string
ExplainFn = Callable[[str, str, str, str, str, str], str]

def _hash_key(goal_idx: int, url: str, step_t: int) -> str:
    h = hashlib.sha256()
    h.update(str(goal_idx).encode()); h.update(b"|"); h.update(url.encode()); h.update(b"|"); h.update(str(step_t).encode())
    return h.hexdigest()

@dataclass
class EdgeStat:
    N: int = 0
    W: float = 0.0
    P: float = 0.25
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
    # Every action is a 4-tuple
    actions: List[dict] = field(default_factory=list)  # [{"plan","thought","env","expl"}...]
    edges: Dict[str, EdgeStat] = field(default_factory=dict)  # env -> stats

@dataclass
class Rollout:
    env_path: List[str]              # from root (env strings only)
    actions_full: List[dict]         # full 4-tuples along this path (root-forward)
    reward: float
    length: int
    terminal: bool

class MCTSReplay:
    """
    PUCT-like MCTS with stateless replays and full 4-tuple actions.
    - Propose (plan,thought,env), simulate next state, then call explainer -> store (plan,thought,env,expl).
    - During sims we pass ONLY envs to ws.replay, but history we maintain is full 4-tuples.
    - EARLY-STOP: if any terminal rollout with reward == 1.0 is found, stop whole search.
      Otherwise stop when at least target_good_paths (≥3) terminal high-quality paths are harvested,
      then choose highest-reward trajectory.
    - Node snapshots for listwise CE training: for each expanded node we export:
        {"state": {...}, "actions_available": {...}, "history": [full 4-tuples], "children": [...]}
    """
    def __init__(
        self,
        propose_fn: ProposeFn,
        explain_fn: ExplainFn,
        c_puct: float = 1.2,
        sims_per_root: int = 32,
        max_depth: int = 12,
        # Harvest policy (root-level)
        require_reward1: bool = True,
        target_high_paths: int = 3,        # at least 3 if reward-1 not found
        min_reward: float = 0.8,
        path_depth_cap: int = 8,
        # Safety cap on total nodes
        max_nodes_per_tree: int = 32,
        uct_c_exp: float = 1.0,   # Agent-Q exploration constant for UCT logging
    ):
        self.propose_fn = propose_fn
        self.explain_fn = explain_fn
        self.c_puct = float(c_puct)
        self.sims = int(sims_per_root)
        self.max_depth = int(max_depth)
        self.uct_c_exp = float(uct_c_exp)
        self.require_reward1 = bool(require_reward1)
        self.target_high_paths = int(target_high_paths)
        self.min_reward = float(min_reward)
        self.path_depth_cap = int(path_depth_cap)
        self.max_nodes_per_tree = int(max_nodes_per_tree)

        self.nodes: Dict[str, Node] = {}
        # per-root caches
        self._cache: Dict[Tuple[int, Tuple[str, ...]], dict] = {}
        self._rollouts: Dict[Tuple[str, ...], Rollout] = {}  # unique by env_path tuple

    # ---------- Stateless env ----------
    def _get_state(self, goal_idx: int, prefix_envs: List[str]) -> dict:
        key = (goal_idx, tuple(prefix_envs))
        if key in self._cache:
            return self._cache[key]
        snap = ws.replay(goal_idx=goal_idx, actions=prefix_envs, observation_mode="text")
        self._cache[key] = snap
        return snap

    # ---------- Expansion (produce 4-tuples) ----------
    def _expand_if_needed(
        self,
        node: Node,
        goal_idx: int,
        local_prefix_envs: List[str],
        local_hist_full: List[dict],
    ):
        if node.expanded:
            return

        # 1) Propose triples on full composite history
        cands = self.propose_fn(node.instr, node.obs, node.avail, local_hist_full, node.step_t)
        ordered: List[dict] = [cands[k] for k in ("1","2","3","4") if k in cands]

        # 2) Validate against affordances
        has_search = bool(node.avail.get("has_search_bar"))
        clickables = set(map(str, node.avail.get("clickables", [])))
        triples: List[dict] = []
        for c in ordered:
            env = c.get("env","")
            if env.startswith("search["):
                if has_search: triples.append(c)
            elif env.startswith("click[") and env.endswith("]"):
                cid = env[6:-1]
                if cid in clickables: triples.append(c)

        if not triples:
            # Fallback: synthesize 3-tuples from clickables
            picks = []
            for p in ["buy now","next >","back to search"]:
                if p in clickables: picks.append({"plan":"fallback","thought":"fallback","env":f"click[{p}]"})
            for cid in clickables:
                if len(picks) >= 4: break
                if cid not in [e["env"][6:-1] for e in picks if e["env"].startswith("click[")]:
                    picks.append({"plan":"fallback","thought":"fallback","env":f"click[{cid}]"})
            triples = picks[:4]

        # 3) Complete to 4-tuples (simulate + explain)
        node.actions = []
        node.edges = {}
        for t in triples:
            plan   = t.get("plan","n/a")
            thought= t.get("thought","n/a")
            env    = t.get("env","")

            next_prefix = local_prefix_envs + [env]
            next_snap   = self._get_state(goal_idx, next_prefix)
            curr_obs    = next_snap.get("observation","")

            try:
                expl = self.explain_fn(
                    node.instr, node.obs,
                    plan, thought, env, curr_obs
                ) or ""
            except Exception:
                expl = ""

            node.actions.append({"plan":plan, "thought":thought, "env":env, "expl":expl})
            # uniform prior among valid actions
            self.node_edge(node, env).P = 1.0 / max(1, len(triples))

        node.expanded = True

    def node_edge(self, node: Node, env: str) -> EdgeStat:
        es = node.edges.get(env)
        if es is None:
            es = EdgeStat()
            node.edges[env] = es
        return es

    # ---------- Selection ----------
    def _select_env(self, node: Node) -> str:
        total_N = sum(s.N for s in node.edges.values())
        best_env, best_score = None, -1e18
        for env, s in node.edges.items():
            u = self.c_puct * s.P * math.sqrt(max(1, total_N)) / (1 + s.N)
            sc = s.Q + u
            if sc > best_score:
                best_score, best_env = sc, env
        assert best_env is not None
        return best_env

    # ---------- Node creation ----------
    def _node_from_snap(self, goal_idx: int, snap: dict, step_t: int) -> Node:
        key = _hash_key(goal_idx, snap["url"], step_t)
        node = self.nodes.get(key)
        if node is None:
            # Respect node cap
            if len(self.nodes) >= self.max_nodes_per_tree:
                # Return a lightweight placeholder over the same key space:
                node = Node(
                    key=key, goal_idx=goal_idx, url=snap["url"],
                    instr=snap["instruction_text"], obs=snap["observation"],
                    avail=snap["available_actions"], step_t=step_t, expanded=True
                )
                self.nodes[key] = node
                return node
            node = Node(
                key=key, goal_idx=goal_idx, url=snap["url"],
                instr=snap["instruction_text"], obs=snap["observation"],
                avail=snap["available_actions"], step_t=step_t
            )
            self.nodes[key] = node
        return node

    # ---------- Backup ----------
    def _backprop(self, visited: List[Tuple[Node, str]], R: float):
        for node, env in visited:
            s = node.edges[env]
            s.N += 1
            s.W += R  # undiscounted end reward

    # ---------- Child dump for listwise CE ----------
    def _child_dump(self, node: Node, total_N: int) -> List[dict]:
        out = []
        log_parent = math.log(max(1, total_N))
        for a in node.actions:
            env = a["env"]
            es = node.edges.get(env, EdgeStat())
            Q = 0.0 if es.N == 0 else es.W / es.N
            U = self.c_puct * es.P * math.sqrt(max(1, total_N)) / (1 + es.N)
            UCB = Q + U
            UCT = Q + self.uct_c_exp * math.sqrt(log_parent / (1 + es.N))
            out.append({
                "action": a,                          # {plan, thought, env, expl}
                "stats":  {"N": es.N, "Q": Q, "P": es.P, "U": U, "UCB": UCB, "UCT": UCT}
            })
        return out

    # ---------- Public API: greedy one-shot with node snapshots ----------
    def search_greedy_full_trajectory(
        self,
        goal_idx: int,
        prefix_envs: List[str],
        exec_history_records: List[dict],  # already executed full 4-tuples (may be empty)
    ) -> Tuple[Optional[Rollout], List[dict], List[dict], Dict[str, Any]]:
        """
        Run sims from the root. EARLY-STOP the *entire* search as soon as a terminal
        reward==1 rollout is found. Otherwise, stop when at least target_high_paths
        terminal rollouts with reward >= min_reward and length <= path_depth_cap are collected.
        Return:
          (best_rollout_or_None, node_snapshots_all, node_snapshots_winning, debug)

        Each node snapshot has exactly the listwise CE shape:
          {
            "state": {"url", "observation"},
            "actions_available": {...},
            "history": [ {plan,thought,env,expl}, ... ],  # full tuples up to this node
            "children": [
               {"action": {plan,thought,env,expl}, "stats": {N,Q,P,U,UCB,UCT}},
               ...
            ]
          }
        """
        # reset per-root caches
        self._cache.clear()
        self._rollouts.clear()

        node_snapshots_all: List[dict] = []

        root_snap = self._get_state(goal_idx, prefix_envs)
        root_node = self._node_from_snap(goal_idx, root_snap, len(prefix_envs))
        self._expand_if_needed(root_node, goal_idx, prefix_envs, exec_history_records)

        # snapshot root with full tuple history (exec_history_records)
        total_N_root = sum(es.N for es in root_node.edges.values())
        node_snapshots_all.append({
            "state": {"url": root_node.url, "observation": root_node.obs},
            "actions_available": root_node.avail,
            "history": list(exec_history_records),                   # FULL 4-tuples so far
            "children": self._child_dump(root_node, total_N_root)
        })

        reward1_found = False
        sims_done = 0
        high_paths = 0

        for _ in range(self.sims):
            sims_done += 1
            if reward1_found:
                break
            if len(self.nodes) >= self.max_nodes_per_tree:
                break
            if (not self.require_reward1) and (high_paths >= self.target_high_paths):
                break  # if user disabled reward1 requirement and quota met
            # Even when require_reward1=True, we still want to be able to stop if no reward1
            # was found but we already have enough high-quality paths; defer this decision until after sims,
            # unless node cap triggers earlier.

            visited: List[Tuple[Node, str]] = []
            local_prefix_envs = list(prefix_envs)              # env-only list
            local_hist_full   = list(exec_history_records)     # FULL 4-tuples

            snap = root_snap
            node = root_node
            depth = 0
            terminal = False

            while True:
                # stop on terminal
                if ws.is_done({"done": snap.get("done", False), "reward": snap.get("reward", 0.0)}):
                    terminal = True
                    R = float(snap.get("reward", 0.0))
                    self._backprop(visited, R)
                    break
                # depth / cap
                if depth >= self.max_depth or len(self.nodes) >= self.max_nodes_per_tree:
                    self._backprop(visited, 0.0)
                    R = 0.0
                    break

                self._expand_if_needed(node, goal_idx, local_prefix_envs, local_hist_full)

                # snapshot this node (post-expansion) with FULL 4-tuple history
                total_N = sum(es.N for es in node.edges.values())
                node_snapshots_all.append({
                    "state": {"url": node.url, "observation": node.obs},
                    "actions_available": node.avail,
                    "history": list(local_hist_full),          # FULL 4-tuples to reach this node
                    "children": self._child_dump(node, total_N)
                })

                env = self._select_env(node)
                visited.append((node, env))
                local_prefix_envs.append(env)

                # append chosen full 4-tuple to composite history
                chosen = next((a for a in node.actions if a.get("env") == env), None)
                if chosen is None:
                    chosen = {"plan":"n/a","thought":"n/a","env":env,"expl":""}
                local_hist_full.append(chosen)

                snap = self._get_state(goal_idx, local_prefix_envs)
                node = self._node_from_snap(goal_idx, snap, len(local_prefix_envs))
                depth += 1

            # record rollout
            path_key = tuple(local_prefix_envs)
            if path_key not in self._rollouts:
                self._rollouts[path_key] = Rollout(
                    env_path=list(local_prefix_envs),
                    actions_full=list(local_hist_full[len(exec_history_records):]),  # only from root forward
                    reward=float(snap.get("reward", 0.0)),
                    length=len(local_prefix_envs) - len(prefix_envs),
                    terminal=terminal
                )
                if terminal:
                    if self._rollouts[path_key].reward >= self.min_reward and self._rollouts[path_key].length <= self.path_depth_cap:
                        high_paths += 1
                    if self._rollouts[path_key].reward >= 0.999:
                        reward1_found = True
                        break  # EARLY STOP WHOLE SEARCH

        # If reward1 not found, but we have ≥ target_high_paths high-quality paths, we can stop now.
        # (If sims exhausted earlier due to caps, we proceed with best-available anyway.)
        # Choose the single trajectory to return.
        best = None
        if reward1_found:
            for r in self._rollouts.values():
                if r.terminal and r.reward >= 0.999:
                    if best is None or r.length < best.length:
                        best = r
        else:
            # prefer highest reward; tie-break on shortest length
            # (works whether or not high_paths reached target_high_paths)
            for r in self._rollouts.values():
                if best is None or r.reward > best.reward or (r.reward == best.reward and r.length < best.length):
                    best = r

        # Build node_snapshots for nodes that lie on the winning trajectory only (for listwise CE focus).
        node_snapshots_winning: List[dict] = []
        if best is not None:
            # We match by env prefixes and reconstruct a running FULL 4-tuple history prefix.
            # We will walk from the root forward and pick the node snapshots whose history length matches.
            def env_seq(hist_full: List[dict]) -> List[str]:
                return [d.get("env","") for d in hist_full]

            winning_envs = best.env_path  # env-only path from root forward
            # We include all nodes whose 'history' env sequence is a prefix of winning_envs
            for snap in node_snapshots_all:
                hist_full = snap.get("history", [])
                if env_seq(hist_full) == winning_envs[:len(hist_full)]:
                    node_snapshots_winning.append(snap)

        debug = {
            "sims_done": sims_done,
            "node_count": len(self.nodes),
            "node_cap": self.max_nodes_per_tree,
            "reward1_found": reward1_found,
            "paths_found": len(self._rollouts),
            "high_paths": high_paths
        }
        return best, node_snapshots_all, node_snapshots_winning, debug

    # ---------- Data export helpers ----------
    def harvested_rollouts(self) -> List[Rollout]:
        return list(self._rollouts.values())

    def export_grpo_trajectories(self, instruction: str) -> List[dict]:
        out = []
        for r in self.harvested_rollouts():
            out.append({
                "instruction": instruction,
                "actions": r.actions_full,         # [{plan,thought,env,expl}, ...]
                "env_path": r.env_path,            # ["search[...]","click[...]", ...]
                "reward": r.reward,
                "length": r.length,
                "terminal": r.terminal
            })
        return out

    def export_node_scores(self) -> Dict[str, Dict[str, Any]]:
        dump: Dict[str, Dict[str, Any]] = {}
        for k, n in self.nodes.items():
            N_parent = sum(es.N for es in n.edges.values())
            log_parent = math.log(max(1, N_parent))
            edges_dump = {}
            for env, es in n.edges.items():
                Q = 0.0 if es.N == 0 else es.W / es.N
                UCT = Q + self.uct_c_exp * math.sqrt(log_parent / (1 + es.N))
                edges_dump[env] = {
                    "N": es.N,
                    "W": es.W,
                    "Q": Q,
                    "P": es.P,
                    "UCT": UCT
                }
            dump[k] = {
                "url": n.url,
                "step_t": n.step_t,
                "edges": edges_dump,
                "actions": n.actions,
            }
        return dump
