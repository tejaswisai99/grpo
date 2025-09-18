# grpo_dpo_targets.py
import os, glob, json, math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# -------- Tunables --------
INPUT_DIR   = "trajs"
OUT_GRPO    = "processed_grpo"
OUT_DPO     = "processed_dpo"
ALPHA_S     = 0.25     # S = R_max + ALPHA_S*Q
GRPO_METRICS= ("N", "Q", "UCT", "R_max", "S")  # which metrics to output z-scores for
DPO_METRIC  = "S"      # which metric to rank pairs for DPO
MARGIN      = 0.05     # min margin to emit a pair
EPS_STD     = 1e-8     # guard for std

# -------- Utils --------
def starts_with_prefix(seq: List[str], prefix: List[str]) -> bool:
    return len(seq) >= len(prefix) and seq[:len(prefix)] == prefix

def rmax_lmin_for_root(all_harvested: List[dict], prefix_envs: List[str]) -> Dict[str, Tuple[float, Optional[int]]]:
    out: Dict[str, Tuple[float, Optional[int]]] = {}
    for r in all_harvested:
        env_path: List[str] = r.get("env_path", [])
        if not starts_with_prefix(env_path, prefix_envs):
            continue
        if len(env_path) <= len(prefix_envs):
            continue
        first = env_path[len(prefix_envs)]
        rew = float(r.get("reward", 0.0))
        L   = int(r.get("length", len(env_path) - len(prefix_envs)))
        prev_R, prev_L = out.get(first, (0.0, None))
        best_R = max(prev_R, rew)
        best_L = L if prev_L is None else min(prev_L, L)
        out[first] = (best_R, best_L)
    return out

def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(v)

def zscores(xs: List[float]) -> Tuple[float, float, List[float]]:
    m, s = mean_std(xs)
    s_eff = s if s > EPS_STD else EPS_STD
    return m, s, [(x - m) / s_eff for x in xs]

# -------- Data holders --------
@dataclass
class Candidate:
    plan: str
    thought: str
    expl: str
    env: str
    N: float
    Q: float
    U: float
    UCB: float
    P: float
    UCT: float
    R_max: float
    L_min: Optional[int]
    S: float   # derived: R_max + ALPHA_S * Q

@dataclass
class GRPOFrame:
    goal_idx: int
    step: int
    instruction: str
    observation: str
    available_actions: dict
    prefix_envs: List[str]
    candidates: List[Candidate]
    stats: Dict[str, Dict[str, object]]   # per-metric: {raw, mean, std, z}
    meta: dict

# DPO pair record: one preference per line
@dataclass
class DPOPair:
    goal_idx: int
    step: int
    instruction: str
    observation: str
    available_actions: dict
    prefix_envs: List[str]
    metric: str
    margin: float
    pos: dict   # full candidate 4-tuple + scores for pos
    neg: dict   # full candidate 4-tuple + scores for neg
    meta: dict

def process_goal(path: str) -> Tuple[List[GRPOFrame], List[DPOPair]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    goal_idx = int(data.get("goal_idx", -1))
    frames = data.get("frames", [])
    all_harvested = data.get("all_harvested", [])

    instr_global = ""
    if frames:
        instr_global = frames[0].get("instruction", "")
    instr_global = instr_global or data.get("instruction", "")

    grpo_frames: List[GRPOFrame] = []
    dpo_pairs: List[DPOPair] = []

    for fr in frames:
        step = int(fr.get("step", 0))
        instr = fr.get("instruction", instr_global)
        obs   = fr.get("observation", "")
        avail = fr.get("available_actions", {})
        hist  = fr.get("history_tuples", [])
        prefix = [h.get("env","") for h in hist if isinstance(h, dict) and "env" in h]

        rmap = rmax_lmin_for_root(all_harvested, prefix)

        # Build candidate rows with derived S
        cands: List[Candidate] = []
        for c in fr.get("proposed", {}).get("candidates", []):
            env = c.get("env","")
            Rmax, Lmin = rmap.get(env, (0.0, None))
            Q = float(c.get("Q", 0.0))
            row = Candidate(
                plan=c.get("plan",""), thought=c.get("thought",""),
                expl=c.get("expl",""), env=env,
                N=float(c.get("N",0.0)), Q=Q,
                U=float(c.get("U",0.0)), UCB=float(c.get("UCB",0.0)),
                P=float(c.get("P",0.0)), UCT=float(c.get("UCT",0.0)),
                R_max=Rmax, L_min=Lmin, S=(Rmax + ALPHA_S*Q)
            )
            cands.append(row)

        # GRPO group stats
        stats: Dict[str, Dict[str, object]] = {}
        vecs = {
            "N":  [c.N for c in cands],
            "Q":  [c.Q for c in cands],
            "UCT":[c.UCT for c in cands],
            "R_max":[c.R_max for c in cands],
            "S":  [c.S for c in cands],
        }
        for name in GRPO_METRICS:
            xs = vecs[name]
            m, s, z = zscores(xs)
            stats[name] = {"raw": xs, "mean": m, "std": s, "z": z}

        grpo_frames.append(GRPOFrame(
            goal_idx=goal_idx, step=step, instruction=instr, observation=obs,
            available_actions=avail, prefix_envs=prefix, candidates=cands,
            stats=stats,
            meta={
                "status": data.get("status",""),
                "final_reward": data.get("final_reward", 0.0),
                "root_total_N": fr.get("proposed",{}).get("total_N", None),
                "sims_done": fr.get("sims_done", None),
                "reward1_found": fr.get("reward1_found", None)
            }
        ))

        # DPO pairs from the chosen metric
        scores = vecs[DPO_METRIC]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                margin = scores[i] - scores[j]
                if margin >= MARGIN:
                    pos = cands[i]; neg = cands[j]
                    dpo_pairs.append(DPOPair(
                        goal_idx=goal_idx, step=step, instruction=instr, observation=obs,
                        available_actions=avail, prefix_envs=prefix,
                        metric=DPO_METRIC, margin=float(margin),
                        pos={
                            "plan":pos.plan, "thought":pos.thought, "expl":pos.expl, "env":pos.env,
                            "N":pos.N, "Q":pos.Q, "UCT":pos.UCT, "R_max":pos.R_max, "S":pos.S
                        },
                        neg={
                            "plan":neg.plan, "thought":neg.thought, "expl":neg.expl, "env":neg.env,
                            "N":neg.N, "Q":neg.Q, "UCT":neg.UCT, "R_max":neg.R_max, "S":neg.S
                        },
                        meta={"margin": float(margin)}
                    ))

    return grpo_frames, dpo_pairs

def main():
    os.makedirs(OUT_GRPO, exist_ok=True)
    os.makedirs(OUT_DPO, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "goal_*.json")))
    global_pairs_path = os.path.join(OUT_DPO, "dpo_pairs_all.jsonl")
    with open(global_pairs_path, "w", encoding="utf-8") as global_w:
        total_pairs = 0
        for path in files:
            grpo_frames, dpo_pairs = process_goal(path)
            # write per-goal GRPO
            goal_idx = grpo_frames[0].goal_idx if grpo_frames else -1
            out_grpo = os.path.join(OUT_GRPO, f"goal_{goal_idx}_grpo.jsonl")
            with open(out_grpo, "w", encoding="utf-8") as w:
                for fr in grpo_frames:
                    obj = asdict(fr)
                    obj["candidates"] = [asdict(c) for c in fr.candidates]
                    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            # write per-goal DPO pairs
            out_dpo = os.path.join(OUT_DPO, f"goal_{goal_idx}_dpo.jsonl")
            with open(out_dpo, "w", encoding="utf-8") as w:
                for p in dpo_pairs:
                    w.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
            # also append to global pool
            for p in dpo_pairs:
                global_w.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
            total_pairs += len(dpo_pairs)
            print(f"[written] GRPO={out_grpo} ({len(grpo_frames)} frames), DPO={out_dpo} ({len(dpo_pairs)} pairs)")
        print(f"[global] DPO pairs written to {global_pairs_path} (total {total_pairs})")

if __name__ == "__main__":
    main()
