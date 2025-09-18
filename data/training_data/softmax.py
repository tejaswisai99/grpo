# softmax_targets.py
import os, glob, json, math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# -------- Tunables --------
#
INPUT_DIR  = "trajs"
OUTPUT_DIR = "processed_softmax"
TAU_Q   = 0.2     # temperature for softmax over Q
TAU_S   = 0.2     # temperature for softmax over S
ALPHA_S = 0.25    # S = R_max + ALPHA_S * Q

# -------- Utils --------
def softmax(xs: List[float], tau: float) -> List[float]:
    if not xs:
        return []
    if tau <= 0:
        m = max(xs)
        return [1.0 if x == m else 0.0 for x in xs]
    m = max(xs)
    exps = [math.exp((x - m)/tau) for x in xs]
    s = sum(exps)
    return [z/s for z in exps] if s > 0 else [1.0/len(xs)]*len(xs)

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

# -------- Data holders --------
from dataclasses import dataclass
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

@dataclass
class Record:
    goal_idx: int
    step: int
    instruction: str
    observation: str
    available_actions: dict
    prefix_envs: List[str]
    candidates: List[Candidate]
    pi_N: List[float]
    pi_Q: List[float]
    pi_S: List[float]
    meta: dict

def process_goal(path: str) -> List[Record]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    goal_idx = int(data.get("goal_idx", -1))
    frames = data.get("frames", [])
    all_harvested = data.get("all_harvested", [])
    out: List[Record] = []

    instr_global = ""
    if frames:
        instr_global = frames[0].get("instruction", "")
    instr_global = instr_global or data.get("instruction", "")

    for fr in frames:
        step = int(fr.get("step", 0))
        instr = fr.get("instruction", instr_global)
        obs   = fr.get("observation", "")
        avail = fr.get("available_actions", {})
        hist  = fr.get("history_tuples", [])
        prefix = [h.get("env","") for h in hist if isinstance(h, dict) and "env" in h]

        # harvested outcome per first action
        rmap = rmax_lmin_for_root(all_harvested, prefix)

        cand_rows = []
        counts, Qs, Ss = [], [], []
        for c in fr.get("proposed", {}).get("candidates", []):
            env = c.get("env","")
            Rmax, Lmin = rmap.get(env, (0.0, None))
            row = Candidate(
                plan=c.get("plan",""), thought=c.get("thought",""),
                expl=c.get("expl",""), env=env,
                N=float(c.get("N",0.0)), Q=float(c.get("Q",0.0)),
                U=float(c.get("U",0.0)), UCB=float(c.get("UCB",0.0)),
                P=float(c.get("P",0.0)), UCT=float(c.get("UCT",0.0)),
                R_max=Rmax, L_min=Lmin
            )
            cand_rows.append(row)
            counts.append(row.N); Qs.append(row.Q); Ss.append(Rmax + ALPHA_S*row.Q)

        total_N = sum(counts)
        pi_N = [n/total_N for n in counts] if total_N > 0 else softmax(Qs, TAU_Q)
        pi_Q = softmax(Qs, TAU_Q)
        pi_S = softmax(Ss, TAU_S)

        rec = Record(
            goal_idx=goal_idx, step=step,
            instruction=instr, observation=obs, available_actions=avail,
            prefix_envs=prefix, candidates=cand_rows,
            pi_N=pi_N, pi_Q=pi_Q, pi_S=pi_S,
            meta={
                "status": data.get("status",""),
                "final_reward": data.get("final_reward", 0.0),
                "root_total_N": fr.get("proposed",{}).get("total_N", None),
                "sims_done": fr.get("sims_done", None),
                "reward1_found": fr.get("reward1_found", None)
            }
        )
        out.append(rec)
    return out

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "goal_*.json")))
    for path in files:
        recs = process_goal(path)
        goal = recs[0].goal_idx if recs else -1
        outp = os.path.join(OUTPUT_DIR, f"goal_{goal}_softmax.jsonl")
        with open(outp, "w", encoding="utf-8") as w:
            for r in recs:
                obj = asdict(r)
                obj["candidates"] = [asdict(c) for c in r.candidates]
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[written] {outp} ({len(recs)} frames)")

if __name__ == "__main__":
    main()
