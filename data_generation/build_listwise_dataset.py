#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Listwise dataset builder with auto-τ calibration for ListNet-style softmax CE.

Output schema per line:
{
  "prompt": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."}
  ],
  "candidates": [
    {"content":"{\"plan\":\"...\",\"thought\":\"...\",\"env\":\"...\"}",
     "score":0.625, "rank":0, "meta":{"stats":{...}}}
    ...
  ],
  "targets": {"type":"listnet", "tau": <float>, "probs":[...softmax...]},
  "meta": {"source":"candidates|messages", "step":0, "score_key":"Q"}
}

Two-pass flow:
  Pass-1: compute normalized scores per item, measure average top-2 gap g
  τ_auto = g / log(p/(1-p))
  Pass-2: build rows with probs = softmax(scores_normalized / τ_final)

Usage:
  python build_listwise_dataset.py \
    --input /path/in.jsonl \
    --output /path/out.listwise.jsonl \
    --mode auto \
    --score-key Q \
    --norm minmax \
    --desired-top-prob 0.6 \
    --tau auto \
    --max-k 4 \
    --dedupe-env \
    --min-candidates 2
"""

import json
import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- STATIC PROMPTS (edit if needed) ----------
SYSTEM_PROMPT  = f"""You are an agent-policy generator for a web-navigation RL agent (Agent Q style).
You must output STRICT JSON with exactly **one object** containing exactly three top-level string fields:
  - "plan": a brief, high-level plan for the NEXT FEW steps (1–2 steps max).
  - "thought": a concise internal reasoning for THIS step only (1–2 sentences).
  - "env": the concrete environment action for THIS step in EXACT format:
        either  search[<free-text query>]
        or      click[<ELEMENT_ID>]

Your task will be completed only when we buy the product. i.e. "env" is click[buy now].

HARD CONSTRAINTS:
1) Output MUST be valid JSON, with double-quoted keys and string values. No trailing commas. No markdown. No extra text.
2) "env" MUST be one of:
      - search[...] ONLY IF "has_search_bar" is true in available_actions.
      - click[ID], where ID MUST be EXACTLY one of the provided "clickables".
3) If no suitable click is possible and has_search_bar is false, choose the best available clickable (including "back to search" or "next >").
4) Keep "plan" short and actionable; keep "thought" minimal (no verbose chain-of-thought).
5) NEVER invent element IDs. NEVER include spaces around brackets. Use EXACT casing for IDs.
6) Use the "instruction" and "history" to stay on task and avoid repeating failed actions.
7) Observation is plain text with [SEP] separators between elements; treat it as read-only state.
8) Ensure that the task ends within 12 steps. Step counter will be provided.

GENERAL GUIDELINES:
1) Look for the closest match. Look in the top results returned, and 1-2 next pages.
2) It's always better not to include price constraint in search, as it uses lucene indexer and price is not part of it while building the index.
3) Explore a bit, look at the entire current observation (web page).
4) Observe the title/description and then choose the best product, best variant and press buy now.
5) Price, color, variant details are sometimes in product page. Don't keep going to last pages.
6) Avoid searching after you got a list of products; don't go back to search once you already searched multiple options during MCTS.
7) The end goal is always to buy a product.
"""


USER_TEMPLATE  = f"""
INSTRUCTION:
{{INSTRUCTION}}

OBSERVATION (Current web page): 
{{OBSERVATION}}

AVAILABLE_ACTIONS (JSON):
{{AVAILABLE_ACTIONS_JSON}}

HISTORY (List of previous actions taken):
{{HISTORY_JSON}}

STEP_COUNTER (NUMBER OF STEPS COMPLETE):
{{STEP_COUNTER}}
"""

TAIL  = """
OUTPUT FORMAT:
Fill your candidates in this format-
{"plan": "string","thought": "string","env": "string}
RETURN JSON. NO extra text. Do not put line breaks and escape characters."""

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield line_no, json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}")

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def is_messages_entry(obj: Dict[str, Any]) -> bool:
    return isinstance(obj.get("messages"), list)

def is_candidates_entry(obj: Dict[str, Any]) -> bool:
    return "candidate_actions" in obj

def extract_from_observation(obs: str, key_prefix: str = "Instruction:") -> Optional[str]:
    if not isinstance(obs, str):
        return None
    parts = [p.strip() for p in obs.split("[SEP]")]
    for i, p in enumerate(parts):
        if p.lower().startswith(key_prefix.lower()):
            if i + 1 < len(parts):
                return parts[i + 1].strip()
    return None

def compact_one_line(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if "\n" in s:
            first, rest = s.split("\n", 1)
            if first.lower().startswith("json"):
                s = rest.strip()
    return " ".join(s.split())

def action_to_json_string(act: Any) -> str:
    if isinstance(act, dict):
        return json.dumps({
            "plan":   act.get("plan", ""),
            "thought": act.get("thought", ""),
            "env":    act.get("env", "")
        }, separators=(",", ":"), ensure_ascii=False)
    if isinstance(act, str):
        return act
    return ""

def softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [v / s for v in exps]

def normalize_scores(scores: List[float], method: str, jitter: float = 1e-8) -> List[float]:
    if not scores:
        return []
    if method == "none":
        return scores[:]
    if method == "minmax":
        lo, hi = min(scores), max(scores)
        rng = hi - lo
        if rng <= 0:
            # all equal -> uniform after tiny jitter
            return [0.5 + (i - len(scores)/2)*jitter for i in range(len(scores))]
        return [(s - lo) / (rng + 1e-12) for s in scores]
    if method == "zscore":
        mu = sum(scores) / len(scores)
        var = sum((s - mu) ** 2 for s in scores) / max(1, len(scores)-1)
        sd = math.sqrt(var) + 1e-12
        return [(s - mu) / sd for s in scores]
    raise ValueError(f"Unknown norm method: {method}")

def rank_from_scores(scores: List[float], higher_is_better: bool = True) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=higher_is_better)
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks

def dedupe_by_env(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in cands:
        env = ""
        act = c.get("action")
        if isinstance(act, dict):
            env = str(act.get("env", ""))
        elif isinstance(act, str):
            try:
                d = json.loads(act)
                env = str(d.get("env", ""))
            except Exception:
                env = ""
        key = env.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(c)
    return out

def gather_from_candidates(entry: Dict[str, Any],
                           score_key: str,
                           dedupe_env_flag: bool,
                           max_k: int) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    cands = entry.get("candidate_actions", []) or []
    if dedupe_env_flag:
        cands = dedupe_by_env(cands)
    texts, scores, metas = [], [], []
    for c in cands:
        content = compact_one_line(action_to_json_string(c.get("action")))
        st = c.get("stats", {}) or {}
        score = float(st.get(score_key, 0.0))
        if not content:
            continue
        texts.append(content)
        scores.append(score)
        metas.append({"stats": st, "action_id": c.get("action_id")})
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if max_k > 0:
        order = order[:max_k]
    texts = [texts[i] for i in order]
    scores = [scores[i] for i in order]
    metas  = [metas[i]  for i in order]
    return texts, scores, metas

def gather_from_messages(entry: Dict[str, Any]) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    msgs = entry.get("messages", [])
    win_msgs = [m for m in msgs if m.get("role") == "assistant_w"]
    los_msgs = [m for m in msgs if m.get("role") == "assistant_l"]
    texts, scores, metas = [], [], []
    if win_msgs:
        texts.append(compact_one_line(win_msgs[-1]["content"]))
        scores.append(1.0)
        metas.append({"source":"assistant_w"})
    if los_msgs:
        texts.append(compact_one_line(los_msgs[-1]["content"]))
        scores.append(0.0)
        metas.append({"source":"assistant_l"})
    return texts, scores, metas

def build_user_content(instruction: str,
                       observation: str,
                       available_actions: Dict[str, Any],
                       history: List[Dict[str, Any]],
                       step_counter: int) -> str:
    return USER_TEMPLATE.format(
        INSTRUCTION=instruction,
        OBSERVATION=observation,
        AVAILABLE_ACTIONS_JSON=json.dumps(available_actions, ensure_ascii=False),
        HISTORY_JSON=json.dumps(history, ensure_ascii=False),
        STEP_COUNTER=step_counter
    ) + "\n\n" + TAIL

def make_prompt(system: str, user: str) -> List[Dict[str, str]]:
    return [{"role":"system","content":system},{"role":"user","content":user}]

def pass1_collect_top2_gaps(input_path: Path,
                            mode: str,
                            score_key: str,
                            norm: str,
                            max_k: int,
                            dedupe_env_flag: bool,
                            min_candidates: int) -> Tuple[float, int]:
    """
    Returns (avg_top2_gap, count_items_used) after normalizing per item.
    """
    gaps = []
    used = 0
    for _, obj in read_jsonl(input_path):
        if mode == "candidates" or (mode == "auto" and is_candidates_entry(obj)):
            texts, scores_raw, _ = gather_from_candidates(obj, score_key, dedupe_env_flag, max_k)
        elif mode == "messages" or (mode == "auto" and is_messages_entry(obj)):
            texts, scores_raw, _ = gather_from_messages(obj)
        else:
            continue
        if len(texts) < max(2, min_candidates):
            continue
        scores_norm = normalize_scores(scores_raw, norm)
        # top-2 gap in normalized domain
        order = sorted(range(len(scores_norm)), key=lambda i: scores_norm[i], reverse=True)
        g = scores_norm[order[0]] - scores_norm[order[1]]
        gaps.append(g)
        used += 1
    if not gaps:
        return 0.0, 0
    return sum(gaps)/len(gaps), used

def probs_from_normalized(scores_norm: List[float], tau: float) -> List[float]:
    scaled = [s / max(tau, 1e-8) for s in scores_norm]
    return softmax(scaled)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=["auto","candidates","messages"], default="auto")
    ap.add_argument("--score-key", choices=["Q","UCT","N"], default="Q")
    ap.add_argument("--norm", choices=["minmax","zscore","none"], default="minmax",
                    help="Per-item normalization before softmax")
    ap.add_argument("--desired-top-prob", type=float, default=0.80,
                    help="Desired probability mass for the best candidate (for τ auto)")
    ap.add_argument("--tau", default="auto",
                    help="'auto' to calibrate from data; or provide a float (e.g., 0.15)")
    ap.add_argument("--max-k", type=int, default=0, help="Cap candidates per item (0=no cap)")
    ap.add_argument("--dedupe-env", action="store_true", help="Drop duplicate env candidates (by env string)")
    ap.add_argument("--min-candidates", type=int, default=2, help="Skip items with fewer than N candidates")
    args = ap.parse_args()

    # ---------- PASS 1: auto-τ if requested ----------
    if args.tau == "auto":
        avg_gap, used = pass1_collect_top2_gaps(
            args.input, args.mode, args.score_key, args.norm,
            args.max_k, args.dedupe_env, args.min_candidates
        )
        if used == 0:
            print("[ERROR] No items eligible for τ calibration. Check input/mode/filters.", file=sys.stderr)
            sys.exit(2)
        p = max(min(args.desired_top_prob, 0.9999), 0.5001)  # clamp to (0.5,1)
        # tau ≈ g / ln(p/(1-p))
        denom = math.log(p/(1.0-p))
        tau_auto = avg_gap / max(denom, 1e-8)
        # safety clamps
        tau_auto = max(min(tau_auto, 5.0), 1e-4)
        tau_final = tau_auto
        print(f"[INFO] τ calibration: used {used} items, avg top-2 gap={avg_gap:.6f}, "
              f"desired p={p:.4f} -> τ={tau_final:.6f}")
    else:
        try:
            tau_final = float(args.tau)
        except Exception:
            print("[ERROR] --tau must be 'auto' or a float.", file=sys.stderr)
            sys.exit(2)
        print(f"[INFO] Using fixed τ={tau_final}")

    # ---------- PASS 2: build rows ----------
    out_rows: List[Dict[str, Any]] = []
    total_in, total_out = 0, 0

    for line_no, obj in read_jsonl(args.input):
        total_in += 1
        try:
            # Build prompt (system, user)
            if args.mode == "candidates" or (args.mode == "auto" and is_candidates_entry(obj)):
                instruction = obj.get("instruction") or extract_from_observation(obj.get("observation",""), "Instruction:")
                history     = obj.get("history", [])
                observation = obj.get("observation", "")
                available   = obj.get("actions_available", {})
                step_ctr    = obj.get("step", 0)

                system = SYSTEM_PROMPT
                user   = USER_TEMPLATE.format(
                    INSTRUCTION=instruction or "",
                    OBSERVATION=observation or "",
                    AVAILABLE_ACTIONS_JSON=json.dumps(available or {}, ensure_ascii=False),
                    HISTORY_JSON=json.dumps(history or [], ensure_ascii=False),
                    STEP_COUNTER=step_ctr if isinstance(step_ctr, int) else 0
                ) + "\n\n" + TAIL

                texts, scores_raw, metas = gather_from_candidates(obj, args.score_key, args.dedupe_env, args.max_k)
                if len(texts) < max(2, args.min_candidates):
                    continue

            elif args.mode == "messages" or (args.mode == "auto" and is_messages_entry(obj)):
                msgs = obj.get("messages", [])
                sys_msgs = [m for m in msgs if m.get("role") == "system"]
                usr_msgs = [m for m in msgs if m.get("role") == "user"]
                if not sys_msgs or not usr_msgs:
                    continue
                system = sys_msgs[-1]["content"]
                user   = usr_msgs[-1]["content"]
                texts, scores_raw, metas = gather_from_messages(obj)
                if len(texts) < max(2, args.min_candidates):
                    continue
            else:
                continue

            scores_norm = normalize_scores(scores_raw, args.norm)
            ranks = rank_from_scores(scores_norm, higher_is_better=True)
            # probs with calibrated τ
            scaled = [s / max(tau_final, 1e-8) for s in scores_norm]
            probs = softmax(scaled)

            candidates = []
            for t, s_raw, r, m in zip(texts, scores_raw, ranks, metas):
                candidates.append({"content": t, "score": float(s_raw), "rank": int(r), "meta": m})

            row = {
                "prompt": [{"role":"system","content": system},
                           {"role":"user","content": user}],
                "candidates": candidates,
                "targets": {"type":"listnet", "tau": float(tau_final), "probs": probs},
                "meta": {"source": "candidates" if "candidate_actions" in obj else "messages",
                         "score_key": args.score_key}
            }
            out_rows.append(row)
            total_out += 1

        except Exception as e:
            print(f"[WARN] Skipping line {line_no}: {e}", file=sys.stderr)

    if not out_rows:
        print("[ERROR] No output rows generated. Check filters.", file=sys.stderr)
        sys.exit(2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, out_rows)
    print(f"[OK] Read {total_in} lines, wrote {total_out} listwise items -> {args.output}")
    print(f"[INFO] Final τ written in each row: {tau_final:.6f}; norm={args.norm}; score_key={args.score_key}")

if __name__ == "__main__":
    main()
