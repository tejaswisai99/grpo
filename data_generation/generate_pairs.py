#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python generate_pairs.py --input 'D:\iisc_project\grpo\greedy_mcts\trajs\best_trajs.jsonl' --output 'D:\iisc_project\grpo\greedy_mcts\trajs\dpo_train.jsonl' --mode auto
"""
Generate Qwen3-ready DPO JSONL from:
  A) WebShop-like candidate_actions states  -> creates (best vs rest) pairs
  B) messages-style data with assistant_w/l -> converts to (prompt, chosen, rejected)

Output schema (per line):
{
  "prompt":   [ {"role":"system","content":...}, {"role":"user","content":...} ],
  "chosen":   [ {"role":"assistant","content":...} ],
  "rejected": [ {"role":"assistant","content":...} ],
  "meta":     { ... optional provenance ... }
}

Usage:
  python build_dpo_pairs.py \
      --input /path/in.jsonl \
      --output /path/out.dpo.jsonl \
      --mode auto \
      --max_pairs 3

Notes:
- For mode=auto, the script detects by presence of 'candidate_actions' or 'messages'.
- For candidate_actions, you can edit SYSTEM_PROMPT, USER_TEMPLATE, TAIL below.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------- CONSTANTS: Edit if needed ---------

SYSTEM_PROMPT = """You are an agent-policy generator for a web-navigation RL agent (Agent Q style).
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
1) Look for the closest match. Look in the top results returned, and 1–2 next pages.
2) It's always better not to include price constraint in search, as it uses lucene indexer and price is not part of it while building the index.
3) Explore a bit, look at the entire current observation (web page).
4) Observe the title/description and then choose the best product, best variant and press buy now.
5) Price, color, variant details are sometimes in product page. Don't keep going to last pages.
6) Avoid searching after you got a list of products; don't go back to search once you already searched multiple options during MCTS.
7) The end goal is always to buy a product.
"""

USER_TEMPLATE = """INSTRUCTION:
{INSTRUCTION}

OBSERVATION (Current web page): 
{OBSERVATION}

AVAILABLE_ACTIONS (JSON):
{AVAILABLE_ACTIONS_JSON}

HISTORY (List of previous actions taken):
{HISTORY_JSON}

STEP_COUNTER (NUMBER OF STEPS COMPLETE):
{STEP_COUNTER}
"""

TAIL = """OUTPUT FORMAT:
Fill your candidates in this format-
{"plan": "string","thought": "string","env": "string}
RETURN JSON. NO extra text. Do not put line breaks and escape characters."""
# ---------------------------------------------

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

def compact_json_str(s: str) -> str:
    """
    Normalize assistant JSON strings:
      - strip whitespace/fences
      - keep as a single line
    """
    s = s.strip()
    # Remove accidental code fences
    if s.startswith("```"):
        s = s.strip("`").strip()
        # If there is a language tag like ```json, remove the first line
        if "\n" in s:
            first_line, rest = s.split("\n", 1)
            if first_line.lower().startswith("json"):
                s = rest.strip()
    # collapse whitespace-only lines
    return " ".join(s.split())

def validate_candidate_json(s: str) -> Optional[str]:
    """
    If s looks like a JSON object string, ensure it parses.
    Return normalized compacted string if OK, else None.
    """
    t = compact_json_str(s)
    try:
        obj = json.loads(t)
        # Re-serialize to enforce canonical no-whitespace style
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return None

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

def is_messages_entry(obj: Dict[str, Any]) -> bool:
    return isinstance(obj.get("messages"), list)

def is_candidates_entry(obj: Dict[str, Any]) -> bool:
    return "candidate_actions" in obj

def pick_best_index(cands: List[Dict[str, Any]]) -> int:
    """
    Choose best by:
      1) max stats.Q
      2) then max stats.UCT
      3) then max stats.N
      4) else lowest index
    """
    def key_fn(c: Dict[str, Any]) -> Tuple:
        st = c.get("stats", {}) or {}
        # Use None-safe fallbacks
        q   = st.get("Q", float("-inf"))
        uct = st.get("UCT", float("-inf"))
        n   = st.get("N", float("-inf"))
        return (q, uct, n)
    # Get argmax with tie-break by original order
    best_idx = 0
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    for i, c in enumerate(cands):
        k = key_fn(c)
        if k > best_key:
            best_key, best_idx = k, i
    return best_idx

def to_qwen_dpo_line_from_messages(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single 'messages' entry having assistant_w / assistant_l
    into one DPO line (prompt, chosen, rejected). If there are multiple
    alternates you can extend this function.
    """
    msgs: List[Dict[str, str]] = entry["messages"]
    system_msgs = [m for m in msgs if m.get("role") == "system"]
    user_msgs   = [m for m in msgs if m.get("role") == "user"]
    chosen_msgs = [m for m in msgs if m.get("role") in ("assistant_w",)]
    rej_msgs    = [m for m in msgs if m.get("role") in ("assistant_l",)]

    if not system_msgs or not user_msgs or not chosen_msgs or not rej_msgs:
        # If any are missing, skip gracefully
        return []

    system = system_msgs[-1]["content"]
    user   = user_msgs[-1]["content"]
    chosen = compact_json_str(chosen_msgs[-1]["content"])
    rejected = compact_json_str(rej_msgs[-1]["content"])

    # Optional validation of assistant JSONs; if invalid, still keep raw
    chosen_valid = validate_candidate_json(chosen) or chosen
    rejected_valid = validate_candidate_json(rejected) or rejected

    out = {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        "chosen": [
            {"role": "assistant", "content": chosen_valid}
        ],
        "rejected": [
            {"role": "assistant", "content": rejected_valid}
        ],
        "meta": entry.get("meta", {})
    }
    return [out]

def to_qwen_dpo_lines_from_candidates(entry: Dict[str, Any],
                                      max_pairs: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    From a single WebShop-like state with candidate_actions,
    emit (best vs rest) pairs as separate DPO lines.
    """
    instruction = entry.get("instruction") or extract_from_observation(entry.get("observation", ""), "Instruction:")
    history     = entry.get("history", [])
    observation = entry.get("observation", "")
    available   = entry.get("actions_available", {})
    step_ctr    = entry.get("step", 0)
    cands       = entry.get("candidate_actions", [])

    if not cands:
        return []

    # Build prompt: system + user
    system_content = SYSTEM_PROMPT
    user_content   = build_user_content(
        instruction=instruction or "",
        observation=observation or "",
        available_actions=available or {},
        history=history or [],
        step_counter=step_ctr if isinstance(step_ctr, int) else 0
    )

    # choose the best candidate
    best_idx = pick_best_index(cands)
    # collect pairs (best vs i) for all i != best_idx
    outputs: List[Dict[str, Any]] = []
    pairs_emitted = 0

    def extract_action_json_string(c: Dict[str, Any]) -> Optional[str]:
        # Prefer c["action"] dict with plan/thought/env
        act = c.get("action")
        if isinstance(act, dict):
            # serialize to minimal JSON
            return json.dumps({
                "plan":   act.get("plan", ""),
                "thought": act.get("thought", ""),
                "env":    act.get("env", "")
            }, separators=(",", ":"), ensure_ascii=False)
        # Or if already a string
        if isinstance(act, str):
            return act
        return None

    best = extract_action_json_string(cands[best_idx])
    if not best:
        return []

    best_norm = validate_candidate_json(best) or compact_json_str(best)

    for j, c in enumerate(cands):
        if j == best_idx:
            continue
        alt = extract_action_json_string(c)
        if not alt:
            continue
        alt_norm = validate_candidate_json(alt) or compact_json_str(alt)

        row = {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content}
            ],
            "chosen":   [ {"role": "assistant", "content": best_norm} ],
            "rejected": [ {"role": "assistant", "content": alt_norm}  ],
            "meta": {
                "source": "candidates",
                "step": step_ctr,
                "best_index": best_idx,
                "rejected_index": j,
                "scores": {
                    "best": cands[best_idx].get("stats", {}),
                    "rejected": c.get("stats", {})
                }
            }
        }
        outputs.append(row)
        pairs_emitted += 1
        if max_pairs is not None and pairs_emitted >= max_pairs:
            break

    return outputs

def extract_from_observation(obs: str, key_prefix: str) -> Optional[str]:
    """
    For lines like: "WebShop [SEP] Instruction: [SEP] ... [SEP] Search"
    try to extract after 'Instruction:' token (best-effort).
    """
    if not isinstance(obs, str):
        return None
    parts = [p.strip() for p in obs.split("[SEP]")]
    # Look for the token 'Instruction:' and return the next segment
    for i, p in enumerate(parts):
        if p.lower().startswith("instruction"):
            if i + 1 < len(parts):
                return parts[i + 1].strip()
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=["auto", "messages", "candidates"], default="auto")
    ap.add_argument("--max_pairs", type=int, default=None, help="Cap pairs per example (best vs rest). Default: no cap.")
    args = ap.parse_args()

    rows_out: List[Dict[str, Any]] = []
    total_in, total_out = 0, 0

    for line_no, obj in read_jsonl(args.input):
        total_in += 1
        try:
            if args.mode == "messages" or (args.mode == "auto" and is_messages_entry(obj)):
                out_rows = to_qwen_dpo_line_from_messages(obj)
            elif args.mode == "candidates" or (args.mode == "auto" and is_candidates_entry(obj)):
                out_rows = to_qwen_dpo_lines_from_candidates(obj, max_pairs=args.max_pairs)
            else:
                # Unknown entry; skip silently
                out_rows = []

            rows_out.extend(out_rows)
            total_out += len(out_rows)
        except Exception as e:
            print(f"[WARN] Skipping line {line_no}: {e}", file=sys.stderr)
            continue

    if not rows_out:
        print("[ERROR] No output rows generated. Check input format and mode.", file=sys.stderr)
        sys.exit(2)

    # Write once at the end
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows_out)
    print(f"[OK] Read {total_in} lines, wrote {total_out} DPO pairs -> {args.output}")

if __name__ == "__main__":
    main()
