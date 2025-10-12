#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build TRL KTO dataset from:
  A) WebShop-like 'candidate_actions' entries  -> multiple (prompt, completion, label) rows
  B) 'messages' entries with assistant_w / assistant_l -> two rows (pos/neg)

Output lines (JSONL):
{
  "prompt":    [ {"role":"system","content":...}, {"role":"user","content":...} ],
  "completion":[ {"role":"assistant","content":...} ],
  "label":     1 or 0,
  "meta":      { ... optional provenance ... }
}

Usage:
  python build_kto_dataset.py \
      --input /path/in.jsonl \
      --output /path/out.kto.jsonl \
      --mode auto \
      --positives-only       # optional
      --negatives-only       # optional
      --max_examples 0       # cap rows per input example from candidate mode (0 = no cap)

Notes:
- TRL's KTOTrainer supports conversational datasets; it applies the chat template automatically. See HF docs.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


USER_TEMPLATE  = """
INSTRUCTION:
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

def compact_one_line(s: str) -> str:
    s = (s or "").strip()
    # strip accidental code fences
    if s.startswith("```"):
        s = s.strip("`").strip()
        if "\n" in s:
            first, rest = s.split("\n", 1)
            if first.lower().startswith("json"):
                s = rest.strip()
    return " ".join(s.split())

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

def build_user_content(instruction: str,
                       observation: str,
                       available_actions: Dict[str, Any],
                       history: List[Dict[str, Any]],
                       step_counter: int) -> str:
    # We keep TAIL appended so the model knows the expected JSON format.
    return USER_TEMPLATE.format(
        INSTRUCTION=instruction,
        OBSERVATION=observation,
        AVAILABLE_ACTIONS_JSON=json.dumps(available_actions, ensure_ascii=False),
        HISTORY_JSON=json.dumps(history, ensure_ascii=False),
        STEP_COUNTER=step_counter
    ) + "\n\n" + TAIL

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
        q   = st.get("Q", float("-inf"))
        uct = st.get("UCT", float("-inf"))
        n   = st.get("N", float("-inf"))
        return (q, uct, n)
    best_idx, best_key = 0, (float("-inf"), float("-inf"), float("-inf"))
    for i, c in enumerate(cands):
        k = key_fn(c)
        if k > best_key:
            best_key, best_idx = k, i
    return best_idx

def action_to_json_string(cand: Dict[str, Any]) -> str:
    # Prefer structured dict with plan/thought/env
    act = cand.get("action")
    if isinstance(act, dict):
        return json.dumps({
            "plan":   act.get("plan", ""),
            "thought": act.get("thought", ""),
            "env":    act.get("env", "")
        }, separators=(",", ":"), ensure_ascii=False)
    # Already a string?
    if isinstance(act, str):
        return act
    return ""

def from_candidates(entry: Dict[str, Any],
                    positives_only: bool,
                    negatives_only: bool,
                    max_examples: int) -> List[Dict[str, Any]]:
    instruction = entry.get("instruction") or extract_from_observation(entry.get("observation", ""), "Instruction:")
    history     = entry.get("history", [])
    observation = entry.get("observation", "")
    available   = entry.get("actions_available", {})
    step_ctr    = entry.get("step", 0)
    cands       = entry.get("candidate_actions", []) or []

    if not cands:
        return []

    system = SYSTEM_PROMPT
    user   = build_user_content(
        instruction=instruction or "",
        observation=observation or "",
        available_actions=available or {},
        history=history or [],
        step_counter=step_ctr if isinstance(step_ctr, int) else 0
    )

    best_idx = pick_best_index(cands)
    best_str = compact_one_line(action_to_json_string(cands[best_idx]))
    outputs: List[Dict[str, Any]] = []

    # Emit positive example for the best
    if not negatives_only:
        outputs.append({
            "prompt":    [{"role":"system","content": system},
                          {"role":"user",  "content": user}],
            "completion":[{"role":"assistant","content": best_str}],
            "label":     1,
            "meta": {
                "source":"candidates",
                "step": step_ctr,
                "kind":"positive",
                "index": best_idx,
                "scores": cands[best_idx].get("stats", {})
            }
        })

    # Emit negatives for all non-best
    if not positives_only:
        emitted = 0
        for j, cand in enumerate(cands):
            if j == best_idx:
                continue
            alt_str = compact_one_line(action_to_json_string(cand))
            outputs.append({
                "prompt":    [{"role":"system","content": system},
                              {"role":"user",  "content": user}],
                "completion":[{"role":"assistant","content": alt_str}],
                "label":     0,
                "meta": {
                    "source":"candidates",
                    "step": step_ctr,
                    "kind":"negative",
                    "index": j,
                    "scores": cand.get("stats", {})
                }
            })
            emitted += 1
            if max_examples > 0 and emitted >= max_examples:
                break

    return outputs

def from_messages(entry: Dict[str, Any],
                  positives_only: bool,
                  negatives_only: bool) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = entry.get("messages", [])
    if not msgs:
        return []

    sys_msgs = [m for m in msgs if m.get("role") == "system"]
    usr_msgs = [m for m in msgs if m.get("role") == "user"]
    win_msgs = [m for m in msgs if m.get("role") in ("assistant_w",)]
    los_msgs = [m for m in msgs if m.get("role") in ("assistant_l",)]

    if not sys_msgs or not usr_msgs:
        return []
    system = sys_msgs[-1]["content"]
    user   = usr_msgs[-1]["content"]

    outputs: List[Dict[str, Any]] = []
    if win_msgs and not negatives_only:
        win = compact_one_line(win_msgs[-1]["content"])
        outputs.append({
            "prompt":    [{"role":"system","content": system},
                          {"role":"user",  "content": user}],
            "completion":[{"role":"assistant","content": win}],
            "label":     1,
            "meta": entry.get("meta", {}) | {"source":"messages","kind":"positive"}
        })
    if los_msgs and not positives_only:
        los = compact_one_line(los_msgs[-1]["content"])
        outputs.append({
            "prompt":    [{"role":"system","content": system},
                          {"role":"user",  "content": user}],
            "completion":[{"role":"assistant","content": los}],
            "label":     0,
            "meta": entry.get("meta", {}) | {"source":"messages","kind":"negative"}
        })
    return outputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=["auto","candidates","messages"], default="auto")
    ap.add_argument("--positives-only", action="store_true", help="Only emit label=1 rows.")
    ap.add_argument("--negatives-only", action="store_true", help="Only emit label=0 rows.")
    ap.add_argument("--max_examples", type=int, default=0,
                    help="For candidates mode: max negatives per input example (0 = no cap).")
    args = ap.parse_args()

    if args.positives_only and args.negatives_only:
        print("[ERROR] Choose at most one of --positives-only / --negatives-only.", file=sys.stderr)
        sys.exit(2)

    rows_out: List[Dict[str, Any]] = []
    total_in, total_out = 0, 0

    for line_no, obj in read_jsonl(args.input):
        total_in += 1
        try:
            if args.mode == "messages" or (args.mode == "auto" and is_messages_entry(obj)):
                out = from_messages(obj, args.positives_only, args.negatives_only)
            elif args.mode == "candidates" or (args.mode == "auto" and is_candidates_entry(obj)):
                out = from_candidates(obj, args.positives_only, args.negatives_only, args.max_examples)
            else:
                out = []
            rows_out.extend(out)
            total_out += len(out)
        except Exception as e:
            print(f"[WARN] Skipping line {line_no}: {e}", file=sys.stderr)

    if not rows_out:
        print("[ERROR] No output rows generated. Check input format and mode.", file=sys.stderr)
        sys.exit(2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows_out)
    print(f"[OK] Read {total_in} lines, wrote {total_out} KTO rows -> {args.output}")

if __name__ == "__main__":
    main()
