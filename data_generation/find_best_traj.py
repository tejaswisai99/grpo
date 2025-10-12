import json, re, os, glob
from pathlib import Path
import webshop_interaction as ws
INPUT_DIR = r"D:\iisc_project\grpo\greedy_mcts\trajs"
OUTPUT = Path(INPUT_DIR) / "train_first5.jsonl"

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
count = 0
def extract_instruction(observation:str):
    m = re.search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]', observation, flags=re.I|re.S)
    return m.group(1).strip() if m else ""

def build_records(doc,file_id):
    out = []
    goal_id = int(file_id)
    rollouts = doc['all_harvested']
    filtered = [p for p in rollouts if p.get("reward", 0) >= 0.6]
    filtered = sorted(filtered, key=lambda x: (-x.get("reward", 0), x.get("steps", float("inf"))))
    if len(filtered) < 1:
        return None
    filtered = filtered[:1]
    for i in range(len(filtered[0]['actions'])+1):
        env_data = [k['env'] for k in filtered[0]['actions']]
        prefix_envs =  env_data[0:i]
        step_counter = i
        snap = ws.replay(goal_idx=goal_id, actions=prefix_envs, observation_mode="text")
        if (snap['reward']>=1.0):
            return out
        instruction = snap['instruction_text']
        state = snap['observation']
        available_actions = snap['available_actions']
        history = filtered[0]['actions'][0:i]
        user = USER_TEMPLATE.format(
            INSTRUCTION=instruction,
            OBSERVATION=state,
            AVAILABLE_ACTIONS_JSON=json.dumps(available_actions, ensure_ascii=False),
            HISTORY_JSON=json.dumps(history, ensure_ascii=False),
            STEP_COUNTER=step_counter,
        ) + "\n" + TAIL

        rec = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(filtered[0]['actions'][i], ensure_ascii=False)},
            ],
            "meta": {"goal_idx": goal_id, "step": step_counter, "is_positive": True}
        }
        out.append(rec)
    return out

def main():
    files = sorted(glob.glob(str(Path(INPUT_DIR) / "*.json")))[:1000]
    counter = 0
    with open(OUTPUT, "w", encoding="utf-8") as fout:
        for f in files:
            try:
                doc = json.load(open(f, "r", encoding="utf-8"))
                temp = f.split("_")[3]
                temp2 = temp.split(".")[0]
                for rec in build_records(doc, temp2):
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counter += 1
            except Exception as e:
                continue
    print("wrote", counter, "records to", OUTPUT)

if __name__ == "__main__":

    main()