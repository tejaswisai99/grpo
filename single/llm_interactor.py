# llm_interactor.py
from __future__ import annotations

import json
from typing import Callable, Any

import regex
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from models import AvailableActions, ActionSuggestion, validate_env_constraints, Explanation

JsonStr = str
CallLLM = Callable[[str, str], JsonStr]  # (system, user) -> raw LLM text

# Extract first top-level JSON object from a string (handles extra tokens around it)
JSON_OBJECT_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}")


def extract_first_json_object(text: str) -> str:
    m = JSON_OBJECT_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    return m.group(0)


def default_serialize(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# Robust local session
s = requests.Session()
s.headers.update({"Connection": "keep-alive", "Content-Type": "application/json"})
retries = Retry(total=2, backoff_factor=0.1, status_forcelist=[502, 503, 504])
s.mount("http://", HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries))
s.mount("https://", HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries))

SYSTEM_PROMPT_ACTION_SUGGESTION = """You are an agent-policy generator for a web-navigation RL agent (Agent Q style). 
You must output STRICT JSON **with exactly three top-level string fields**.:
  - "plan": a brief, high-level plan for the NEXT FEW steps (1–4 steps max).
  - "thought": a concise internal reasoning for THIS step only (1–2 sentences).
  - "env": the concrete environment action for THIS step in EXACT format:
        either  search[<free-text query>]
        or      click[<ELEMENT_ID>]
Your task will be completed only when we buy the product. i.e. "env" is click[buy now].

HARD CONSTRAINTS:
1) Output MUST be valid JSON, with double-quoted keys and string values. No trailing commas. No markdown. No extra text.
2) "env" MUST be one of:
      - search[...], ONLY IF "has_search_bar" is true in available_actions.
      - click[ID], where ID MUST be EXACTLY one of the provided "clickables".
3) If no suitable click is possible and has_search_bar is false, choose the best available clickable (including "back to search" or "next >").
4) Keep "plan" short and actionable; keep "thought" minimal (no verbose chain-of-thought).
5) NEVER invent element IDs. NEVER include spaces around brackets. Use EXACT casing for IDs.
6) Use the "instruction" and "history" to stay on task and avoid repeating failed actions.
7) Observation is plain text with [SEP] separators between elements; treat it as read-only state.
8) Ensure that the task ends with in 12 steps. Step counter will be provided.

GENERAL GUIDELINES:
1) Look for the closest match. Look in the top results returned, and 1-2 next pages.
2) It's always better not to include price constraint in search, as it uses lucence indexer and price is not part of it while building the index. 
3) Explore a bit, look at the entire current observation(web page).
4) Observe the title/description and then choose the best product, best variant and press buy now. 
5) Price, color, variant details are sometimes in product page. Don't keep going to last pages. 
6) Rarely do you find useful product at the end. Come back and search with a different query
7) The end goal is to buy a product always.


DEMONSTRATIONS (FEW-SHOT)
[Example-1]
INSTRUCTION:
i need a blink outdoor camera kit that has motion detection, and price lower than 260.00 dollars

OBSERVATION:
"WebShop [SEP] Instruction: [SEP] i need a blink outdoor camera kit that has motion detection, and price lower than 260.00 dollars [SEP] Search"

AVAILABLE_ACTIONS:
{"available_actions":{"clickables":["search"],"has_search_bar":true}}

STEP_COUNTER: 
0

HISTORY:
[]

OUTPUT:
{"plan": "let's try to search the product, look at the results, and then find the suitable product and click buy now", "thought": "let's search 'blink outdoor camera motion detection'", "env": "search[blink outdoor camera motion detection]"}

[Example-2]
INSTRUCTION:
i want to shop for some sulfate free, paraben free conditioner for dry, damaged hair, and price lower than 40.00 dollars

OBSERVATION:
"Instruction: [SEP] i want to shop for some sulfate free, paraben free conditioner for dry, damaged hair, and price lower than 40.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B08F2L92SR [SEP] Khadi Veda Shampoo & Conditioner with Goat Milk- SLS Free, Sulfate Free, Paraben Free, Mineral OIl Free - For Frizz-free, Strong, & Shiny Hair, Safe for Color & Keratin Treated Hair - 6.76 Oz./200ml [SEP] $16.99 [SEP] B00LJQUUD6 [SEP] OGX Renewing + Argan Oil of Morocco Hydrating Hair Conditioner, Cold-Pressed Argan Oil to Help Moisturize, Soften & Strengthen Hair, Paraben-Free with Sulfate-Free Surfactants, 25.4 fl oz(Pack of 4) [SEP] $10.57 [SEP] B0845D32KQ [SEP] TRESemmé Pro Pure Sulfate Free Shampoo, Conditioner and Styler For Light Moisture and Volume Light Moisture Sulfate Free, Paraben Free and Dye-Free Formulas for Dry Hair 3 Count [SEP] $17.97 [SEP] B0845D1CJH [SEP] TRESemmé Pro Pure Sulfate Free Shampoo, Conditioner and Styler To Repair Damage and Add Volume Damage Repair Sulfate Free, Paraben Free and Dye-Free Hair Care 3 Count [SEP] $17.97 [SEP] B07JFLDJQ5 [SEP] Luseta Biotin & Collagen Conditioner Thickening for Hair Loss & Fast Hair Growth - Infused with Argan Oil to Repair Damaged Dry Hair - Sulfate Free Paraben Free 16.9oz [SEP] $16.35 [SEP] B094YKWM5M [SEP] GoodMood Moroccan Argan Oil Shampoo and Conditioner Set - Enriched with Keratin, Volume and Moisture, For Frizzy, Dry And Damaged Hair 2x16oz [SEP] $24.95 [SEP] B07N3BN1KS [SEP] Pantene, Shampoo and Sulfate Free Conditioner Kit, Paraben and Dye Free, Pro-V Blends, Soothing Rose Water, 17.9 fl oz, Twin Pack [SEP] $19.5 [SEP] B08M7D6X65 [SEP] OGX Extra Strength Hydrate & Repair + Argan Oil of Morocco Conditioner for Dry, Damaged Hair, Cold-Pressed Argan Oil to Moisturize Hair, Paraben-Free, Sulfate-Free Surfactants, 25.4 Fl Oz [SEP] $11.97 [SEP] B08P5YMPGS [SEP] SheaMoisture Silicone Free Conditioner for Dry Hair, Sugarcane and Meadowfoam, Sulfate Free Conditioner, 13 Oz [SEP] $10.99 [SEP] B08X7D69S1 [SEP] Head Wear Phytophusion Repair Maintenance Shampoo and Conditioner Set - Color Safe, All Hair Types, 20 Fl. Oz. Each [SEP] $37.99"

AVAILABLE_ACTIONS:
{"available_actions":{"clickables":["b00ljquud6","b07jfldjq5","b07n3bn1ks","b0845d1cjh","b0845d32kq","b08f2l92sr","b08m7d6x65","b08p5ympgs","b08x7d69s1","b094ykwm5m","back to search","next >"],"has_search_bar":false}}

HISTORY:
[{"plan":"Let's try to search the product, look at the results and choose the best product and click buy now","thought":"search for 'sulfate free, paraben free conditioner for dry hair'","env":"search[sulfate free, paraben free conditioner for dry hair]", "expl": "Searched for the product using key strings, and found the best matching products"}]

STEP_COUNTER:
1

OUTPUT:
{"plan": "let's choose the best product from the available products, and click buy now", "thought": "Let's click on the B094YKWM5M, it matches the price constraint and looks fitting for user", "env": "click[b094ykwm5m]"}

[Example-3]
INSTRUCTION:
i want to shop for some sulfate free, paraben free conditioner for dry, damaged hair, and price lower than 40.00 dollars

OBSERVATION:
"Instruction: [SEP] i am looking for a pair of women's high heel stilettos in a size 7, and price lower than 80.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] c1-black [SEP] c1-brown [SEP] c1-green [SEP] size [SEP] 7 [SEP] 7.5 [SEP] 8 [SEP] 8.5 [SEP] 9 [SEP] 10 [SEP] Gibobby Sandals for Women Dressy Heel 2022 Fashion Sandals Ankle Strap Hook Loop Pumps Roman Thin High Heels Summer Sandals [SEP] Price: $17.49 to $18.49 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"

AVAILABLE_ACTIONS:
{"available_actions":{"clickables":["10","7","7.5","8","8.5","9","< prev","back to search","buy now","c1-black","c1-brown","c1-green","description","features","reviews"],"has_search_bar":false}}

HISTORY:
[{"plan": "search for the product, explore for the search results, look at 1-2 products before choosing the final product", "thought": "search for 'women's high heel stilettos'", "env": "search[women's high heel stilettos]", "expl":"The products are displayed after the searching"}}},
{"plan": "let's choose the best product from the available products, and click buy now", "thought": "product B09Q69NN5T seems closest to the description out of all products displayed", "env": "click[b09q69nn5t]", "expl": "clicked the best possible product of all the available for review and we have the product description"}}}]

STEP_COUNTER:
2

OUTPUT:
{"plan": "let's click buy now, we have found the product", "thought": "we are on the product page, click buy", "env": "click[buy now]"},{"3": {"plan": "let's go back and find for a better match from available products list, and choose the product and buy it", "thought": "to go back, we need to click '< prev'", "env": "click[< prev]"}
"""

USER_PROMPT_ACTION_SUGGESTION = f"""
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

SYSTEM_PROMPT_ACTION_EXPLANATION = """
You are producing a brief, post-hoc explanation for the previously selected action by you, based on 6 keys, 
"plan", "thought", "action_taken", "instruction", "prev_state", and "current_state"
Looking at this, you will give an explanation on why we took
Output STRICT JSON with exactly one key: "expl" (string).
Keep it to 1–2 sentences, referencing the chosen "env" and the user "instruction".
Do NOT reveal long reasoning; be concise and professional.

DEMONSTRATION (ONE-SHOT)
PREVIOUS_STEP:
{"plan":"Open a cheap sandal candidate.","thought":"Cheapest option fits instruction.","env":"click[b0demo1]", "prev_state":"...previous web page..", "current_state":"..current_web..", "instruction":"I want to buy cheap sandals".}
OUTPUT:
{"expl":"I clicked the lowest-priced sandal because it matched the user’s request for affordable footwear."}
"""

USER_PROMPT_ACTION_EXPLANATION = f"""
INSTRUCTION:
{{INSTRUCTION}}

PLAN (Plan chosen for the task):
{{plan}}

THOUGHT (Thinking behind choosing the action):
{{thought}}

ACTION (action taken on previous webpage):
{{env}}

PREVIOUS_PAGE (page before taking action):
{{observation}}

CURRENT_PAGE (current page):
{{OBSERVATION}}
"""


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    payload = {
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 40000,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "stream": False
    }
    resp = s.post("http://localhost:8001/v1/chat/completions", data=json.dumps(payload))
    if resp.status_code != 200:
        raise RuntimeError(f"LLM call failed: {resp.status_code} {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


def suggest_action(
        call_llm_fn: CallLLM,
        instruction: str,
        observation: str,
        available_actions: AvailableActions,
        history: list[dict[str, Any]] | None = None,
        step_counter: int = 0,
) -> ActionSuggestion:
    history = history or []
    user_prompt = USER_PROPMPT_FOR_SUGGESTION(
        instruction=instruction,
        observation=observation,
        available_actions=available_actions,
        history=history,
        step_counter=step_counter,
    )
    raw = call_llm_fn(SYSTEM_PROMPT_ACTION_SUGGESTION, user_prompt)
    json_blob = extract_first_json_object(raw)
    parsed = ActionSuggestion.model_validate_json(json_blob)
    # Safety: enforce env constraints
    validate_env_constraints(parsed, available_actions)
    return parsed


def explain_action(
        call_llm_fn: CallLLM,
        instruction: str,
        prev_observation: str,
        current_observation: str,
        plan: str,
        thought: str,
        env: str,
) -> Explanation:
    user_prompt = USER_PROMPT_ACTION_EXPLANATION.format(
        INSTRUCTION=instruction,
        plan=plan,
        thought=thought,
        env=env,
        observation=prev_observation,
        OBSERVATION=current_observation,
    )
    raw = call_llm_fn(SYSTEM_PROMPT_ACTION_EXPLANATION, user_prompt)
    json_blob = extract_first_json_object(raw)
    return Explanation.model_validate_json(json_blob)


# small helper to keep prompt-assembly consistent with your tester formatting quirks
def USER_PROPMPT_FOR_SUGGESTION(
        instruction: str,
        observation: str,
        available_actions: AvailableActions | dict,
        history: list[dict],
        step_counter: int,
) -> str:
    # Accept dict or pydantic for convenience
    if hasattr(available_actions, "model_dump"):
        aa_json = default_serialize(available_actions.model_dump())
    else:
        aa_json = default_serialize(available_actions)
    hist_json = default_serialize(history)
    base = USER_PROMPT_ACTION_SUGGESTION.format(
        INSTRUCTION=instruction,
        OBSERVATION=observation,
        AVAILABLE_ACTIONS_JSON=aa_json,
        HISTORY_JSON=hist_json,
        STEP_COUNTER=step_counter,
    )
    # Match your tester’s "OUTPUT FORMAT" tail to coerce strict JSON from some OSS models
    tail = """
OUTPUT FORMAT:
Fill your candidates in this format-
{"plan": "string","thought": "string","env": "string}
RETURN JSON. NO extra text. Do not put line breaks and escape characters."""
    return base + tail
