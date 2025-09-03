import json
import re
from typing import Callable, Any

import requests
from pydantic import ValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from models import AvailableActions, ActionSuggestion, validate_env_constraints, Explanation
import regex

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


common_url = "http://127.0.0.1:5000/api"
common_body = {"session_id": "1"}
s = requests.Session()
s.headers.update({"Connection": "keep-alive"})
retries = Retry(total=2, backoff_factor=0.1, status_forcelist=[502, 503, 504])
s.mount("http://", HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries))
s.mount("https://", HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries))

SYSTEM_PROMPT_ACTION_SUGGESTION = """
You are an agent-policy generator for a web-navigation RL agent (Agent Q style). 
You must output STRICT JSON with exactly three top-level string fields:
  - "plan": a brief, high-level plan for the NEXT FEW steps (1–4 steps max).
  - "thought": a concise internal reasoning for THIS step only (1–2 sentences).
  - "env": the concrete environment action for THIS step in EXACT format:
        either  search[<free-text query>]
        or      click[<ELEMENT_ID>]
Your task will be completed only when we buy the product. i.e. "env" is click[buy now].
You will be given some goals for web-navigation for buying products in WebShop -- a very well known dataset. 
For each goal, you'll have category, attributes, price match etc., defined by humans before.
Once you complete the task, a reward is calculated based on the bought product's details compared to goal's expected details.
        
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
2) Explore a bit, look at the entire current observation(web page).
3) Observe the title/description and then choose the best product, best variant and press buy now. 
4) Price, color, variant details are sometimes in product page. Don't keep going to last pages. 
5) Rarely do you find useful product at the end. Come back and search with a different query
6) The end goal is to buy a product always.


DEMONSTRATIONS (FEW-SHOT)

[SHOT 1: Has search bar; looking for products]
INSTRUCTION:
i want silver and noise cancelling earbuds

OBSERVATION:
"WebShop [SEP] Instruction: [SEP] i want silver and noise cancelling earbuds [SEP] Search"

AVAILABLE_ACTIONS:
{"available_actions":{"clickables":["search"],"has_search_bar":true}}

STEP_COUNTER: 
0

HISTORY:
[]

OUTPUT:
{"plan":"Let's try to search the product, look at the results and choose the best product and click buy now","thought":"search noise cancelling earbuds silver","env":"search[noise cancelling earbuds silver]"}

[SHOT 2: No search bar; picking an option]
INSTRUCTION:
i want silver and noise cancelling earbuds

OBSERVATION:
"Instruction: [SEP] i want silver and noise cancelling earbuds [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09HV7RM6Z [SEP] Sony WF-1000XM4 True Wireless Noise Canceling in-Ear Headphones (Silver) with Kratos Power Dual Pad Wireless Charger Bundle (2 Items) [SEP] $278.0 [SEP] B09BDTP2S3 [SEP] Sony WF-1000XM4 Noise Canceling Wireless Earbud Headphones - Silver (Renewed) [SEP] $149.99 [SEP] B0844HF4VP [SEP] Sony WF-1000XM3 True Wireless Noise-Canceling Earbud Headphones (Black, USA Warranty) with Hardshell Travel/Storage case and Noise Isolating Memory Foam & Silicone Tips Bundle (3 Items) [SEP] $199.99 [SEP] B0849QHQ1T [SEP] Sony WF-1000XM3 True Wireless Noise-Canceling Earbuds (Silver, USA Warranty) Bundle Earbud mic Headset and Bluetooth USB dongle (3 Items) [SEP] $199.99 [SEP] B071NZSCZZ [SEP] Bluetooth Headphones Headset Neckband by Gladton, Wireless Stereo Headphones Bluetooth Earbuds with Mic, Noise Cancelling Magnetic Sweatproof Sports Earphones for Running, Music, Jogging, Gym - Black [SEP] $25.0 [SEP] B094C4VDJZ [SEP] Sony WF-1000XM4 Industry Leading Noise Canceling Truly Wireless Earbud Headphones with Alexa Built-in, Black [SEP] $278.0 [SEP] B09GJSS5PP [SEP] HUAWEI FreeBuds 4, True Wireless Bluetooth Earbuds, Open-fit Active Noise Cancellation Headphones, 3-Microphone System, Wired Charging Case, Silver [SEP] $148.99 [SEP] B07T81554H [SEP] Sony WF-1000XM3 Industry Leading Noise Canceling Truly Wireless Earbuds Headset/Headphones with AlexaVoice Control And Mic For Phone Call, Black [SEP] $198.0 [SEP] B098SRVLSC [SEP] Wireless Earbuds Active Noise Cancelling Digdiy D10X ANC Bluetooth Earbuds Wireless Earphones Transparency Mode Clear Calls with ENC Wireless Charge 40H Battery Deep Bass and Immersive Sound Headsets [SEP] $49.99 [SEP] B096SBSTFL [SEP] Bose QuietComfort\u00ae Noise Cancelling Earbuds \u2013 True Wireless Earphones, Sandstone, World Class Bluetooth Noise Cancelling Earbuds with Charging Case - Limited Edition [SEP] $219.0"

AVAILABLE_ACTIONS:
{"available_actions":{"clickables":["b071nzsczz","b07t81554h","b0844hf4vp","b0849qhq1t","b094c4vdjz","b096sbstfl","b098srvlsc","b09bdtp2s3","b09gjss5pp","b09hv7rm6z","back to search","next >"],"has_search_bar":false}}

HISTORY:
[{"plan":"Let's try to search the product, look at the results and choose the best product and click buy now","thought":"search noise cancelling earbuds silver","env":"search[noise cancelling earbuds silver]"}]

STEP_COUNTER:
1

OUTPUT:
{"plan":"Let's look at the products displayed and choose the best one, and buy","thought":"click[b09gjss5pp] - this product matches the description well","env":"click[b09gjss5pp]"}
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

DEMONSTRATION (FEW-SHOT)
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

def suggest_action(
    call_llm: CallLLM,
    instruction: str,
    observation: str,
    available_actions: AvailableActions,
    history: list[dict[str, Any]] | None = None,
) -> ActionSuggestion:
    """
    Returns a validated ActionSuggestion (plan/thought/env).
    Enforces environment constraints post-parse.
    """
    history = history or []
    user_prompt = USER_PROMPT_ACTION_SUGGESTION.format(
        instruction=instruction,
        observation=observation,
        available_actions_json=default_serialize(available_actions.model_dump()),
        history_json=default_serialize(history),
    )

    raw = call_llm(SYSTEM_PROMPT_ACTION_SUGGESTION, user_prompt)
    json_blob = extract_first_json_object(raw)
    try:
        parsed = ActionSuggestion.model_validate_json(json_blob)
    except ValidationError as e:
        raise ValueError(f"LLM output failed schema validation: {e}\nRaw: {raw}")

    # Environment constraints
    validate_env_constraints(parsed, available_actions)
    return parsed


def explain_action(
    call_llm: CallLLM,
    instruction: str,
    observation: str,
    available_actions: AvailableActions,
    history: list[dict[str, Any]] | None,
    prev_output: ActionSuggestion | dict[str, Any],
) -> Explanation:
    """
    Returns a validated Explanation (expl).
    prev_output can be the Pydantic ActionSuggestion or its dict.
    """
    history = history or []
    prev_payload = (
        prev_output.model_dump() if isinstance(prev_output, ActionSuggestion) else prev_output
    )

    user_prompt = SYSTEM_PROMPT_ACTION_EXPLANATION.format(
        instruction=instruction,
        observation=observation,
        available_actions_json=default_serialize(available_actions.model_dump()),
        history_json=default_serialize(history),
        prev_output_json=default_serialize(prev_payload),
    )

    raw = call_llm(SYSTEM_PROMPT_ACTION_EXPLANATION, user_prompt)
    json_blob = extract_first_json_object(raw)
    try:
        parsed = Explanation.model_validate_json(json_blob)
    except ValidationError as e:
        raise ValueError(f"LLM output failed schema validation (expl): {e}\nRaw: {raw}")

    return parsed


def call_llm(system_prompt: str, user_prompt: str):
    #print(user_prompt)
    payload = {
        "model": "gpt-oss-20b",  # or whatever you named the model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 40000,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "stream": False
    }
    response = requests.post("http://localhost:8001/v1/chat/completions", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return None