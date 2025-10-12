import json, re, os, glob
from pathlib import Path

INPUT_DIR = r"D:\iisc_project\grpo\greedy_mcts\trajs"
FILE = Path(INPUT_DIR) / "best_trajs.jsonl"
FILE2 = Path(INPUT_DIR) / "best_critiques.jsonl"

SYSTEM_PROMPT = """You are an action evaluator for a web-navigation agent. 
Your job is to read the current instruction, history, available_actions, state, and a list of candidate actions and then pick EXACTLY ONE candidate that best progresses toward the goal of completing the instruction according to YOU. 
You will have to return the action_id and reason why that action is the best. Keep the reason to 25-30 words.
Each action will have following components 
- "plan": a brief, high-level plan for the NEXT FEW steps (1–4 steps max).
- "thought": a concise internal reasoning for THIS step only (1–2 sentences).
- "env": the concrete environment action for THIS step in EXACT format:
        either  search[<free-text query>]
        or      click[<ELEMENT_ID>]

Additionally in history, you'll have additional component in action called expl 
- "expl": a brief, high-level explanation of what this step achieved.
You can use this for your evaluation as well. Sometimes this may be missing.
However, note that for the candidate actions, this field will be NOT present.

Evaluation criteria (in order of importance):
Goal fit: Does the action directly move toward satisfying the instruction’s constraints (attributes, price, brand, etc.)?
Feasibility now: Is the action executable given available_actions and the current state?
Information gain & precision: Prefer actions that add the most targeted progress with minimal ambiguity (queries that encode key constraints without unnecessary noise).
Safety & efficiency: Avoid obviously redundant, low-signal, or risky steps. Prefer fewer steps to reach the goal when plausible.
History coherence: Use history to avoid repeating unsuccessful attempts or dead ends.
Tie-breakers: pick the more specific and constraint-aware action; if still tied, pick the action likely to reduce future branching (higher precision).

DEMONSTRATIONS:
[EXAMPLE-1]
INSTRUCTION
i am looking for ladies large size black06 colored jeans with straight leg fit, and price lower than 120.00 dollars

HISTORY
[{"plan": "search using a slightly different phrasing to capture variations in product descriptions","thought": "use 'black jeans large size straight leg' to match common product listings","env": "search[black jeans large size straight leg]","expl": "I performed the search using 'black jeans large size straight leg' to align with common product descriptions and better capture the user's request for ladies' large size, black, straight leg jeans under $120."},{"plan": "evaluate the products for black color, large size, and straight leg fit, then select the best match","thought": "B09D3YV6B5 has black color and straight leg, matches size and price constraints","env": "click[b09d3yv6b5]","expl": "I clicked on B09D3YV6B5 because it matches the user's request for black color, large size, straight leg fit, and price below $120.00."}]

AVAILABLE_ACTIONS
{"available_actions":{"clickables":["3x-large","< prev","back to search","beige","black","brown","buy now","description","features","green","khaki","large","m-01 beige","m-02 black","m-04 blue","m-05 white","medium","reviews","small","x-large","xx-large"],"has_search_bar":false}}

STATE
Instruction: [SEP] i am looking for ladies large size black06 colored jeans with straight leg fit, and price lower than 120.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] beige [SEP] black [SEP] brown [SEP] green [SEP] khaki [SEP] m-05 white [SEP] m-04 blue [SEP] m-01 beige [SEP] m-02 black [SEP] size [SEP] small [SEP] medium [SEP] large [SEP] x-large [SEP] xx-large [SEP] 3x-large [SEP] Fudule Y2K Jeans for Women High Waisted Straight Pants Fall Ribbed Trousers 90s Vintage Baggy Pant Fashion Y2K Clothes [SEP] Price: $12.64 to $24.87 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now

CANDIDATE_ACTIONS
[{"action_id":"1","action":{"plan":"confirm the correct size is selected (large) and ensure color is black06","thought":"user requested large size and black color, so select size large and verify color is black","env":"click[large]"}},{"action_id":"2","action":{"plan":"verify product details and proceed to buy now if all criteria are met","thought":"product matches size, color, and fit; proceed to purchase","env":"click[buy now]"}}]

OUTPUT:
{"action_id":2, "reason": "The product description from the webpage present in state seems to be matching the user's instruction, so it's better to buy the product"}


EXAMPLE-2:
INSTRUCTION
i am looking for a 12 ounce jar of raspberry preserve that is nut and gluten free, and price lower than 130.00 dollars

HISTORY
[{"plan":"perform a search for the product with exact requirements to find relevant results","thought":"search for '12 ounce jar raspberry preserve nut free gluten free'","env":"search[12 ounce jar raspberry preserve nut free gluten free]","expl":"I performed a search for '12 ounce jar raspberry preserve nut free gluten free' to find products matching the user's specific requirements for size, dietary restrictions, and price."}]

AVAILABLE ACTIONS
{"actions_available":{"clickables":["b003gs5fmq","b00ciw2kf6","b00d8vgda4","b00jiwbmz8","b01cittq2s","b078wjf5wc","b07gp2dt86","b08frv67t5","b08fwfdrqw","b08xn4p6fz","back to search","next >"],"has_search_bar":false}}

STATE
Instruction: [SEP] i am looking for a 12 ounce jar of raspberry preserve that is nut and gluten free, and price lower than 130.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B00CIW2KF6 [SEP] Bonne Maman Preserves, Variety Pack (Strawberry, Raspberry, Wild Blueberry, Cherry), 13 Ounce Jars (Pack of 4) [SEP] $24.57 [SEP] B08FRV67T5 [SEP] Tiptree Strawberry Preserve, 12 Ounce Jar & 'Old Times' Orange Marmalade, 12 Ounce Jar [SEP] $26.61 [SEP] B07GP2DT86 [SEP] Sahale Snacks Trail Mix Variety Pack, 1.5 Ounces (Pack of 12) [SEP] $21.75 [SEP] B08FWFDRQW [SEP] Sahale Snacks Pomegranate Vanilla Flavored Cashews Glazed Mix, 1.5 Ounces (Pack of 9) & Raspberry Crumble Cashew Trail Mix, 1.5 Ounces, (Pack of 9) [SEP] $21.42 [SEP] B08XN4P6FZ [SEP] Smucker's Sugar Free Apricot Preserves and Red Raspberry Preserves, 12.75 Ounce (Pack of 2) - with Spice of Life Spreader [SEP] $14.99 [SEP] B003GS5FMQ [SEP] Sahale Snacks Maple Pecans Glazed Mix, 1.5 Ounces (Pack of 18) [SEP] $19.62 [SEP] B00JIWBMZ8 [SEP] Bonne Maman Strawberry Preserves, 1 Ounce Jars (Pack of 15) [SEP] $15.0 [SEP] B01CITTQ2S [SEP] Linzer Cookies with Raspberry Jam, 6 Ounce [2 Count], Gluten Free Cookies | Shortbread Cookies, Kosher, Nut Free, Gluten Free Dairy Free Cookies by Gluten Free Palace [SEP] $17.56 [SEP] B078WJF5WC [SEP] Guava Gourmet Guava Jam (12oz, Jar), Fresh Tropical Guava Fruit Jam, All-Natural, Non-GMO, Vegan, Gluten and Cholesterol-Free Premium Artisan Craft Jam, No Fillers or Preservatives, Certified Kosher [SEP] $12.99 [SEP] B00D8VGDA4 [SEP] Sahale Snacks Classic Fruit and Nut Trail Mix, 1.5 Ounces (Pack of 18) [SEP] $23.22

CANDIDATE_ACTIONS
[{"action_id":1,"action":{"plan":"select the product that matches 12 ounce jar, raspberry preserve, nut and gluten free, and price under $130","thought":"B078WJF5WC is raspberry preserve, 12oz, gluten and nut free, fits all criteria","env":"click[b078wjf5wc]"}},{"action_id":2,"action":{"plan":"check alternative products in case of mismatch, ensure correct dietary requirements","thought":"B08XN4P6FZ has raspberry preserve but is 12.75oz and not explicitly nut/gluten free","env":"click[b08xn4p6fz]"}},{"action_id":3,"action":{"plan":"go back to search and refine query to ensure no missed options","thought":"search results may have missed key filters; go back to search for better match","env":"click[back to search]"}}]

OUTPUT:
{"action_id":1, "reason": "The product with product id B078WJF5WC is matching with user description, so clicking on it is the best possible action from available actions"}
"""

USER_PROMPT = f"""
INSTRUCTION
{{INSTRUCTION}}

HISTORY
{{HISTORY}}

AVAILABLE ACTIONS
{{AVAILABLE_ACTIONS}}

STATE
{{OBSERVATION}}

CANDIDATE_ACTIONS
{{CANDIDATE_ACTIONS}}
"""

TAIL ="""
OUTPUT FORMAT:
Fill your candidates in this format-
{"action_id": "Integer" ,"reason": "String"}
RETURN JSON. NO extra text. Do not put line breaks and escape characters."""


def update_row_with_critique_scores(doc):



def main():
    with open(FILE, "r", encoding="utf-8") as fin, open(FILE2, "w", encoding="utf-8") as fout:
        counter = 0
        for f in fin:
            try:
                doc = json.loads(f)
                updated_doc = update_row_with_critique_scores(doc)
                fout.write(json.dumps(updated_doc, ensure_ascii=False) + "\n")
            except Exception as e:
                continue
        #print("wrote", counter, "records to", OUTPUT)

if __name__ == "__main__":
    main()
