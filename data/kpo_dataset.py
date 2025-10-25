import json

SYSTEM_PROMPT = f"""You are helping in web navigation task. There are only two things to do search[free text query] or click[something].
You'll be given a webpage, actions available and history so far. Using that you need to predict the next best action."""

USER_TEMPLATE = """
WEBPAGE (actual webpage):
{page}
-------------------------------------
ACTIONS AVAILABLE (current actions you can perform on web page):
{actions_available}
-------------------------------------
HISTORY (a list of actions taken so far)
{history}
-------------------------------------
predict the next action:
"""

def best_action_fn(actions):
    sorted_actions = sorted(actions, key=lambda a: a['stats']['Q'], reverse=True)
    return sorted_actions[0]['action']['env'], sorted_actions

with open("../greedy_mcts/trajs/best_trajs.jsonl","r",encoding="utf-8") as f:
    count = 0
    history = []
    prompts = []
    for line in f:
        entry = json.loads(line)
        if entry['actions_available']['has_search_bar']:
            history = []

        clickables = entry['actions_available']['clickables']
        state =entry['observation']
        original_actions = entry['actions_available']['clickables']
        actions_available = []
        for action in clickables:
            if entry['actions_available']['has_search_bar']:
                action_available = 'search'
            else:
                idx = "[SEP] " + action.upper() + " [SEP]"
                state_split = state.split(idx)
                if len(state_split) == 2:
                    product_info_and_beyond = state_split[1]
                    product_info = product_info_and_beyond.split('[SEP]')[0]
                    action_available = "click["+ product_info.strip() + "]"
                else:
                    action_available = "click["+ action + "]"
            actions_available.append(action_available)
        best_action,rest_action = best_action_fn(entry["candidate_actions"])
        old_history = history
        if entry['actions_available']['has_search_bar']:
            original_actions_with_click = ["search"]
        else:
            original_actions_with_click = ["[click"+action+"]" for action in original_actions]

        if "search[" not in best_action:
            label = original_actions.index(best_action.split("click[")[1].split("]")[0])
            current_action = actions_available[label]
        else:
            current_action = best_action
        result_rest_action = []
        set_result_action = set()
        for x in rest_action:
            if x['action']['env'] not in set_result_action:
                result_rest_action.append(x)
                set_result_action.add(x['action']['env'])
        envs = []
        for action in result_rest_action:
            env = action["action"]["env"]
            if "search[" not in env:
                label_rejected = original_actions.index(env.split("click[")[1].split("]")[0])
                rejected_act = actions_available[label_rejected]
            else:
                rejected_act = env
            envs.append(rejected_act)
        if len(result_rest_action) > 2:
            prompt = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    page = state.lower(),
                    actions_available = actions_available,
                    history = old_history
                )},
            ],
                "candidates": [{"role": "assistant", "content": k} for k in envs],
                "q_values": [k['stats']['Q'] for k in result_rest_action],
                "generation_order": [k for k in range(len(result_rest_action))]
            }
            prompts.append(prompt)
        history.append(current_action)
with open("../greedy_mcts/trajs/prompts.jsonl","w",encoding="utf-8") as f:
    for r in prompts:
        f.write(json.dumps(r, ensure_ascii=False)+ "\n")
