import json

import llm_interactor
import webshop_interaction

out_file = 'D:/WebShop/baseline_models/data/human_goals.json'

with open(out_file, 'r') as f:
    data = json.load(f)

goal = data[100]

goal_response = webshop_interaction.initialize_goal(goal)
#print(goal_response)
step_counter = 0
history_json=[]
user_prompt_suggestion = llm_interactor.USER_PROMPT_ACTION_SUGGESTION.format(
        INSTRUCTION=goal_response['instruction_text'],
        OBSERVATION=goal_response['observation'],
        AVAILABLE_ACTIONS_JSON=llm_interactor.default_serialize(goal_response['available_actions']),
        HISTORY_JSON=llm_interactor.default_serialize(history_json),
    STEP_COUNTER=step_counter,
    ) +  """REQUIRED OUTPUT SCHEMA:
{
    "plan": "string",
  "thought": "string",
  "env": "string"
}
Return ONLY the JSON. No extra text.
"""
llm_plan = llm_interactor.call_llm(llm_interactor.SYSTEM_PROMPT_ACTION_SUGGESTION, user_prompt_suggestion)
llm_json = json.loads(llm_plan)
step_response = webshop_interaction.take_step(goal_response['session_id'],llm_json['env'])
response = goal_response
print(llm_json)
print(step_response)
while not webshop_interaction.is_done(step_response):
    step_counter += 1
    #llm_json['env'] = webshop_interaction.explain(llm_json['env'],response['observation'])
    response=step_response
    history_json.append(llm_json)
    user_prompt_suggestion = llm_interactor.USER_PROMPT_ACTION_SUGGESTION.format(
        INSTRUCTION=step_response['instruction_text'],
        OBSERVATION=step_response['observation'],
        AVAILABLE_ACTIONS_JSON=llm_interactor.default_serialize(step_response['available_actions']),
        HISTORY_JSON=llm_interactor.default_serialize(history_json),
        STEP_COUNTER=step_counter
    ) +  """REQUIRED OUTPUT SCHEMA:
{
    "plan": "string",
  "thought": "string",
  "env": "string"
}
Return ONLY the JSON. No extra text.
"""
    llm_plan = llm_interactor.call_llm(llm_interactor.SYSTEM_PROMPT_ACTION_SUGGESTION, user_prompt_suggestion)
    llm_json = json.loads(llm_plan)
    print(json.dumps(llm_json))
    step_response = webshop_interaction.take_step(goal_response['session_id'],llm_json['env'])
    print(json.dumps(step_response))
