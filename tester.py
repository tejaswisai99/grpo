import json

import llm_interactor
import webshop_interaction
import logging

logging.basicConfig(
    filename='test.log',
    filemode='a',
    format='%(asctime)s %(message)s',
    level=logging.INFO)

goal_count = 0
success_count = 0
completion_count = 0
#print(goal_response)
while goal_count < 1000:
    goal_response = webshop_interaction.initialize_goal()
    try:
        step_counter = 0
        history_json=[]
        user_prompt_suggestion = llm_interactor.USER_PROMPT_ACTION_SUGGESTION.format(
                INSTRUCTION=goal_response['instruction_text'],
                OBSERVATION=goal_response['observation'],
                AVAILABLE_ACTIONS_JSON=llm_interactor.default_serialize(goal_response['available_actions']),
                HISTORY_JSON=llm_interactor.default_serialize(history_json),
            STEP_COUNTER=step_counter,
            ) +"""
OUTPUT FORMAT:
Fill your candidates in this format-
{"1": {"plan": "string","thought": "string","env": "string},"2": {"plan": "string","thought": "string","env": "string},"3": {"plan": "string","thought": "string","env": "string},"4": {"plan": "string","thought": "string","env": "string}} 
RETURN JSON. NO extra text. Do not put line breaks and escape characters."""
        llm_plan = llm_interactor.call_llm(llm_interactor.SYSTEM_PROMPT_ACTION_SUGGESTION, user_prompt_suggestion, 0.2)
        llm_json = json.loads(llm_plan)["1"]
        print(json.dumps(llm_json))
        step_response = webshop_interaction.take_step(goal_response['session_id'],llm_json['env'])
        expl_response = llm_interactor.call_llm(llm_interactor.SYSTEM_PROMPT_ACTION_EXPLANATION,
                                                llm_interactor.USER_PROMPT_ACTION_EXPLANATION.format(
                                                    INSTRUCTION=step_response['instruction_text'],
                                                    plan=llm_json['plan'],
                                                    thought=llm_json['thought'],
                                                    env=llm_json['env'],
                                                    observation=step_response['observation'],
                                                    OBSERVATION=goal_response['observation'],
                                                ), 0.2)
        response = goal_response

        # print(step_response)
        llm_json['explanation'] = expl_response
        while not webshop_interaction.is_done(step_response):
            step_counter += 1
            #llm_json['env'] = webshop_interaction.explain(llm_json['env'],response['observation'])
            response=step_response
            history_json.append(llm_json)
            prev_observation = step_response['observation']
            user_prompt_suggestion = llm_interactor.USER_PROMPT_ACTION_SUGGESTION.format(
                INSTRUCTION=step_response['instruction_text'],
                OBSERVATION=step_response['observation'],
                AVAILABLE_ACTIONS_JSON=llm_interactor.default_serialize(step_response['available_actions']),
                HISTORY_JSON=llm_interactor.default_serialize(history_json),
                STEP_COUNTER=step_counter
            ) + """
OUTPUT FORMAT:
Fill your candidates in this format-
{"1": {"plan": "string","thought": "string","env": "string},"2": {"plan": "string","thought": "string","env": "string},"3": {"plan": "string","thought": "string","env": "string},"4": {"plan": "string","thought": "string","env": "string}} 
RETURN JSON. NO extra text. DO not put line breaks and escape characters."""
            llm_plan = llm_interactor.call_llm(llm_interactor.SYSTEM_PROMPT_ACTION_SUGGESTION, user_prompt_suggestion,0.2)
            llm_json = json.loads(llm_plan)["1"]
            # print(json.dumps(llm_json))
            step_response = webshop_interaction.take_step(goal_response['session_id'],llm_json['env'])
            expl_response = llm_interactor.call_llm(llm_interactor.SYSTEM_PROMPT_ACTION_EXPLANATION,
                                                    llm_interactor.USER_PROMPT_ACTION_EXPLANATION.format(
                                                        INSTRUCTION=step_response['instruction_text'],
                                                        plan=llm_json['plan'],
                                                        thought=llm_json['thought'],
                                                        env=llm_json['env'],
                                                        observation=step_response['observation'],
                                                        OBSERVATION=prev_observation
                                                    ), 0.2)
            llm_json['explanation'] = expl_response
            if webshop_interaction.is_done(step_response) or step_counter==12:
                if step_response['reward']>0.999:
                    success_count += 1
                if step_response['reward']>0:
                    completion_count += 1
                log_ = {"instruction": goal_response['instruction_text'],"reward": step_response['reward']}
            # print(json.dumps(step_response))
    except Exception as ex:
        log_ = {"instruction": goal_response['instruction_text'],"reward": 0.0}
        pass
    finally:
        goal_count += 1
        log_['success_rate'] = success_count / goal_count
        log_['completion_rate'] = completion_count / goal_count
        log_['completion_count'] = completion_count
        logging.info(log_)