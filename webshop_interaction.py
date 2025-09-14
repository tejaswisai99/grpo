import json
import uuid

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from models import InitiateGoalResponse, TakeStepResponse

common_url = "http://127.0.0.1:5000/api"
common_body = {"session_id": "1"}
s = requests.Session()
s.headers.update({"Connection": "keep-alive"})
retries = Retry(total=2, backoff_factor=0.1, status_forcelist=[502, 503, 504])
s.mount("http://", HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries))
s.mount("https://", HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries))


def initialize_goal():
    data = {'session_id': ' 1', 'observation_mode': "text"}
    #print(json.dumps(data))
    response = s.post(common_url+"/start", data=json.dumps(data))
    #print(response.json())
    InitiateGoalResponse.model_validate(response.json())
    return response.json()

def take_step(session_id: str, action: str):
    data = {"session_id": session_id, "action": action}
    response = s.post(common_url+"/step", data=json.dumps(data))
    TakeStepResponse.model_validate(response.json())
    return response.json()

def is_done(step_response) -> bool:
    return step_response['done']


def explain(env_action, webpage):
    print (env_action, webpage)
    if "search" in env_action:
        return env_action
    clicked_element = env_action.split("click[")[1].split("]")[0]
    title = webpage.split(clicked_element)[1].split(" [SEP] ")[1]
    return "click["+title+"]"