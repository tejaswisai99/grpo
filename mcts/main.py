# tester_mcts_replay.py
import json, logging
from typing import List, Dict

import llm_interactor as lli
import webshop_interaction as ws
from mcts_replay import MCTSReplay

logging.basicConfig(filename='test_tree.log', filemode='a',
                    format='%(asctime)s %(message)s', level=logging.INFO)

# LLM proposer (strict 4 candidates)
def propose_candidates(instr: str, obs: str, avail: dict, history: List[dict], step_t: int) -> Dict[str, dict]:
    prompt = lli.USER_PROPMPT_FOR_SUGGESTION(
        instruction=instr,
        observation=obs,
        available_actions=avail,   # adapter accepts dict
        history=history,
        step_counter=step_t,
    )
    raw = lli.call_llm(lli.SYSTEM_PROMPT_ACTION_SUGGESTION, prompt, temperature=0.2)
    blob = lli.extract_first_json_object(raw)
    return json.loads(blob)

def run_one_goal(goal_idx: int, sims_per_root: int = 32, max_steps: int = 12, c_puct: float = 1.2):
    # Start a LIVE episode deterministically bound to goal_idx
    live = ws.initialize_goal(goal_idx=goal_idx, observation_mode="text")
    session_id = live["session_id"]
    instr = live["instruction_text"]
    print(f"\n=== Goal {goal_idx} ===\n{instr}")

    mcts = MCTSReplay(propose_fn=propose_candidates, c_puct=c_puct, sims_per_root=sims_per_root, max_depth=max_steps)

    history_envs: List[str] = []
    steps = 0
    final_R = 0.0
    done = False
    while not done and steps < max_steps:
        # Plan at root from (goal_idx, history_envs) using *stateless replays*
        chosen_env, dbg = mcts.plan_at_root(goal_idx=goal_idx, prefix_envs=history_envs)
        print(f"step {steps} :: chose {chosen_env}  | root_edges={dbg['root_edges']}")

        # Execute once on the LIVE episode
        step_resp = ws.take_step(session_id, chosen_env)
        history_envs.append(chosen_env)

        done = ws.is_done(step_resp)
        steps += 1
        if done:
            final_R = float(step_resp.get("reward", 0.0))
            print(f"TERMINATED: reward={final_R:.3f}")
            break

    # Bookkeeping/log
    logging.info({"goal_idx": goal_idx, "steps": steps, "reward": final_R})
    return final_R, steps

def run_many(goals: List[int], sims_per_root: int = 32, max_steps: int = 12):
    total, succ = 0, 0
    for g in goals:
        R, steps = run_one_goal(g, sims_per_root=sims_per_root, max_steps=max_steps)
        total += 1
        succ += int(R > 0.999)
        print(f"[agg] success_rate={succ/total:.3f}")

if __name__ == "__main__":
    # Example: try first few deterministic goals
    run_many(goals=[0,1,2], sims_per_root=16, max_steps=12)
