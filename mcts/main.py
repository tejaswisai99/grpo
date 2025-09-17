# main.py
import json, logging
from typing import List, Dict, Any, Optional

import llm_interactor as lli
import webshop_interaction as ws
from mcts_replay import MCTSReplay

logging.basicConfig(filename='test_tree.log', filemode='a',
                    format='%(asctime)s %(message)s', level=logging.INFO)

# ---- LLM wrappers (do NOT modify llm_interactor.py) ----

def propose_candidates(instr: str, obs: str, avail: dict, history: List[dict], step_t: int) -> Dict[str, dict]:
    """
    Propose 4 candidate triples (plan,thought,env) conditioning on FULL composite history.
    """
    try:
        user_prompt = lli.USER_PROMPT_ACTION_SUGGESTION.format(
            INSTRUCTION=instr,
            OBSERVATION=obs,
            AVAILABLE_ACTIONS_JSON=json.dumps(avail, ensure_ascii=False),
            HISTORY_JSON=json.dumps(history, ensure_ascii=False),
            STEP_COUNTER=step_t
        )
        raw = lli.call_llm(lli.SYSTEM_PROMPT_ACTION_SUGGESTION, user_prompt, temperature=0.2)
        blob = lli.extract_first_json_object(raw)
        return json.loads(blob)
    except Exception:
        # conservative fallback from available actions
        clickables = list(map(str, (avail.get("available_actions", avail).get("clickables") or [])))
        picks: List[dict] = []
        for p in ["buy now", "next >", "back to search"]:
            if p in clickables:
                picks.append({"plan": "fallback", "thought": "fallback", "env": f"click[{p}]"})
        for cid in clickables:
            if len(picks) >= 4: break
            if cid not in [c["env"][6:-1] for c in picks if c["env"].startswith("click[")]:
                picks.append({"plan": "fallback", "thought": "fallback", "env": f"click[{cid}]"})
        return {str(i+1): picks[i] for i in range(min(4, len(picks)))} or {"1":{"plan":"fallback","thought":"fallback","env":"click[back to search]"}}

def explain_short(instr: str, prev_obs: str, prev_avail: dict, history: List[dict],
                  plan: str, thought: str, env: str, curr_obs: str) -> str:
    """
    Call lli.explain_action and return expl string. We give prev page and current page context.
    """
    try:
        exp = lli.explain_action(
            call_llm=lli.call_llm,
            instruction=instr,
            observation=prev_obs,                # previous page
            available_actions=prev_avail,
            history=history,                     # full composite so far
            prev_output={"plan": plan, "thought": thought, "env": env}
        )
        # pydantic DTO or dict
        expl_text = getattr(exp, "expl", None)
        if expl_text is None and hasattr(exp, "model_dump"):
            expl_text = exp.model_dump().get("expl")
        return expl_text or ""
    except Exception:
        # deterministic short fallback
        return f"Chose {env} consistent with the plan."

# ---- Runner ----

def run_one_goal(goal_idx: int, sims_per_root: int = 32, max_steps: int = 12, c_puct: float = 1.2):
    # Start a live episode
    live = ws.initialize_goal(goal_idx=goal_idx, observation_mode="text")
    session_id = live["session_id"]
    instr = live["instruction_text"]
    last_obs = live["observation"]
    last_avail = live["available_actions"]

    print(f"\n=== Goal {goal_idx} ===\n{instr}")

    # MCTS with full 4-tuple actions; expl is computed during expansion
    mcts = MCTSReplay(
        propose_fn=propose_candidates,
        explain_fn=explain_short,
        c_puct=c_puct,
        sims_per_root=sims_per_root,
        max_depth=max_steps
    )

    # Keep two forms:
    #  - env-only for actual WebShop replay/exec
    #  - full composite history for the proposer/explainer
    history_envs: List[str] = []
    history_records: List[dict] = []  # [{plan,thought,env,expl}, ...]

    steps = 0
    final_R = 0.0
    done = False

    while not done and steps < max_steps:
        # Plan from root with full composite history
        chosen_env, dbg = mcts.plan_at_root(
            goal_idx=goal_idx,
            prefix_envs=history_envs,
            exec_history_records=history_records
        )
        print(f"step {steps} :: chose {chosen_env}  | root_edges={dbg['root_edges']}")

        # Fetch the chosen full 4-tuple from root candidates
        chosen_full = next((a for a in (dbg.get("root_candidates") or []) if a.get("env") == chosen_env), None)
        if chosen_full is None:
            # Safety (shouldn't happen)
            chosen_full = {"plan":"n/a","thought":"n/a","env":chosen_env,"expl":""}

        # Execute on LIVE episode
        step_resp = ws.take_step(session_id, chosen_env)
        history_envs.append(chosen_env)

        # Append the exact 4-tuple to composite history (this conditions the next root)
        history_records.append(chosen_full)

        # Update loop state
        done = ws.is_done(step_resp)
        steps += 1
        last_obs = step_resp.get("observation", last_obs)
        last_avail = step_resp.get("available_actions", last_avail)

        if done:
            final_R = float(step_resp.get("reward", 0.0))
            print(f"TERMINATED: reward={final_R:.3f}")
            break

    logging.info({"goal_idx": goal_idx, "steps": steps, "reward": final_R})
    return final_R, steps

def run_many(goals: List[int], sims_per_root: int = 32, max_steps: int = 12):
    total, succ = 0, 0
    for g in goals:
        try:
            R, steps = run_one_goal(g, sims_per_root=sims_per_root, max_steps=max_steps)
            total += 1
            succ += int(R > 0.999)
            print(f"[agg] success_rate={succ/total:.3f}")
        except Exception as e:
            logging.exception({"goal_idx": g, "error": str(e)})
            total += 1
            print(f"[agg] (goal {g} failed) success_rate={succ/total:.3f}")

if __name__ == "__main__":
    run_many(goals=[0,1,2], sims_per_root=16, max_steps=12)
