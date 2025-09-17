# main.py
import json, logging, os
from typing import List, Dict, Any, Optional

import llm_interactor as lli
import webshop_interaction as ws
from mcts_replay import MCTSReplay

logging.basicConfig(filename='test_tree.log', filemode='a',
                    format='%(asctime)s %(message)s', level=logging.INFO)


def propose_candidates(instr: str, obs: str, avail: dict, history: List[dict], step_t: int) -> Dict[str, dict]:
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

def explain_short(instr: str, prev_obs: str,
                  plan: str, thought: str, env: str, curr_obs: str) -> str:
    exp = lli.explain_action(
        call_llm_fn=lli.call_llm,
        instruction=instr,
        prev_observation=prev_obs,  # previous page
        plan=plan,  # full composite so far
        thought=thought,
        env=env,
        current_observation=curr_obs
    )
    return getattr(exp, "expl", None) or (exp.model_dump().get("expl") if hasattr(exp, "model_dump") else "")


def run_one_goal(goal_idx: int,
                 sims_per_root: int = 64,
                 max_steps: int = 12,
                 c_puct: float = 1.2,
                 require_reward1: bool = True,
                 target_high_paths: int = 3,
                 min_reward: float = 0.8,
                 path_depth_cap: int = 8,
                 max_nodes_per_tree: int = 32):

    live = ws.initialize_goal(goal_idx=goal_idx, observation_mode="text")
    session_id = live["session_id"]
    instr = live["instruction_text"]
    prev_obs = live["observation"]
    prev_avail = live["available_actions"]

    print(f"\n=== Goal {goal_idx} ===\n{instr}")

    mcts = MCTSReplay(
        propose_fn=propose_candidates,
        explain_fn=explain_short,
        c_puct=c_puct,
        sims_per_root=sims_per_root,
        max_depth=max_steps,
        require_reward1=require_reward1,
        target_high_paths=target_high_paths,
        min_reward=min_reward,
        path_depth_cap=path_depth_cap,
        max_nodes_per_tree=max_nodes_per_tree
    )

    history_envs: List[str] = []
    history_records: List[dict] = []  # [{plan,thought,env,expl}, ...]

    # For training export: collect one frame per root decision
    frames: List[dict] = []

    steps = 0
    final_R = 0.0
    done = False

    while not done and steps < max_steps:
        chosen_env, dbg = mcts.plan_at_root(
            goal_idx=goal_idx,
            prefix_envs=history_envs,
            exec_history_records=history_records
        )

        # Build training frame for this root decision
        # Proposed actions with N/Q/U/UCB
        root_actions = []
        edges = dbg["root_edges"]
        # Map env -> stats
        for cand in (dbg.get("root_candidates") or []):
            env = cand.get("env", "")
            est = edges.get(env, {"N": 0, "Q": 0.0, "P": 0.0, "U": 0.0, "UCB": 0.0})
            root_actions.append({
                "plan": cand.get("plan", ""),
                "thought": cand.get("thought", ""),
                "env": env,
                "expl": cand.get("expl", ""),
                "N": est["N"],
                "Q": est["Q"],
                "U": est["U"],
                "UCB": est["UCB"],
                "P": est["P"]
            })

        frame = {
            "step": steps,
            "instruction": instr,
            "observation": prev_obs,
            "available_actions": prev_avail,
            "history_tuples": list(history_records),   # full composite so far
            "proposed": {
                "c_puct": dbg.get("c_puct"),
                "total_N": dbg.get("root_total_N"),
                "candidates": root_actions
            },
            "harvested_rollouts": mcts.export_grpo_trajectories(instr),  # from this root
            "node_count": dbg.get("node_count"),
            "node_cap": dbg.get("node_cap"),
            "sims_done": dbg.get("sims_done"),
            "paths_found": dbg.get("paths_found"),
            "good_paths": dbg.get("good_paths"),
            "reward1_found": dbg.get("reward1_found")
        }
        frames.append(frame)

        print(f"step {steps} :: chose {chosen_env} | root_edges={dbg['root_edges']} | "
              f"sims={dbg['sims_done']} paths_found={dbg['paths_found']} good={dbg['good_paths']} "
              f"reward1_found={dbg['reward1_found']} nodes={dbg['node_count']}/{dbg['node_cap']}")

        # Pull the chosen 4-tuple from root
        chosen_full = next((a for a in (dbg.get("root_candidates") or []) if a.get("env") == chosen_env), None)
        if chosen_full is None:
            chosen_full = {"plan":"n/a","thought":"n/a","env":chosen_env,"expl":""}

        # Execute on LIVE env
        step_resp = ws.take_step(session_id, chosen_env)
        history_envs.append(chosen_env)
        history_records.append(chosen_full)

        done = ws.is_done(step_resp)
        steps += 1
        prev_obs = step_resp.get("observation", prev_obs)
        prev_avail = step_resp.get("available_actions", prev_avail)

        if done:
            final_R = float(step_resp.get("reward", 0.0))
            print(f"TERMINATED: reward={final_R:.3f}")
            break

    out = {
        "goal_idx": goal_idx,
        "final_reward": final_R,
        "steps": steps,
        "frames": frames,
        "final_history": history_records,            # full composite history for the live path
        "final_env_path": history_envs,              # env-only
        "final_node_table": mcts.export_node_scores()
    }
    fname = f"trajs/goal_{goal_idx}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] {fname}")

    logging.info({"goal_idx": goal_idx, "steps": steps, "reward": final_R})
    return final_R, steps

def run_many(goals: List[int],
             sims_per_root: int = 64,
             max_steps: int = 12,
             require_reward1: bool = True,
             target_high_paths: int = 3,
             min_reward: float = 0.8,
             path_depth_cap: int = 8,
             max_nodes_per_tree: int = 32):
    total, succ = 0, 0
    for g in goals:
        try:
            R, steps = run_one_goal(
                g,
                sims_per_root=sims_per_root,
                max_steps=max_steps,
                require_reward1=require_reward1,
                target_high_paths=target_high_paths,
                min_reward=min_reward,
                path_depth_cap=path_depth_cap,
                max_nodes_per_tree=max_nodes_per_tree
            )
            total += 1
            succ += int(R > 0.999)
            print(f"[agg] success_rate={succ/total:.3f}")
        except Exception as e:
            logging.exception({"goal_idx": g, "error": str(e)})
            total += 1
            print(f"[agg] (goal {g} failed) success_rate={succ/total:.3f}")

if __name__ == "__main__":
    run_many(goals=[1000],
             sims_per_root=1024,
             max_steps=12,
             require_reward1=True,
             target_high_paths=3,
             min_reward=0.3,
             path_depth_cap=20,
             max_nodes_per_tree=6000)
