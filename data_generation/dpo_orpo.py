import json, re, glob
from pathlib import Path
from typing import List, Dict, Any, Tuple

INPUT_DIR = r"D:\iisc_project\grpo\greedy_mcts\trajs"
OUTPUT = Path(INPUT_DIR) / "best_trajs.jsonl"


def extract_instruction(observation:str):
    m = re.search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]', observation, flags=re.I|re.S)
    return m.group(1).strip() if m else ""


def build_records(doc):
    final_env_path = doc['final_env_path']
    steps = doc['steps']
    final_node_table = doc['final_node_table']
    node_snapshots_all = doc['node_snapshots_all']
    out = []
    for step in range(steps):
        curr_env_action = final_env_path[step]
        filtered_nodes_for_curr_step = [k for k in final_node_table.values() if k['step_t']==step]
        edge_list_for_step = [k['edges'] for k in filtered_nodes_for_curr_step]
        best_id, key = choose_best_loc(edge_list_for_step, by="Q")
        best_node = filtered_nodes_for_curr_step[best_id]
        actions = best_node['actions']
        id_counter = 1
        action_data = []
        for action in actions:
            action_dict = {'action_id': id_counter}
            id_counter += 1
            action_dict['action'] = action
            for edge in best_node['edges'].keys():
                if action['env'] == edge:
                    action_dict['stats'] = best_node['edges'][edge]
            action_data.append(action_dict)
        url = best_node['url']
        data = {}
        for node in node_snapshots_all:
            children_actions = [k['action'] for k in node['children']]
            if url == node['state']['url'] and actions == children_actions:
                data = {'instruction': extract_instruction(node['state']['observation']),
                        'history': node['history'],
                        'observation': node['state']['observation'],
                        'actions_available': node['actions_available'],
                        'step': step,
                        'candidate_actions': action_data }
                break
        out.append(data)
    return out

def choose_best_loc(
    nodes: List[Dict[str, Dict[str, Any]]],
    by: str = "UCT",
    maximize: bool = True,
    missing_policy: str = "infer"  # "infer" -> -inf/+inf; or "error"
) -> Tuple[int, str]:
    if not nodes:
        raise ValueError("nodes is empty")

    best_idx = -1
    best_key = None
    best_val = float("-inf") if maximize else float("inf")

    for i, d in enumerate(nodes):
        if not isinstance(d, dict) or not d:
            continue
        for k, stats in d.items():
            if not isinstance(stats, dict):
                continue
            if by not in stats:
                if missing_policy == "error":
                    raise KeyError(f"Metric '{by}' missing in node {i}, key '{k}'")
                v = float("-inf") if maximize else float("inf")
            else:
                v = stats[by]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    if missing_policy == "error":
                        raise TypeError(f"Metric '{by}' not numeric in node {i}, key '{k}'")
                    v = float("-inf") if maximize else float("inf")

            if (maximize and v > best_val) or ((not maximize) and v < best_val):
                best_val = v
                best_idx = i
                best_key = k

    if best_idx == -1 or best_key is None:
        raise ValueError(f"No valid candidates for metric '{by}'")
    return best_idx, best_key


def main():
    files = sorted(glob.glob(str(Path(INPUT_DIR) / "*.json")))[:1000]
    with open(OUTPUT, "w", encoding="utf-8") as fout:
        counter = 0
        for f in files:
            try:
                doc = json.load(open(f, "r", encoding="utf-8"))
                for rec in build_records(doc):
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counter += 1
            except Exception as e:
                continue
        print("wrote", counter, "records to", OUTPUT)


if __name__ == "__main__":
    main()