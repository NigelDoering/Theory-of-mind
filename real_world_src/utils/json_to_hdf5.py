import json
import h5py
import numpy as np

def convert_jsonlines_to_hdf5(path_json, goal_json, out_path_h5, out_goal_h5):
    # Convert path_data.jsonl
    print(f"Converting {path_json} to {out_path_h5}")
    with h5py.File(out_path_h5, 'w') as h5f, open(path_json, 'r') as f:
        for line in f:
            ep_obj = json.loads(line)
            for ep, agents in ep_obj.items():
                grp = h5f.create_group(str(ep))
                for agent_id, path in agents.items():
                    grp.create_dataset(str(agent_id), data=np.array(path, dtype=np.int64))
    # Convert goal_data.jsonl
    print(f"Converting {goal_json} to {out_goal_h5}")
    with h5py.File(out_goal_h5, 'w') as h5f, open(goal_json, 'r') as f:
        for line in f:
            ep_obj = json.loads(line)
            for ep, agents in ep_obj.items():
                grp = h5f.create_group(str(ep))
                for agent_id, goal in agents.items():
                    grp.create_dataset(str(agent_id), data=np.array(goal, dtype=np.int64))

if __name__ == "__main__":
    convert_jsonlines_to_hdf5('./data/10k/path_data.json', './data/10k/goal_data.json', './data/10k/path_data.h5', './data/10k/goal_data.h5')