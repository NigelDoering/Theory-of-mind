import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )

import numpy as np
from real_world_src.models.tom_graph_encoder import CampusDataLoader

def main():
    data_loader = CampusDataLoader()
    path_data = data_loader.path_data
    goal_data = data_loader.goal_data
    node_id_mapping = data_loader.node_id_mapping
    max_seq_len = 100

    all_trajs, all_targets, all_true_goals, all_ts = [], [], [], []
    valid_node_ids = set(node_id_mapping.keys())
    for episode, agent_dict in goal_data.items():
        for agent_id, goal_node in agent_dict.items():
            if goal_node not in valid_node_ids:
                continue
            if agent_id in path_data.get(episode, {}):
                traj = path_data[episode][agent_id]
                traj = [n for n in traj if n in valid_node_ids]
                if len(traj) < 2:
                    continue
                traj_idx = [node_id_mapping[n] for n in traj]
                goal_idx = node_id_mapping[goal_node]
                for t in range(2, min(len(traj_idx)+1, max_seq_len+1)):
                    partial = traj_idx[:t]
                    pad_len = max_seq_len - len(partial)
                    if pad_len > 0:
                        partial = partial + [0]*pad_len
                    if t == len(traj_idx):
                        target_goal = goal_idx
                    else:
                        target_goal = partial[t-1]
                    all_trajs.append(partial)
                    all_targets.append(target_goal)
                    all_true_goals.append(goal_idx)
                    all_ts.append(t)
    # Save as numpy arrays
    np.savez_compressed(
        'stepwise_data.npz',
        trajs=np.array(all_trajs, dtype=np.int32),
        targets=np.array(all_targets, dtype=np.int32),
        true_goals=np.array(all_true_goals, dtype=np.int32),
        ts=np.array(all_ts, dtype=np.int32)
    )
    print(f"Saved stepwise data: {len(all_trajs)} samples to stepwise_data.npz")

if __name__ == "__main__":
    main() 