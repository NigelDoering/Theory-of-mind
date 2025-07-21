import os
import sys
import numpy as np
import json
import pickle
import h5py

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

data_dir = os.path.join(project_root, 'data/500')
os.makedirs(data_dir, exist_ok=True)
agents_path = os.path.join(data_dir, 'agents.pkl')
path_data_path = os.path.join(data_dir, 'path_data.json')
goal_data_path = os.path.join(data_dir, 'goal_data.json')
path_h5_path = os.path.join(data_dir, 'path_data.h5')
goal_h5_path = os.path.join(data_dir, 'goal_data.h5')

from real_world_src.agents.agent_factory import AgentFactory
from real_world_src.environment.campus_env import CampusEnvironment
import networkx as nx

campus = None
goals = None

def new_episode(agents):
    global campus, goals
    for agent in agents:
        agent.path = []
        agent.goal_node = int(np.random.choice(goals, size=1, p=agent.goal_distribution)[0])
        while True:
            start_node = campus.get_random_node()
            try:
                path = nx.shortest_path(
                    campus.G_undirected,
                    source=start_node,
                    target=agent.goal_node,
                    weight='length'
                )
                if len(path) > 15:
                    break
            except Exception:
                continue
        agent.start_node = start_node
        agent.current_node = start_node

def main_json():
    global campus, goals
    campus = CampusEnvironment()
    # Landmark nodes as goals
    goals = [
        469084068, 49150691, 768264666, 1926666015, 1926673385, 49309735,
        273627682, 445989107, 445992528, 446128310, 1772230346, 1926673336,
        2872424923, 3139419286, 4037576308
    ]

    print(f"Building the required agents and environment...")

    # Create agents
    n_agents = 100
    agents = [AgentFactory.create_agent("shortest") for _ in range(n_agents)]
    for i, agent in enumerate(agents):
        agent.id = i

    n_goals = len(goals)
    ag_alpha = np.random.normal(1, 0.2, size=n_goals)
    for agent in agents:
        agent.goal_distribution = np.random.dirichlet(alpha=np.ones(n_goals) * ag_alpha, size=1)[0]
        agent.environment = campus

    # Save agents to file
    with open(agents_path, 'wb') as f:
        pickle.dump(agents, f)

    print(f"Agents and environment are ready. Starting data generation...")

    # Generate data for each agent
    path_data = {}
    goal_data = {}
    n_episodes = 500

    # Write to file in chunks to avoid memory issues
    chunk_size = 10000
    for episode in range(n_episodes):
        new_episode(agents)
        episode_path_data = {}
        episode_goal_data = {}
        for agent in agents:
            agent.plan_path()
            episode_path_data[agent.id] = agent.path
            episode_goal_data[agent.id] = agent.goal_node
        path_data[episode] = episode_path_data
        goal_data[episode] = episode_goal_data

        # Periodically write to disk and clear memory
        if (episode + 1) % chunk_size == 0:
            with open(path_data_path, "a") as file:
                json.dump({k: path_data[k] for k in range(episode - chunk_size + 1, episode + 1)}, file)
                file.write('\n')
            with open(goal_data_path, "a") as file:
                json.dump({k: goal_data[k] for k in range(episode - chunk_size + 1, episode + 1)}, file)
                file.write('\n')
            path_data.clear()
            goal_data.clear()

    # Write any remaining data
    if path_data:
        with open(path_data_path, "a") as file:
            json.dump(path_data, file)
            file.write('\n')
    if goal_data:
        with open(goal_data_path, "a") as file:
            json.dump(goal_data, file)
            file.write('\n')

    print(f"Data generation completed. {n_episodes} episodes generated.")

def main_h5():
    global campus, goals
    campus = CampusEnvironment()
    goals = [
        469084068, 49150691, 768264666, 1926666015, 1926673385, 49309735,
        273627682, 445989107, 445992528, 446128310, 1772230346, 1926673336,
        2872424923, 3139419286, 4037576308
    ]

    print(f"Building the required agents and environment...")

    n_agents = 100
    n_episodes = 1000
    max_path_len = 100  # adjust as needed

    agents = [AgentFactory.create_agent("shortest") for _ in range(n_agents)]
    for i, agent in enumerate(agents):
        agent.id = i

    n_goals = len(goals)
    ag_alpha = np.random.normal(1, 0.2, size=n_goals)
    for agent in agents:
        agent.goal_distribution = np.random.dirichlet(alpha=np.ones(n_goals) * ag_alpha, size=1)[0]
        agent.environment = campus

    # Save agents as pickle (efficient for Python objects)
    with open(agents_path, 'wb') as f:
        pickle.dump(agents, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Agents and environment are ready. Starting data generation...")

    # Preallocate HDF5 datasets
    with h5py.File(path_h5_path, "w") as path_h5, h5py.File(goal_h5_path, "w") as goal_h5:
        path_ds = path_h5.create_dataset(
            "paths",
            shape=(n_episodes, n_agents, max_path_len),
            dtype='int64',
            compression="gzip",
            chunks=(1, n_agents, max_path_len)
        )
        path_len_ds = path_h5.create_dataset(
            "path_lengths",
            shape=(n_episodes, n_agents),
            dtype='int32',
            compression="gzip",
            chunks=(1, n_agents)
        )
        goal_ds = goal_h5.create_dataset(
            "goals",
            shape=(n_episodes, n_agents),
            dtype='int64',
            compression="gzip",
            chunks=(1, n_agents)
        )

        for episode in range(n_episodes):
            new_episode(agents)
            for agent in agents:
                agent.plan_path()
                path = agent.path[:max_path_len]
                path_len = len(path)
                # Pad path if shorter than max_path_len
                padded_path = np.zeros(max_path_len, dtype=np.int64)
                padded_path[:path_len] = path
                path_ds[episode, agent.id, :] = padded_path
                path_len_ds[episode, agent.id] = path_len
                goal_ds[episode, agent.id] = agent.goal_node

            if (episode + 1) % 100 == 0:
                print(f"Generated {episode+1}/{n_episodes} episodes...")

    print(f"Data generation completed. {n_episodes} episodes generated.")

if __name__ == "__main__":
    # main_h5()
    main_json()