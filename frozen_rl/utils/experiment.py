import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any

def save_experiment_config(config: Dict[str, Any], filename: str) -> None:
    """Save experiment configuration to a JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert any non-serializable objects
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            serializable_config[key] = {str(k): float(v) for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            serializable_config[key] = value.tolist()
        elif isinstance(value, (list, tuple)) and all(isinstance(pos, tuple) for pos in value):
            serializable_config[key] = [list(pos) for pos in value]
        else:
            serializable_config[key] = value
    
    # Add timestamp
    serializable_config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'w') as f:
        json.dump(serializable_config, f, indent=4)
    
    print(f"Experiment configuration saved to {filename}")

def load_experiment_config(filename: str) -> Dict[str, Any]:
    """Load experiment configuration from a JSON file."""
    with open(filename, 'r') as f:
        config = json.load(f)
    
    # Convert types back
    if 'goal_ids' in config:
        config['goal_ids'] = [tuple(pos) for pos in config['goal_ids']]
    
    return config

def create_experiment_config(
    goal_ids: list,
    goal_rewards: list,
    alpha_values: Dict[str, float],
    dirichlet_probabilities: Dict[str, float],
    final_goal: Any,
    seed: int
) -> Dict[str, Any]:
    return {
        "goal_ids": goal_ids,
        "goal_rewards": goal_rewards,
        "alpha_values": alpha_values,
        "dirichlet_probabilities": dirichlet_probabilities,
        "final_goal": final_goal,
        "seed": seed
    }

def visualize_episode(env, agent, max_steps=100):
    """
    Visualize an episode with the agent's policy.
    
    Args:
        env: The environment
        agent: The agent
        max_steps: Maximum steps per episode
    """
    # Reset environment
    state, _ = env.reset()
    
    # Select initial goal using policy
    goal = agent._select_goal(state, eval_mode=True)
    
    # Track episode
    total_reward = 0
    goals_reached = 0
    
    for step in range(max_steps):
        # Select action
        action = agent._select_action(state, goal, eval_mode=True)
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update metrics
        total_reward += reward
        
        # Check if we reached the goal and collected
        goal_pos = env.state_to_pos[goal]
        agent_pos = env.state_to_pos[next_state]
        
        if agent_pos == goal_pos and action == 4:  # Collect action
            goals_reached += 1
            if not done:
                goal = agent._select_goal(next_state, eval_mode=True)
        
        # Update state
        state = next_state
        
        # Render
        env.render()
        
        # Break if done
        if done:
            print(f"Episode finished after {step+1} steps, reward: {total_reward:.2f}, goals: {goals_reached}")
            break
    
    return total_reward, step+1, goals_reached