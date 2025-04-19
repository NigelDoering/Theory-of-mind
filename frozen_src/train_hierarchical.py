import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frozen_src.hierarchical_agent import HierarchicalAgent

def train_hierarchical_agent(env_name='FrozenLake-v1', map_size=5, is_slippery=False, 
                            n_episodes=1000, max_steps=100, seed=123):
    """
    Train a hierarchical agent on the FrozenLake environment.
    
    Args:
        env_name: Name of the environment
        map_size: Size of the grid
        is_slippery: Whether the environment is slippery
        n_episodes: Number of training episodes
        max_steps: Maximum number of steps per episode
        seed: Random seed
        
    Returns:
        Trained agent and training statistics
    """
    # Create the environment
    env = gym.make(
        env_name,
        is_slippery=is_slippery,
        desc=generate_random_map(size=map_size, p=0.9, seed=seed),
        render_mode="rgb_array"
    )
    env.action_space.seed(seed)
    np.random.seed(seed)
    
    # Create the agent
    agent = HierarchicalAgent(
        env=env,
        budget=max_steps,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.2
    )
    
    # Training statistics
    rewards_history = []
    steps_history = []
    goals_reached_history = []
    
    # Training loop
    for episode in range(n_episodes):
        # Reset environment and agent
        state = env.reset(seed=seed)[0]
        agent.reset(max_steps)
        
        # Episode statistics
        total_reward = 0
        steps = 0
        done = False
        
        # Simulate discovery of start position
        agent.update_observations(state, None)
        
        while not done and steps < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Take the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update observations
            goal_reached = agent.update_observations(next_state, None)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state
            
            # Break if final goal reached
            if next_state == agent.final_goal:
                break
        
        # Record episode statistics
        rewards_history.append(total_reward)
        steps_history.append(steps)
        goals_reached_history.append(len(agent.reached_goals))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_steps = np.mean(steps_history[-10:])
            avg_goals = np.mean(goals_reached_history[-10:])
            print(f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.2f} | Avg Goals Reached: {avg_goals:.2f}")
    
    # Plot training statistics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(steps_history)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(1, 3, 3)
    plt.plot(goals_reached_history)
    plt.title('Goals Reached per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Goals Reached')
    
    plt.tight_layout()
    plt.savefig('hierarchical_agent_training.png')
    plt.show()
    
    return agent, {
        'rewards': rewards_history,
        'steps': steps_history,
        'goals_reached': goals_reached_history
    }

if __name__ == "__main__":
    # Train the agent
    agent, stats = train_hierarchical_agent(
        map_size=8,
        is_slippery=False,
        n_episodes=500,
        max_steps=200
    )