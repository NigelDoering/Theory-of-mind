import os
import sys
import numpy as np

from environments.multi_goal_frozen_lake import MultiGoalFrozenLakeEnv
from agents.hierarchical_agent import HierarchicalQLearningAgent
from utils.experiment import save_experiment_config, visualize_episode
from utils.visualization import plot_policy, plot_reward_distribution

def main():
    # Experiment parameters
    size = 8
    n_goals = 5
    n_holes = 10
    is_slippery = False
    seed = 42
    n_episodes = 5000
    save_path = "models/hierarchical_agent.pkl"
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments/config_samples", exist_ok=True)
    
    # Set up environment
    env = MultiGoalFrozenLakeEnv(
        size=size,
        max_steps=int(size**2 + size*np.sqrt(2)),
        n_goals=n_goals, 
        n_holes=n_holes,
        is_slippery=is_slippery, 
        seed=seed,
        render_mode='human'
    )
    
    # Display initial map
    print("\nInitial Map:")
    env.render()
    
    # Visualize reward distribution
    plot_reward_distribution(env)

    # Create agent
    agent = HierarchicalQLearningAgent(
        env=env,
        high_alpha=0.2,
        low_alpha=0.5,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )

    # Save experiment configuration
    config = {
        "env_params": {
            "size": size,
            "n_goals": n_goals,
            "n_holes": n_holes,
            "is_slippery": is_slippery,
            "seed": seed
        },
        "goal_ids": env.goal_positions,
        "goal_rewards": env.goal_rewards,
        "agent_alpha": env.agent_alpha.tolist() if hasattr(env, 'agent_alpha') else None,
        "dirichlet_probabilities": env.dirichlet_probs.tolist() if hasattr(env, 'dirichlet_probs') else None,
        "final_goal": env.final_goal_pos,
        "alpha_values": {
            "high_alpha": agent.high_alpha,
            "low_alpha": agent.low_alpha
        }
    }
    save_experiment_config(config, "experiments/config_samples/example_config.json")

    # Train agent with live rendering occasionally
    print("\nTraining agent...")
    agent.train(n_episodes=n_episodes, max_steps=size*size*4, eval_interval=500, render=True)

    # Plot training progress
    agent.plot_training_progress()
    
    # Save trained agent
    agent.save(save_path)
    
    # Visualize policy for final goal
    plot_policy(env, agent, agent.final_goal_state)
    
    # Run and visualize final evaluation episode
    print("\nFinal evaluation:")
    visualize_episode(env, agent, max_steps=size*size*4)

    print("Training complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)