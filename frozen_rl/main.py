import os
import sys
import numpy as np
import random
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

from environments.multi_goal_frozen_lake import MultiGoalFrozenLakeEnv
from agents.flat_q_agent import FlatQLearningAgent  # Use the new flat agent
from utils.experiment import save_experiment_config, visualize_episode
from utils.visualization import plot_policy, plot_reward_distribution

# Configuration for all agents
N_AGENTS = 10
MIN_SIZE = 8
MAX_SIZE = 12
BASE_SEED = 14
MAX_EPISODES = 10000  # Limit for training episodes
CONVERGENCE_THRESHOLD = 0.95  # Consider converged at 95% of optimal performance
MIN_EPISODES = 2000  # Minimum episodes even if converged early

def train_single_agent(agent_id, size, n_holes, seed, output_dir):
    """
    Train a single agent with specific parameters and save results.
    
    Args:
        agent_id: Unique identifier for this agent
        size: Grid size (NxN)
        n_holes: Number of holes to place
        seed: Random seed
        output_dir: Directory to save results
    """
    print(f"\n{'='*50}")
    print(f"Training Agent {agent_id} (Size: {size}×{size}, Seed: {seed})")
    print(f"{'='*50}")
    
    # Calculate number of goals based on size
    n_goals = size + 1
    
    # Create environment with NO rendering during training
    env = MultiGoalFrozenLakeEnv(
        size=size,
        max_steps=int(size**2 * 1.5),  # Give enough steps to reach multiple goals
        n_goals=n_goals, 
        n_holes=n_holes,
        is_slippery=False,  # Deterministic movement for easier learning
        seed=seed,
        render_mode=None,
        randomize_start=True,
        fixed_map=True
    )
    
    # Create flat Q-learning agent with improved parameters
    agent = FlatQLearningAgent(
        env=env,
        alpha=0.7,            # Higher learning rate
        gamma=0.99,           # Keep high discount factor
        epsilon=1.0,          # Start with full exploration
        epsilon_decay=0.998,  # Slower decay for more exploration
        min_epsilon=0.02      # Lower minimum exploration
    )
    
    # Save agent and environment configuration
    agent_config = {
        "agent_id": agent_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_params": {
            "size": size,
            "n_goals": n_goals,
            "n_holes": n_holes,
            "is_slippery": False,
            "seed": seed
        },
        "goal_positions": [(pos[0], pos[1]) for pos in env.goal_positions],
        "goal_rewards": {f"{pos[0]},{pos[1]}": reward for pos, reward in env.goal_rewards.items()},
        "agent_alpha": env.agent_alpha.tolist() if hasattr(env, 'agent_alpha') else None,
        "dirichlet_probabilities": env.dirichlet_probs.tolist() if hasattr(env, 'dirichlet_probs') else None,
        "learning_params": {
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "initial_epsilon": agent.epsilon
        }
    }
    
    # Save configuration
    config_path = os.path.join(output_dir, f"agent_{agent_id}_config.json")
    save_experiment_config(agent_config, config_path)
    
    # Training with convergence check
    print(f"\nTraining Agent {agent_id}...")
    best_reward = float('-inf')
    converged = False
    
    for episode in range(MAX_EPISODES):
        # Add periodic evaluation to monitor progress
        if episode % 5000 == 0:
            eval_reward, eval_steps, eval_goals = agent.evaluate(n_episodes=5, max_steps=size*size*2)
            print(f"Episode {episode}: Reward={eval_reward:.2f}, Steps={eval_steps:.1f}, Goals={eval_goals:.1f}/{env.n_goals}")
        
        # Train for one episode with increased max steps 
        agent.train(n_episodes=1, max_steps=int(size**2 * 2),  # Increased max_steps for better exploration
                   eval_interval=float('inf'), render=False)
        
        # Print progress
        if episode % 1000 == 0:
            print(f"Episode {episode}/{MAX_EPISODES} complete.")
    
    # Switch to human rendering mode for final visualization
    env.render_mode = 'human'
    
    # Generate all visualizations
    print("\nGenerating visualizations and saving results...")
    visualize_agent_performance(env, agent, output_dir, agent_id)
    
    # Save trained agent
    model_path = os.path.join(output_dir, f"agent_{agent_id}_model.pkl")
    agent.save(model_path)
    
    # Final evaluation episode
    print(f"\nFinal evaluation for Agent {agent_id}:")
    eval_reward, eval_steps, eval_goals = agent.evaluate(n_episodes=1, max_steps=size*size*2, render=True)
    print(f"Final evaluation: reward={eval_reward:.2f}, steps={eval_steps}, goals={eval_goals}/{env.n_goals}")
    
    # Close environment
    env.close()
    
    print(f"\nAgent {agent_id} training {'converged' if converged else 'completed'}!")
    
    # Add this section inside the train_single_agent function after training
    # First, make sure the improved agent is working properly by evaluating

    # Standard evaluation
    print(f"\nEvaluating Agent {agent_id} on standard metrics:")
    eval_reward, eval_steps, eval_goals = agent.evaluate(n_episodes=5, max_steps=size*size*2)
    print(f"Reward: {eval_reward:.2f}, Steps: {eval_steps:.1f}, Goals: {eval_goals:.1f}/{env.n_goals}")

    # Goal collection rate
    print("\nGoal collection analysis:")
    for i in range(5):  # 5 test episodes 
        state, _ = env.reset()
        visited_goals = set()
        steps = 0

        for step in range(size*size*2):
            action = agent.select_action(state, visited_goals, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Check for goals
            pos = env.state_to_pos[next_state]
            if pos in env.goal_positions and pos not in visited_goals:
                visited_goals.add(pos)
                goal_reward = env.goal_rewards.get(pos, 0)
                print(f"  Step {step}: Collected goal at {pos} with reward {goal_reward:.2f}")

            state = next_state
            steps += 1

            if terminated or truncated:
                break

        print(f"  Episode {i+1}: Collected {len(visited_goals)}/{env.n_goals} goals in {steps} steps")
    
    return agent_config

def plot_policy(env, agent, save_path=None):
    """
    Create an enhanced visualization of the agent's policy with integrated Q-values.
    
    Args:
        env: The environment
        agent: The agent
        save_path: Optional path to save the figure
    """
    # Get policy (best action for each state)
    policy = agent.get_policy()
    
    # Create a grid to visualize the policy
    size = env.size
    policy_grid = np.zeros((size, size), dtype=int)
    q_max_grid = np.zeros((size, size))
    
    # Action names and symbols
    action_names = ['Left', 'Down', 'Right', 'Up']
    action_symbols = ['←', '↓', '→', '↑']
    action_colors = ['blue', 'green', 'red', 'purple']
    
    # Fill grids with policy and max Q-values
    for i in range(size):
        for j in range(size):
            if (i, j) in env.pos_to_state:
                state = env.pos_to_state[(i, j)]
                policy_grid[i, j] = policy[state]
                # Get maximum Q-value for this state
                q_values = [agent.q_table[state][a] for a in range(agent.n_actions)]
                q_max_grid[i, j] = max(q_values)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Create heatmap background based on max Q-values
    im = plt.imshow(q_max_grid, cmap='Blues', alpha=0.7)
    cbar = plt.colorbar(im)
    cbar.set_label('Max Q-value')
    
    # Add gridlines
    plt.grid(True, color='black', alpha=0.2)
    
    # Plot policy as arrows
    for i in range(size):
        for j in range(size):
            if (i, j) in env.hole_positions:
                # Draw holes as X markers
                plt.plot(j, i, 'X', markersize=15, color='red')
                plt.text(j, i, 'H', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            elif (i, j) in env.goal_positions:
                # Draw goals as colored circles with rewards
                reward = env.goal_rewards.get((i, j), 0)
                plt.plot(j, i, 'o', markersize=15, 
                        color='green', alpha=0.8)
                plt.text(j, i, f'{reward:.1f}', ha='center', va='center', 
                        color='white', fontsize=10, fontweight='bold')
            else:
                # Draw policy action
                if (i, j) in env.pos_to_state:
                    state = env.pos_to_state[(i, j)]
                    action = policy[state]
                    color = action_colors[action]
                    plt.text(j, i, action_symbols[action], ha='center', va='center', 
                            color=color, fontsize=18, fontweight='bold')
    
    # Add legend - FIXED: Don't use symbols as markers
    for i, (name, symbol, color) in enumerate(zip(action_names, action_symbols, action_colors)):
        plt.plot([], [], color=color, marker='o', linestyle='None', 
                markersize=10, label=f'{symbol} {name} ({i})')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
    
    # Labels and title
    plt.title("Agent Policy with Q-Value Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    # Set integer ticks
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Enhanced policy visualization saved to {save_path}")
    
    plt.show()

def visualize_agent_performance(env, agent, output_dir, agent_id):
    """
    Create a comprehensive visualization of agent performance.
    
    Args:
        env: The environment
        agent: The trained agent
        output_dir: Directory to save visualizations
        agent_id: Agent identifier for filenames
    """
    # 1. Training progress
    print("\nGenerating enhanced learning curve and statistics...")
    plot_path = os.path.join(output_dir, f"agent_{agent_id}_training_progress.png")
    agent.plot_training_progress(window_size=50, save_path=plot_path)
    
    # 2. Policy visualization with Q-value heatmap
    print("\nGenerating policy visualization...")
    policy_path = os.path.join(output_dir, f"agent_{agent_id}_policy.png")
    plot_policy(env, agent, save_path=policy_path)
    
    # 3. Q-value visualizations for all actions
    print("\nSaving detailed Q-value visualization...")
    q_viz_path = os.path.join(output_dir, f"agent_{agent_id}_q_values.png")
    agent.visualize_q_values(save_path=q_viz_path)
    
    # 4. Summary statistics
    print("\nAgent Performance Summary:")
    print(f"  - Total training episodes: {len(agent.episode_rewards)}")
    print(f"  - Final epsilon: {agent.epsilon:.5f}")
    print(f"  - Average reward (last 100 episodes): {np.mean(agent.episode_rewards[-100:]):.2f}")
    print(f"  - Average steps (last 100 episodes): {np.mean(agent.episode_steps[-100:]):.2f}")
    print(f"  - Average goals (last 100 episodes): {np.mean(agent.goals_reached[-100:]):.2f}")
    print(f"  - Maximum goals in any episode: {agent.max_goals_per_episode}")

def main():
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"frozen_rl/experiments/results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all agent configurations
    all_configs = []
    
    # Train N different agents
    for agent_id in range(1, N_AGENTS + 1):
        # Generate unique configuration for this agent
        size = random.randint(MIN_SIZE, MAX_SIZE)
        n_holes = int(size*1.5)  # FIXED: Convert to integer!
        seed = BASE_SEED + agent_id
        
        # Train agent
        agent_config = train_single_agent(agent_id, size, n_holes, seed, output_dir)
        all_configs.append(agent_config)
    
    # Save summary of all agents
    summary_path = os.path.join(output_dir, "all_agents_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_configs, f, indent=2)
    
    print("\nTraining complete! All agent data saved to:", output_dir)

if __name__ == "__main__":
    main()