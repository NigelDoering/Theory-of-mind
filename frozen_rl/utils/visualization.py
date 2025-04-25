import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(episode_rewards, episode_steps, goals_reached):

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot rewards
    axs[0].plot(episode_rewards)
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')

    # Plot steps
    axs[1].plot(episode_steps)
    axs[1].set_title('Episode Steps')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')

    # Plot goals reached
    axs[2].plot(goals_reached)
    axs[2].set_title('Goals Reached per Episode')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Goals Reached')

    plt.tight_layout()
    plt.show()


def visualize_episode(env, agent, max_steps=100):
    """
    Visualize a single episode.
    
    Args:
        env: The environment
        agent: The agent 
        max_steps: Maximum steps per episode
    """
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    visited_goals = set()
    
    print("\nStarting episode visualization...")
    env.render()
    
    for step in range(max_steps):
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        
        # Check if reached a goal
        agent_pos = env.state_to_pos[next_state]
        if agent_pos in env.goal_positions and agent_pos not in visited_goals:
            visited_goals.add(agent_pos)
            goal_reward = env.goal_rewards.get(agent_pos, 0)
            print(f"Reached goal at {agent_pos} with reward {goal_reward:.2f}")
        
        # Render and pause
        env.render()
        time.sleep(0.3)  # Slow down for better visualization
        
        # Update state
        state = next_state
        
        if done:
            if terminated and agent_pos in env.hole_positions:
                print(f"Episode ended: Fell in hole at {agent_pos}")
            elif truncated:
                print(f"Episode ended: Maximum steps reached ({max_steps})")
            else:
                print(f"Episode ended: Terminated")
            break
    
    print(f"Episode complete: {steps} steps, {total_reward:.2f} total reward, {len(visited_goals)} goals visited")


# Replace the existing plot_policy function

def plot_policy(env, agent, goal=None, save_path=None):
    """
    Visualize the agent's policy for a specific goal with probability colorbar.
    
    Args:
        env: The environment
        agent: The agent
        goal: Goal state (if None, use final goal)
        save_path: Optional path to save the figure
    """
    if goal is None:
        goal = agent.final_goal_state
    
    # Create a grid to visualize the policy
    size = env.size
    policy_grid = np.zeros((size, size), dtype=int)
    confidence_grid = np.zeros((size, size))
    
    # Map of action to arrow
    action_arrows = ['←', '↓', '→', '↑', 'C']
    action_colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # For each state, determine the best action
    for state in range(agent.n_states):
        pos = env.state_to_pos.get(state)
        if pos:
            i, j = pos
            
            # Find best action for this state-goal pair
            q_values = [agent.low_q_tables[goal][(state, a)] for a in range(agent.n_actions)]
            best_action = np.argmax(q_values)
            
            # Calculate confidence (softmax probability of best action)
            q_values = np.array(q_values)
            q_values = q_values - np.max(q_values)  # For numerical stability
            probs = np.exp(q_values) / np.sum(np.exp(q_values))
            confidence = probs[best_action]
            
            # Store in grid
            policy_grid[i, j] = best_action
            confidence_grid[i, j] = confidence
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Get a sample map
    state, _ = env.reset()
    map_grid = env.get_map_with_agent()
    
    # Visualize the map
    grid = np.zeros((size, size, 3))  # RGB
    
    # Fill grid with colors
    for i in range(size):
        for j in range(size):
            cell = map_grid[i][j]
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            
            if cell == 'H':  # Hole
                grid[i, j] = [0.1, 0.1, 0.1]  # Dark gray
            elif cell == 'G':  # Goal
                grid[i, j] = [0.0, 0.8, 0.0]  # Green
            elif cell == 'A':  # Agent
                grid[i, j] = [1.0, 0.0, 0.0]  # Red
            else:  # Frozen
                grid[i, j] = [0.8, 0.8, 1.0]  # Light blue
    
    # Plot grid
    plt.imshow(grid)
    
    # Mark the current goal
    goal_pos = env.state_to_pos[goal]
    plt.plot(goal_pos[1], goal_pos[0], 'yo', markersize=15, alpha=0.5)
    
    # Overlay policy arrows with confidence-based transparency
    for i in range(size):
        for j in range(size):
            cell = map_grid[i][j]
            if isinstance(cell, bytes):
                cell = cell.decode('utf-8')
            
            # Skip holes and goals for clarity
            if cell == 'H' or cell == 'G':
                continue
                
            action = policy_grid[i, j]
            confidence = confidence_grid[i, j]
            
            # Use confidence for both size and color intensity
            arrow_size = 100 + 300 * confidence  # Scale arrow size
            color = action_colors[action]
            
            # Plot arrow with direction based on action
            if action == 0:  # Left
                plt.arrow(j, i, -0.3, 0, head_width=0.3, head_length=0.2, 
                         fc=color, ec=color, alpha=confidence, length_includes_head=True)
            elif action == 1:  # Down
                plt.arrow(j, i, 0, 0.3, head_width=0.3, head_length=0.2, 
                         fc=color, ec=color, alpha=confidence, length_includes_head=True)
            elif action == 2:  # Right
                plt.arrow(j, i, 0.3, 0, head_width=0.3, head_length=0.2, 
                         fc=color, ec=color, alpha=confidence, length_includes_head=True)
            elif action == 3:  # Up
                plt.arrow(j, i, 0, -0.3, head_width=0.3, head_length=0.2, 
                         fc=color, ec=color, alpha=confidence, length_includes_head=True)
            elif action == 4:  # Collect
                plt.scatter(j, i, s=arrow_size, marker='*', color=color, alpha=confidence)
    
    # Add colorbar for confidence - FIX: specify the axis
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Action Confidence')  # Add ax parameter
    
    # Add legend for actions
    for i, (arrow, color) in enumerate(zip(action_arrows, action_colors)):
        plt.plot([], [], color=color, marker='s', markersize=10, label=f'{arrow} ({i})')
    plt.legend(title="Actions", loc='upper right')
    
    # Set limits and labels
    plt.title(f"Policy for Goal at {goal_pos}")
    plt.grid(True, color='black', alpha=0.2)
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy visualization saved to {save_path}")
    
    plt.show()

# Update plot_reward_distribution to accept save_path
def plot_reward_distribution(env, save_path=None):
    """Visualize the reward distribution across goals."""
    plt.figure(figsize=(10, 6))
    
    goals = list(env.goal_rewards.keys())
    rewards = [env.goal_rewards[g] for g in goals]
    
    # Convert goals to strings for x-axis labels
    goal_labels = [f"Goal {i+1}: {g}" for i, g in enumerate(goals)]
    
    # Create bar chart
    plt.bar(range(len(goals)), rewards)
    plt.xticks(range(len(goals)), goal_labels, rotation=45, ha='right')
    plt.ylabel('Reward Value')
    plt.title('Goal Reward Distribution')
    
    # Mark final goal
    final_idx = goals.index(env.final_goal_pos) if env.final_goal_pos in goals else -1
    if final_idx >= 0:
        plt.bar(final_idx, rewards[final_idx], color='gold', label='Final Goal')
        plt.legend()
        
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
