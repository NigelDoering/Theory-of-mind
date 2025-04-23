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

    # Reset environment
    state, _ = env.reset()
    
    # Select initial goal using policy
    goal = agent._select_goal(state, eval_mode=True)
    
    # Track episode
    total_reward = 0
    steps = 0
    goals_reached = 0
    visited_goals = set()
    
    # Create figure for visualization
    plt.figure(figsize=(10, 10))
    
    for step in range(max_steps):
        # Select action
        action = agent._select_action(state, goal, eval_mode=True)
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update metrics
        total_reward += reward
        steps += 1
        
        # Check if we reached the current goal
        goal_reached = False
        if next_state == goal:
            goal_reached = True
            goals_reached += 1
            visited_goals.add(next_state)
            
            # Select new goal if not done
            if not done:
                goal = agent._select_goal(next_state, eval_mode=True)
        
        # Update state
        state = next_state
        
        # Visualize current state
        plt.clf()
        
        # Get map with agent position
        map_with_agent = env.get_map_with_agent()
        
        # Create visualization grid
        size = env.size
        grid = np.zeros((size, size, 3))  # RGB
        
        # Fill grid with colors
        for i in range(size):
            for j in range(size):
                cell = map_with_agent[i][j]
                
                if cell == 'H':  # Hole
                    grid[i, j] = [0.1, 0.1, 0.1]  # Dark gray
                elif cell == 'G':  # Goal
                    if env.pos_to_state[(i, j)] in visited_goals:
                        grid[i, j] = [0.8, 0.4, 0.8]  # Purple (visited goal)
                    else:
                        grid[i, j] = [0.0, 0.8, 0.0]  # Green (unvisited goal)
                elif cell == 'A':  # Agent
                    grid[i, j] = [1.0, 0.0, 0.0]  # Red
                else:  # Frozen
                    grid[i, j] = [0.8, 0.8, 1.0]  # Light blue
        
        # Plot grid
        plt.imshow(grid)
        
        # Mark the current goal
        goal_pos = env.state_to_pos[goal]
        plt.plot(goal_pos[1], goal_pos[0], 'yo', markersize=15, alpha=0.5)
        
        # Add text labels
        for i in range(size):
            for j in range(size):
                cell = map_with_agent[i][j]
                
                if cell == 'A':
                    plt.text(j, i, 'A', ha='center', va='center', color='white', fontweight='bold')
                elif cell == 'G':
                    state_id = env.pos_to_state[(i, j)]
                    reward = env.state_rewards[state_id]
                    plt.text(j, i, f'G\n{reward:.1f}', ha='center', va='center', 
                             color='white' if state_id in visited_goals else 'black')
                elif cell == 'H':
                    plt.text(j, i, 'H', ha='center', va='center', color='white')
        
        # Add status text
        plt.title(f"Step: {step}, Reward: {total_reward:.2f}, Goals: {goals_reached}/{len(env.goal_rewards)}")
        plt.grid(True, color='black', alpha=0.2)
        plt.tight_layout()
        
        # Capture frame
        plt.pause(0.1)
        
        # Break if done
        if done:
            print(f"Episode finished after {steps} steps, reward: {total_reward:.2f}, goals: {goals_reached}")
            break
    
    plt.close()

def plot_policy(env, agent, goal=None):
    """
    Visualize the agent's policy for a specific goal.
    
    Args:
        env: The environment
        agent: The agent
        goal: Goal state (if None, use final goal)
    """
    if goal is None:
        goal = agent.final_goal_state
    
    # Create a grid to visualize the policy
    size = env.size
    policy_grid = np.zeros((size, size), dtype=int)
    
    # Map of action to arrow
    action_arrows = ['←', '↓', '→', '↑', 'C']
    
    # For each state, determine the best action
    for state in range(agent.n_states):
        pos = env.state_to_pos.get(state)
        if pos:
            i, j = pos
            
            # Find best action for this state-goal pair
            q_values = [agent.low_q_tables[goal][(state, a)] for a in range(agent.n_actions)]
            best_action = np.argmax(q_values)
            
            # Store in grid
            policy_grid[i, j] = best_action
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
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
            elif cell == 'F':  # Final goal
                grid[i, j] = [1.0, 0.8, 0.0]  # Gold
            elif cell == 'A':  # Agent
                grid[i, j] = [1.0, 0.0, 0.0]  # Red
            elif cell == 'S':  # Start
                grid[i, j] = [0.5, 0.5, 1.0]  # Light blue
            else:  # Frozen
                grid[i, j] = [0.8, 0.8, 1.0]  # Light blue
    
    # Plot grid
    plt.imshow(grid)
    
    # Plot policy
    for i in range(size):
        for j in range(size):
            action = policy_grid[i, j]
            plt.text(j, i, action_arrows[action], ha='center', va='center', 
                     color='black' if grid[i, j].sum() > 1.5 else 'white',
                     fontsize=12, fontweight='bold')
    
    # Mark the goal
    goal_pos = env.state_to_pos[goal]
    plt.plot(goal_pos[1], goal_pos[0], 'yo', markersize=15, alpha=0.5)
    
    # Set limits and labels
    plt.title(f"Policy for Goal at {goal_pos}")
    plt.grid(True)
    plt.xticks(np.arange(0, size, 1))
    plt.yticks(np.arange(0, size, 1))
    plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
    plt.tight_layout()
    plt.savefig(f"policy_goal_{goal_pos[0]}_{goal_pos[1]}.png")
    plt.show()

def plot_reward_distribution(env):
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
    plt.savefig("reward_distribution.png")
    plt.show()