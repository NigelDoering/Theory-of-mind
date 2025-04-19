import os
import sys
import json

import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from IPython.display import clear_output, display, HTML

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frozen_src.hierarchical_agent import HierarchicalAgent
from frozen_src.goal_space_generator import GoalSpaceGenerator

def visualize_environment(desc):
    """Visualize the FrozenLake environment using matplotlib"""
    plt.figure(figsize=(7, 7))
    
    # Create a grid of the same size as the environment
    nrow, ncol = desc.shape
    grid = np.zeros((nrow, ncol, 3), dtype=float)  # RGB array
    
    # Define colors for different tiles
    colors = {
        b'S': [0.0, 0.5, 0.0],    # Start: Dark Green
        b'F': [0.9, 0.9, 1.0],    # Frozen: Light Blue
        b'H': [0.2, 0.2, 0.8],    # Hole: Dark Blue
        b'G': [1.0, 0.9, 0.0]     # Goal: Gold
    }
    
    # Fill the grid with colors based on the environment
    for i in range(nrow):
        for j in range(ncol):
            grid[i, j] = colors[desc[i][j]]
    
    # Plot the grid
    plt.imshow(grid)
    
    # Add text annotations
    tile_dict = {b'S': 'S', b'F': 'F', b'H': 'H', b'G': 'G'}
    for i in range(nrow):
        for j in range(ncol):
            plt.text(j, i, tile_dict[desc[i][j]], 
                     ha="center", va="center", color="black", fontsize=15)
    
    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    
    # Add a title with map size
    plt.title(f"FrozenLake Environment ({nrow}x{ncol})")
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('frozen_lake_map.png')  # Save the visualization
    plt.show()

def print_text_map(desc):
    """Print a text-based visualization of the map to the console"""
    nrow, ncol = desc.shape
    print("FrozenLake Map:")
    print("┌" + "─" * (ncol * 2 - 1) + "┐")
    for i in range(nrow):
        row_str = "│"
        for j in range(ncol):
            tile = desc[i][j]
            if tile == b'S':
                symbol = "S"  # Start
            elif tile == b'F':
                symbol = "·"  # Frozen tile
            elif tile == b'H':
                symbol = "O"  # Hole
            elif tile == b'G':
                symbol = "G"  # Goal
            row_str += symbol
            if j < ncol - 1:
                row_str += " "
        row_str += "│"
        print(row_str)
    print("└" + "─" * (ncol * 2 - 1) + "┘")
    print("S: Start, ·: Frozen, O: Hole, G: Goal")

def run_with_goal_space(map_size=8, n_goals=None, seed=None, is_slippery=False,
                     n_episodes=5, max_steps=200, delay=0.3, 
                     perception_radius=2, visualize_every=5, load_config=None):
    """
    Run the hierarchical agent with a rich goal space.
    
    Args:
        map_size: Size of the grid (nxn)
        n_goals: Number of goals (default: map_size+1)
        seed: Random seed for reproducibility
        is_slippery: Whether the environment is slippery
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        delay: Delay between frames in seconds
        perception_radius: How far the agent can perceive
        visualize_every: Create visualizations every X steps
        load_config: Path to existing goal space config to load
    """
    # Create output directory for visualizations
    os.makedirs("exploration_visuals", exist_ok=True)
    
    # Generate or load goal space
    if load_config:
        print(f"Loading goal space from {load_config}")
        goal_generator = GoalSpaceGenerator.load_from_config(load_config)
    else:
        print(f"Generating new goal space (size: {map_size}x{map_size}, goals: {n_goals or map_size+1})")
        goal_generator = GoalSpaceGenerator(map_size=map_size, n_goals=n_goals, seed=seed)
        goal_generator.generate_goal_space()
    
    # Visualize the goal space
    goal_generator.visualize_goal_space(save_path="goal_space.png")
    
    # Get the custom map with goals
    custom_map = goal_generator.map
    
    # Create the environment with the custom map
    env = gym.make(
        'FrozenLake-v1',
        is_slippery=is_slippery,
        desc=custom_map,
        render_mode="human"
    )
    env.action_space.seed(goal_generator.seed)
    np.random.seed(goal_generator.seed)
    
    # Print environment details
    print(f"Environment initialized with {len(goal_generator.goal_positions)} goals")
    print(f"Final goal at position: {goal_generator.final_goal}")
    
    # Visualize the environment
    print_text_map(env.unwrapped.desc)
    visualize_environment(env.unwrapped.desc)
    
    # Create the agent
    agent = HierarchicalAgent(
        env=env,
        budget=max_steps,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.1,
        perception_radius=perception_radius
    )
    
    # Create a dictionary for tracking discovered goal rewards
    discovered_goal_rewards = {}
    
    # Display visualization legend
    print("\n--- VISUALIZATION LEGEND ---")
    print("• Gray tiles: Unknown areas")
    print("• Light Blue: Frozen tiles")
    print("• Dark Blue: Holes")
    print("• Gold: Goals")
    print("• Red border: Current position")
    print("• Yellow border: Perception field")
    print("• Green border: Discovered goals")
    print("• Purple border: Reached goals")
    print("• Cyan border: Current subgoal")
    print("• Orange border: Final goal")
    print("• Red line: Agent's path history")
    print("---------------------------\n")
    
    # Run episodes
    for episode in range(n_episodes):
        print(f"Episode {episode+1}/{n_episodes}")
        
        # Reset environment and agent
        state = env.reset()[0]
        agent.reset(max_steps)
        
        # Episode statistics
        total_reward = 0
        discovered_goals_this_episode = set()
        steps = 0
        done = False
        
        # Simulate discovery of start position
        agent.update_observations(state, None)
        
        # Initial visualization
        agent.visualize_knowledge(episode+1, steps)
        agent.visualize_plan(episode+1, steps)
        
        while not done and steps < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Take the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Check if we've reached a goal and get its real reward
            row, col = next_state // env.unwrapped.ncol, next_state % env.unwrapped.ncol
            pos = (row, col)
            
            # If it's a goal position and not yet discovered, apply the true reward
            if pos in goal_generator.goal_positions and pos not in discovered_goal_rewards:
                true_reward = goal_generator.goal_rewards[pos]
                discovered_goal_rewards[pos] = true_reward
                discovered_goals_this_episode.add(pos)
                
                # Special handling for final goal
                is_final = (pos == goal_generator.final_goal)
                
                print(f"\n*** GOAL DISCOVERED at {pos} ***")
                print(f"Goal ID: {goal_generator.goal_ids[pos]}")
                print(f"Reward: {true_reward:.2f}")
                if is_final:
                    print("This is the FINAL GOAL!")
                print("***************************\n")
                
                # Update reward with true value
                reward = true_reward
                
                # Only terminate if this is the final goal AND we've discovered at least 3 goals
                # This encourages the agent to find more goals before finishing
                if is_final and len(discovered_goal_rewards) >= 3:
                    print(f"Final goal reached after discovering {len(discovered_goal_rewards)} goals!")
                    done = True
                elif is_final:
                    print("Found the final goal, but continuing to explore for more goals...")
            
            # Update observations and get exploration bonus
            exploration_bonus, goal_reached = agent.update_observations(next_state, None)
            reward += exploration_bonus  # Add exploration bonus to reward
            
            # Special override for the agent's rewards for discovered goals
            for pos, true_reward in discovered_goal_rewards.items():
                row, col = pos
                goal_state = row * env.unwrapped.ncol + col
                agent.rewards[goal_state] = true_reward
                
                # If it's the final goal, mark it
                if pos == goal_generator.final_goal:
                    agent.final_goal = goal_state
                    
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Render and pause
            env.render()
            time.sleep(delay)
            
            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state
            
            # Create visualizations
            if steps % visualize_every == 0 or done:
                agent.visualize_knowledge(episode+1, steps)
                agent.visualize_plan(episode+1, steps)
            
            # Display info
            clear_output(wait=True)
            print(f"Episode {episode+1}/{n_episodes} | Step {steps}/{max_steps}")
            print(f"Current State: {state} | Total Reward: {total_reward:.2f}")
            print(f"Current Subgoal: {agent.current_subgoal}")
            print(f"Goals Reached: {agent.reached_goals}")
            print(f"Budget Remaining: {agent.remaining_budget}")
            print(f"Known Map: {np.sum(agent.known_map)}/{agent.known_map.size} tiles")
            
            # Show discovered goals and their rewards
            if discovered_goal_rewards:
                print("\nDiscovered Goals:")
                for pos, reward_value in discovered_goal_rewards.items():
                    goal_state = pos[0] * env.unwrapped.ncol + pos[1]
                    reached = goal_state in agent.reached_goals
                    goal_id = goal_generator.goal_ids[pos]
                    is_final = (pos == goal_generator.final_goal)
                    status = "REACHED" if reached else "DISCOVERED"
                    final_marker = " (FINAL)" if is_final else ""
                    print(f"  - Goal {goal_id}{final_marker}: Position {pos}, Reward: {reward_value:.2f}, Status: {status}")
            
            # Show visualizations
            if steps % visualize_every == 0 or done:
                latest_knowledge = f"exploration_visuals/knowledge_ep{episode+1}_step{steps}.png"
                latest_plan = f"exploration_visuals/plan_ep{episode+1}_step{steps}.png"
                
                display(HTML(f"""
                <div style="display: flex; justify-content: center;">
                    <div style="margin: 10px;">
                        <img src="{latest_knowledge}" width="500"/>
                    </div>
                    <div style="margin: 10px;">
                        <img src="{latest_plan}" width="500"/>
                    </div>
                </div>
                """))
        
        # Episode summary
        print(f"\nEpisode {episode+1} completed: Total Reward = {total_reward:.2f}, Steps = {steps}")
        print(f"Goals discovered this episode: {len(discovered_goals_this_episode)}")
        print(f"Total goals discovered so far: {len(discovered_goal_rewards)} out of {len(goal_generator.goal_positions)}")
        time.sleep(2)  # Pause between episodes
    
    # Final summary
    print("\n===== EXPERIMENT SUMMARY =====")
    print(f"Total goals in environment: {len(goal_generator.goal_positions)}")
    print(f"Goals discovered: {len(discovered_goal_rewards)}")
    print(f"Goal configuration saved at: {goal_generator.save_dir}")
    print("=============================")

if __name__ == "__main__":
    run_with_goal_space(
        map_size=8,
        n_goals=9,  # Will default to map_size+1
        seed=42,
        n_episodes=3,
        max_steps=200,
        perception_radius=2,
        visualize_every=5
    )