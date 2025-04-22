import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from datetime import datetime
import imageio
from IPython.display import clear_output, display, HTML

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frozen_src.hierarchical_agent import HierarchicalAgent
from frozen_src.goal_space_generator import GoalSpaceGenerator
from frozen_src.visualization import visualize_environment, print_text_map


def max_steps_per_grid(map_size):
    """ Optimal + extra number of steps to explore the grid and complete the task"""
    return int(map_size**2 + map_size*np.sqrt(2))

def train_hierarchical_agent(env_name='FrozenLake-v1', map_size=5, n_goals=None, is_slippery=False, 
                            n_episodes=500, max_steps=200, seed=42, delay=0.1,
                            perception_radius=2, visualize_every=1):
    """
    Train a hierarchical agent on the FrozenLake environment with multiple goals.
    
    Args:
        env_name: Name of the environment
        map_size: Size of the grid
        n_goals: Number of goals (default: map_size+1)
        is_slippery: Whether the environment is slippery
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        seed: Random seed
        delay: Delay between steps for visualization
        perception_radius: Agent's perception radius
        visualize_every: Create visualizations every X steps
        
    Returns:
        Trained agent and training statistics
    """
    # Create timestamp-based directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"BASE_RUNS/training_run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Generate goal space with multiple rewards
    print(f"Generating goal space (size: {map_size}x{map_size}, goals: {n_goals or map_size+1})")
    goal_generator = GoalSpaceGenerator(map_size=map_size, n_goals=n_goals, seed=seed)
    goal_generator.generate_goal_space()
    
    # Visualize the goal space
    goal_generator.visualize_goal_space(save_path=f"{run_dir}/goal_space.png")
    
    # Get the custom map with goals
    custom_map = goal_generator.map
    
    # Create the environment
    env = gym.make(
        env_name,
        is_slippery=is_slippery,
        desc=custom_map,
        render_mode="rgb_array"
    )
    env.action_space.seed(seed)
    np.random.seed(seed)
    
    # Print environment details
    print(f"Environment initialized with {len(goal_generator.goal_positions)} goals")
    print(f"Final goal at position: {goal_generator.final_goal}")
    
    # Visualize the environment
    print_text_map(env.unwrapped.desc)
    environment_img = visualize_environment(env.unwrapped.desc, save_path=f"{run_dir}/environment.png")
    
    # Create the agent
    agent = HierarchicalAgent(
        env=env,
        budget=max_steps,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.2,
        perception_radius=perception_radius
    )
    
    # Create a dictionary for tracking discovered goal rewards
    discovered_goal_rewards = {}
    
    # Training statistics
    rewards_history = []
    steps_history = []
    goals_discovered_history = []
    goals_reached_history = []
    
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
    
    # Save configuration
    with open(f"{run_dir}/config.txt", "w") as f:
        f.write(f"Environment: {env_name}\n")
        f.write(f"Map Size: {map_size}x{map_size}\n")
        f.write(f"Number of Goals: {n_goals or map_size+1}\n")
        f.write(f"Is Slippery: {is_slippery}\n")
        f.write(f"Number of Episodes: {n_episodes}\n")
        f.write(f"Max Steps: {max_steps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Perception Radius: {perception_radius}\n")
    
    # Training loop
    for episode in range(n_episodes):
        print(f"Episode {episode+1}/{n_episodes}")
        
        # Create episode-specific directory
        episode_dir = f"{run_dir}/episode_{episode+1}"
        os.makedirs(episode_dir, exist_ok=True)
        
        # Create frames directory for this episode
        frames_dir = f"{episode_dir}/frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Reset environment and agent
        state = env.reset()[0]
        agent.reset(max_steps)
        
        # Get initial render
        frame = env.render()
        
        # Episode statistics
        total_reward = 0
        steps = 0
        discovered_goals_this_episode = set()
        done = False
        
        # Store frames for video
        frames = [frame]
        
        # Adjust exploration rate based on episode progress
        # Higher exploration early on, more exploitation later
        if episode < n_episodes * 0.3:  # First 30% of episodes
            agent.low_level_controller.epsilon = 0.3
        elif episode < n_episodes * 0.7:  # Middle 40% of episodes
            agent.low_level_controller.epsilon = 0.2
        else:  # Last 30% of episodes
            agent.low_level_controller.epsilon = 0.1
        
        # Simulate discovery of start position
        agent.update_observations(state, None)
        
        # Initial visualization
        agent.visualize_knowledge(episode+1, steps, save_dir=episode_dir)
        agent.visualize_plan(episode+1, steps, save_dir=episode_dir)
        
        # Save initial frame
        plt.figure(figsize=(8, 8))
        plt.imshow(frame)
        plt.title(f"Episode {episode+1}, Step 0")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{frames_dir}/step_0.png")
        plt.close()
        
        # Track unique goals reached this episode
        goals_reached_this_episode = set()
        
        # Create a log file for this episode
        log_file = open(f"{episode_dir}/log.txt", "w")
        log_file.write(f"Episode {episode+1} Log\n")
        log_file.write("=" * 20 + "\n\n")
        
        # Training episode loop
        while not done and steps < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Log action
            action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
            log_file.write(f"Step {steps+1}: Taking action {action_names[action]} from state {state}\n")
            
            # Take the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Get rendered frame after action
            frame = env.render()
            frames.append(frame)
            
            # Check if we've reached a goal and get its real reward
            row, col = next_state // env.unwrapped.ncol, next_state % env.unwrapped.ncol
            pos = (row, col)
            
            # Track if we hit a hole (for early termination)
            hit_hole = env.unwrapped.desc[row][col] == b'H'
            
            # Log position and observation
            log_file.write(f"  → New position: {pos}, State: {next_state}\n")
            if hit_hole:
                log_file.write(f"  → HIT A HOLE!\n")
            
            # If it's a goal position and not yet discovered, apply the true reward
            if pos in goal_generator.goal_positions and pos not in discovered_goal_rewards:
                true_reward = goal_generator.goal_rewards[pos]
                discovered_goal_rewards[pos] = true_reward
                discovered_goals_this_episode.add(pos)
                
                # Special handling for final goal
                is_final = (pos == goal_generator.final_goal)
                
                goal_msg = f"\n*** GOAL DISCOVERED at {pos} ***\n"
                goal_msg += f"Goal ID: {goal_generator.goal_ids.get(pos, 'Unknown')}\n"
                goal_msg += f"Reward: {true_reward:.2f}\n"
                if is_final:
                    goal_msg += "This is the FINAL GOAL!\n"
                goal_msg += "***************************\n"
                
                print(goal_msg)
                log_file.write(goal_msg)
                
                # Update reward with true value
                reward = true_reward
            
            # If it's a goal we've already discovered but not visited this episode,
            # we still get the reward once per episode
            elif pos in goal_generator.goal_positions and pos in discovered_goal_rewards:
                if pos not in goals_reached_this_episode:
                    goals_reached_this_episode.add(pos)
                    true_reward = discovered_goal_rewards[pos]
                    reward = true_reward
                    msg = f"Reached previously discovered goal at {pos}, reward: {true_reward:.2f}\n"
                    print(msg)
                    log_file.write(msg)
                else:
                    # Already visited this goal in this episode, no additional reward
                    reward = 0.01  # Small reward for revisiting
                    msg = f"Revisited goal at {pos}, no additional reward\n"
                    print(msg)
                    log_file.write(msg)
            
            # Update observations and get exploration bonus
            exploration_bonus, goal_reached = agent.update_observations(next_state, None)
            reward += exploration_bonus  # Add exploration bonus to reward
            
            if exploration_bonus > 0:
                log_file.write(f"  → Exploration bonus: {exploration_bonus:.4f}\n")
            
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
            log_file.write(f"  → Received reward: {reward:.4f}\n")
            
            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state
            
            # Save frame with step number
            plt.figure(figsize=(8, 8))
            plt.imshow(frame)
            plt.title(f"Episode {episode+1}, Step {steps}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{frames_dir}/step_{steps}.png")
            plt.close()
            
            # Create visualizations at each step
            agent.visualize_knowledge(episode+1, steps, save_dir=episode_dir)
            agent.visualize_plan(episode+1, steps, save_dir=episode_dir)
            
            # Display progress info
            clear_output(wait=True)
            print(f"Episode {episode+1}/{n_episodes} | Step {steps}/{max_steps}")
            print(f"Current State: {state} | Total Reward: {total_reward:.2f}")
            print(f"Goals Discovered: {len(discovered_goal_rewards)}/{len(goal_generator.goal_positions)}")
            print(f"Goals Reached: {len(agent.reached_goals)}")
            print(f"Known Map: {np.sum(agent.known_map)}/{agent.known_map.size} tiles")
            
            # Show the latest visualizations
            latest_knowledge = f"{episode_dir}/knowledge_ep{episode+1}_step{steps}.png"
            latest_plan = f"{episode_dir}/plan_ep{episode+1}_step{steps}.png"
            latest_frame = f"{frames_dir}/step_{steps}.png"
            
            display(HTML(f"""
            <div style="display: flex; justify-content: center;">
                <div style="margin: 5px;">
                    <img src="{latest_frame}" width="300"/>
                    <p style="text-align: center;">Environment</p>
                </div>
                <div style="margin: 5px;">
                    <img src="{latest_knowledge}" width="300"/>
                    <p style="text-align: center;">Knowledge Map</p>
                </div>
                <div style="margin: 5px;">
                    <img src="{latest_plan}" width="300"/>
                    <p style="text-align: center;">Plan</p>
                </div>
            </div>
            """))
            
            # Small delay to see progress
            time.sleep(delay)
            
            # Break early if we hit a hole or reach final goal late in training
            if hit_hole:
                log_file.write("Hit a hole, ending episode early\n")
                print("Hit a hole, ending episode early")
                break
                
            # Only terminate early if we're in later stages and have discovered most goals
            if episode > n_episodes * 0.7 and len(discovered_goal_rewards) > len(goal_generator.goal_positions) * 0.75:
                if pos == goal_generator.final_goal:
                    log_file.write("Final goal reached and most goals discovered, ending episode\n")
                    print("Final goal reached and most goals discovered, ending episode")
                    break
        
        # Close log file
        log_file.write(f"\nEpisode Summary:\n")
        log_file.write(f"Total Reward: {total_reward:.2f}\n")
        log_file.write(f"Steps: {steps}\n")
        log_file.write(f"Goals discovered this episode: {len(discovered_goals_this_episode)}\n")
        log_file.write(f"Total goals discovered so far: {len(discovered_goal_rewards)} out of {len(goal_generator.goal_positions)}\n")
        log_file.close()
        
        # Create video of the episode
        print("Creating video of the episode...")
        video_path = f"{episode_dir}/episode_{episode+1}_video.mp4"
        
        # Use imageio to create video
        writer = imageio.get_writer(video_path, fps=5)  # 5 frames per second
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        
        print(f"Video saved to {video_path}")
        
        # Record episode statistics
        rewards_history.append(total_reward)
        steps_history.append(steps)
        goals_discovered_history.append(len(discovered_goals_this_episode))
        goals_reached_history.append(len(agent.reached_goals))
        
        # Create episode summary plot
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        # Create a temporary reward history for this episode
        episode_rewards = []
        reward_so_far = 0
        with open(f"{episode_dir}/log.txt", "r") as f:
            for line in f:
                if "Received reward:" in line:
                    try:
                        reward = float(line.split("Received reward:")[1].strip())
                        reward_so_far += reward
                        episode_rewards.append(reward_so_far)
                    except:
                        pass

        # Plot the rewards (handle the case of empty rewards)
        if episode_rewards:
            plt.plot(range(1, len(episode_rewards)+1), episode_rewards, marker='o')
        else:
            plt.plot([0], [0])  # Fallback if no rewards
        plt.title('Cumulative Reward')
        plt.xlabel('Step')
        plt.ylabel('Total Reward')
        
        plt.subplot(2, 2, 2)
        discovered_goals = len(discovered_goal_rewards)
        total_goals = len(goal_generator.goal_positions)
        plt.bar(['Discovered', 'Remaining'], [discovered_goals, total_goals - discovered_goals])
        plt.title('Goals Progress')
        plt.ylabel('Number of Goals')
        
        plt.subplot(2, 2, 3)
        known_tiles = np.sum(agent.known_map)
        total_tiles = agent.known_map.size
        plt.bar(['Known', 'Unknown'], [known_tiles, total_tiles - known_tiles])
        plt.title('Map Exploration')
        plt.ylabel('Number of Tiles')
        
        plt.subplot(2, 2, 4)
        actions_taken = [0, 0, 0, 0]  # LEFT, DOWN, RIGHT, UP
        for i, pos in enumerate(agent.path_history[:-1]):
            next_pos = agent.path_history[i+1]
            row, col = pos // env.unwrapped.ncol, pos % env.unwrapped.ncol
            next_row, next_col = next_pos // env.unwrapped.ncol, next_pos % env.unwrapped.ncol
            
            if next_col < col:  # LEFT
                actions_taken[0] += 1
            elif next_row > row:  # DOWN
                actions_taken[1] += 1
            elif next_col > col:  # RIGHT
                actions_taken[2] += 1
            elif next_row < row:  # UP
                actions_taken[3] += 1
        
        plt.bar(['LEFT', 'DOWN', 'RIGHT', 'UP'], actions_taken)
        plt.title('Actions Taken')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"{episode_dir}/episode_summary.png")
        plt.close()
        
        # Print episode summary
        print(f"\nEpisode {episode+1} completed: Total Reward = {total_reward:.2f}, Steps = {steps}")
        print(f"Goals discovered this episode: {len(discovered_goals_this_episode)}")
        print(f"Total goals discovered so far: {len(discovered_goal_rewards)} out of {len(goal_generator.goal_positions)}")
        
        # Print progress summary every few episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_steps = np.mean(steps_history[-10:])
            avg_goals_discovered = np.mean(goals_discovered_history[-10:])
            avg_goals_reached = np.mean(goals_reached_history[-10:])
            progress_msg = (f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.2f} | Avg Goals Discovered: {avg_goals_discovered:.2f} | "
                  f"Avg Goals Reached: {avg_goals_reached:.2f}")
            print(progress_msg)
            
            # Save progress to run summary
            with open(f"{run_dir}/progress.txt", "a") as f:
                f.write(f"{progress_msg}\n")
            
            # Gradually decrease exploration as training progresses
            if episode > n_episodes * 0.5:
                agent.epsilon = max(0.05, agent.epsilon * 0.995)
                agent.low_level_controller.epsilon = max(0.05, agent.low_level_controller.epsilon * 0.995)
    
    # Plot training statistics
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(rewards_history)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 4, 2)
    plt.plot(steps_history)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(1, 4, 3)
    plt.plot(goals_discovered_history)
    plt.title('Goals Discovered per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Goals Discovered')
    
    plt.subplot(1, 4, 4)
    plt.plot(goals_reached_history)
    plt.title('Goals Reached per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Goals Reached')
    
    plt.tight_layout()
    plt.savefig(f'{run_dir}/training_stats.png')
    plt.show()
    
    # Final summary
    print("\n===== TRAINING SUMMARY =====")
    print(f"Total goals in environment: {len(goal_generator.goal_positions)}")
    print(f"Goals discovered: {len(discovered_goal_rewards)}")
    print(f"Average reward in last 50 episodes: {np.mean(rewards_history[-50:]):.2f}")
    print(f"Average steps in last 50 episodes: {np.mean(steps_history[-50:]):.2f}")
    print(f"Average goals reached in last 50 episodes: {np.mean(goals_reached_history[-50:]):.2f}")
    print("=============================")
    
    # Save final summary to file
    with open(f"{run_dir}/final_summary.txt", "w") as f:
        f.write("===== TRAINING SUMMARY =====\n")
        f.write(f"Total goals in environment: {len(goal_generator.goal_positions)}\n")
        f.write(f"Goals discovered: {len(discovered_goal_rewards)}\n")
        f.write(f"Average reward in last 50 episodes: {np.mean(rewards_history[-50:]):.2f}\n")
        f.write(f"Average steps in last 50 episodes: {np.mean(steps_history[-50:]):.2f}\n")
        f.write(f"Average goals reached in last 50 episodes: {np.mean(goals_reached_history[-50:]):.2f}\n")
        f.write("=============================\n")
    
    return agent, {
        'rewards': rewards_history,
        'steps': steps_history,
        'goals_discovered': goals_discovered_history,
        'goals_reached': goals_reached_history,
        'goal_generator': goal_generator,
        'discovered_goal_rewards': discovered_goal_rewards,
        'run_dir': run_dir
    }

if __name__ == "__main__":
    # Calculate max steps based on grid size
    map_size = 8
    max_steps = max_steps_per_grid(map_size)
    print(f"Using max steps: {max_steps} for grid size: {map_size}")

    agent, stats = train_hierarchical_agent(
        map_size=map_size,
        n_goals=9,  # More goals for better learning
        is_slippery=False,
        n_episodes=20,
        max_steps=max_steps,
        seed=14,
        visualize_every=1
    )
    
    print(f"Training complete! All outputs saved to: {stats['run_dir']}")