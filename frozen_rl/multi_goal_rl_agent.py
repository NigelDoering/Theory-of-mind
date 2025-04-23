import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any
import os
import pickle
import time
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class MultiGoalFrozenLakeEnv(gym.Env):
    """
    Custom Frozen Lake environment with multiple goals.
    
    Extends the standard Frozen Lake environment to include:
    - Multiple goals with varying rewards
    - A final goal that completes the episode
    - Custom reward structure
    - Random map generation
    """
    
    def __init__(
        self,
        size: int = 8,
        n_goals: int = 5,
        is_slippery: bool = False,
        desc: Optional[List[str]] = None,
        goal_rewards: Optional[Dict[Tuple[int, int], float]] = None,
        final_goal_reward: float = 10.0,
        hole_reward: float = -1.0,
        step_reward: float = -0.1,
        p_frozen: float = 0.9,
        seed: Optional[int] = None,
        render_mode: str = "rgb_array"
    ):
        """
        Initialize the multi-goal Frozen Lake environment.
        
        Args:
            size: Size of the grid world (size x size)
            n_goals: Number of sub-goals to add (excluding final goal)
            is_slippery: Whether the frozen tiles are slippery
            desc: Optional custom map as list of strings
            goal_rewards: Optional dict mapping (row, col) to reward values
            final_goal_reward: Reward for reaching the final goal
            hole_reward: Reward for falling in a hole
            step_reward: Default reward for each step
            p_frozen: Probability of frozen tile when generating random map
            seed: Random seed for reproducibility
            render_mode: Rendering mode
        """
        self.size = size
        self.n_goals = n_goals
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        self.p_frozen = p_frozen
        self.hole_reward = hole_reward
        self.step_reward = step_reward
        self.final_goal_reward = final_goal_reward
        
        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate map if not provided
        if desc is None:
            self.desc = self._generate_multi_goal_map(size, n_goals, p_frozen)
        else:
            self.desc = desc
        
        # Create the base environment
        self.env = gym.make(
            "FrozenLake-v1", 
            desc=self.desc, 
            is_slippery=is_slippery,
            render_mode=render_mode
        )
        
        # Set up observation and action spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Build reward mapping
        self.goal_rewards = goal_rewards or self._generate_goal_rewards()
        self._build_state_map()
        
        # Initialize state tracking
        self.current_state = 0
        self.visited_goals = set()
        self.steps_taken = 0
        
    def _generate_multi_goal_map(self, size, n_goals, p_frozen):
        """
        Generate a random map with multiple goals.
        
        Args:
            size: Size of the map
            n_goals: Number of intermediate goals
            p_frozen: Probability of frozen tile
            
        Returns:
            List of strings representing the map
        """
        # Start with a base random map
        base_map = generate_random_map(size=size, p=p_frozen)
        
        # Convert to list of lists for easier modification
        grid = [list(row) for row in base_map]
        
        # Find all valid positions for goals (only frozen tiles)
        valid_positions = []
        for i in range(size):
            for j in range(size):
                # Skip start position (0,0) and existing holes
                if (i == 0 and j == 0) or grid[i][j] == 'H':
                    continue
                valid_positions.append((i, j))
        
        # Place final goal at bottom right if it's valid
        if (size-1, size-1) in valid_positions:
            grid[size-1][size-1] = 'G'
            valid_positions.remove((size-1, size-1))
            self.final_goal_pos = (size-1, size-1)
        else:
            # Find another position for final goal
            if valid_positions:
                pos = random.choice(valid_positions)
                grid[pos[0]][pos[1]] = 'G'
                valid_positions.remove(pos)
                self.final_goal_pos = pos
        
        # Place intermediate goals
        self.goal_positions = []
        goal_count = min(n_goals, len(valid_positions))
        
        for _ in range(goal_count):
            if not valid_positions:
                break
                
            pos = random.choice(valid_positions)
            grid[pos[0]][pos[1]] = 'G'
            valid_positions.remove(pos)
            self.goal_positions.append(pos)
        
        # Convert back to list of strings
        return [''.join(row) for row in grid]
    
    def _generate_goal_rewards(self):
        """
        Generate rewards for each goal position.
        
        Returns:
            Dict mapping (row, col) to reward values
        """
        rewards = {}
        
        # Set reward for final goal
        rewards[self.final_goal_pos] = self.final_goal_reward
        
        # Set random rewards for intermediate goals
        for pos in self.goal_positions:
            rewards[pos] = np.random.uniform(1.0, self.final_goal_reward * 0.8)
        
        return rewards
    
    def _build_state_map(self):
        """
        Build mappings between states and positions.
        """
        self.state_rewards = {}
        self.state_to_pos = {}
        self.pos_to_state = {}
        
        state = 0
        for i, row in enumerate(self.desc):
            for j, cell in enumerate(row):
                pos = (i, j)
                self.state_to_pos[state] = pos
                self.pos_to_state[pos] = state
                
                # Set rewards for goals
                if pos in self.goal_rewards:
                    self.state_rewards[state] = self.goal_rewards[pos]
                # Set rewards for holes
                elif cell == b'H' or cell == 'H':
                    self.state_rewards[state] = self.hole_reward
                # Default step reward
                else:
                    self.state_rewards[state] = self.step_reward
                
                state += 1
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        observation, info = self.env.reset(seed=seed)
        self.current_state = observation
        self.visited_goals = set()
        self.steps_taken = 0
        return observation, info
    
    def step(self, action):
        """Take a step in the environment."""
        next_state, _, terminated, truncated, info = self.env.step(action)
        
        # Calculate reward
        reward = self.state_rewards[next_state]
        
        # Check if this is a goal and has not been visited
        pos = self.state_to_pos[next_state]
        if pos in self.goal_rewards and next_state not in self.visited_goals:
            self.visited_goals.add(next_state)
        
        # Check if we've reached the final goal
        if pos == self.final_goal_pos:
            terminated = True
        
        self.current_state = next_state
        self.steps_taken += 1
        
        return next_state, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()
        
    def get_map_with_agent(self):
        """Get the current map with agent position marked."""
        grid = []
        for i, row in enumerate(self.desc):
            grid_row = []
            for j, cell in enumerate(row):
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                grid_row.append(cell)
            grid.append(grid_row)
        
        # Mark agent position
        pos = self.state_to_pos[self.current_state]
        grid[pos[0]][pos[1]] = 'A'  # Agent
        
        return grid


class HierarchicalQLearningAgent:
    """
    Hierarchical Q-learning agent for multi-goal environments.
    
    This agent uses a two-level hierarchy:
    1. High level: Choose which goal to pursue next
    2. Low level: Learn how to reach the chosen goal
    """
    
    def __init__(
        self,
        env: MultiGoalFrozenLakeEnv,
        high_alpha: float = 0.2,
        low_alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        """
        Initialize the hierarchical Q-learning agent.
        
        Args:
            env: The multi-goal environment
            high_alpha: Learning rate for high-level policy
            low_alpha: Learning rate for low-level policies
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            min_epsilon: Minimum exploration rate
        """
        self.env = env
        self.high_alpha = high_alpha
        self.low_alpha = low_alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Action space
        self.n_actions = env.action_space.n
        
        # State space
        self.n_states = env.observation_space.n
        
        # Initialize Q-tables
        self._init_q_tables()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.goals_reached = []
        
        # Current goal tracking
        self.current_goal = None
        self.goal_reached = False
        
    def _init_q_tables(self):
        """Initialize Q-tables for high and low level policies."""
        # High-level Q-table: state -> goal -> value
        self.high_q_table = defaultdict(lambda: defaultdict(float))
        
        # Low-level Q-tables: goal -> (state, action) -> value
        self.low_q_tables = defaultdict(lambda: defaultdict(float))
        
        # Find all goal states
        self.goal_states = []
        for state, pos in self.env.state_to_pos.items():
            if pos in self.env.goal_rewards:
                self.goal_states.append(state)
        
        # Add final goal
        final_goal_state = self.env.pos_to_state[self.env.final_goal_pos]
        if final_goal_state not in self.goal_states:
            self.goal_states.append(final_goal_state)
        
        self.final_goal_state = final_goal_state
        
    def _select_goal(self, state, eval_mode=False):
        """
        Select the next goal to pursue using epsilon-greedy.
        
        Args:
            state: Current state
            eval_mode: If True, use greedy selection
            
        Returns:
            Selected goal state
        """
        # Find available goals (not visited yet)
        available_goals = [g for g in self.goal_states if g not in self.env.visited_goals]
        
        # Always include final goal
        if self.final_goal_state not in available_goals:
            available_goals.append(self.final_goal_state)
            
        # If no goals available, return final goal
        if not available_goals:
            return self.final_goal_state
            
        # Epsilon-greedy selection
        if eval_mode or random.random() > self.epsilon:
            # Greedy selection: choose goal with highest Q-value
            q_values = [self.high_q_table[state][goal] for goal in available_goals]
            max_value = max(q_values)
            best_goals = [g for i, g in enumerate(available_goals) if q_values[i] == max_value]
            return random.choice(best_goals)
        else:
            # Random selection
            return random.choice(available_goals)
    
    def _select_action(self, state, goal, eval_mode=False):
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            goal: Current goal
            eval_mode: If True, use greedy selection
            
        Returns:
            Selected action
        """
        # Epsilon-greedy selection
        if eval_mode or random.random() > self.epsilon:
            # Greedy selection
            q_values = [self.low_q_tables[goal][(state, a)] for a in range(self.n_actions)]
            max_value = max(q_values)
            best_actions = [a for a, v in enumerate(q_values) if v == max_value]
            return random.choice(best_actions)
        else:
            # Random selection
            return random.randint(0, self.n_actions - 1)
    
    def train(self, n_episodes=5000, max_steps=100, eval_interval=100):
        """
        Train the agent.
        
        Args:
            n_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            eval_interval: How often to evaluate the agent
            
        Returns:
            Training metrics
        """
        # Reset performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.goals_reached = []
        
        # Progress bar
        pbar = tqdm(range(n_episodes), desc="Training")
        
        for episode in pbar:
            # Reset environment
            state, _ = self.env.reset()
            
            # Select initial goal
            goal = self._select_goal(state)
            self.current_goal = goal
            
            # Track episode performance
            total_reward = 0
            steps = 0
            goals_reached = 0
            
            # Episode loop
            for step in range(max_steps):
                # Select action based on current state and goal
                action = self._select_action(state, goal)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update total reward
                total_reward += reward
                steps += 1
                
                # Check if we reached the current goal
                if next_state == goal:
                    goals_reached += 1
                    
                    # High-level update: reward for reaching goal
                    self.high_q_table[state][goal] += self.high_alpha * (
                        reward + self.gamma * max(
                            [self.high_q_table[next_state][g] for g in self.goal_states]
                        ) - self.high_q_table[state][goal]
                    )
                    
                    # Select new goal
                    if not done:
                        goal = self._select_goal(next_state)
                        self.current_goal = goal
                else:
                    # Low-level update
                    self.low_q_tables[goal][(state, action)] += self.low_alpha * (
                        reward + self.gamma * max(
                            [self.low_q_tables[goal][(next_state, a)] 
                             for a in range(self.n_actions)]
                        ) - self.low_q_tables[goal][(state, action)]
                    )
                
                # Update state
                state = next_state
                
                # Break if done
                if done:
                    break
            
            # Record episode performance
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.goals_reached.append(goals_reached)
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Update progress bar
            if episode % 10 == 0:
                pbar.set_postfix({
                    'reward': f"{np.mean(self.episode_rewards[-10:]):.2f}",
                    'steps': f"{np.mean(self.episode_steps[-10:]):.1f}",
                    'goals': f"{np.mean(self.goals_reached[-10:]):.1f}",
                    'epsilon': f"{self.epsilon:.3f}"
                })
            
            # Run evaluation
            if episode % eval_interval == 0:
                eval_reward, eval_steps, eval_goals = self.evaluate(5)
                print(f"\nEval: reward={eval_reward:.2f}, steps={eval_steps:.1f}, goals={eval_goals:.1f}")
        
        return {
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'goals_reached': self.goals_reached
        }
    
    def evaluate(self, n_episodes=5, max_steps=100, render=False):
        """
        Evaluate the agent.
        
        Args:
            n_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            
        Returns:
            Average reward, steps and goals reached
        """
        eval_rewards = []
        eval_steps = []
        eval_goals_reached = []
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            goal = self._select_goal(state, eval_mode=True)
            
            total_reward = 0
            steps = 0
            goals_reached = 0
            
            for step in range(max_steps):
                # Select best action
                action = self._select_action(state, goal, eval_mode=True)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update metrics
                total_reward += reward
                steps += 1
                
                # Render if requested
                if render:
                    self.env.render()
                    time.sleep(0.1)  # Slow down for visualization
                
                # Check if goal reached
                if next_state == goal:
                    goals_reached += 1
                    
                    # Select new goal
                    if not done:
                        goal = self._select_goal(next_state, eval_mode=True)
                
                # Update state
                state = next_state
                
                # Break if done
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_steps.append(steps)
            eval_goals_reached.append(goals_reached)
        
        return np.mean(eval_rewards), np.mean(eval_steps), np.mean(eval_goals_reached)

    def visualize_policy(self, goal=None):
        """
        Visualize the agent's policy for a specific goal.
        
        Args:
            goal: Goal state (if None, use final goal)
        """
        if goal is None:
            goal = self.final_goal_state
        
        # Create a grid to visualize the policy
        size = self.env.size
        policy_grid = np.zeros((size, size), dtype=int)
        
        # Map of action to arrow
        action_arrows = ['←', '↓', '→', '↑']
        
        # For each state, determine the best action
        for state in range(self.n_states):
            pos = self.env.state_to_pos[state]
            i, j = pos
            
            # Find best action for this state-goal pair
            q_values = [self.low_q_tables[goal][(state, a)] for a in range(self.n_actions)]
            best_action = np.argmax(q_values)
            
            # Store in grid
            policy_grid[i, j] = best_action
        
        # Create figure
        plt.figure(figsize=(8, 8))
        
        # Visualize the map
        map_grid = np.zeros((size, size), dtype='<U1')
        for i in range(size):
            for j in range(size):
                cell = self.env.desc[i][j]
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                map_grid[i, j] = cell
        
        # Mark the goal
        goal_pos = self.env.state_to_pos[goal]
        map_grid[goal_pos] = 'G'
        
        # Plot policy
        for i in range(size):
            for j in range(size):
                cell = map_grid[i, j]
                if cell == 'H':  # Hole
                    plt.text(j, i, 'H', ha='center', va='center', color='white',
                             bbox=dict(facecolor='black', alpha=0.8))
                elif cell == 'G':  # Goal
                    plt.text(j, i, 'G', ha='center', va='center', color='white',
                             bbox=dict(facecolor='green', alpha=0.8))
                elif cell == 'S':  # Start
                    plt.text(j, i, 'S', ha='center', va='center', color='black',
                             bbox=dict(facecolor='yellow', alpha=0.8))
                elif cell == 'F':  # Frozen
                    action = policy_grid[i, j]
                    plt.text(j, i, action_arrows[action], ha='center', va='center')
        
        # Set limits and labels
        plt.title(f"Policy for Goal at {goal_pos}")
        plt.grid(True)
        plt.xticks(np.arange(0, size, 1))
        plt.yticks(np.arange(0, size, 1))
        plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
        plt.tight_layout()
        plt.savefig(f"policy_goal_{goal_pos}.png")
        plt.show()
    
    def plot_training_progress(self):
        """Plot the training progress."""
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot rewards
        axs[0].plot(self.episode_rewards)
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        
        # Plot steps
        axs[1].plot(self.episode_steps)
        axs[1].set_title('Episode Steps')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Steps')
        
        # Plot goals reached
        axs[2].plot(self.goals_reached)
        axs[2].set_title('Goals Reached per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Goals Reached')
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filename='hierarchical_agent.pkl'):
        """Save the agent to a file."""
        data = {
            'high_q_table': dict(self.high_q_table),
            'low_q_tables': {k: dict(v) for k, v in self.low_q_tables.items()},
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'goals_reached': self.goals_reached
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Agent saved to {filename}")
    
    def load(self, filename='hierarchical_agent.pkl'):
        """Load the agent from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Convert dictionaries back to defaultdicts
        self.high_q_table = defaultdict(lambda: defaultdict(float))
        for s, goals in data['high_q_table'].items():
            for g, v in goals.items():
                self.high_q_table[s][g] = v
                
        self.low_q_tables = defaultdict(lambda: defaultdict(float))
        for g, states in data['low_q_tables'].items():
            for sa, v in states.items():
                self.low_q_tables[g][sa] = v
        
        self.epsilon = data['epsilon']
        self.episode_rewards = data['episode_rewards']
        self.episode_steps = data['episode_steps']
        self.goals_reached = data['goals_reached']
        
        print(f"Agent loaded from {filename}")


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
    steps = 0
    goals_reached = 0
    visited_goals = set()
    
    # Create figure for visualization
    plt.figure(figsize=(10, 10))
    
    frames = []
    
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


def run_experiment(size=8, n_goals=5, is_slippery=False, seed=42, n_episodes=5000, save_path=None):
    """
    Run a complete experiment: create environment, train agent, evaluate and visualize.
    
    Args:
        size: Size of the grid world
        n_goals: Number of intermediate goals
        is_slippery: Whether frozen tiles are slippery
        seed: Random seed
        n_episodes: Number of training episodes
        save_path: Where to save the trained agent (None = don't save)
        
    Returns:
        The trained agent and environment
    """
    # Set up environment
    env = MultiGoalFrozenLakeEnv(
        size=size,
        n_goals=n_goals,
        is_slippery=is_slippery,
        seed=seed,
        final_goal_reward=10.0,
        hole_reward=-1.0,
        step_reward=-0.1
    )
    
    # Print environment details
    print(f"Created environment with {len(env.goal_rewards)} goals")
    print(f"Goal rewards: {env.goal_rewards}")
    print(f"Final goal at {env.final_goal_pos}")
    
    # Print map
    print("\nInitial Map:")
    for row in env.desc:
        if isinstance(row, bytes):
            print(row.decode('utf-8'))
        else:
            print(row)
    
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
    
    # Train agent
    print("\nTraining agent...")
    metrics = agent.train(n_episodes=n_episodes, max_steps=size*size*4, eval_interval=500)
    
    # Plot training progress
    agent.plot_training_progress()
    
    # Save trained agent if requested
    if save_path:
        agent.save(save_path)
    
    # Evaluate agent
    print("\nFinal evaluation:")
    reward, steps, goals = agent.evaluate(n_episodes=10, max_steps=size*size*4, render=False)
    print(f"Average reward: {reward:.2f}, steps: {steps:.2f}, goals: {goals:.2f}")
    
    # Visualize policy for final goal
    print("\nPolicy for final goal:")
    agent.visualize_policy(agent.final_goal_state)
    
    # Visualize an episode
    print("\nVisualizing episode:")
    visualize_episode(env, agent, max_steps=size*size*4)
    
    return agent, env


if __name__ == "__main__":
    # Set parameters
    size = 8
    n_goals = 5
    is_slippery = False
    seed = 42
    n_episodes = 5000
    
    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Run experiment
    agent, env = run_experiment(
        size=size,
        n_goals=n_goals,
        is_slippery=is_slippery,
        seed=seed,
        n_episodes=n_episodes,
        save_path="models/hierarchical_agent.pkl"
    )
    
    print("Experiment complete!")