
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import json

class MultiGoalFrozenLakeEnv(gym.Env):
    """
    Multi-goal FrozenLake environment with randomized goals and holes.
    
    The agent must navigate to collect rewards from multiple goals before
    reaching a final goal. The environment is regenerated for each episode.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 4}
    
    def __init__(
        self,
        size: int = 8,
        max_steps: int = 100,
        n_goals: int = 5,
        n_holes: int = 10,
        is_slippery: bool = False,
        seed: Optional[int] = None,
        base_reward: float = 10.0,
        final_goal_reward: float = 10.0,
        hole_reward: float = -1.0,
        step_reward: float = -0.1,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the environment.
        
        Args:
            size: Size of the grid world (size x size)
            n_goals: Number of goals
            n_holes: Number of holes
            is_slippery: Whether movement is stochastic
            seed: Random seed
            base_reward: Base reward value for scaling Dirichlet distribution
            final_goal_reward: Reward for reaching the final goal
            hole_reward: Reward for falling into a hole
            step_reward: Reward for each step taken
            render_mode: Rendering mode
        """
        self.size = size
        self.max_steps = max_steps
        self.n_goals = n_goals
        self.n_holes = n_holes
        self.is_slippery = is_slippery
        self.base_reward = base_reward
        self.final_goal_reward = final_goal_reward
        self.hole_reward = hole_reward
        self.step_reward = step_reward
        
        # Set random seed
        self.np_random = np.random.RandomState(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Set up action and observation spaces
        # 0: Left, 1: Down, 2: Right, 3: Up, 4: Collect
        self.action_space = spaces.Discrete(5)
        # Observation is the current state (position)
        self.observation_space = spaces.Discrete(size * size)
        
        # State mappings
        self.agent_pos = (0, 0)  # Agent's position
        self.steps_taken = 0
        self.agent_state = 0
        self.state_to_pos = {}
        self.pos_to_state = {}
        
        # Set up state mappings
        for i in range(size):
            for j in range(size):
                state = i * size + j
                self.state_to_pos[state] = (i, j)
                self.pos_to_state[(i, j)] = state
        
        # Initialize tracking variables
        self.state_rewards = {}
        self.goal_rewards = {}
        self.goal_positions = []
        self.hole_positions = []
        self.start_pos = None
        self.final_goal_pos = None
        self.visited_goals = set()
        self.collected_goals = set()
        
        # Initialize Dirichlet parameters
        self.agent_alpha = None
        self.dirichlet_probs = None
        
        # Initialize descriptor
        self.desc = None
        
        # Generate initial map
        self._generate_map()
        
        # Save original descriptor for reset
        self.initial_desc = self.desc.copy()
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
    def _generate_map(self):
        """Generate a random map with goals and holes."""
        # Initialize grid with frozen tiles
        self.desc = np.array(['F'] * (self.size * self.size), dtype='c')
        self.desc = self.desc.reshape(self.size, self.size)
        
        # Clear previous state
        self.goal_rewards = {}
        self.goal_positions = []
        self.hole_positions = []
        self.state_rewards = {}
        
        # Generate agent starting position (random)
        valid_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.start_pos = random.choice(valid_positions)
        valid_positions.remove(self.start_pos)
        start_state = self.pos_to_state[self.start_pos]
        self.desc[self.start_pos] = b'S'
        
        # Generate hole positions (random)
        self.hole_positions = random.sample(valid_positions, min(self.n_holes, len(valid_positions)))
        for pos in self.hole_positions:
            valid_positions.remove(pos)
            self.desc[pos] = b'H'
            state = self.pos_to_state[pos]
            self.state_rewards[state] = self.hole_reward
        
        # Generate goal positions (random)
        n_goals_to_place = min(self.n_goals, len(valid_positions))
        self.goal_positions = random.sample(valid_positions, n_goals_to_place)
        for pos in self.goal_positions:
            self.desc[pos] = b'G'
        
        # Generate rewards using Dirichlet distribution
        self._generate_goal_rewards()
        
        # Build state reward map
        for state, pos in self.state_to_pos.items():
            if pos in self.goal_rewards:
                self.state_rewards[state] = self.goal_rewards[pos]
        
        return self.desc
    
    def _generate_goal_rewards(self):
        """Generate goal rewards using Dirichlet distribution."""
        # Create alpha parameter with slight variation for each goal
        self.agent_alpha = np.random.normal(1, 0.2, size=len(self.goal_positions))
        
        # Generate Dirichlet distribution
        self.dirichlet_probs = np.random.dirichlet(alpha=np.ones(len(self.goal_positions))*self.agent_alpha, size=1)[0]
        
        # Scale probabilities by base reward
        rewards = self.dirichlet_probs * self.base_reward
        
        # Print rewards for debugging
        print(f"Debug - Goal rewards: {rewards}")
        
        # Set the reward for the final goal
        final_goal_index = random.randint(0, len(self.goal_positions) - 1)
        self.final_goal_pos = self.goal_positions[final_goal_index]
        state = self.pos_to_state[self.final_goal_pos]
        self.goal_rewards[self.final_goal_pos] = self.final_goal_reward
        self.state_rewards[state] = self.final_goal_reward
        self.desc[self.final_goal_pos] = b'O'  # Mark final goal
        
        # Set the rewards for the other goals
        for i in range(len(self.goal_positions)):
            if i != final_goal_index:
                pos = self.goal_positions[i]
                # Set the reward for the goal position
                self.goal_rewards[pos] = rewards[i]
                state = self.pos_to_state[pos]
                self.state_rewards[state] = rewards[i]
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        # Set seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate a new map with different positions
        self._generate_map()
        
        # Reset agent state
        self.agent_pos = self.start_pos
        self.agent_state = self.pos_to_state[self.agent_pos]
        
        # Reset tracking variables
        self.visited_goals = set()
        self.collected_goals = set()
        self.steps_taken = 0
        
        # Return initial state and info
        observation = self.agent_state
        info = {
            "agent_pos": self.agent_pos,
            "goal_positions": self.goal_positions,
            "goal_rewards": self.goal_rewards,
            "final_goal_pos": self.final_goal_pos,
            "map": self.get_map_with_agent()
        }
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: 0-3 for movement, 4 for collect
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Check if action is collect
        if action == 4:  # Collect action
            reward = self._handle_collect_action()
        else:  # Movement action
            reward = self._handle_movement_action(action)
        
        # Add step penalty
        reward += self.step_reward
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        # Check if agent fell into a hole
        if self.agent_pos in self.hole_positions:
            terminated = True
        
        # Check if agent reached the final goal
        if self.agent_pos == self.final_goal_pos:
            terminated = True
        
        # Check if maximum steps exceeded
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            truncated = True
        
        # Return state, reward, done flags, and info
        observation = self.pos_to_state[self.agent_pos]
        info = {
            "agent_pos": self.agent_pos,
            "visited_goals": self.visited_goals,
            "collected_goals": self.collected_goals
        }
        
        return observation, reward, terminated, truncated, info
    
    def _handle_collect_action(self):
        """Handle the collect action."""
        state = self.pos_to_state[self.agent_pos]
        reward = 0
        
        # Check if agent is at a goal
        if self.agent_pos in self.goal_positions:
            # Check if goal has already been collected
            if self.agent_pos not in self.collected_goals:
                # Collect goal
                reward = self.goal_rewards[self.agent_pos]
                self.collected_goals.add(self.agent_pos)
                print(f"Collected goal at {self.agent_pos} with reward {reward}")
        
        return reward
    
    def _handle_movement_action(self, action):
        """Handle movement action."""
        reward = 0
        
        # Current position
        i, j = self.agent_pos
        
        # Calculate next position
        if action == 0:  # Left
            j = max(0, j - 1)
        elif action == 1:  # Down
            i = min(self.size - 1, i + 1)
        elif action == 2:  # Right
            j = min(self.size - 1, j + 1)
        elif action == 3:  # Up
            i = max(0, i - 1)
        
        # Apply slippery effect if enabled
        if self.is_slippery and random.random() < 0.33:
            # Random movement in a different direction
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            di, dj = random.choice(directions)
            i = max(0, min(self.size - 1, i + di))
            j = max(0, min(self.size - 1, j + dj))
        
        # Update agent position
        self.agent_pos = (i, j)
        
        # Check if agent has visited a goal
        if self.agent_pos in self.goal_positions and self.agent_pos not in self.visited_goals:
            self.visited_goals.add(self.agent_pos)
            # Note: Reward is only given when explicitly collecting
        
        return reward
    
    def get_map_with_agent(self):
        """Get a copy of the map with the agent's position marked."""
        map_with_agent = np.copy(self.desc)
        ai, aj = self.agent_pos
        map_with_agent[ai, aj] = b'A'
        return map_with_agent
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human' or self.render_mode == 'rgb_array':
            return self._render_gui()
        else:
            return self._render_text()
    
    def _render_gui(self):
        """Render the environment with matplotlib."""
        # Create figure if not already created
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 10))
        
        plt.clf()
        
        # Get map with agent position
        map_with_agent = self.get_map_with_agent()
        
        # Create visualization grid
        grid = np.zeros((self.size, self.size, 3))  # RGB
        
        # Fill grid with colors
        for i in range(self.size):
            for j in range(self.size):
                cell = map_with_agent[i][j]
                
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                
                if cell == 'H':  # Hole
                    grid[i, j] = [0.1, 0.1, 0.1]  # Dark gray
                elif cell == 'G':  # Goal
                    if (i, j) in self.collected_goals:
                        grid[i, j] = [0.5, 0.5, 0.5]  # Gray (collected)
                    elif (i, j) in self.visited_goals:
                        grid[i, j] = [0.8, 0.4, 0.8]  # Purple (visited)
                    else:
                        grid[i, j] = [0.0, 0.8, 0.0]  # Green (unvisited)
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
        
        # Add text labels
        for i in range(self.size):
            for j in range(self.size):
                cell = map_with_agent[i][j]
                
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                
                if cell == 'A':
                    plt.text(j, i, 'A', ha='center', va='center', color='white', fontweight='bold')
                elif cell == 'G':
                    state_id = self.pos_to_state[(i, j)]
                    reward = self.state_rewards.get(state_id, 0)
                    if (i, j) in self.collected_goals:
                        text = f'G\n{reward:.1f}\n[C]'
                    else:
                        text = f'G\n{reward:.1f}'
                    plt.text(j, i, text, ha='center', va='center', 
                             color='white' if (i, j) in self.collected_goals else 'black')
                elif cell == 'F':
                    plt.text(j, i, f'F\n{self.final_goal_reward}', ha='center', va='center', color='black')
                elif cell == 'H':
                    plt.text(j, i, 'H', ha='center', va='center', color='white')
                elif cell == 'S':
                    plt.text(j, i, 'S', ha='center', va='center', color='black')
        
        # Add status text
        collected_reward = sum(self.goal_rewards[pos] for pos in self.collected_goals)
        plt.title(f"Steps: {self.steps_taken}, Collected: {collected_reward:.2f}, "
                 f"Goals: {len(self.collected_goals)}/{len(self.goal_positions)}")
        plt.grid(True, color='black', alpha=0.2)
        plt.tight_layout()
        
        plt.pause(0.1)
        
        if self.render_mode == 'rgb_array':
            # Convert plot to RGB array
            fig_canvas = plt.gcf().canvas
            fig_canvas.draw()
            img = np.array(fig_canvas.renderer.buffer_rgba())
            return img
    
    def _render_text(self):
        """Render the environment as text."""
        outfile = sys.stdout

        map_with_agent = self.get_map_with_agent()
        
        for row in map_with_agent:
            outfile.write(" ".join(row.decode("utf-8") if isinstance(row, bytes) else row))
            outfile.write("\n")
            
        collected_reward = sum(self.goal_rewards[pos] for pos in self.collected_goals)
        outfile.write(f"Steps: {self.steps_taken}, Collected: {collected_reward:.2f}, "
                     f"Goals: {len(self.collected_goals)}/{len(self.goal_positions)}\n")
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
