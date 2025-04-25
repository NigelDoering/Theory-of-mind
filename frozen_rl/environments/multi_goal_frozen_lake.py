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
    Multi-goal FrozenLake environment with support for multiple weighted goals.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 4}
    
    def __init__(
        self,
        size=8,
        max_steps=100,
        n_goals=5,
        n_holes=8,
        is_slippery=False,
        seed=None,
        render_mode=None,
        randomize_start=True,
        fixed_map=False
    ):
        """Initialize the environment with specified parameters."""
        self.size = size
        self.max_steps = max_steps
        self.n_goals = n_goals
        self.n_holes = n_holes
        self.is_slippery = is_slippery
        self.base_reward = 10.0
        self.hole_reward = -1.0
        self.step_reward = -0.01
        self.randomize_start = randomize_start
        self.fixed_map = fixed_map
        
        # Set random seed
        self.np_random = np.random.RandomState(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Setup spaces
        self.action_space = spaces.Discrete(4)  # Left, down, right, up
        self.observation_space = spaces.Discrete(size * size)
        
        # Initialize state mappings
        self.agent_pos = (0, 0)
        self.steps_taken = 0
        self.agent_state = 0
        self.state_to_pos = {}
        self.pos_to_state = {}
        
        # Map states to positions and vice versa
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
        self.visited_goals = set()
        
        # Initialize parameters for reward distribution
        self.agent_alpha = None
        self.dirichlet_probs = None
        
        # Generate initial map
        self._generate_map()
        
        # Save map if using fixed layout
        if self.fixed_map:
            self.fixed_goal_positions = self.goal_positions.copy()
            self.fixed_hole_positions = self.hole_positions.copy()
            self.fixed_goal_rewards = dict(self.goal_rewards)
            self.fixed_state_rewards = dict(self.state_rewards) 
            self.fixed_desc = self.desc.copy()
        
        # Rendering settings
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
        """Generate goal rewards using Dirichlet distribution and scale to 1-10 range."""
        # Create alpha parameter with slight variation for each goal
        self.agent_alpha = np.random.normal(1, 0.2, size=self.n_goals)
        
        # Generate Dirichlet distribution
        self.dirichlet_probs = np.random.dirichlet(alpha=self.agent_alpha, size=1)[0]
        
        # Scale rewards to be between 1-10 (min-max scaling)
        min_val = 1.0
        max_val = 10.0
        scaled_rewards = min_val + (self.dirichlet_probs * (max_val - min_val))
        
        # Set rewards for goals
        for i, pos in enumerate(self.goal_positions):
            self.goal_rewards[pos] = scaled_rewards[i]
            state = self.pos_to_state[pos]
            self.state_rewards[state] = scaled_rewards[i]
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # Set seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        if self.fixed_map:
            # Restore the saved map
            self.goal_positions = self.fixed_goal_positions.copy()
            self.hole_positions = self.fixed_hole_positions.copy() 
            self.goal_rewards = dict(self.fixed_goal_rewards)
            self.state_rewards = dict(self.fixed_state_rewards)
            self.desc = self.fixed_desc.copy()
        else:
            # Generate a new map
            self._generate_map()
        
        # Reset agent state
        if self.randomize_start:
            # Choose a random valid starting position
            valid_positions = []
            for i in range(self.size):
                for j in range(self.size):
                    pos = (i, j)
                    if (pos not in self.goal_positions and 
                        pos not in self.hole_positions):
                        valid_positions.append(pos)
            
            if valid_positions:
                self.start_pos = random.choice(valid_positions)
                
        self.agent_pos = self.start_pos
        self.agent_state = self.pos_to_state[self.agent_pos]
        
        # Reset tracking variables
        self.visited_goals = set()
        self.steps_taken = 0
        
        # Reset loop detection
        self.previous_positions = []
        
        # Return initial state and info
        observation = self.agent_state
        info = {
            "agent_pos": self.agent_pos,
            "goal_positions": self.goal_positions,
            "goal_rewards": self.goal_rewards,
            "map": self.get_map_with_agent()
        }
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment with diminishing returns for revisits"""
        self.steps_taken += 1
        
        # Initialize termination flags
        terminated = False
        truncated = False
        
        # Default step reward - smaller penalty for efficiency
        reward = -0.05  # Reduced penalty to encourage exploration
        
        # Check for maximum steps
        if self.steps_taken >= self.max_steps:
            truncated = True
        
        # Process the movement action
        old_pos = self.agent_pos
        row, col = old_pos
        
        # Apply selected action (0: left, 1: down, 2: right, 3: up)
        if action == 0:  # Left
            col = max(0, col - 1)
        elif action == 1:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 3:  # Up
            row = max(0, row - 1)
        
        new_pos = (row, col)
        
        # Check if movement is valid
        if new_pos in self.hole_positions:
            # Fell in a hole
            reward = -1.0  # Fixed penalty for falling in hole
            terminated = True
        else:
            # Valid move
            self.agent_pos = new_pos
            
            # Add reward shaping to encourage movement toward unvisited goals
            unvisited_goals = [g for g in self.goal_positions if g not in self.visited_goals]
            if unvisited_goals:
                # Find closest unvisited goal distance
                min_dist_before = min(abs(old_pos[0] - g[0]) + abs(old_pos[1] - g[1]) for g in unvisited_goals)
                min_dist_after = min(abs(new_pos[0] - g[0]) + abs(new_pos[1] - g[1]) for g in unvisited_goals)
                
                # Reward for getting closer to a goal
                if min_dist_after < min_dist_before:
                    reward += 0.1  # Small shaping reward for progress
                elif min_dist_after > min_dist_before:
                    reward -= 0.05  # Small penalty for moving away
            
            # Check if at a goal position
            if self.agent_pos in self.goal_positions:
                if self.agent_pos not in self.visited_goals:
                    # First visit - full reward
                    goal_reward = self.goal_rewards.get(self.agent_pos, 0)
                    reward += goal_reward
                    self.visited_goals.add(self.agent_pos)
                else:
                    # Revisit - significantly diminished reward (10% of original)
                    goal_reward = self.goal_rewards.get(self.agent_pos, 0) * 0.05
                    reward += goal_reward
        
        # Return state, reward, done flags, and info
        observation = self.pos_to_state[self.agent_pos]
        info = {
            "agent_pos": self.agent_pos,
            "visited_goals": list(self.visited_goals)
        }
        
        return observation, reward, terminated, truncated, info
    
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
                    if (i, j) in self.visited_goals:
                        grid[i, j] = [0.8, 0.4, 0.8]  # Purple (visited)
                    else:
                        grid[i, j] = [0.0, 0.8, 0.0]  # Green (unvisited)
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
                    reward = self.goal_rewards.get((i, j), 0)
                    text = f'G\n{reward:.1f}'
                    plt.text(j, i, text, ha='center', va='center', 
                             color='white' if (i, j) in self.visited_goals else 'black')
                elif cell == 'H':
                    plt.text(j, i, 'H', ha='center', va='center', color='white')
                elif cell == 'S':
                    plt.text(j, i, 'S', ha='center', va='center', color='black')
        
        # Add status text
        collected_reward = sum(self.goal_rewards.get(pos, 0) for pos in self.visited_goals)
        plt.title(f"Steps: {self.steps_taken}, Collected: {collected_reward:.2f}, "
                 f"Goals: {len(self.visited_goals)}/{len(self.goal_positions)}")
        
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
            
        collected_reward = sum(self.goal_rewards[pos] for pos in self.visited_goals)
        outfile.write(f"Steps: {self.steps_taken}, Collected: {collected_reward:.2f}, "
                     f"Goals: {len(self.visited_goals)}/{len(self.goal_positions)}\n")
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig is None
