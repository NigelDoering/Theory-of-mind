import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class GoalSpaceGenerator:
    """Generate a rich goal space with Dirichlet-distributed rewards."""
    
    def __init__(self, map_size=8, n_goals=None, seed=None, base_reward=10.0,
                 save_dir="goal_spaces"):
        """
        Initialize the goal space generator.
        
        Args:
            map_size: Size of the grid (nxn)
            n_goals: Number of sub-goals (default: map_size+1)
            seed: Random seed for reproducibility
            base_reward: Base reward value (R_0)
            save_dir: Directory to save goal space configurations
        """
        self.map_size = map_size
        self.n_goals = n_goals if n_goals is not None else map_size + 1
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.base_reward = base_reward
        self.save_dir = save_dir
        
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Internal state
        self.map = None
        self.goal_positions = []
        self.goal_ids = {}
        self.goal_rewards = {}
        self.dirichlet_probs = None
        self.final_goal = None
        
    def generate_goal_space(self, p=0.9):
        """
        Generate a goal space with Dirichlet-distributed rewards.
        
        Args:
            p: Probability of a tile being frozen (higher = fewer holes)
        """
        # Create a base map first (with no goals)
        self.map = self._create_base_map(p=p)
        
        # Select goal positions randomly
        self._select_goal_positions()
        
        # Sample Dirichlet distribution for rewards
        self._assign_rewards()
        
        # Update the map with goal positions
        self._update_map_with_goals()
        
        # Save configuration to JSON
        config_path = self._save_configuration()
        
        return {
            'map': self.map,
            'goal_positions': self.goal_positions,
            'goal_ids': self.goal_ids,
            'goal_rewards': self.goal_rewards,
            'final_goal': self.final_goal,
            'config_path': config_path
        }
    
    def _create_base_map(self, p=0.9):
        """
        Create a base map with randomized start position and no goals.
        
        Args:
            p: Probability of a tile being frozen (higher value = fewer holes)
        """
        # Generate a random map with holes but no goals
        raw_map = generate_random_map(size=self.map_size, p=p, seed=self.seed)
        
        # Debug output - handle both bytes and strings
        print("Generated initial map:")
        for row in raw_map:
            if isinstance(row, bytes):
                print(row.decode('utf-8'))
            else:
                print(row)
        
        # Convert to array for easier manipulation - but use a list of lists first
        map_list = []
        start_pos = None
        goal_positions = []
        
        # First pass: identify positions and create a modifiable list structure
        for i, row in enumerate(raw_map):
            new_row = []
            for j, cell in enumerate(row):
                # Convert cell to bytes if it's not already
                cell_bytes = cell.encode('utf-8') if isinstance(cell, str) else cell
                
                if cell_bytes == b'S':
                    start_pos = (i, j)
                    # Replace start with frozen tile - it will be repositioned later
                    new_row.append(b'F')
                elif cell_bytes == b'G':
                    goal_positions.append((i, j))
                    # Replace goal with frozen tile
                    new_row.append(b'F')
                else:
                    new_row.append(cell_bytes)
            map_list.append(new_row)
        
        # Convert to numpy array
        map_array = np.array(map_list, dtype='object')
        
        # Report the changes
        if start_pos:
            print(f"Found and removed start position at {start_pos}")
        if goal_positions:
            print(f"Found and removed {len(goal_positions)} goal positions")
        
        # Count free spaces
        free_spaces = sum(row.count(b'F') for row in map_list)
        print(f"Initial free spaces: {free_spaces} (need {self.n_goals} for goals)")
        
        # If not enough free spaces, convert some holes to frozen tiles
        if free_spaces < self.n_goals:
            holes_needed = self.n_goals - free_spaces + 2  # Add buffer
            
            # Find all hole positions
            hole_positions = []
            for i, row in enumerate(map_list):
                for j, cell in enumerate(row):
                    if cell == b'H':
                        hole_positions.append((i, j))
            
            # Convert randomly selected holes to frozen
            if hole_positions:
                to_convert = min(len(hole_positions), holes_needed)
                for i, j in random.sample(hole_positions, to_convert):
                    map_array[i, j] = b'F'
                    map_list[i][j] = b'F'  # Update both for consistency
                print(f"Converted {to_convert} holes to frozen tiles")
        
        # Place the start randomly on a frozen tile
        frozen_tiles = []
        for i, row in enumerate(map_list):
            for j, cell in enumerate(row):
                if cell == b'F':
                    frozen_tiles.append((i, j))
        
        if frozen_tiles:
            # Select random frozen tile for start
            start_i, start_j = random.choice(frozen_tiles)
            map_array[start_i, start_j] = b'S'
            print(f"Placed start randomly at ({start_i}, {start_j})")
        else:
            # Rare case: no frozen tiles left
            map_array[0, 0] = b'S'
            print("No frozen tiles available, placed start at (0, 0)")
        
        # Final count of available positions
        final_free_spaces = 0
        for i in range(self.map_size):
            for j in range(self.map_size):
                if map_array[i, j] == b'F':
                    final_free_spaces += 1
        
        print(f"Final map has {final_free_spaces} free spaces for {self.n_goals} goals")
        
        # Display the final map
        print("Final map with randomized start:")
        for row in map_array:
            print(''.join([c.decode('utf-8') for c in row]))
        
        return map_array
    
    def _select_goal_positions(self):
        """Randomly select goal positions."""
        # Get all possible positions (excluding start and holes)
        possible_positions = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map[i][j] == b'F':  # Only frozen tiles can be goals
                    possible_positions.append((i, j))
        
        # Check if we have enough positions
        if len(possible_positions) < self.n_goals:
            raise ValueError(f"Not enough valid positions for {self.n_goals} goals")
        
        # Randomly select goal positions
        selected_indices = random.sample(range(len(possible_positions)), self.n_goals)
        self.goal_positions = [possible_positions[i] for i in selected_indices]
        
        # Assign goal IDs (1 to n_goals)
        for i, pos in enumerate(self.goal_positions):
            self.goal_ids[pos] = i + 1  # 1-based IDs
    
    def _assign_rewards(self):
        """Sample from Dirichlet distribution and assign rewards."""

        ag_alpha = np.random.normal(1, 0.2, size=self.n_goals)
        self.dirichlet_probs = np.random.dirichlet(alpha=np.ones(self.n_goals)*(ag_alpha), size=1)[0]
        
        # Scale probabilities by base reward
        rewards = self.dirichlet_probs * self.base_reward

        # Set the reward for the final goal
        final_goal_index = random.randint(0, self.n_goals - 1)
        self.final_goal = self.goal_positions[final_goal_index]
        self.goal_rewards[self.final_goal] = self.base_reward

        # Set the rewards for the other goals
        for i in range(self.n_goals):
            if i != final_goal_index:
                pos = self.goal_positions[i]
                # Set the reward for the goal position
                self.goal_rewards[pos] = rewards[i]
        
    def _update_map_with_goals(self):
        """Update the map with goal positions."""
        for pos in self.goal_positions:
            i, j = pos
            self.map[i][j] = b'G'
    
    def _save_configuration(self):
        """Save goal space configuration to JSON file."""
        # Convert goal positions to serializable format (tuple -> string)
        serializable_goals = {}
        for pos, goal_id in self.goal_ids.items():
            pos_str = f"{pos[0]}_{pos[1]}"
            serializable_goals[pos_str] = {
                'id': goal_id,
                'reward': self.goal_rewards[pos] if pos in self.goal_rewards else 0.0,
                'is_final': pos == self.final_goal
            }
        
        # Prepare the configuration dictionary
        config = {
            'seed': self.seed,
            'map_size': self.map_size,
            'n_goals': self.n_goals,
            'base_reward': self.base_reward,
            'goals': serializable_goals,
            'dirichlet_probs': self.dirichlet_probs.tolist(),
            'final_goal': f"{self.final_goal[0]}_{self.final_goal[1]}"
        }
        
        # Save to JSON file
        filename = f"goal_space_m{self.map_size}_g{self.n_goals}_s{self.seed}.json"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        return filepath
    
    def visualize_goal_space(self, save_path=None):
        """Visualize the generated goal space with rewards."""
        if self.map is None:
            raise ValueError("Generate goal space first using generate_goal_space()")
        
        plt.figure(figsize=(10, 10))
        
        # Create a grid for visualization
        grid = np.zeros((self.map_size, self.map_size, 3), dtype=float)
        
        # Define colors for different tiles
        colors = {
            b'S': [0.0, 0.5, 0.0],    # Start: Dark Green
            b'F': [0.9, 0.9, 1.0],    # Frozen: Light Blue
            b'H': [0.2, 0.2, 0.8],    # Hole: Dark Blue
            b'G': [1.0, 0.9, 0.0]     # Goal: Gold
        }
        
        # Fill the grid with colors
        for i in range(self.map_size):
            for j in range(self.map_size):
                grid[i, j] = colors[self.map[i][j]]
        
        # Plot the grid
        plt.imshow(grid)
        
        # Add text annotations for all cells
        for i in range(self.map_size):
            for j in range(self.map_size):
                # Basic tile label
                tile_dict = {b'S': 'S', b'F': 'F', b'H': 'H', b'G': 'G'}
                label = tile_dict[self.map[i][j]]
                
                # Add goal ID and reward if applicable
                if (i, j) in self.goal_positions:
                    goal_id = self.goal_ids[(i, j)]
                    reward = self.goal_rewards[(i, j)]
                    if (i, j) == self.final_goal:
                        label = f"{label}\nID:{goal_id}\nFINAL\n{reward:.2f}"
                    else:
                        label = f"{label}\nID:{goal_id}\n{reward:.2f}"
                
                plt.text(j, i, label, ha="center", va="center", 
                         color="black", fontsize=8)
        
        # Add markers for goals with size proportional to reward
        for pos in self.goal_positions:
            i, j = pos
            reward = self.goal_rewards[pos]
            marker_size = 100 + (reward / self.base_reward) * 500
            
            if pos == self.final_goal:
                # Final goal gets a star marker
                plt.plot(j, i, 'r*', markersize=20)
            else:
                # Regular goals get circles with size proportional to reward
                plt.plot(j, i, 'go', markersize=marker_size/100, alpha=0.5)
        
        # Remove ticks
        plt.xticks([])
        plt.yticks([])
        
        # Add a title
        plt.title(f"Goal Space (Size: {self.map_size}x{self.map_size}, Goals: {self.n_goals})\n"
                 f"Seed: {self.seed}, Base Reward: {self.base_reward}")
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    @classmethod
    def load_from_config(cls, config_path):
        """Load a goal space configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create a new generator with the loaded parameters
        generator = cls(
            map_size=config['map_size'],
            n_goals=config['n_goals'],
            seed=config['seed'],
            base_reward=config['base_reward']
        )
        
        # Reconstruct the map
        generator.map = generator._create_base_map()
        
        # Reconstruct goal positions, IDs, and rewards
        for pos_str, goal_info in config['goals'].items():
            i, j = map(int, pos_str.split('_'))
            pos = (i, j)
            
            # Add to goal positions
            generator.goal_positions.append(pos)
            
            # Set goal ID
            generator.goal_ids[pos] = goal_info['id']
            
            # Set reward
            generator.goal_rewards[pos] = goal_info['reward']
            
            # Set final goal if applicable
            if goal_info['is_final']:
                generator.final_goal = pos
        
        # Set Dirichlet probabilities
        generator.dirichlet_probs = np.array(config['dirichlet_probs'])
        
        # Update map with goals
        generator._update_map_with_goals()
        
        return generator