import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from frozen_src.high_level_planner import HighLevelPlanner
from frozen_src.low_level_controller import LowLevelController

class HierarchicalAgent:
    """Hierarchical agent combining high-level planning with low-level control."""
    
    def __init__(self, env, budget=1000, learning_rate=0.8, gamma=0.95, epsilon=0.2, perception_radius=2):
        """
        Initialize the hierarchical agent.
        
        Args:
            env: FrozenLake environment
            budget: Maximum number of steps allowed
            learning_rate: Learning rate for Q-learning
            gamma: Discount factor
            epsilon: Exploration probability
            perception_radius: How far the agent can perceive around itself
        """
        self.env = env
        self.env_unwrapped = env.unwrapped
        self.high_level_planner = HighLevelPlanner(env, budget)
        self.low_level_controller = LowLevelController(env, learning_rate, gamma, epsilon)
        self.total_steps = 0
        self.max_budget = budget
        self.remaining_budget = budget
        self.perception_radius = perception_radius
        self.nrow = self.env_unwrapped.nrow
        self.ncol = self.env_unwrapped.ncol
        
        # Maps for tracking environment knowledge
        self.known_map = np.zeros((self.nrow, self.ncol), dtype=bool)
        self.tile_types = np.zeros((self.nrow, self.ncol), dtype=object)
        
        # Goal tracking
        self.goals = []  # Will be discovered through exploration
        self.rewards = {}  # Default rewards
        self.final_goal = None
        self.current_subgoal = None
        self.discovered_goals = set()
        self.reached_goals = set()
        self.replanning_needed = True
        
        # Position and path tracking
        self.current_position = None
        self.path_history = []  # Store agent's path
        
        # Ensure visualization directory exists
        os.makedirs("exploration_visuals", exist_ok=True)
    
    def update_observations(self, state, observation):
        """Update agent's knowledge based on observations within perception radius."""
        # Update current position
        self.current_position = state
        self.path_history.append(state)
        
        # Convert flat state to 2D coordinates
        row, col = state // self.env_unwrapped.ncol, state % self.env_unwrapped.ncol
        exploration_bonus = 0.0  # Initialize exploration bonus
        
        # Check cells within perception radius
        for dr in range(-self.perception_radius, self.perception_radius + 1):
            for dc in range(-self.perception_radius, self.perception_radius + 1):
                # Calculate Manhattan distance
                if abs(dr) + abs(dc) <= self.perception_radius:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.env_unwrapped.nrow and 0 <= nc < self.env_unwrapped.ncol:
                        # Add exploration bonus for newly discovered cells
                        if not self.known_map[nr, nc]:
                            exploration_bonus += 0.05  # Small bonus for each new cell
                        
                        # Mark as known
                        self.known_map[nr, nc] = True
                        
                        # Update tile type
                        cell_type = self.env_unwrapped.desc[nr][nc]
                        self.tile_types[(nr, nc)] = 'hole' if cell_type == b'H' else 'normal'
                        
                        # Update discovered goals
                        if cell_type == b'G':
                            goal_state = nr * self.env_unwrapped.ncol + nc
                            if goal_state not in self.discovered_goals:
                                self.discovered_goals.add(goal_state)
                                # Add to goals list and assign default reward
                                if goal_state not in self.goals:
                                    self.goals.append(goal_state)
                                    self.rewards[goal_state] = 1.0
                                self.replanning_needed = True
                                
        # Check if current state is a goal
        goal_reached = self.env_unwrapped.desc[row][col] == b'G' and state not in self.reached_goals
        if goal_reached:
            self.reached_goals.add(state)
            self.replanning_needed = True
        
        return exploration_bonus, goal_reached

    def select_action(self, state):
        """Select action based on hierarchical planning."""
        # Decrease remaining budget
        self.total_steps += 1
        self.remaining_budget -= 1
        
        # Check if we need to replan
        if self.replanning_needed:
            self.replan(state)
        
        # Check deadline for switching to final goal
        if self.final_goal is not None:
            dist_to_final = self.high_level_planner.estimate_distance(state, self.final_goal)
            if dist_to_final != float('inf') and self.remaining_budget <= dist_to_final + 5:
                self.high_level_planner.switch_to_final()
                self.current_subgoal = self.final_goal
                print("Switching to final goal due to budget constraints!")
        
        # Select next sub-goal if needed
        if self.current_subgoal is None or state == self.current_subgoal:
            self.current_subgoal = self.high_level_planner.next_subgoal()
            if self.current_subgoal is None and self.final_goal is not None:
                # No more sub-goals, try to reach final goal
                self.current_subgoal = self.final_goal
        
        # If no clear subgoal, find nearest unexplored area
        if self.current_subgoal is None:
            self.current_subgoal = self._find_nearest_unexplored(state)
            if self.current_subgoal is not None:
                print(f"Exploring towards nearest unexplored area at {self.current_subgoal}")
        
        # If still no subgoal, use random exploration
        if self.current_subgoal is None:
            # Use low-level controller with random exploration
            return self.low_level_controller.random_action()
        
        # Use low-level controller to select action toward current sub-goal
        return self.low_level_controller.select_action(state, self.current_subgoal)
    
    def replan(self, state):
        """Replan the high-level path."""
        # Get remaining goals (not yet reached)
        remaining_goals = [g for g in self.goals if g not in self.reached_goals]
        
        # If we have very few discovered goals, prioritize exploration
        if len(self.discovered_goals) < 4:  # Adjust this threshold as needed
            # Find the nearest unexplored area
            temp_goal = self._find_nearest_unexplored(state)
            if temp_goal is not None:
                print(f"  → Adding exploration goal at {temp_goal}")
                if temp_goal not in remaining_goals:
                    remaining_goals.append(temp_goal)
                # Give exploration a moderate reward
                self.rewards[temp_goal] = 0.5
        
        # Print remaining goals and their rewards
        if remaining_goals:
            print(f"  - Replanning with remaining goals: {remaining_goals}")
            
            # Use the assigned final_goal if it exists, otherwise use highest reward goal
            if self.final_goal is not None and self.final_goal in remaining_goals:
                print(f"  → Using known final goal: {self.final_goal}")
            elif remaining_goals:
                # Use highest reward goal as temporary final goal
                self.final_goal = max(remaining_goals, key=lambda g: self.rewards.get(g, 0))
                print(f"  → Final goal set to: {self.final_goal}")
                
            # Plan the path to the final goal
            self.high_level_planner.plan(state, self.final_goal)
            
            # Get next subgoal from planner
            self.current_subgoal = self.high_level_planner.get_next_subgoal()
            
            # Update low-level controller's goal
            self.low_level_controller.set_goal(self.current_subgoal)
        else:
            print("  - No remaining goals, exploring randomly")
            self.current_subgoal = None
            
        self.replanning_needed = False
    
    def _find_nearest_unexplored(self, state):
        """Find the nearest state adjacent to unexplored territory."""
        row, col = state // self.env_unwrapped.ncol, state % self.env_unwrapped.ncol
        
        # First, find all states that are known but adjacent to unknown
        exploration_candidates = []
        for i in range(self.env_unwrapped.nrow):
            for j in range(self.env_unwrapped.ncol):
                if self.known_map[i, j] and self.tile_types.get((i, j)) != 'hole':
                    # Check if this is adjacent to any unknown cells
                    has_unknown_neighbor = False
                    for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.env_unwrapped.nrow and 
                            0 <= nj < self.env_unwrapped.ncol and
                            not self.known_map[ni, nj]):
                            has_unknown_neighbor = True
                            break
                    
                    if has_unknown_neighbor:
                        state_idx = i * self.env_unwrapped.ncol + j
                        # Calculate Manhattan distance from current state
                        distance = abs(i - row) + abs(j - col)
                        exploration_candidates.append((state_idx, distance))
        
        # Sort by distance and return the closest
        if exploration_candidates:
            exploration_candidates.sort(key=lambda x: x[1])
            return exploration_candidates[0][0]
        
        return None
    
    def learn(self, state, action, reward, next_state, done):
        """Learn from experience."""
        # Update low-level controller's knowledge
        self.low_level_controller.learn(state, action, reward, next_state, done)
        
        # Decay exploration rate
        if done:
            self.low_level_controller.decay_epsilon()
    
    def reset(self, budget=None):
        """Reset the agent for a new episode while retaining map knowledge."""
        if budget is not None:
            self.max_budget = budget
        
        self.remaining_budget = self.max_budget
        self.current_position = None
        self.path_history = []
        self.current_subgoal = None
        self.replanning_needed = True
        
        # Don't reset knowledge between episodes
        # self.known_map = np.zeros((self.nrow, self.ncol), dtype=bool)
        # self.tile_types = np.zeros((self.nrow, self.ncol), dtype=object)
        # self.discovered_goals = set()
        
        # Instead, just reset which goals have been reached this episode
        self.reached_goals = set()
    
    def visualize_knowledge(self, episode, step):
        """Visualize the agent's current knowledge of the environment."""
        plt.figure(figsize=(10, 10))
        
        # Grid visualization
        grid = np.zeros((self.nrow, self.ncol, 3), dtype=float)
        
        # Fill grid based on knowledge
        for i in range(self.nrow):
            for j in range(self.ncol):
                state_idx = i * self.ncol + j
                
                if not self.known_map[i, j]:
                    # Unknown - dark gray
                    grid[i, j] = [0.3, 0.3, 0.3]
                else:
                    # Known areas
                    tile = self.env_unwrapped.desc[i][j]
                    if tile == b'S':
                        grid[i, j] = [0.0, 0.5, 0.0]  # Start: dark green
                    elif tile == b'F':
                        grid[i, j] = [0.9, 0.9, 1.0]  # Frozen: light blue
                    elif tile == b'H':
                        grid[i, j] = [0.2, 0.2, 0.8]  # Hole: dark blue
                    elif tile == b'G':
                        grid[i, j] = [1.0, 0.9, 0.0]  # Goal: gold
        
        # Plot grid
        plt.imshow(grid)
        
        # Add annotations and highlights
        for i in range(self.nrow):
            for j in range(self.ncol):
                state_idx = i * self.ncol + j
                
                # Mark the current position
                if self.current_position is not None and state_idx == self.current_position:
                    plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                                    edgecolor='red', linewidth=3))
                
                # Mark perception field
                if self.current_position is not None:
                    current_row, current_col = self.current_position // self.ncol, self.current_position % self.ncol
                    if abs(i - current_row) + abs(j - current_col) <= self.perception_radius:
                        plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                                        edgecolor='yellow', linewidth=1))
                
                # Mark discovered goals
                if state_idx in self.discovered_goals and state_idx not in self.reached_goals:
                    plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                                    edgecolor='green', linewidth=2))
                
                # Mark reached goals
                if state_idx in self.reached_goals:
                    plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                                    edgecolor='purple', linewidth=2))
                
                # Mark current subgoal
                if self.current_subgoal is not None and state_idx == self.current_subgoal:
                    plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                                    edgecolor='cyan', linewidth=2))
                    
                # Mark final goal
                if self.final_goal is not None and state_idx == self.final_goal:
                    plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                                    edgecolor='orange', linewidth=3))
                
                # Add labels to known tiles
                if self.known_map[i, j]:
                    tile_dict = {b'S': 'S', b'F': '·', b'H': 'H', b'G': 'G'}
                    plt.text(j, i, tile_dict[self.env_unwrapped.desc[i][j]], 
                             ha="center", va="center", color="black", fontsize=10)
                
                # Add reward text for discovered goals
                if state_idx in self.discovered_goals:
                    reward = self.rewards.get(state_idx, 0.0)
                    plt.text(j, i-0.3, f"R:{reward:.1f}", 
                             ha="center", va="center", color="green", fontsize=8)
        
        # Add path history
        if len(self.path_history) > 1:
            path_x = []
            path_y = []
            for state in self.path_history:
                row, col = state // self.ncol, state % self.ncol
                path_y.append(row)
                path_x.append(col)
            plt.plot(path_x, path_y, 'r-', linewidth=1, alpha=0.7)
        
        # Remove ticks
        plt.xticks([])
        plt.yticks([])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.3, 0.3, 0.3], edgecolor='black', label='Unknown'),
            Patch(facecolor=[0.9, 0.9, 1.0], edgecolor='black', label='Frozen'),
            Patch(facecolor=[0.2, 0.2, 0.8], edgecolor='black', label='Hole'),
            Patch(facecolor=[1.0, 0.9, 0.0], edgecolor='black', label='Goal'),
            Patch(facecolor='none', edgecolor='red', label='Current Position'),
            Patch(facecolor='none', edgecolor='yellow', label='Perception Field'),
            Patch(facecolor='none', edgecolor='green', label='Discovered Goal'),
            Patch(facecolor='none', edgecolor='purple', label='Reached Goal'),
            Patch(facecolor='none', edgecolor='cyan', label='Current Subgoal'),
            Patch(facecolor='none', edgecolor='orange', label='Final Goal')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add title with stats
        plt.title(f"Agent Knowledge - Episode {episode}, Step {step}\n"
                 f"Known: {np.sum(self.known_map)}/{self.nrow * self.ncol} tiles")
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(f"exploration_visuals/knowledge_ep{episode}_step{step}.png")
        plt.close()
        
    def visualize_plan(self, episode, step):
        """Visualize the agent's current plan as a graph."""
        plt.figure(figsize=(12, 12))
        
        # Create a graph
        G = nx.DiGraph()
        
        # Add nodes for each position
        for i in range(self.nrow):
            for j in range(self.ncol):
                node_id = i * self.ncol + j
                if self.known_map[i, j]:
                    # Use tile type for known nodes
                    tile = self.env_unwrapped.desc[i][j]
                    G.add_node(node_id, pos=(j, -i), type=tile.decode('utf-8'))
                else:
                    # Add unknown nodes but mark them differently
                    G.add_node(node_id, pos=(j, -i), type='unknown')
        
        # Add edges for valid moves
        for i in range(self.nrow):
            for j in range(self.ncol):
                node_id = i * self.ncol + j
                
                if not self.known_map[i, j]:
                    continue  # Skip unknown nodes
                
                # Skip holes
                if self.env_unwrapped.desc[i, j] == b'H':
                    continue
                
                # Add edges to adjacent nodes
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nrow and 0 <= nj < self.ncol:
                        if self.known_map[ni, nj] and self.env_unwrapped.desc[ni][nj] != b'H':
                            # Valid edge between known non-hole tiles
                            neighbor_id = ni * self.ncol + nj
                            G.add_edge(node_id, neighbor_id, weight=1)
        
        # Get positions for visualization
        pos = nx.get_node_attributes(G, 'pos')
        
        # Node colors based on type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node]['type']
            if node_type == 'unknown':
                node_colors.append('gray')
            elif node_type == 'S':
                node_colors.append('darkgreen')
            elif node_type == 'F':
                node_colors.append('lightblue')  
            elif node_type == 'H':
                node_colors.append('blue')
            elif node_type == 'G':
                node_colors.append('gold')
            else:
                node_colors.append('gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='black', width=1.0, alpha=0.5)
        
        # Draw planned path if available
        if hasattr(self.high_level_planner, 'tour') and self.high_level_planner.tour:
            path_edges = [(self.high_level_planner.tour[i], self.high_level_planner.tour[i+1]) 
                          for i in range(len(self.high_level_planner.tour)-1)
                          if self.high_level_planner.tour[i] in G.nodes and self.high_level_planner.tour[i+1] in G.nodes]
            
            if path_edges:
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.5)
        
        # Draw path history
        if len(self.path_history) > 1:
            history_edges = [(self.path_history[i], self.path_history[i+1]) 
                           for i in range(len(self.path_history)-1)
                           if self.path_history[i] in G.nodes and self.path_history[i+1] in G.nodes]
            
            if history_edges:
                nx.draw_networkx_edges(G, pos, edgelist=history_edges, edge_color='orange', 
                                      width=1.5, style='dashed', alpha=0.7)
        
        # Add labels to known nodes
        labels = {}
        for node in G.nodes():
            if G.nodes[node]['type'] != 'unknown':
                labels[node] = f"{node}"
            else:
                labels[node] = "?"
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Highlight special nodes
        if self.current_position in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[self.current_position], 
                                  node_color='red', node_shape='o', node_size=500)
        
        if self.discovered_goals:
            goal_nodes = [g for g in self.discovered_goals if g in G.nodes]
            if goal_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=goal_nodes, 
                                      node_color='none', node_shape='o', node_size=700,
                                      edgecolors='green', linewidths=2)
        
        if self.current_subgoal in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[self.current_subgoal], 
                                  node_color='none', node_shape='o', node_size=900,
                                  edgecolors='cyan', linewidths=2)
        
        if self.final_goal in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[self.final_goal], 
                                  node_color='none', node_shape='o', node_size=1000,
                                  edgecolors='orange', linewidths=3)
        
        # Add title
        plt.title(f"Agent Plan Graph - Episode {episode}, Step {step}\n"
                 f"Current: {self.current_position}, Subgoal: {self.current_subgoal}, Final: {self.final_goal}")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', label='Unknown'),
            Patch(facecolor='lightblue', label='Frozen'),
            Patch(facecolor='blue', label='Hole'),
            Patch(facecolor='gold', label='Goal'),
            Patch(facecolor='red', label='Current Position'),
            Patch(facecolor='none', edgecolor='green', label='Discovered Goal'),
            Patch(facecolor='none', edgecolor='cyan', label='Current Subgoal'),
            Patch(facecolor='none', edgecolor='orange', label='Final Goal'),
            Patch(facecolor='none', edgecolor='red', label='Planned Path'),
            Patch(facecolor='none', edgecolor='orange', label='Path History', linestyle='--')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"exploration_visuals/plan_ep{episode}_step{step}.png")
        plt.close()