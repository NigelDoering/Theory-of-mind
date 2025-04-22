import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from frozen_src.high_level_planner import HighLevelPlanner
from frozen_src.low_level_controller import LowLevelController

class HierarchicalAgent:
    """Hierarchical agent combining high-level planning with low-level control."""
    
    def __init__(self, env, budget=100, learning_rate=0.1, gamma=0.99, epsilon=0.1, perception_radius=1):
        """Initialize the hierarchical agent."""
        self.env = env
        self.env_unwrapped = env.unwrapped
        self.max_budget = budget
        self.remaining_budget = budget
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.perception_radius = perception_radius

        self.nrow = self.env_unwrapped.nrow
        self.ncol = self.env_unwrapped.ncol
        
        # Knowledge representation
        self.known_map = np.zeros((self.env_unwrapped.nrow, self.env_unwrapped.ncol), dtype=bool)
        self.tile_types = {}
        
        # Goal tracking
        self.goals = []
        self.rewards = {}
        self.discovered_goals = set()
        self.reached_goals = set()
        self.final_goal = None
        
        # Path tracking
        self.current_position = None
        self.path_history = []
        self.current_subgoal = None
        
        # Track known holes for safety
        self.known_holes = set()
        
        # High-level planner
        self.high_level_planner = HighLevelPlanner(self.env, self.rewards)
        
        # Low-level controller
        self.low_level_controller = LowLevelController(self.env, learning_rate=learning_rate, gamma=gamma, epsilon=epsilon)
        
        # Flags
        self.replanning_needed = True
    
    def update_observations(self, state, observation):
        """Update agent's knowledge based on observations within perception radius."""
        # Update current position
        self.current_position = state
        self.path_history.append(state)
        
        # Convert flat state to 2D coordinates
        row, col = state // self.env_unwrapped.ncol, state % self.env_unwrapped.ncol
        exploration_bonus = 0.0  # Initialize exploration bonus
        
        # Keep track of newly discovered holes
        newly_discovered_holes = set()
        
        # Check cells within perception radius
        for dr in range(-self.perception_radius, self.perception_radius + 1):
            for dc in range(-self.perception_radius, self.perception_radius + 1):
                # Calculate Manhattan distance
                if abs(dr) + abs(dc) <= self.perception_radius:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.env_unwrapped.nrow and 0 <= nc < self.env_unwrapped.ncol:
                        # Add exploration bonus for newly discovered cells
                        if not self.known_map[nr, nc]:
                            exploration_bonus += 0.02  # Increased bonus for new cells
                        
                        # Mark as known
                        self.known_map[nr, nc] = True
                        
                        # Update tile type
                        cell_type = self.env_unwrapped.desc[nr][nc]
                        is_hole = (cell_type == b'H')
                        
                        self.tile_types[(nr, nc)] = 'hole' if is_hole else 'normal'
                        
                        # Record hole state for safety
                        if is_hole:
                            hole_state = nr * self.env_unwrapped.ncol + nc
                            self.known_holes.add(hole_state)
                            newly_discovered_holes.add(hole_state)
                        
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
        
        # If we discovered holes, update the high-level planner's graph
        if newly_discovered_holes:
            self.high_level_planner.update_graph(self.known_map, self.tile_types)
            self.replanning_needed = True
            
        # Check if current state is a goal
        goal_reached = self.env_unwrapped.desc[row][col] == b'G'
        
        # For newly reached goals, mark as reached but don't remove from consideration
        if goal_reached and state not in self.reached_goals:
            self.reached_goals.add(state)
            self.replanning_needed = True
        
        return exploration_bonus, goal_reached

    def select_action(self, state):
        """
        Select an action using the hierarchical approach.
        
        First, use high-level planner to select a subgoal.
        Then, use low-level controller to navigate to that subgoal.
        """
        # Update the remaining budget
        self.remaining_budget -= 1
        
        # If planning is needed, replan
        if self.replanning_needed:
            self.replan(state)
        
        # Use low-level controller to navigate to current subgoal
        # Pass the known_holes to avoid selecting actions that lead to holes
        if self.current_subgoal is not None:
            action = self.low_level_controller.select_action(state, known_holes=self.known_holes)
        else:
            # If no subgoal, take a random action (with safety check)
            valid_actions = []
            row, col = state // self.env_unwrapped.ncol, state % self.env_unwrapped.ncol
            
            for action in range(4):  # LEFT, DOWN, RIGHT, UP
                dr, dc = 0, 0
                if action == 0:  # LEFT
                    dc = -1
                elif action == 1:  # DOWN
                    dr = 1
                elif action == 2:  # RIGHT
                    dc = 1
                elif action == 3:  # UP
                    dr = -1
                
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.env_unwrapped.nrow and 0 <= nc < self.env_unwrapped.ncol:
                    next_state = nr * self.env_unwrapped.ncol + nc
                    if next_state not in self.known_holes:
                        valid_actions.append(action)
            
            if not valid_actions:
                valid_actions = list(range(4))  # Fallback if all actions are risky
            
            action = np.random.choice(valid_actions)
        
        return action
    
    def replan(self, state):
        """Replan the high-level path."""
        # Get all goals, without excluding reached ones
        # We will adjust their rewards instead
        remaining_goals = self.goals.copy()
        
        # Adjust rewards for goals we've already reached - they're less valuable now
        for goal in remaining_goals:
            if goal in self.reached_goals:
                # Reduce the reward for already visited goals but don't make it zero
                # This lets the agent consider revisiting if it makes sense in a larger tour
                original_reward = self.rewards.get(goal, 1.0)
                self.rewards[goal] = original_reward * 0.2  # Significant reduction but still positive
        
        # Boost exploration by adding exploration goals, especially early on
        exploration_candidates = self._find_multiple_exploration_candidates(state, count=3)
        for i, candidate in enumerate(exploration_candidates):
            if candidate is not None and candidate not in remaining_goals:
                print(f"  → Adding exploration goal at {candidate}")
                remaining_goals.append(candidate)
                # Assign decreasing rewards to farther exploration candidates
                self.rewards[candidate] = 0.15 - (i * 0.05)  # 0.15, 0.1, 0.05
        
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
            self.high_level_planner.plan(state, remaining_goals, self.rewards, self.remaining_budget)
            
            # Get next subgoal from planner
            self.current_subgoal = self.high_level_planner.next_subgoal()
            
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
                if self.known_map[i, j] and self.tile_types[i, j] != 'hole':
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
    
    def _find_multiple_exploration_candidates(self, state, count=3):
        """Find the nearest states adjacent to unexplored territory."""
        row, col = state // self.env_unwrapped.ncol, state % self.env_unwrapped.ncol
        
        # First, find all states that are known but adjacent to unknown
        exploration_candidates = []
        for i in range(self.env_unwrapped.nrow):
            for j in range(self.env_unwrapped.ncol):
                if self.known_map[i, j] and self.tile_types[i, j] != 'hole':
                    # Check if this is adjacent to any unknown cells
                    unknown_neighbor_count = 0
                    for di, dj in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.env_unwrapped.nrow and 
                            0 <= nj < self.env_unwrapped.ncol and
                            not self.known_map[ni, nj]):
                            unknown_neighbor_count += 1
                
                    if unknown_neighbor_count > 0:
                        state_idx = i * self.env_unwrapped.ncol + j
                        # Calculate Manhattan distance from current state
                        distance = abs(i - row) + abs(j - col)
                        # Prioritize locations with more unknown neighbors
                        exploration_score = unknown_neighbor_count / (distance + 1)
                        exploration_candidates.append((state_idx, distance, exploration_score))
        
        # Sort by exploration score (higher is better) and return the top candidates
        if exploration_candidates:
            exploration_candidates.sort(key=lambda x: x[2], reverse=True)
            return [candidate[0] for candidate in exploration_candidates[:count]]
        
        return [None] * count

    def learn(self, state, action, reward, next_state, done):
        """Learn from experience."""
        # Update low-level controller's knowledge
        self.low_level_controller.learn(state, action, reward, next_state, done)
        
        # Decay exploration rate
        if done:
            self.low_level_controller.decay_epsilon()
    
    def reset(self, budget=None):
        """Reset the agent for a new episode while retaining map knowledge."""
        # Calculate max steps based on grid size if not provided
        if budget is None:
            n = self.env_unwrapped.nrow
            budget = int(n**2 + n*np.sqrt(2))
        
        self.max_budget = budget
        self.remaining_budget = self.max_budget
        self.current_position = None
        self.path_history = []
        self.current_subgoal = None
        self.replanning_needed = True
        
        # Keep map knowledge and discovered goals between episodes
        # But reset reached goals for this episode to encourage revisiting goals
        self.reached_goals = set()
        
        # Reset the high-level planner's internal state
        if hasattr(self.high_level_planner, 'reached_goals'):
            self.high_level_planner.reached_goals = set()
    
    def visualize_knowledge(self, episode, step, save_dir="exploration_visuals"):
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
        plt.savefig(f"{save_dir}/knowledge_ep{episode}_step{step}.png")
        plt.close()
        
    def visualize_plan(self, episode, step, save_dir="exploration_visuals"):
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
        plt.savefig(f"{save_dir}/plan_ep{episode}_step{step}.png")
        plt.close()