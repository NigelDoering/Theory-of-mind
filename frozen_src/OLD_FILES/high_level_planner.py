import numpy as np
import networkx as nx
from heapq import heappush, heappop

class HighLevelPlanner:
    """High-level planner that solves an orienteering problem to maximize reward collection."""
    
    def __init__(self, env, rewards=None):
        """
        Initialize the high-level planner.
        
        Args:
            env: FrozenLake environment
            rewards: Dictionary mapping goal positions to rewards
        """
        self.env = env.unwrapped
        self.rewards = rewards or {}
        self.start = None
        self.current = None
        self.final_goal = None
        self.tour = []
        self.sub_goals = []
        self.remaining_budget = 0
        self.reached_goals = set()
        self.heading_to_final = False
        
        # Create a graph representation of the environment
        self.graph = self._create_initial_graph()
        
        # Keep track of hole positions
        self.holes = set()
    
    def _create_initial_graph(self):
        """Create an initial graph representation with unknown weights."""
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                node_id = i * self.env.ncol + j
                G.add_node(node_id, pos=(i, j), known=False, visited=False)
        
        # Add edges (4-way connectivity)
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                node_id = i * self.env.ncol + j
                
                # Add edges to adjacent cells (right, down, left, up)
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.env.nrow and 0 <= nj < self.env.ncol:
                        neighbor_id = ni * self.env.ncol + nj
                        G.add_edge(node_id, neighbor_id, weight=10.0, known=False)
        
        return G
    
    def update_graph(self, observed_map, tile_types):
        """
        Update the graph with observed information including complete hole removal.
        
        Args:
            observed_map: Boolean array of observed positions
            tile_types: Dictionary mapping positions to tile types
        """
        # Process all observed tiles
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                if observed_map[i, j]:
                    node_id = i * self.env.ncol + j
                    
                    # Mark node as known
                    if node_id in self.graph:
                        self.graph.nodes[node_id]['known'] = True
                    
                    # If it's a hole, remove it completely from the graph
                    if (i, j) in tile_types and tile_types[(i, j)] == 'hole':
                        if node_id in self.graph:
                            self.graph.remove_node(node_id)
                            self.holes.add(node_id)
                            print(f"SAFETY: Removed hole at position ({i},{j}) [node {node_id}] from planning graph")
                    else:
                        # For non-hole tiles, update edge weights to known values
                        for neighbor in list(self.graph.neighbors(node_id)):
                            if neighbor not in self.holes:  # Skip edges to holes
                                self.graph[node_id][neighbor]['weight'] = 1.0
                                self.graph[node_id][neighbor]['known'] = True
    
    def estimate_distance(self, start, goal):
        """Estimate the distance between two states using the graph."""
        # If either node is a hole, return infinity
        if start in self.holes or goal in self.holes:
            return float('inf')
            
        try:
            # Use A* search for pathfinding
            path = nx.astar_path(self.graph, start, goal, heuristic=lambda u, v: 0)
            distance = len(path) - 1
            return distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path is found, return a large value
            return float('inf')
    
    def plan(self, current_pos, goals, rewards, remaining_budget):
        """
        Plan a tour through goals using a greedy approach with modified cost function.
        
        Cost = Distance(current_node, target_node) - Reward(target_node) + Distance(target_node, final_goal)
        Lower cost is better.
        
        Args:
            current_pos: Current position in the environment
            goals: List of goal positions
            rewards: Dictionary mapping goal positions to rewards
            remaining_budget: Remaining number of steps
            
        Returns:
            Ordered list of sub-goals to visit
        """
        self.start = current_pos
        self.current = current_pos
        self.remaining_budget = remaining_budget
        
        # Filter out already reached goals
        valid_goals = [(g, rewards[g]) for g in goals if g not in self.reached_goals]
        
        # Find the final goal (highest reward)
        if valid_goals:
            self.final_goal = max(valid_goals, key=lambda x: x[1])[0]
        else:
            self.final_goal = None
            return []
        
        # Initialize the greedy tour
        self.tour = [self.start]
        self.sub_goals = []
        
        # Make a copy of all unvisited goals
        unvisited_goals = [g for g, _ in valid_goals]
        total_distance = 0
        
        # Keep adding goals until we run out of budget or goals
        while unvisited_goals and total_distance < self.remaining_budget:
            best_goal = None
            best_cost = float('inf')
            
            # For each unvisited goal, calculate its cost
            for goal in unvisited_goals:
                # Distance from current position to this goal
                distance_to_goal = self.estimate_distance(self.current, goal)
                
                # Distance from this goal to final goal
                distance_to_final = self.estimate_distance(goal, self.final_goal)
                
                # Reward value (negative to convert to cost)
                reward_value = -rewards.get(goal, 0)  # Negative because lower cost is better
                
                # Calculate combined cost
                # We want to minimize: distance_to_goal - reward_value + distance_to_final
                cost = distance_to_goal + reward_value + distance_to_final
                
                # Check if we have enough budget
                if distance_to_goal + total_distance <= self.remaining_budget and cost < best_cost:
                    best_cost = cost
                    best_goal = goal
            
            # If we found a goal we can reach within budget
            if best_goal:
                # Add it to our tour
                self.tour.append(best_goal)
                unvisited_goals.remove(best_goal)
                
                # Update our current position and total distance
                distance_moved = self.estimate_distance(self.current, best_goal)
                total_distance += distance_moved
                self.current = best_goal
            else:
                # None of the remaining goals are reachable within budget
                break
        
        # Ensure final goal is in the tour if there's enough budget
        if (self.final_goal not in self.tour and 
                total_distance + self.estimate_distance(self.current, self.final_goal) <= self.remaining_budget):
            self.tour.append(self.final_goal)
        
        # Extract sub-goals (excluding current position)
        self.sub_goals = self.tour[1:] if self.tour else []
        
        return self.sub_goals
    
    def _compute_tour_length(self, tour):
        """Compute the length of a tour."""
        length = 0
        for i in range(len(tour) - 1):
            length += self.estimate_distance(tour[i], tour[i+1])
        return length
    
    def next_subgoal(self):
        """Return the next sub-goal to visit."""
        if self.heading_to_final:
            return self.final_goal
            
        if not self.sub_goals:
            return None
            
        return self.sub_goals[0]
    
    def subgoal_reached(self, goal):
        """Mark a sub-goal as reached."""
        if goal in self.sub_goals:
            self.sub_goals.remove(goal)
        self.reached_goals.add(goal)
        self.current = goal
    
    def switch_to_final(self):
        """Switch to heading directly to the final goal."""
        self.heading_to_final = True
        self.sub_goals = [self.final_goal] if self.final_goal else []
    
    def needs_replanning(self, current_pos, new_goals_discovered, new_goal_reached):
        """Determine if replanning is needed."""
        return new_goals_discovered or new_goal_reached or current_pos != self.current