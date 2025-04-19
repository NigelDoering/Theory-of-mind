import numpy as np
import networkx as nx
from heapq import heappush, heappop

class HighLevelPlanner:
    """High-level planner that solves an orienteering problem to maximize reward collection."""
    
    def __init__(self, env, budget, unknown_cost=2.0):
        """
        Initialize the high-level planner.
        
        Args:
            env: FrozenLake environment
            budget: Maximum number of steps allowed
            unknown_cost: Estimated cost for unknown edges
        """
        self.env = env
        self.env.nrow = int(np.sqrt(env.observation_space.n))
        self.env.ncol = self.env.nrow
        self.max_budget = budget
        self.remaining_budget = budget
        self.unknown_cost = unknown_cost
        self.graph = self._build_initial_graph()
        self.start = None
        self.current = None
        self.final_goal = None
        self.sub_goals = []
        self.tour = []
        self.reached_goals = set()
        self.heading_to_final = False
        
    def _build_initial_graph(self):
        """Build the initial graph representation of the environment."""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each position in the grid
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                pos = i * self.env.ncol + j
                G.add_node(pos, pos=(i, j), visited=False, known=False)
                
                # Add edges to adjacent cells (initially with unknown cost)
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.env.nrow and 0 <= nj < self.env.ncol:
                        neighbor_pos = ni * self.env.ncol + nj
                        G.add_edge(pos, neighbor_pos, weight=self.unknown_cost, known=False)
        
        return G
    
    def update_graph(self, current_pos, observed):
        """
        Update the graph with observed information.
        
        Args:
            current_pos: Current position in the environment
            observed: Dictionary of observed states and their properties
        """
        # Mark current node as visited and known
        self.graph.nodes[current_pos]['visited'] = True
        self.graph.nodes[current_pos]['known'] = True
        
        # Update knowledge about observed positions
        for pos, properties in observed.items():
            node_id = pos[0] * self.env.ncol + pos[1]
            self.graph.nodes[node_id]['known'] = True
            
            # Update edge weights based on observed tiles
            if properties['type'] == 'hole':
                # Make holes very expensive to visit
                for neighbor in self.graph.neighbors(node_id):
                    self.graph[node_id][neighbor]['weight'] = float('inf')
                    self.graph[node_id][neighbor]['known'] = True
            else:
                # Update weights for normal tiles
                for neighbor in self.graph.neighbors(node_id):
                    if self.graph.has_edge(node_id, neighbor):
                        self.graph[node_id][neighbor]['weight'] = 1.0
                        self.graph[node_id][neighbor]['known'] = True
    
    def estimate_distance(self, start, goal):
        """Estimate shortest path distance using A*."""
        try:
            path = nx.astar_path(self.graph, start, goal, weight='weight')
            return sum(self.graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path is found, return a large value
            return float('inf')
    
    def plan(self, current_pos, goals, rewards, remaining_budget):
        """
        Plan a tour through goals to maximize reward collection within budget.
        
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
        
        # Find the final goal (highest reward or designated goal)
        if valid_goals:
            self.final_goal = max(valid_goals, key=lambda x: x[1])[0]
        else:
            self.final_goal = None
            return []
        
        # Initialize tour with just start and final goal
        self.tour = [self.start]
        self.sub_goals = []
        
        # Use greedy insertion algorithm
        unvisited_goals = [g for g, _ in valid_goals if g != self.final_goal]
        
        while unvisited_goals:
            best_insertion = None
            best_marginal_value = float('-inf')
            best_goal = None
            
            # Try inserting each unvisited goal
            for goal in unvisited_goals:
                # Try inserting at each position
                for i in range(1, len(self.tour) + 1):
                    new_tour = self.tour.copy()
                    new_tour.insert(i, goal)
                    
                    # Check if new tour fits within budget
                    tour_length = self._compute_tour_length(new_tour)
                    if tour_length <= self.remaining_budget:
                        # Compute marginal value (reward/distance)
                        reward = rewards[goal]
                        prev_length = self._compute_tour_length(self.tour)
                        marginal_value = reward / (tour_length - prev_length) if tour_length > prev_length else reward
                        
                        if marginal_value > best_marginal_value:
                            best_marginal_value = marginal_value
                            best_insertion = (i, goal)
                            best_goal = goal
            
            # If a valid insertion was found, update the tour
            if best_insertion:
                i, goal = best_insertion
                self.tour.insert(i, goal)
                unvisited_goals.remove(best_goal)
            else:
                # No more goals can be inserted within budget
                break
        
        # Ensure final goal is in the tour if there's enough budget
        if self.final_goal not in self.tour:
            tour_with_final = self.tour + [self.final_goal]
            if self._compute_tour_length(tour_with_final) <= self.remaining_budget:
                self.tour = tour_with_final
        
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