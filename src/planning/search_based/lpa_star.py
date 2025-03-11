import heapq
import numpy as np
from src.planning.base_planner import BasePlanner

class LPAStarPlanner(BasePlanner):
    """
    Lifelong Planning A* (LPA*) algorithm for path planning.
    
    LPA* is an incremental version of A* that efficiently replans when the graph changes.
    It maintains estimates of the start distance for each node and only updates nodes
    affected by graph changes.
    
    Key features:
    - Efficient replanning in dynamic environments
    - Maintains consistency between g-values and rhs-values
    - Only updates nodes affected by graph changes
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.g_values = {}  # True start distances
        self.rhs_values = {}  # One-step lookahead start distances
        self.open_set = []  # Priority queue
        self.open_set_hash = set()  # Set for faster lookups
        self.start = None
        self.goal = None
        self.changed_edges = []  # List of changed edges
        self.obstacles_added = []  # List of added obstacles
        self.obstacles_removed = []  # List of removed obstacles
    
    def heuristic(self, a, b):
        """Calculate Manhattan distance between points a and b."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, point):
        """Get valid neighboring points (4-connected grid)."""
        x, y = point
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_point((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors
    
    def calculate_key(self, s):
        """Calculate the key for a state in the priority queue."""
        if s in self.g_values:
            g = self.g_values[s]
        else:
            g = float('inf')
            
        if s in self.rhs_values:
            rhs = self.rhs_values[s]
        else:
            rhs = float('inf')
            
        # Primary key: min(g, rhs) + h
        # Secondary key: min(g, rhs)
        return (min(g, rhs) + self.heuristic(s, self.goal), min(g, rhs))
    
    def update_vertex(self, u):
        """Update the vertex u and its position in the priority queue."""
        if u != self.start:
            # Update rhs value
            min_rhs = float('inf')
            for s in self.get_neighbors(u):
                if s in self.g_values:
                    g_s = self.g_values[s]
                else:
                    g_s = float('inf')
                min_rhs = min(min_rhs, g_s + 1)  # Assuming uniform cost of 1
            self.rhs_values[u] = min_rhs
        
        # Update position in priority queue
        if u in self.open_set_hash:
            self.open_set_hash.remove(u)
            # Remove from priority queue (will be re-added if needed)
            self.open_set = [(k, s) for k, s in self.open_set if s != u]
            heapq.heapify(self.open_set)
        
        if u in self.g_values and self.g_values[u] != self.rhs_values.get(u, float('inf')):
            heapq.heappush(self.open_set, (self.calculate_key(u), u))
            self.open_set_hash.add(u)
    
    def compute_shortest_path(self):
        """Compute the shortest path using LPA*."""
        while (self.open_set and 
               (self.calculate_key(self.open_set[0][1]) < self.calculate_key(self.goal) or 
                self.rhs_values.get(self.goal, float('inf')) != self.g_values.get(self.goal, float('inf')))):
            _, u = heapq.heappop(self.open_set)
            self.open_set_hash.remove(u)
            
            if self.g_values.get(u, float('inf')) > self.rhs_values.get(u, float('inf')):
                # Locally overconsistent
                self.g_values[u] = self.rhs_values[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                # Locally underconsistent
                self.g_values[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
    
    def initialize(self, start, goal):
        """Initialize the LPA* algorithm."""
        self.start = start
        self.goal = goal
        self.g_values = {}
        self.rhs_values = {}
        self.g_values[start] = float('inf')
        self.rhs_values[start] = 0
        self.open_set = []
        self.open_set_hash = set()
        heapq.heappush(self.open_set, (self.calculate_key(start), start))
        self.open_set_hash.add(start)
    
    def update_graph(self):
        """Update the graph based on added/removed obstacles."""
        # Process added obstacles
        for pos in self.obstacles_added:
            # Update affected vertices
            for neighbor in self.get_neighbors(pos):
                self.update_vertex(neighbor)
        
        # Process removed obstacles
        for pos in self.obstacles_removed:
            # Update affected vertices
            for neighbor in self.get_neighbors(pos):
                self.update_vertex(neighbor)
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Lifelong Planning A* (LPA*).
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            obstacles_added: Optional list of newly added obstacles.
            obstacles_removed: Optional list of newly removed obstacles.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Get changes if provided
        self.obstacles_added = kwargs.get('obstacles_added', [])
        self.obstacles_removed = kwargs.get('obstacles_removed', [])
        
        # Initialize if this is a new search or if start/goal changed
        if self.start != start or self.goal != goal:
            self.initialize(start, goal)
        
        # Update graph based on changes
        self.update_graph()
        
        # Compute shortest path
        self.compute_shortest_path()
        
        # Check if goal is reachable
        if self.g_values.get(self.goal, float('inf')) == float('inf'):
            return None
        
        # Reconstruct path
        path = [self.goal]
        current = self.goal
        
        while current != self.start:
            # Find the neighbor with minimum g-value + cost
            min_g = float('inf')
            next_node = None
            
            for neighbor in self.get_neighbors(current):
                if neighbor in self.g_values:
                    g_val = self.g_values[neighbor] + 1  # Cost is 1
                    if g_val < min_g:
                        min_g = g_val
                        next_node = neighbor
            
            if next_node is None:
                # No valid neighbor found
                return None
            
            path.append(next_node)
            current = next_node
        
        path.reverse()  # Reverse to get path from start to goal
        return path
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of LPA* that can provide step-by-step visualization.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            callback: Function to call after each step with current state and path.
            **kwargs: Additional parameters.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Get changes if provided
        self.obstacles_added = kwargs.get('obstacles_added', [])
        self.obstacles_removed = kwargs.get('obstacles_removed', [])
        
        # Initialize if this is a new search or if start/goal changed
        if self.start != start or self.goal != goal:
            self.initialize(start, goal)
            
            # Call callback with initial state
            if callback:
                callback(start, [], self.g_values, self.rhs_values, self.open_set)
        
        # Update graph based on changes
        self.update_graph()
        
        # Compute shortest path with visualization
        while (self.open_set and 
               (self.calculate_key(self.open_set[0][1]) < self.calculate_key(self.goal) or 
                self.rhs_values.get(self.goal, float('inf')) != self.g_values.get(self.goal, float('inf')))):
            _, u = heapq.heappop(self.open_set)
            self.open_set_hash.remove(u)
            
            if self.g_values.get(u, float('inf')) > self.rhs_values.get(u, float('inf')):
                # Locally overconsistent
                self.g_values[u] = self.rhs_values[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                # Locally underconsistent
                self.g_values[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            
            # Call callback with current state
            if callback:
                # Reconstruct current best path
                current_path = self.extract_current_path()
                callback(u, current_path, self.g_values, self.rhs_values, self.open_set)
        
        # Final path
        path = self.extract_current_path()
        
        # Call callback with final state
        if callback:
            callback(goal, path, self.g_values, self.rhs_values, self.open_set)
        
        return path
    
    def extract_current_path(self):
        """Extract the current best path based on g-values."""
        if self.g_values.get(self.goal, float('inf')) == float('inf'):
            return []
        
        path = [self.goal]
        current = self.goal
        
        while current != self.start:
            # Find the neighbor with minimum g-value + cost
            min_g = float('inf')
            next_node = None
            
            for neighbor in self.get_neighbors(current):
                if neighbor in self.g_values:
                    g_val = self.g_values[neighbor] + 1  # Cost is 1
                    if g_val < min_g:
                        min_g = g_val
                        next_node = neighbor
            
            if next_node is None:
                # No valid neighbor found
                return []
            
            path.append(next_node)
            current = next_node
        
        path.reverse()  # Reverse to get path from start to goal
        return path 