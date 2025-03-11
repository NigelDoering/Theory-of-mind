import heapq
import numpy as np
from src.planning.base_planner import BasePlanner

class DStarLitePlanner(BasePlanner):
    """
    D* Lite algorithm for path planning.
    
    D* Lite is an incremental search algorithm that efficiently replans paths
    when the graph changes. It's particularly useful for dynamic environments
    where obstacles may appear or disappear.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.g_values = {}  # Cost from node to goal
        self.rhs_values = {}  # One-step lookahead values
        self.open_set = []  # Priority queue
        self.open_set_hash = set()  # Set for faster lookups
        self.start = None
        self.goal = None
        self.k_m = 0  # Accumulated heuristic inflation
        self.changes = []  # List of changed edges
    
    def heuristic(self, a, b):
        """Calculate Manhattan distance between points a and b."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, point):
        """Get valid neighboring points (4-connected grid)."""
        x, y = point
        # Four-connected grid: up, right, down, left
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
            
        # Primary key: min(g, rhs) + h(start, s) + k_m
        # Secondary key: min(g, rhs)
        return (min(g, rhs) + self.heuristic(self.start, s) + self.k_m, min(g, rhs))
    
    def update_vertex(self, u):
        """Update the vertex u and its position in the priority queue."""
        if u != self.goal:
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
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using D* Lite.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            changes: Optional list of changed edges.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Get changes if provided
        self.changes = kwargs.get('changes', [])
        
        # Initialize if this is a new search or if start/goal changed
        if self.start != start or self.goal != goal:
            self.start = start
            self.goal = goal
            self.g_values = {s: float('inf') for s in self.g_values}
            self.rhs_values = {s: float('inf') for s in self.rhs_values}
            self.rhs_values[start] = 0
            self.open_set = []
            self.open_set_hash = set()
            heapq.heappush(self.open_set, (self.calculate_key(start), start))
            self.open_set_hash.add(start)
        
        # Process changes
        for u in self.changes:
            self.update_vertex(u)
            for s in self.get_neighbors(u):
                self.update_vertex(s)
        
        # Compute shortest path
        if self.g_values.get(self.goal, float('inf')) == float('inf'):
            return None
        
        # Reconstruct path
        path = [self.goal]
        current = self.goal
        
        while current != self.start:
            # Find the neighbor with minimum g-value
            min_g = float('inf')
            next_node = None
            
            for neighbor in self.get_neighbors(current):
                if neighbor in self.g_values:
                    g_val = self.g_values[neighbor]
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