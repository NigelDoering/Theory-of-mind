import numpy as np
from src.planning.base_planner import BasePlanner

class DFSPlanner(BasePlanner):
    """
    Depth-First Search algorithm for path planning.
    
    DFS explores as far as possible along each branch before backtracking.
    Note: DFS does not guarantee the shortest path.
    """
    
    def __init__(self, world):
        super().__init__(world)
    
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
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Depth-First Search.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Dictionary to store visited nodes and their parents
        visited = {start: None}
        # Stack for DFS
        stack = [start]
        
        while stack:
            current = stack.pop()
            
            # Check if goal is reached
            if current == goal:
                # Reconstruct the path
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                path.reverse()  # Reverse path to get start to goal
                return path
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    stack.append(neighbor)
                    visited[neighbor] = current
        
        # No path found
        return None 