import heapq
import numpy as np
from src.planning.base_planner import BasePlanner

class BestFirstPlanner(BasePlanner):
    """
    Best-First Search algorithm for path planning.
    
    Best-First Search selects the node that appears closest to the goal according to a heuristic function.
    It's a greedy algorithm and does not guarantee the shortest path.
    """
    
    def __init__(self, world):
        super().__init__(world)
    
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
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Best-First Search.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Priority queue for Best-First Search
        open_set = [(self.heuristic(start, goal), id(start), start)]
        # Dictionary to store visited nodes and their parents
        came_from = {start: None}
        # Set of nodes in the open set
        open_set_hash = {start}
        
        while open_set:
            # Get node with lowest heuristic value
            _, _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # Check if goal is reached
            if current == goal:
                # Reconstruct the path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()  # Reverse path to get start to goal
                return path
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    # Only care about the heuristic to goal in Best-First Search
                    h = self.heuristic(neighbor, goal)
                    # Add to open set if not already there
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (h, id(neighbor), neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return None 