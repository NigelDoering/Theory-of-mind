import heapq
import numpy as np
from src.planning.base_planner import BasePlanner

class DijkstraPlanner(BasePlanner):
    """
    Dijkstra's algorithm for path planning.
    
    Dijkstra's algorithm finds the shortest path from the start node to all nodes in the graph,
    including the goal node. It guarantees the shortest path but explores in all directions.
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
        Plan a path from start to goal using Dijkstra's algorithm.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Priority queue for Dijkstra's algorithm
        open_set = [(0, id(start), start)]  # (cost, unique_id, position)
        # Dictionary to store visited nodes and their parents
        came_from = {start: None}
        # Dictionary to store cost from start to each node
        cost_so_far = {start: 0}
        
        while open_set:
            # Get node with lowest cost
            current_cost, _, current = heapq.heappop(open_set)
            
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
                # Assuming uniform cost of 1 for each step
                new_cost = cost_so_far[current] + 1
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (new_cost, id(neighbor), neighbor))
        
        # No path found
        return None 