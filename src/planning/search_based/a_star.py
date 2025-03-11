import heapq
import numpy as np
from src.planning.base_planner import BasePlanner
from src.planning.node import Node

class AStarPlanner(BasePlanner):
    """
    A* path planning algorithm.
    
    A* combines the advantages of Dijkstra's algorithm (guarantees shortest path)
    and Best-First Search (uses heuristic to guide search toward goal).
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
        Plan a path from start to goal using A* algorithm.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Priority queue for A*
        open_set = [(0 + self.heuristic(start, goal), 0, id(start), start)]  # (f_score, g_score, unique_id, position)
        # Dictionary to store visited nodes and their parents
        came_from = {start: None}
        # Dictionary to store cost from start to each node
        g_score = {start: 0}
        # Dictionary to store estimated total cost from start to goal through each node
        f_score = {start: self.heuristic(start, goal)}
        # Set of positions in the open set (for faster lookup)
        open_set_hash = {start}
        
        while open_set:
            # Get node with lowest f_score
            _, _, _, current = heapq.heappop(open_set)
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
                # Assuming uniform cost of 1 for each step
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], id(neighbor), neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return None 