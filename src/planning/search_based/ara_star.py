import heapq
import numpy as np
from src.planning.base_planner import BasePlanner

class ARAStarPlanner(BasePlanner):
    """
    Anytime Repairing A* (ARA*) algorithm for path planning.
    
    ARA* is an anytime search algorithm that quickly finds a suboptimal solution
    and then iteratively improves it as time allows by decreasing the inflation factor.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.initial_epsilon = 3.0  # Initial inflation factor
        self.final_epsilon = 1.0    # Final inflation factor (1.0 = optimal A*)
        self.delta_epsilon = 0.5    # Amount to decrease epsilon by in each iteration
        self.max_iterations = 10    # Maximum number of iterations
    
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
    
    def ara_star_search(self, start, goal, epsilon):
        """
        Perform inflated A* search with the given epsilon.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            epsilon: Inflation factor for the heuristic.
            
        Returns:
            Tuple containing the path and whether the search was successful.
        """
        # Priority queue for A*
        open_set = [(epsilon * self.heuristic(start, goal), 0, id(start), start)]
        # Dictionary to store visited nodes and their parents
        came_from = {start: None}
        # Dictionary to store cost from start to each node
        g_score = {start: 0}
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
                return path, True
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                # Assuming uniform cost of 1 for each step
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + epsilon * self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score, tentative_g_score, id(neighbor), neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return None, False
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Anytime Repairing A* (ARA*).
        
        This method implements the required abstract method from BasePlanner.
        It runs ARA* with decreasing epsilon values to improve the solution over time.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            **kwargs: Additional parameters that can override default settings:
                - initial_epsilon: Starting inflation factor
                - final_epsilon: Final inflation factor
                - delta_epsilon: Amount to decrease epsilon by in each iteration
                - max_iterations: Maximum number of iterations
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Override default parameters if provided
        initial_epsilon = kwargs.get('initial_epsilon', self.initial_epsilon)
        final_epsilon = kwargs.get('final_epsilon', self.final_epsilon)
        delta_epsilon = kwargs.get('delta_epsilon', self.delta_epsilon)
        max_iterations = kwargs.get('max_iterations', self.max_iterations)
        
        # Initialize best path
        best_path = None
        
        # Start with initial epsilon and decrease until final_epsilon
        epsilon = initial_epsilon
        iteration = 0
        
        while epsilon >= final_epsilon and iteration < max_iterations:
            # Run ARA* search with current epsilon
            path, success = self.ara_star_search(start, goal, epsilon)
            
            if success:
                # Update best path
                best_path = path
                
                # If we've reached the optimal solution (epsilon = 1.0), we can stop
                if epsilon <= 1.0 + 1e-6:  # Add small epsilon for floating point comparison
                    break
            else:
                # If search failed, no need to continue
                break
            
            # Decrease epsilon for next iteration
            epsilon = max(epsilon - delta_epsilon, final_epsilon)
            iteration += 1
        
        return best_path 