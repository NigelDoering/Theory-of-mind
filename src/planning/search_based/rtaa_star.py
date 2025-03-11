import heapq
import numpy as np
from src.planning.base_planner import BasePlanner

class RTAAStarPlanner(BasePlanner):
    """
    Real-time Adaptive A* (RTAA*) algorithm for path planning.
    
    RTAA* is a real-time heuristic search algorithm that interleaves planning and execution.
    It updates heuristic values of states in the closed list to make them more accurate.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.heuristic_values = {}  # Dictionary to store updated heuristic values
        self.lookahead = 10  # Number of expansions before returning a partial path
    
    def manhattan_distance(self, a, b):
        """Calculate Manhattan distance between points a and b."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def heuristic(self, pos, goal):
        """
        Get heuristic value for a position.
        Uses stored value if available, otherwise calculates Manhattan distance.
        """
        if pos in self.heuristic_values:
            return self.heuristic_values[pos]
        return self.manhattan_distance(pos, goal)
    
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
        Plan a path from start to goal using Real-time Adaptive A* (RTAA*).
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            lookahead: Optional parameter to set the lookahead value.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Set lookahead if provided
        self.lookahead = kwargs.get('lookahead', self.lookahead)
        
        # Full path to be returned
        full_path = [start]
        current = start
        
        # Continue until goal is reached or no path is found
        while current != goal:
            # Run A* search with lookahead limit
            path = self.rtaa_search(current, goal)
            
            if not path or len(path) <= 1:
                # No path found or no progress made
                return None
            
            # Move to the next position in the path
            next_pos = path[1]  # Skip the current position
            full_path.append(next_pos)
            current = next_pos
            
            # If we've reached the goal, return the full path
            if current == goal:
                return full_path
        
        return full_path
    
    def rtaa_search(self, start, goal):
        """
        Perform a single RTAA* search iteration with lookahead limit.
        
        Parameters:
            start: Current position.
            goal: Goal position.
            
        Returns:
            Partial path to follow.
        """
        # Priority queue for A*
        open_set = [(self.heuristic(start, goal), 0, id(start), start)]
        # Dictionary to store visited nodes and their parents
        came_from = {start: None}
        # Dictionary to store cost from start to each node
        g_score = {start: 0}
        # Set of positions in the open set (for faster lookup)
        open_set_hash = {start}
        # Set of closed nodes
        closed_set = set()
        
        expansions = 0
        
        while open_set and expansions < self.lookahead:
            # Get node with lowest f_score
            _, g, _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # Add to closed set
            closed_set.add(current)
            expansions += 1
            
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
                
                if neighbor in closed_set:
                    continue
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score, tentative_g_score, id(neighbor), neighbor))
                        open_set_hash.add(neighbor)
        
        # Update heuristics for nodes in the closed set
        if open_set:
            # Find minimum f-value in open set
            min_f = float('inf')
            for f, _, _, _ in open_set:
                min_f = min(min_f, f)
            
            # Update heuristics for nodes in closed set
            for node in closed_set:
                if node != start:  # Don't update the start node
                    self.heuristic_values[node] = min_f - g_score[node]
        
        # Reconstruct the best partial path
        if not open_set:
            return None
        
        # Get the best node from the open set
        best_f, best_g, _, best_node = min(open_set)
        
        # Reconstruct path to this node
        path = []
        current = best_node
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()  # Reverse path to get start to best_node
        
        return path 