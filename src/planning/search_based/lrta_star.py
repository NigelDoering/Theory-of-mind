import numpy as np
from collections import defaultdict
from src.planning.base_planner import BasePlanner

class LRTAStarPlanner(BasePlanner):
    """
    Learning Real-Time A* (LRTA*) algorithm for path planning.
    
    LRTA* is a real-time heuristic search algorithm that interleaves planning and execution.
    It updates heuristic values during search to improve future searches.
    
    Key features:
    - Real-time planning with limited lookahead
    - Learning heuristic values through experience
    - Guaranteed to find a path in finite state spaces
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.heuristic_values = defaultdict(float)  # Learned heuristic values
        self.lookahead = 3  # Default lookahead depth
        self.max_iterations = 1000  # Maximum iterations to prevent infinite loops
        self.visited_states = set()  # Track visited states to detect loops
    
    def manhattan_distance(self, a, b):
        """Calculate Manhattan distance between points a and b."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_heuristic(self, state, goal):
        """
        Get heuristic value for a state.
        Returns learned value if available, otherwise Manhattan distance.
        """
        if state in self.heuristic_values:
            return self.heuristic_values[state]
        return self.manhattan_distance(state, goal)
    
    def get_neighbors(self, point):
        """Get valid neighboring points (4-connected grid)."""
        x, y = point
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_point((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors
    
    def get_min_f_value(self, state, goal):
        """
        Calculate the minimum f-value among all neighbors of a state.
        f-value = cost to neighbor + heuristic of neighbor
        """
        neighbors = self.get_neighbors(state)
        min_f = float('inf')
        
        for neighbor in neighbors:
            # Cost to neighbor is 1 (uniform grid)
            cost = 1
            h_value = self.get_heuristic(neighbor, goal)
            f_value = cost + h_value
            min_f = min(min_f, f_value)
        
        return min_f
    
    def update_heuristic(self, state, goal):
        """
        Update the heuristic value of the current state based on its neighbors.
        This is the learning component of LRTA*.
        """
        if state == goal:
            self.heuristic_values[state] = 0
            return
        
        min_f = self.get_min_f_value(state, goal)
        self.heuristic_values[state] = min_f
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Learning Real-Time A* (LRTA*).
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            lookahead: Optional parameter to set the lookahead depth.
            max_iterations: Optional parameter to set the maximum iterations.
            reset_learning: Optional parameter to reset learned heuristics.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.lookahead = kwargs.get('lookahead', self.lookahead)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        
        # Reset learning if requested
        if kwargs.get('reset_learning', False):
            self.heuristic_values = defaultdict(float)
        
        # Initialize path and current state
        path = [start]
        current = start
        self.visited_states = set([current])
        iterations = 0
        
        # Main loop
        while current != goal and iterations < self.max_iterations:
            iterations += 1
            
            # Find the best next state within lookahead
            best_next = self.find_best_next_state(current, goal)
            
            if best_next is None:
                # No valid next state found
                return None
            
            # Update heuristic value of current state
            self.update_heuristic(current, goal)
            
            # Move to the best next state
            path.append(best_next)
            current = best_next
            
            # Check for loops
            if current in self.visited_states:
                # We're in a loop, try to break it by increasing heuristic
                self.heuristic_values[current] += 1
            
            self.visited_states.add(current)
        
        # Check if we reached the goal
        if current != goal:
            return None
        
        return path
    
    def find_best_next_state(self, state, goal):
        """
        Find the best next state to move to from the current state.
        Performs a limited lookahead search.
        """
        if state == goal:
            return goal
        
        neighbors = self.get_neighbors(state)
        if not neighbors:
            return None
        
        best_neighbor = None
        best_f = float('inf')
        
        for neighbor in neighbors:
            # Cost to neighbor is 1 (uniform grid)
            cost = 1
            h_value = self.get_heuristic(neighbor, goal)
            f_value = cost + h_value
            
            if f_value < best_f:
                best_f = f_value
                best_neighbor = neighbor
        
        return best_neighbor
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of LRTA* that can provide step-by-step visualization.
        
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
        
        # Update parameters if provided
        self.lookahead = kwargs.get('lookahead', self.lookahead)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        
        # Reset learning if requested
        if kwargs.get('reset_learning', False):
            self.heuristic_values = defaultdict(float)
        
        # Initialize path and current state
        path = [start]
        current = start
        self.visited_states = set([current])
        iterations = 0
        
        # Call callback with initial state
        if callback:
            callback(current, path, self.heuristic_values, iterations)
        
        # Main loop
        while current != goal and iterations < self.max_iterations:
            iterations += 1
            
            # Find the best next state within lookahead
            best_next = self.find_best_next_state(current, goal)
            
            if best_next is None:
                # No valid next state found
                return None
            
            # Update heuristic value of current state
            self.update_heuristic(current, goal)
            
            # Move to the best next state
            path.append(best_next)
            current = best_next
            
            # Check for loops
            if current in self.visited_states:
                # We're in a loop, try to break it by increasing heuristic
                self.heuristic_values[current] += 1
            
            self.visited_states.add(current)
            
            # Call callback with updated state
            if callback:
                callback(current, path, self.heuristic_values, iterations)
        
        # Check if we reached the goal
        if current != goal:
            return None
        
        return path 