import heapq
import numpy as np
from src.planning.base_planner import BasePlanner

class BidirectionalAStarPlanner(BasePlanner):
    """
    Bidirectional A* search algorithm for path planning.
    
    This algorithm performs A* search from both the start and goal simultaneously,
    stopping when the two searches meet in the middle.
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
        Plan a path from start to goal using Bidirectional A* algorithm.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Forward search (from start to goal)
        open_forward = [(self.heuristic(start, goal), 0, id(start), start)]
        closed_forward = set()  # Set of visited nodes
        g_forward = {start: 0}  # Cost from start
        parent_forward = {start: None}  # Parent pointers
        
        # Backward search (from goal to start)
        open_backward = [(self.heuristic(goal, start), 0, id(goal), goal)]
        closed_backward = set()  # Set of visited nodes
        g_backward = {goal: 0}  # Cost from goal
        parent_backward = {goal: None}  # Parent pointers
        
        # Sets for faster lookups
        open_forward_set = {start}
        open_backward_set = {goal}
        
        # Keep track of the best meeting point found so far
        best_meeting_point = None
        best_meeting_cost = float('inf')
        
        while open_forward and open_backward:
            # Check if there's any meeting point found already
            for pos in open_forward_set.intersection(open_backward_set):
                # Calculate total path cost through this meeting point
                total_cost = g_forward[pos] + g_backward[pos]
                if total_cost < best_meeting_cost:
                    best_meeting_cost = total_cost
                    best_meeting_point = pos
            
            # If we found a meeting point, we can terminate if the next nodes to expand have a higher cost
            if best_meeting_point is not None:
                f_forward, g_f, _, _ = open_forward[0]
                f_backward, g_b, _, _ = open_backward[0]
                if g_f + g_b >= best_meeting_cost:
                    # Reconstruct the path
                    forward_path = []
                    current = best_meeting_point
                    while current != start:
                        forward_path.append(current)
                        current = parent_forward[current]
                    forward_path.append(start)
                    forward_path.reverse()
                    
                    backward_path = []
                    current = best_meeting_point
                    while current != goal:
                        backward_path.append(current)
                        current = parent_backward[current]
                    backward_path.append(goal)
                    
                    # Return the full path (without duplicating the meeting point)
                    return forward_path + backward_path[1:]
            
            # Expand forward search
            if open_forward:
                _, g_f, _, current_forward = heapq.heappop(open_forward)
                open_forward_set.remove(current_forward)
                
                # Add to closed set if not already there
                if current_forward not in closed_forward:
                    closed_forward.add(current_forward)
                    
                    # Explore neighbors
                    for neighbor in self.get_neighbors(current_forward):
                        tentative_g = g_forward[current_forward] + 1
                        
                        if neighbor in closed_forward:
                            continue
                        
                        if neighbor not in g_forward or tentative_g < g_forward[neighbor]:
                            g_forward[neighbor] = tentative_g
                            f_forward = tentative_g + self.heuristic(neighbor, goal)
                            
                            # Update parent pointer
                            parent_forward[neighbor] = current_forward
                            
                            if neighbor not in open_forward_set:
                                heapq.heappush(open_forward, (f_forward, tentative_g, id(neighbor), neighbor))
                                open_forward_set.add(neighbor)
            
            # Expand backward search
            if open_backward:
                _, g_b, _, current_backward = heapq.heappop(open_backward)
                open_backward_set.remove(current_backward)
                
                # Add to closed set if not already there
                if current_backward not in closed_backward:
                    closed_backward.add(current_backward)
                    
                    # Explore neighbors
                    for neighbor in self.get_neighbors(current_backward):
                        tentative_g = g_backward[current_backward] + 1
                        
                        if neighbor in closed_backward:
                            continue
                        
                        if neighbor not in g_backward or tentative_g < g_backward[neighbor]:
                            g_backward[neighbor] = tentative_g
                            f_backward = tentative_g + self.heuristic(neighbor, start)
                            
                            # Update parent pointer
                            parent_backward[neighbor] = current_backward
                            
                            if neighbor not in open_backward_set:
                                heapq.heappush(open_backward, (f_backward, tentative_g, id(neighbor), neighbor))
                                open_backward_set.add(neighbor)
        
        # No path found
        return None 