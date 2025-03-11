import numpy as np
import random
from src.planning.base_planner import BasePlanner

class RRTNode:
    """Node class for RRT algorithm."""
    def __init__(self, position):
        self.position = position  # (x, y) tuple
        self.parent = None
        self.cost = 0.0  # Cost from start to this node

class RRTPlanner(BasePlanner):
    """
    Rapidly-exploring Random Tree (RRT) algorithm for path planning.
    
    RRT is a sampling-based algorithm that incrementally builds a tree
    by randomly sampling points in the configuration space and connecting
    them to the nearest node in the tree.
    
    Key features:
    - Efficient exploration of the configuration space
    - Probabilistically complete
    - Works well in high-dimensional spaces
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.nodes = []
        self.step_size = 1.0
        self.max_iterations = 1000
        self.goal_sample_rate = 0.1
        self.goal_threshold = 1.0
        
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def nearest_node(self, position):
        """Find the nearest node in the tree to the given position."""
        return min(self.nodes, key=lambda node: self.distance(node.position, position))
    
    def steer(self, from_pos, to_pos):
        """
        Steer from from_pos towards to_pos with maximum step size.
        Returns a new position that is at most step_size away from from_pos.
        """
        dist = self.distance(from_pos, to_pos)
        if dist <= self.step_size:
            return to_pos
        
        # Calculate the direction vector
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Normalize and scale by step_size
        theta = np.arctan2(dy, dx)
        new_x = from_pos[0] + self.step_size * np.cos(theta)
        new_y = from_pos[1] + self.step_size * np.sin(theta)
        
        return (int(round(new_x)), int(round(new_y)))
    
    def is_collision_free(self, pos1, pos2):
        """
        Check if the path between pos1 and pos2 is collision-free.
        Uses line interpolation to check for obstacles along the path.
        """
        dist = self.distance(pos1, pos2)
        if dist < 1e-6:
            return self.is_valid_point(pos1)
        
        # Number of points to check along the line
        n_points = max(2, int(np.ceil(dist / 0.5)))
        
        for i in range(n_points + 1):
            t = i / n_points
            x = int(round(pos1[0] * (1 - t) + pos2[0] * t))
            y = int(round(pos1[1] * (1 - t) + pos2[1] * t))
            
            if not self.is_valid_point((x, y)):
                return False
        
        return True
    
    def random_position(self):
        """Generate a random position within the world bounds."""
        if random.random() < self.goal_sample_rate:
            # Sample the goal with probability goal_sample_rate
            return self.goal
        else:
            # Sample a random position
            x = random.randint(0, self.world.width - 1)
            y = random.randint(0, self.world.height - 1)
            return (x, y)
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using RRT.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            step_size: Optional step size for extending the tree.
            max_iterations: Optional maximum number of iterations.
            goal_sample_rate: Optional probability of sampling the goal.
            goal_threshold: Optional distance threshold to consider goal reached.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.step_size = kwargs.get('step_size', self.step_size)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        self.goal_threshold = kwargs.get('goal_threshold', self.goal_threshold)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        self.nodes.append(start_node)
        
        # Main loop
        for i in range(self.max_iterations):
            # Sample random position
            random_pos = self.random_position()
            
            # Find nearest node
            nearest = self.nearest_node(random_pos)
            
            # Steer towards random position
            new_pos = self.steer(nearest.position, random_pos)
            
            # Check if new position is collision-free
            if self.is_collision_free(nearest.position, new_pos):
                # Create new node
                new_node = RRTNode(new_pos)
                new_node.parent = nearest
                new_node.cost = nearest.cost + self.distance(nearest.position, new_pos)
                self.nodes.append(new_node)
                
                # Check if goal is reached
                if self.distance(new_pos, goal) <= self.goal_threshold:
                    # Construct path
                    path = [goal]
                    current = new_node
                    while current.parent is not None:
                        path.append(current.position)
                        current = current.parent
                    path.append(start)
                    path.reverse()
                    return path
        
        # No path found
        return None
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of RRT that can provide step-by-step visualization.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            callback: Function to call after each iteration with current tree and path.
            **kwargs: Additional parameters.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.step_size = kwargs.get('step_size', self.step_size)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        self.goal_threshold = kwargs.get('goal_threshold', self.goal_threshold)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        self.nodes.append(start_node)
        
        # Call callback with initial state
        if callback:
            callback(self.nodes, None, 0)
        
        # Main loop
        for i in range(self.max_iterations):
            # Sample random position
            random_pos = self.random_position()
            
            # Find nearest node
            nearest = self.nearest_node(random_pos)
            
            # Steer towards random position
            new_pos = self.steer(nearest.position, random_pos)
            
            # Check if new position is collision-free
            if self.is_collision_free(nearest.position, new_pos):
                # Create new node
                new_node = RRTNode(new_pos)
                new_node.parent = nearest
                new_node.cost = nearest.cost + self.distance(nearest.position, new_pos)
                self.nodes.append(new_node)
                
                # Check if goal is reached
                current_path = None
                if self.distance(new_pos, goal) <= self.goal_threshold:
                    # Construct path
                    current_path = [goal]
                    current = new_node
                    while current.parent is not None:
                        current_path.append(current.position)
                        current = current.parent
                    current_path.append(start)
                    current_path.reverse()
                
                # Call callback with current state
                if callback:
                    callback(self.nodes, current_path, i + 1)
                
                # Return path if goal is reached
                if current_path:
                    return current_path
        
        # No path found
        if callback:
            callback(self.nodes, None, self.max_iterations)
        
        return None 