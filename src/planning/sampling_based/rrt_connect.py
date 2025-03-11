import numpy as np
import random
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode

class RRTConnectPlanner(BasePlanner):
    """
    RRT-Connect algorithm for path planning.
    
    RRT-Connect is a bidirectional variant of RRT that grows two trees:
    one from the start and one from the goal. The algorithm attempts to
    connect the trees whenever a new node is added.
    
    Key features:
    - Bidirectional search for faster convergence
    - Efficient exploration of the configuration space
    - Typically finds solutions faster than standard RRT
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.start_tree = []  # Tree growing from start
        self.goal_tree = []   # Tree growing from goal
        self.step_size = 1.0
        self.max_iterations = 1000
        self.connect_threshold = 1.0
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def nearest_node(self, tree, position):
        """Find the nearest node in the tree to the given position."""
        return min(tree, key=lambda node: self.distance(node.position, position))
    
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
        x = random.randint(0, self.world.width - 1)
        y = random.randint(0, self.world.height - 1)
        return (x, y)
    
    def extend(self, tree, target_pos):
        """
        Extend the tree towards the target position.
        Returns:
            0: Trapped - could not extend
            1: Advanced - extended but did not reach target
            2: Reached - extended and reached target
        """
        nearest = self.nearest_node(tree, target_pos)
        new_pos = self.steer(nearest.position, target_pos)
        
        if self.is_collision_free(nearest.position, new_pos):
            new_node = RRTNode(new_pos)
            new_node.parent = nearest
            new_node.cost = nearest.cost + self.distance(nearest.position, new_pos)
            tree.append(new_node)
            
            if self.distance(new_pos, target_pos) < 1e-6:
                return 2, new_node  # Reached
            else:
                return 1, new_node  # Advanced
        
        return 0, None  # Trapped
    
    def connect(self, tree, target_pos):
        """
        Connect the tree to the target position.
        Repeatedly extends the tree until it reaches the target or gets trapped.
        """
        result = 1  # Advanced
        node = None
        
        while result == 1:
            result, node = self.extend(tree, target_pos)
        
        return result, node
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using RRT-Connect.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            step_size: Optional step size for extending the trees.
            max_iterations: Optional maximum number of iterations.
            connect_threshold: Optional distance threshold for connecting trees.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.step_size = kwargs.get('step_size', self.step_size)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.connect_threshold = kwargs.get('connect_threshold', self.connect_threshold)
        
        # Initialize trees
        self.start_tree = []
        self.goal_tree = []
        
        start_node = RRTNode(start)
        goal_node = RRTNode(goal)
        
        self.start_tree.append(start_node)
        self.goal_tree.append(goal_node)
        
        # Main loop
        for i in range(self.max_iterations):
            # Sample random position
            random_pos = self.random_position()
            
            # Extend start tree towards random position
            extend_result, new_node = self.extend(self.start_tree, random_pos)
            
            if extend_result != 0:  # If not trapped
                # Try to connect goal tree to the new node
                connect_result, connect_node = self.connect(self.goal_tree, new_node.position)
                
                if connect_result == 2:  # If reached
                    # Construct path
                    path = []
                    
                    # Add path from start to new_node
                    current = new_node
                    while current is not None:
                        path.append(current.position)
                        current = current.parent
                    path.reverse()
                    
                    # Add path from connect_node to goal
                    current = connect_node
                    while current is not None:
                        if current.position != new_node.position:  # Avoid duplicating the connection point
                            path.append(current.position)
                        current = current.parent
                    
                    return path
            
            # Swap trees for the next iteration
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
        
        # No path found
        return None
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of RRT-Connect that can provide step-by-step visualization.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            callback: Function to call after each iteration with current trees and path.
            **kwargs: Additional parameters.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.step_size = kwargs.get('step_size', self.step_size)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.connect_threshold = kwargs.get('connect_threshold', self.connect_threshold)
        
        # Initialize trees
        self.start_tree = []
        self.goal_tree = []
        
        start_node = RRTNode(start)
        goal_node = RRTNode(goal)
        
        self.start_tree.append(start_node)
        self.goal_tree.append(goal_node)
        
        # Call callback with initial state
        if callback:
            callback(self.start_tree, self.goal_tree, None, 0)
        
        # Main loop
        for i in range(self.max_iterations):
            # Sample random position
            random_pos = self.random_position()
            
            # Extend start tree towards random position
            extend_result, new_node = self.extend(self.start_tree, random_pos)
            
            # Call callback after extension
            current_path = None
            if callback and extend_result != 0:
                callback(self.start_tree, self.goal_tree, current_path, i + 1)
            
            if extend_result != 0:  # If not trapped
                # Try to connect goal tree to the new node
                connect_result, connect_node = self.connect(self.goal_tree, new_node.position)
                
                # Call callback after connection attempt
                if callback:
                    callback(self.start_tree, self.goal_tree, current_path, i + 1)
                
                if connect_result == 2:  # If reached
                    # Construct path
                    path = []
                    
                    # Add path from start to new_node
                    current = new_node
                    while current is not None:
                        path.append(current.position)
                        current = current.parent
                    path.reverse()
                    
                    # Add path from connect_node to goal
                    current = connect_node
                    while current is not None:
                        if current.position != new_node.position:  # Avoid duplicating the connection point
                            path.append(current.position)
                        current = current.parent
                    
                    # Call callback with final path
                    if callback:
                        callback(self.start_tree, self.goal_tree, path, i + 1)
                    
                    return path
            
            # Swap trees for the next iteration
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
        
        # No path found
        if callback:
            callback(self.start_tree, self.goal_tree, None, self.max_iterations)
        
        return None 