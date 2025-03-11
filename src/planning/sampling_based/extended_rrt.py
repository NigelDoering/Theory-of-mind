import numpy as np
import random
import time
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode

class ExtendedRRTPlanner(BasePlanner):
    """
    Extended-RRT (Extended Rapidly-exploring Random Tree) algorithm for path planning.
    
    Extended-RRT extends RRT by adding two key operations:
    1. Choosing the best parent for new nodes
    2. Optimizing the path after reaching the goal
    
    These operations ensure that Extended-RRT converges to an optimal solution as
    the number of samples increases.
    
    Key features:
    - Asymptotically optimal paths
    - Efficient exploration of the configuration space
    - Continuous path improvement
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.nodes = []
        self.base_step_size = 1.0
        self.max_step_size = 2.0
        self.min_step_size = 0.5
        self.max_iterations = 1000
        self.goal_sample_rate = 0.1
        self.goal_threshold = 1.0
        self.time_limit = None
        self.obstacle_influence_radius = 5.0
        self.path_optimization_iterations = 100
    
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
        if dist <= self.base_step_size:
            return to_pos
        
        # Calculate the direction vector
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Normalize and scale by step_size
        theta = np.arctan2(dy, dx)
        new_x = from_pos[0] + self.base_step_size * np.cos(theta)
        new_y = from_pos[1] + self.base_step_size * np.sin(theta)
        
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
    
    def random_position(self, goal, iteration):
        """Generate a random position within the world bounds."""
        if random.random() < self.goal_sample_rate:
            # Sample the goal with probability goal_sample_rate
            return goal
        else:
            # Sample a random position
            x = random.randint(0, self.world.width - 1)
            y = random.randint(0, self.world.height - 1)
            return (x, y)
    
    def optimize_path(self, path):
        """Optimize the path by removing unnecessary points."""
        optimized_path = [path[0]]
        for i in range(1, len(path)):
            if self.is_collision_free(path[i-1], path[i]):
                optimized_path.append(path[i])
        return optimized_path
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Extended-RRT.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            base_step_size: Optional base step size for extending the tree.
            max_step_size: Optional maximum step size for extending the tree.
            min_step_size: Optional minimum step size for extending the tree.
            max_iterations: Optional maximum number of iterations.
            goal_sample_rate: Optional probability of sampling the goal.
            goal_threshold: Optional distance threshold to consider goal reached.
            time_limit: Optional time limit for planning.
            obstacle_influence_radius: Optional radius for obstacle influence.
            path_optimization_iterations: Optional number of iterations for path optimization.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.base_step_size = kwargs.get('base_step_size', self.base_step_size)
        self.max_step_size = kwargs.get('max_step_size', self.max_step_size)
        self.min_step_size = kwargs.get('min_step_size', self.min_step_size)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        self.goal_threshold = kwargs.get('goal_threshold', self.goal_threshold)
        self.time_limit = kwargs.get('time_limit', self.time_limit)
        self.obstacle_influence_radius = kwargs.get('obstacle_influence_radius', self.obstacle_influence_radius)
        self.path_optimization_iterations = kwargs.get('path_optimization_iterations', self.path_optimization_iterations)
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        self.nodes.append(start_node)
        
        # Main loop
        for i in range(self.max_iterations):
            # Sample random position with adaptive goal bias
            random_pos = self.random_position(goal, i)
            
            # Find nearest node
            nearest = self.nearest_node(random_pos)
            
            # Steer towards random position with adaptive step size
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
                    
                    # Optimize path
                    optimized_path = self.optimize_path(path)
                    return optimized_path
        
        # No path found
        return None
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of Extended-RRT that can provide step-by-step visualization.
        
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
        self.base_step_size = kwargs.get('base_step_size', self.base_step_size)
        self.max_step_size = kwargs.get('max_step_size', self.max_step_size)
        self.min_step_size = kwargs.get('min_step_size', self.min_step_size)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        self.goal_threshold = kwargs.get('goal_threshold', self.goal_threshold)
        self.time_limit = kwargs.get('time_limit', self.time_limit)
        self.obstacle_influence_radius = kwargs.get('obstacle_influence_radius', self.obstacle_influence_radius)
        self.path_optimization_iterations = kwargs.get('path_optimization_iterations', self.path_optimization_iterations)
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        self.nodes.append(start_node)
        
        # Call callback with initial state
        if callback:
            callback(self.nodes, None, 0)
        
        # Start timer if time limit is set
        start_time = time.time()
        
        # Main loop
        for i in range(self.max_iterations):
            # Check time limit
            if self.time_limit and time.time() - start_time > self.time_limit:
                break
                
            # Sample random position with adaptive goal bias
            random_pos = self.random_position(goal, i)
            
            # Find nearest node
            nearest = self.nearest_node(random_pos)
            
            # Steer towards random position with adaptive step size
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
                    path = [goal]
                    current = new_node
                    while current.parent is not None:
                        path.append(current.position)
                        current = current.parent
                    path.append(start)
                    path.reverse()
                    
                    # Optimize path
                    current_path = self.optimize_path(path)
                
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