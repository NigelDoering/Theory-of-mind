import numpy as np
import random
import math
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode
from src.planning.sampling_based.rrt_star import RRTStarPlanner

class InformedRRTStarPlanner(RRTStarPlanner):
    """
    Informed RRT* algorithm for path planning.
    
    Informed RRT* improves upon RRT* by focusing the sampling to an ellipsoidal
    region that contains all paths that could improve the current solution.
    This significantly speeds up convergence to the optimal path.
    
    Key features:
    - Focused sampling in an ellipsoidal region
    - Faster convergence to optimal solutions
    - Maintains all optimality guarantees of RRT*
    - Efficient use of computational resources
    
    Reference:
    Gammell, J. D., Srinivasa, S. S., & Barfoot, T. D. (2014, July). 
    "Informed RRT*: Optimal sampling-based path planning focused via direct sampling of an admissible ellipsoidal heuristic."
    In 2014 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2997-3004). IEEE.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.c_best = float('inf')  # Length of the best path found so far
        self.c_min = 0.0  # Minimum distance between start and goal
        self.ellipse_center = None  # Center of the sampling ellipse
        self.ellipse_axes = None    # Axes of the sampling ellipse
        self.ellipse_rotation = None  # Rotation matrix for the ellipse
    
    def setup_informed_sampling(self, start, goal):
        """
        Set up the informed sampling ellipse based on start and goal positions.
        """
        # Calculate minimum distance between start and goal
        self.c_min = self.distance(start, goal)
        
        # Calculate center of the ellipse
        self.ellipse_center = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
        
        # Calculate direction vector from start to goal
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        # Calculate rotation angle
        if abs(dx) < 1e-6:
            # Vertical line
            theta = np.pi / 2 if dy > 0 else -np.pi / 2
        else:
            theta = np.arctan2(dy, dx)
        
        # Create rotation matrix
        self.ellipse_rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Initially, no best path is known, so no ellipse is defined yet
        self.ellipse_axes = None
    
    def update_ellipse(self, c_best):
        """
        Update the ellipse axes based on the current best path length.
        """
        if c_best < self.c_best:
            self.c_best = c_best
            
            # Calculate the ellipse axes
            a = c_best / 2  # Semi-major axis
            b = np.sqrt(c_best**2 - self.c_min**2) / 2  # Semi-minor axis
            self.ellipse_axes = (a, b)
    
    def sample_from_ellipse(self):
        """
        Sample a point from the informed ellipsoidal region.
        If no solution has been found yet, sample from the entire space.
        """
        if self.ellipse_axes is None or self.c_best == float('inf'):
            # No solution found yet, sample from the entire space
            return self.random_position()
        
        # Sample from a unit circle
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x**2 + y**2 <= 1:
                break
        
        # Transform to ellipse
        a, b = self.ellipse_axes
        point = np.array([a * x, b * y])
        
        # Rotate
        point = np.dot(self.ellipse_rotation, point)
        
        # Translate
        point[0] += self.ellipse_center[0]
        point[1] += self.ellipse_center[1]
        
        # Ensure point is within world bounds
        point[0] = max(0, min(self.world.width - 1, int(round(point[0]))))
        point[1] = max(0, min(self.world.height - 1, int(round(point[1]))))
        
        return (int(point[0]), int(point[1]))
    
    def random_position(self):
        """
        Generate a random position using informed sampling if a solution exists.
        """
        if random.random() < self.goal_sample_rate:
            # Sample the goal with probability goal_sample_rate
            return self.goal
        else:
            # Sample from the ellipse if a solution exists
            return self.sample_from_ellipse()
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Informed RRT*.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            step_size: Optional step size for extending the tree.
            max_iterations: Optional maximum number of iterations.
            goal_sample_rate: Optional probability of sampling the goal.
            goal_threshold: Optional distance threshold to consider goal reached.
            search_radius: Optional radius for nearest neighbors search.
            rewire_factor: Optional factor to scale search radius.
            
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
        self.search_radius = kwargs.get('search_radius', self.search_radius)
        self.rewire_factor = kwargs.get('rewire_factor', self.rewire_factor)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Set up informed sampling
        self.setup_informed_sampling(start, goal)
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        start_node.cost = 0.0
        self.nodes.append(start_node)
        
        # Track best goal node
        best_goal_node = None
        best_goal_cost = float('inf')
        
        # Main loop
        for i in range(self.max_iterations):
            # Sample random position
            random_pos = self.random_position()
            
            # Find nearest node
            nearest = self.nearest_node(random_pos)
            
            # Steer towards random position
            new_pos = self.steer(nearest.position, random_pos)
            
            # Skip if new position is not valid
            if not self.is_valid_point(new_pos):
                continue
            
            # Calculate search radius (grows with log(n))
            n = len(self.nodes)
            radius = min(self.search_radius * (np.log(n + 1) / n) ** (1/2), self.rewire_factor * self.step_size)
            
            # Find near nodes
            near_nodes_list = self.near_nodes(new_pos, radius)
            
            # Choose best parent
            best_parent, min_cost = self.choose_parent(new_pos, near_nodes_list)
            
            if best_parent is None:
                # If no valid parent found, use nearest as parent
                if self.is_collision_free(nearest.position, new_pos):
                    best_parent = nearest
                    min_cost = self.calculate_cost(nearest, new_pos)
                else:
                    continue  # Skip this iteration if no valid parent
            
            # Create new node
            new_node = RRTNode(new_pos)
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            
            # Rewire tree
            self.rewire(new_node, near_nodes_list)
            
            # Check if goal is reached
            if self.distance(new_pos, goal) <= self.goal_threshold:
                # Check if this path to goal is better
                goal_cost = new_node.cost + self.distance(new_pos, goal)
                if goal_cost < best_goal_cost:
                    best_goal_node = new_node
                    best_goal_cost = goal_cost
                    
                    # Update the sampling ellipse
                    self.update_ellipse(best_goal_cost)
        
        # Construct path if goal was reached
        if best_goal_node is not None:
            path = [goal]
            current = best_goal_node
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
        Interactive version of Informed RRT* that can provide step-by-step visualization.
        
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
        self.search_radius = kwargs.get('search_radius', self.search_radius)
        self.rewire_factor = kwargs.get('rewire_factor', self.rewire_factor)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Set up informed sampling
        self.setup_informed_sampling(start, goal)
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        start_node.cost = 0.0
        self.nodes.append(start_node)
        
        # Track best goal node
        best_goal_node = None
        best_goal_cost = float('inf')
        
        # Call callback with initial state
        if callback:
            callback(self.nodes, None, 0, "Initial tree")
        
        # Main loop
        for i in range(self.max_iterations):
            # Sample random position
            random_pos = self.random_position()
            
            # Find nearest node
            nearest = self.nearest_node(random_pos)
            
            # Steer towards random position
            new_pos = self.steer(nearest.position, random_pos)
            
            # Skip if new position is not valid
            if not self.is_valid_point(new_pos):
                continue
            
            # Calculate search radius (grows with log(n))
            n = len(self.nodes)
            radius = min(self.search_radius * (np.log(n + 1) / n) ** (1/2), self.rewire_factor * self.step_size)
            
            # Find near nodes
            near_nodes_list = self.near_nodes(new_pos, radius)
            
            # Choose best parent
            best_parent, min_cost = self.choose_parent(new_pos, near_nodes_list)
            
            if best_parent is None:
                # If no valid parent found, use nearest as parent
                if self.is_collision_free(nearest.position, new_pos):
                    best_parent = nearest
                    min_cost = self.calculate_cost(nearest, new_pos)
                else:
                    continue  # Skip this iteration if no valid parent
            
            # Create new node
            new_node = RRTNode(new_pos)
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            
            # Rewire tree
            self.rewire(new_node, near_nodes_list)
            
            # Check if goal is reached
            current_path = None
            if self.distance(new_pos, goal) <= self.goal_threshold:
                # Check if this path to goal is better
                goal_cost = new_node.cost + self.distance(new_pos, goal)
                if goal_cost < best_goal_cost:
                    best_goal_node = new_node
                    best_goal_cost = goal_cost
                    
                    # Update the sampling ellipse
                    self.update_ellipse(best_goal_cost)
                    
                    # Construct current best path
                    current_path = [goal]
                    current = best_goal_node
                    while current.parent is not None:
                        current_path.append(current.position)
                        current = current.parent
                    current_path.append(start)
                    current_path.reverse()
            
            # Call callback with current state
            ellipse_info = ""
            if self.ellipse_axes is not None:
                ellipse_info = f", Ellipse: {self.ellipse_axes}"
            callback(self.nodes, current_path, i + 1, f"Iteration {i+1}{ellipse_info}")
        
        # Construct final path if goal was reached
        final_path = None
        if best_goal_node is not None:
            final_path = [goal]
            current = best_goal_node
            while current.parent is not None:
                final_path.append(current.position)
                current = current.parent
            final_path.append(start)
            final_path.reverse()
            
            # Call callback with final state
            if callback:
                callback(self.nodes, final_path, self.max_iterations, "Final path")
        else:
            # No path found
            if callback:
                callback(self.nodes, None, self.max_iterations, "No path found")
        
        return final_path 