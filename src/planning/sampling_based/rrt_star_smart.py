import numpy as np
import random
import math
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode
from src.planning.sampling_based.rrt_star import RRTStarPlanner

class RRTStarSmartPlanner(RRTStarPlanner):
    """
    RRT* Smart algorithm for path planning.
    
    RRT* Smart enhances RRT* by adding two key features:
    1. Path optimization through intelligent node rejection
    2. Path biasing to focus sampling near the current best path
    
    These features significantly speed up convergence to the optimal solution.
    
    Key features:
    - Faster convergence to optimal solutions
    - Intelligent node rejection for path optimization
    - Path biasing for focused sampling
    - Maintains all optimality guarantees of RRT*
    
    Reference:
    Nasir, J., Islam, F., Malik, U., Ayaz, Y., Hasan, O., Khan, M., & Muhammad, M. S. (2013).
    "RRT*-SMART: A rapid convergence implementation of RRT*."
    International Journal of Advanced Robotic Systems, 10(7), 299.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.path_bias = 0.3  # Probability of sampling near the current path
        self.path_width = 5.0  # Width of the path corridor for biased sampling
        self.current_path = None  # Current best path
        self.smart_radius = 10.0  # Radius for node rejection
        self.path_optimization_freq = 10  # Frequency of path optimization
    
    def sample_near_path(self):
        """
        Sample a point near the current best path.
        """
        if self.current_path is None or len(self.current_path) < 2:
            return self.random_position()
        
        # Randomly select a segment of the path
        segment_idx = random.randint(0, len(self.current_path) - 2)
        p1 = self.current_path[segment_idx]
        p2 = self.current_path[segment_idx + 1]
        
        # Sample a point along the segment
        t = random.random()
        point_on_segment = (
            p1[0] * (1 - t) + p2[0] * t,
            p1[1] * (1 - t) + p2[1] * t
        )
        
        # Add random offset perpendicular to the segment
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate perpendicular direction
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            # If segment is too short, use random direction
            angle = random.uniform(0, 2 * np.pi)
            perp_x = np.cos(angle)
            perp_y = np.sin(angle)
        else:
            # Normalize and rotate 90 degrees
            length = np.sqrt(dx**2 + dy**2)
            perp_x = -dy / length
            perp_y = dx / length
        
        # Random offset within path width
        offset = random.uniform(-self.path_width, self.path_width)
        
        # Final point
        x = int(round(point_on_segment[0] + offset * perp_x))
        y = int(round(point_on_segment[1] + offset * perp_y))
        
        # Ensure point is within world bounds
        x = max(0, min(self.world.width - 1, x))
        y = max(0, min(self.world.height - 1, y))
        
        return (x, y)
    
    def random_position(self):
        """
        Generate a random position with path biasing.
        """
        if random.random() < self.goal_sample_rate:
            # Sample the goal with probability goal_sample_rate
            return self.goal
        elif self.current_path is not None and random.random() < self.path_bias:
            # Sample near the current path with probability path_bias
            return self.sample_near_path()
        else:
            # Sample a random position
            x = random.randint(0, self.world.width - 1)
            y = random.randint(0, self.world.height - 1)
            return (x, y)
    
    def optimize_path(self):
        """
        Optimize the current path by removing unnecessary nodes.
        This is the "Smart" part of RRT* Smart.
        """
        if self.current_path is None or len(self.current_path) <= 2:
            return
        
        # Find nodes that can be directly connected
        i = 0
        while i < len(self.current_path) - 2:
            # Check if we can connect node i to node i+2 directly
            if self.is_collision_free(self.current_path[i], self.current_path[i+2]):
                # Remove the intermediate node
                self.current_path.pop(i+1)
            else:
                i += 1
    
    def node_rejection(self, new_pos, new_cost):
        """
        Reject nodes that are unlikely to improve the solution.
        Returns True if the node should be rejected, False otherwise.
        """
        if self.current_path is None:
            return False
        
        # Check if the new node is near the current path
        for i in range(len(self.current_path) - 1):
            p1 = self.current_path[i]
            p2 = self.current_path[i + 1]
            
            # Calculate distance from new_pos to the line segment p1-p2
            # First, calculate the projection of new_pos onto the line
            line_vec = (p2[0] - p1[0], p2[1] - p1[1])
            point_vec = (new_pos[0] - p1[0], new_pos[1] - p1[1])
            line_len = self.distance(p1, p2)
            
            if line_len < 1e-6:
                # Line segment is too short, use distance to p1
                dist_to_segment = self.distance(new_pos, p1)
            else:
                # Calculate projection
                t = (line_vec[0] * point_vec[0] + line_vec[1] * point_vec[1]) / (line_len**2)
                
                if t < 0:
                    # Projection is before p1
                    dist_to_segment = self.distance(new_pos, p1)
                elif t > 1:
                    # Projection is after p2
                    dist_to_segment = self.distance(new_pos, p2)
                else:
                    # Projection is on the segment
                    proj_x = p1[0] + t * line_vec[0]
                    proj_y = p1[1] + t * line_vec[1]
                    dist_to_segment = self.distance(new_pos, (proj_x, proj_y))
            
            if dist_to_segment <= self.smart_radius:
                # Node is near the current path
                # Calculate the cost of the path through this segment
                segment_cost = self.distance(p1, p2)
                
                # Calculate the cost of the path through the new node
                new_segment_cost = self.distance(p1, new_pos) + self.distance(new_pos, p2)
                
                # Reject if the new path is not significantly better
                if new_segment_cost >= segment_cost * 0.95:
                    return True
        
        return False
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using RRT* Smart.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            step_size: Optional step size for extending the tree.
            max_iterations: Optional maximum number of iterations.
            goal_sample_rate: Optional probability of sampling the goal.
            goal_threshold: Optional distance threshold to consider goal reached.
            search_radius: Optional radius for nearest neighbors search.
            rewire_factor: Optional factor to scale search radius.
            path_bias: Optional probability of sampling near the current path.
            path_width: Optional width of the path corridor for biased sampling.
            smart_radius: Optional radius for node rejection.
            path_optimization_freq: Optional frequency of path optimization.
            
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
        self.path_bias = kwargs.get('path_bias', self.path_bias)
        self.path_width = kwargs.get('path_width', self.path_width)
        self.smart_radius = kwargs.get('smart_radius', self.smart_radius)
        self.path_optimization_freq = kwargs.get('path_optimization_freq', self.path_optimization_freq)
        
        # Store goal for random sampling
        self.goal = goal
        
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
            
            # Smart node rejection
            if self.node_rejection(new_pos, min_cost):
                continue
            
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
                    
                    # Update current path
                    self.current_path = [goal]
                    current = best_goal_node
                    while current.parent is not None:
                        self.current_path.append(current.position)
                        current = current.parent
                    self.current_path.append(start)
                    self.current_path.reverse()
                    
                    # Periodically optimize the path
                    if i % self.path_optimization_freq == 0:
                        self.optimize_path()
        
        # Final path optimization
        if self.current_path is not None:
            self.optimize_path()
            return self.current_path
        
        # No path found
        return None
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of RRT* Smart that can provide step-by-step visualization.
        
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
        self.path_bias = kwargs.get('path_bias', self.path_bias)
        self.path_width = kwargs.get('path_width', self.path_width)
        self.smart_radius = kwargs.get('smart_radius', self.smart_radius)
        self.path_optimization_freq = kwargs.get('path_optimization_freq', self.path_optimization_freq)
        
        # Store goal for random sampling
        self.goal = goal
        
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
            
            # Smart node rejection
            if self.node_rejection(new_pos, min_cost):
                continue
            
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
                    
                    # Update current path
                    self.current_path = [goal]
                    current = best_goal_node
                    while current.parent is not None:
                        self.current_path.append(current.position)
                        current = current.parent
                    self.current_path.append(start)
                    self.current_path.reverse()
                    
                    # Periodically optimize the path
                    if i % self.path_optimization_freq == 0:
                        self.optimize_path()
            
            # Call callback with current state
            if callback:
                callback(self.nodes, self.current_path, i + 1, f"Iteration {i+1}")
        
        # Final path optimization
        if self.current_path is not None:
            self.optimize_path()
            
            # Call callback with final state
            if callback:
                callback(self.nodes, self.current_path, self.max_iterations, "Final optimized path")
            
            return self.current_path
        
        # No path found
        if callback:
            callback(self.nodes, None, self.max_iterations, "No path found")
        
        return None