import numpy as np
import random
from scipy.interpolate import CubicSpline
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode
from src.planning.sampling_based.rrt_star import RRTStarPlanner

class SplineRRTStarPlanner(RRTStarPlanner):
    """
    Spline-RRT* algorithm for path planning.
    
    Spline-RRT* extends RRT* by using cubic splines to connect nodes instead of
    straight lines. This results in smoother paths that are more suitable for
    vehicles with kinematic constraints.
    
    Key features:
    - Smooth paths using cubic splines
    - Maintains optimality guarantees of RRT*
    - Better suited for vehicles with kinematic constraints
    - Improved path quality
    
    Reference:
    Qureshi, A. H., & Ayaz, Y. (2016).
    "Optimal path planning based on spline-RRT* for fixed-wing UAVs operating in three-dimensional environments."
    In 2016 IEEE 14th International Conference on Industrial Informatics (INDIN) (pp. 810-815). IEEE.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.spline_resolution = 10  # Number of points to sample along each spline
        self.curvature_weight = 0.5  # Weight for curvature in cost calculation
    
    def generate_spline(self, from_pos, to_pos, via_point=None):
        """
        Generate a cubic spline from from_pos to to_pos.
        Optionally uses a via_point to control the shape of the spline.
        
        Returns a list of points along the spline.
        """
        # If no via_point is provided, create one
        if via_point is None:
            # Create a via point perpendicular to the line from from_pos to to_pos
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            # Calculate perpendicular direction
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                # If points are too close, use a random direction
                angle = random.uniform(0, 2 * np.pi)
                perp_x = np.cos(angle)
                perp_y = np.sin(angle)
            else:
                # Normalize and rotate 90 degrees
                length = np.sqrt(dx**2 + dy**2)
                perp_x = -dy / length
                perp_y = dx / length
            
            # Create via point at the midpoint with a small offset
            offset = min(2.0, length / 4)  # Limit offset to avoid extreme curves
            mid_x = (from_pos[0] + to_pos[0]) / 2 + offset * perp_x
            mid_y = (from_pos[1] + to_pos[1]) / 2 + offset * perp_y
            
            via_point = (int(round(mid_x)), int(round(mid_y)))
        
        # Create spline using three points
        x = [from_pos[0], via_point[0], to_pos[0]]
        y = [from_pos[1], via_point[1], to_pos[1]]
        
        # Parameter t for interpolation
        t = np.linspace(0, 1, 3)
        t_fine = np.linspace(0, 1, self.spline_resolution)
        
        # Create cubic spline
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        
        # Sample points along the spline
        spline_points = []
        for ti in t_fine:
            xi = int(round(cs_x(ti)))
            yi = int(round(cs_y(ti)))
            spline_points.append((xi, yi))
        
        return spline_points
    
    def is_spline_collision_free(self, spline_points):
        """
        Check if a spline is collision-free.
        """
        for point in spline_points:
            if not self.is_valid_point(point):
                return False
        
        return True
    
    def calculate_spline_cost(self, spline_points):
        """
        Calculate the cost of a spline based on length and curvature.
        """
        if len(spline_points) < 3:
            return float('inf')
        
        # Calculate length
        length = 0.0
        for i in range(len(spline_points) - 1):
            length += self.distance(spline_points[i], spline_points[i + 1])
        
        # Calculate curvature (approximated by angle changes)
        curvature = 0.0
        for i in range(1, len(spline_points) - 1):
            p1 = spline_points[i - 1]
            p2 = spline_points[i]
            p3 = spline_points[i + 1]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate magnitudes
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            # Skip if vectors are too small
            if mag1 < 1e-6 or mag2 < 1e-6:
                continue
            
            # Calculate dot product
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            
            # Calculate angle
            cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            angle = np.arccos(cos_angle)
            
            # Add to curvature
            curvature += angle
        
        # Weighted sum of length and curvature
        cost = length + self.curvature_weight * curvature
        
        return cost
    
    def choose_parent(self, new_pos, near_nodes):
        """
        Choose the best parent for a new node from a list of near nodes.
        Uses splines instead of straight lines.
        """
        if not near_nodes:
            return None, None, float('inf')
        
        # Find the node that would give the lowest cost
        min_cost = float('inf')
        best_parent = None
        best_spline = None
        
        for node in near_nodes:
            # Generate spline
            spline_points = self.generate_spline(node.position, new_pos)
            
            # Check if the spline is collision-free
            if not self.is_spline_collision_free(spline_points):
                continue
            
            # Calculate spline cost
            spline_cost = self.calculate_spline_cost(spline_points)
            
            # Calculate total cost
            total_cost = node.cost + spline_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_parent = node
                best_spline = spline_points
        
        return best_parent, best_spline, min_cost
    
    def rewire(self, new_node, near_nodes):
        """
        Rewire the tree by checking if paths through the new node are better.
        Uses splines instead of straight lines.
        """
        for node in near_nodes:
            # Skip the parent of the new node
            if node == new_node.parent:
                continue
            
            # Generate spline
            spline_points = self.generate_spline(new_node.position, node.position)
            
            # Check if the spline is collision-free
            if not self.is_spline_collision_free(spline_points):
                continue
            
            # Calculate spline cost
            spline_cost = self.calculate_spline_cost(spline_points)
            
            # Calculate total cost
            new_cost = new_node.cost + spline_cost
            
            # If new path is better, rewire
            if new_cost < node.cost:
                node.parent = new_node
                node.cost = new_cost
                node.spline = spline_points
                
                # Update costs of descendants
                self.update_descendants_cost(node)
    
    def update_descendants_cost(self, node):
        """
        Recursively update the costs of all descendants of a node.
        Called after rewiring to propagate cost changes.
        """
        # Find all children of this node
        children = [n for n in self.nodes if n.parent == node]
        
        for child in children:
            # Calculate spline cost
            spline_cost = self.calculate_spline_cost(child.spline)
            
            # Update child's cost
            child.cost = node.cost + spline_cost
            
            # Recursively update descendants
            self.update_descendants_cost(child)
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Spline-RRT*.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            step_size: Optional step size for extending the tree.
            max_iterations: Optional maximum number of iterations.
            goal_sample_rate: Optional probability of sampling the goal.
            goal_threshold: Optional distance threshold to consider goal reached.
            search_radius: Optional radius for nearest neighbors search.
            rewire_factor: Optional factor to scale search radius.
            spline_resolution: Optional number of points to sample along each spline.
            curvature_weight: Optional weight for curvature in cost calculation.
            
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
        self.spline_resolution = kwargs.get('spline_resolution', self.spline_resolution)
        self.curvature_weight = kwargs.get('curvature_weight', self.curvature_weight)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        start_node.cost = 0.0
        start_node.spline = [start]  # Initialize spline
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
            
            # Skip if the position is invalid
            if not self.is_valid_point(new_pos):
                continue
            
            # Calculate search radius (grows with log(n))
            n = len(self.nodes)
            radius = min(self.search_radius * (np.log(n + 1) / n) ** (1/2), self.rewire_factor * self.step_size)
            
            # Find near nodes
            near_nodes_list = self.near_nodes(new_pos, radius)
            
            # Choose best parent
            best_parent, best_spline, min_cost = self.choose_parent(new_pos, near_nodes_list)
            
            if best_parent is None:
                # If no valid parent found, use nearest as parent
                spline_points = self.generate_spline(nearest.position, new_pos)
                
                # Skip if the spline is invalid
                if not self.is_spline_collision_free(spline_points):
                    continue
                
                best_parent = nearest
                best_spline = spline_points
                
                # Calculate cost
                spline_cost = self.calculate_spline_cost(spline_points)
                min_cost = best_parent.cost + spline_cost
            
            # Create new node
            new_node = RRTNode(new_pos)
            new_node.parent = best_parent
            new_node.cost = min_cost
            new_node.spline = best_spline
            self.nodes.append(new_node)
            
            # Rewire tree
            self.rewire(new_node, near_nodes_list)
            
            # Check if goal is reached
            if self.distance(new_pos, goal) <= self.goal_threshold:
                # Check if this path to goal is better
                goal_cost = new_node.cost
                if goal_cost < best_goal_cost:
                    best_goal_node = new_node
                    best_goal_cost = goal_cost
        
        # Construct path if goal was reached
        if best_goal_node is not None:
            # Extract path
            path = []
            current = best_goal_node
            
            while current is not None:
                # Add spline points in reverse
                if hasattr(current, 'spline') and current.spline:
                    # Add all points except the first one (to avoid duplicates)
                    for point in reversed(current.spline[1:]):
                        path.insert(0, point)
                
                current = current.parent
            
            # Add start position
            path.insert(0, start)
            
            return path
        
        # No path found
        return None
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of Spline-RRT* that can provide step-by-step visualization.
        
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
        self.spline_resolution = kwargs.get('spline_resolution', self.spline_resolution)
        self.curvature_weight = kwargs.get('curvature_weight', self.curvature_weight)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        start_node.cost = 0.0
        start_node.spline = [start]  # Initialize spline
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
            
            # Skip if the position is invalid
            if not self.is_valid_point(new_pos):
                continue
            
            # Calculate search radius (grows with log(n))
            n = len(self.nodes)
            radius = min(self.search_radius * (np.log(n + 1) / n) ** (1/2), self.rewire_factor * self.step_size)
            
            # Find near nodes
            near_nodes_list = self.near_nodes(new_pos, radius)
            
            # Choose best parent
            best_parent, best_spline, min_cost = self.choose_parent(new_pos, near_nodes_list)
            
            if best_parent is None:
                # If no valid parent found, use nearest as parent
                spline_points = self.generate_spline(nearest.position, new_pos)
                
                # Skip if the spline is invalid
                if not self.is_spline_collision_free(spline_points):
                    continue
                
                best_parent = nearest
                best_spline = spline_points
                
                # Calculate cost
                spline_cost = self.calculate_spline_cost(spline_points)
                min_cost = best_parent.cost + spline_cost
            
            # Create new node
            new_node = RRTNode(new_pos)
            new_node.parent = best_parent
            new_node.cost = min_cost
            new_node.spline = best_spline
            self.nodes.append(new_node)
            
            # Rewire tree
            self.rewire(new_node, near_nodes_list)
            
            # Check if goal is reached
            current_path = None
            if self.distance(new_pos, goal) <= self.goal_threshold:
                # Check if this path to goal is better
                goal_cost = new_node.cost
                if goal_cost < best_goal_cost:
                    best_goal_node = new_node
                    best_goal_cost = goal_cost
                    
                    # Extract path for visualization
                    path = []
                    current = best_goal_node
                    
                    while current is not None:
                        # Add spline points in reverse
                        if hasattr(current, 'spline') and current.spline:
                            # Add all points except the first one (to avoid duplicates)
                            for point in reversed(current.spline[1:]):
                                path.insert(0, point)
                        
                        current = current.parent
                    
                    # Add start position
                    path.insert(0, start)
                    current_path = path
            
            # Call callback with current state
            if callback and i % 10 == 0:  # Call every 10 iterations to avoid slowdown
                callback(self.nodes, current_path, i + 1, f"Iteration {i+1}")
        
        # Construct final path if goal was reached
        final_path = None
        if best_goal_node is not None:
            # Extract path
            path = []
            current = best_goal_node
            
            while current is not None:
                # Add spline points in reverse
                if hasattr(current, 'spline') and current.spline:
                    # Add all points except the first one (to avoid duplicates)
                    for point in reversed(current.spline[1:]):
                        path.insert(0, point)
                
                current = current.parent
            
            # Add start position
            path.insert(0, start)
            final_path = path
            
            # Call callback with final state
            if callback:
                callback(self.nodes, final_path, self.max_iterations, "Final path")
        else:
            # No path found
            if callback:
                callback(self.nodes, None, self.max_iterations, "No path found")
        
        return final_path