import numpy as np
import random
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode

class RRTStarPlanner(BasePlanner):
    """
    RRT* (Optimal Rapidly-exploring Random Tree) algorithm for path planning.
    
    RRT* extends RRT by adding two key operations:
    1. Choosing the best parent for new nodes
    2. Rewiring the tree to improve path costs
    
    These operations ensure that RRT* converges to an optimal solution as
    the number of samples increases.
    
    Key features:
    - Asymptotically optimal paths
    - Efficient exploration of the configuration space
    - Continuous path improvement
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.nodes = []
        self.step_size = 1.0
        self.max_iterations = 1000
        self.goal_sample_rate = 0.1
        self.goal_threshold = 1.0
        self.search_radius = 5.0  # Radius for nearest neighbors search
        self.rewire_factor = 2.0  # Factor to scale search radius based on log(n)
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def nearest_node(self, position):
        """Find the nearest node in the tree to the given position."""
        return min(self.nodes, key=lambda node: self.distance(node.position, position))
    
    def near_nodes(self, position, radius):
        """Find all nodes within a certain radius of the given position."""
        return [node for node in self.nodes 
                if self.distance(node.position, position) <= radius]
    
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
    
    def calculate_cost(self, from_node, to_position):
        """Calculate the cost to reach to_position from from_node."""
        return from_node.cost + self.distance(from_node.position, to_position)
    
    def choose_parent(self, new_position, near_nodes):
        """
        Choose the best parent for a new node from a list of near nodes.
        Returns the best parent node and the associated cost.
        """
        if not near_nodes:
            return None, float('inf')
        
        # Find the node that would give the lowest cost
        min_cost = float('inf')
        best_parent = None
        
        for node in near_nodes:
            # Check if the path is collision-free
            if self.is_collision_free(node.position, new_position):
                # Calculate cost through this node
                cost = self.calculate_cost(node, new_position)
                
                if cost < min_cost:
                    min_cost = cost
                    best_parent = node
        
        return best_parent, min_cost
    
    def rewire(self, new_node, near_nodes):
        """
        Rewire the tree by checking if paths through the new node are better.
        Updates parent pointers and costs for affected nodes.
        """
        for node in near_nodes:
            # Skip the parent of the new node
            if node == new_node.parent:
                continue
            
            # Check if path through new_node is collision-free
            if self.is_collision_free(new_node.position, node.position):
                # Calculate potential new cost
                new_cost = new_node.cost + self.distance(new_node.position, node.position)
                
                # If new path is better, rewire
                if new_cost < node.cost:
                    node.parent = new_node
                    node.cost = new_cost
                    
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
            # Update child's cost
            child.cost = node.cost + self.distance(node.position, child.position)
            # Recursively update descendants
            self.update_descendants_cost(child)
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using RRT*.
        
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
        Interactive version of RRT* that can provide step-by-step visualization.
        
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
                    
                    # Construct current best path
                    current_path = [goal]
                    current = best_goal_node
                    while current.parent is not None:
                        current_path.append(current.position)
                        current = current.parent
                    current_path.append(start)
                    current_path.reverse()
            
            # Call callback with current state
            if callback:
                callback(self.nodes, current_path, i + 1, f"Iteration {i+1}")
        
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
    