import numpy as np
import random
import heapq
from src.planning.base_planner import BasePlanner

class FMTNode:
    """
    Node class for FMT* algorithm.
    """
    def __init__(self, position):
        self.position = position
        self.cost = float('inf')
        self.parent = None
        self.in_open = False
        self.in_closed = False
        self.in_unvisited = True
        self.neighbors = []
    
    def __lt__(self, other):
        """
        Comparison operator for priority queue.
        """
        return self.cost < other.cost

class FMTStarPlanner(BasePlanner):
    """
    Fast Marching Trees (FMT*) algorithm for path planning.
    
    FMT* is a sampling-based algorithm that uses a lazy collision checking strategy
    and a marching-type approach to grow a tree of paths. It offers asymptotic
    optimality guarantees similar to RRT* but with faster convergence in many cases.
    
    Key features:
    - Single-query, asymptotically optimal planner
    - Lazy collision checking for efficiency
    - Marching-type approach for tree growth
    - Fast convergence to optimal solutions
    
    Reference:
    Janson, L., Schmerling, E., Clark, A., & Pavone, M. (2015).
    "Fast marching tree: A fast marching sampling-based method for optimal motion planning in many dimensions."
    The International Journal of Robotics Research, 34(7), 883-921.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.num_samples = 1000  # Number of samples to generate
        self.connection_radius = 15.0  # Radius for connecting nodes
        self.collision_check_resolution = 5  # Number of points to check for collisions
    
    def distance(self, p1, p2):
        """
        Calculate Euclidean distance between two points.
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def is_collision_free(self, p1, p2):
        """
        Check if the line segment from p1 to p2 is collision-free.
        """
        # Calculate number of points to check
        dist = self.distance(p1, p2)
        num_points = max(2, int(dist / self.collision_check_resolution))
        
        # Check points along the line
        for i in range(1, num_points):
            t = i / (num_points - 1)
            x = int(p1[0] * (1 - t) + p2[0] * t)
            y = int(p1[1] * (1 - t) + p2[1] * t)
            
            if not self.is_valid_point((x, y)):
                return False
        
        return True
    
    def find_neighbors(self, node, nodes, radius):
        """
        Find all nodes within radius of the given node.
        """
        neighbors = []
        for other in nodes:
            if other != node and self.distance(node.position, other.position) <= radius:
                neighbors.append(other)
        
        return neighbors
    
    def near_unvisited(self, node, unvisited, radius):
        """
        Find all unvisited nodes within radius of the given node.
        """
        return [n for n in unvisited if self.distance(node.position, n.position) <= radius]
    
    def near_open(self, node, open_set, radius):
        """
        Find all open nodes within radius of the given node.
        """
        return [n for n in open_set if self.distance(node.position, n.position) <= radius]
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using FMT*.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            num_samples: Optional number of samples to generate.
            connection_radius: Optional radius for connecting nodes.
            collision_check_resolution: Optional resolution for collision checking.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.num_samples = kwargs.get('num_samples', self.num_samples)
        self.connection_radius = kwargs.get('connection_radius', self.connection_radius)
        self.collision_check_resolution = kwargs.get('collision_check_resolution', self.collision_check_resolution)
        
        # Generate samples (including start and goal)
        samples = [start, goal]
        while len(samples) < self.num_samples:
            x = random.randint(0, self.world.width - 1)
            y = random.randint(0, self.world.height - 1)
            
            if self.is_valid_point((x, y)):
                samples.append((x, y))
        
        # Create nodes
        nodes = [FMTNode(pos) for pos in samples]
        start_node = nodes[0]
        goal_node = nodes[1]
        
        # Initialize sets
        unvisited = set(nodes)
        open_set = set([start_node])
        closed_set = set()
        
        # Initialize start node
        start_node.cost = 0
        start_node.in_unvisited = False
        start_node.in_open = True
        unvisited.remove(start_node)
        
        # Precompute neighbors for all nodes
        for node in nodes:
            node.neighbors = self.find_neighbors(node, nodes, self.connection_radius)
        
        # Main loop
        while open_set and goal_node not in closed_set:
            # Find node in open with lowest cost
            z = min(open_set, key=lambda n: n.cost)
            
            # Move z from open to closed
            open_set.remove(z)
            closed_set.add(z)
            z.in_open = False
            z.in_closed = True
            
            # Find unvisited neighbors of z
            x_near = [n for n in z.neighbors if n.in_unvisited]
            
            # For each unvisited neighbor
            for x in x_near:
                # Find neighbors of x that are in open
                y_near = [n for n in x.neighbors if n.in_open]
                
                # Skip if no open neighbors
                if not y_near:
                    continue
                
                # Find best parent for x
                y_min = min(y_near, key=lambda n: n.cost + self.distance(n.position, x.position))
                
                # Check if the connection is collision-free
                if self.is_collision_free(y_min.position, x.position):
                    # Connect x to y_min
                    x.parent = y_min
                    x.cost = y_min.cost + self.distance(y_min.position, x.position)
                    
                    # Move x from unvisited to open
                    unvisited.remove(x)
                    open_set.add(x)
                    x.in_unvisited = False
                    x.in_open = True
        
        # Check if goal was reached
        if goal_node in closed_set:
            # Extract path
            path = []
            current = goal_node
            
            while current is not None:
                path.insert(0, current.position)
                current = current.parent
            
            return path
        
        # No path found
        return None
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of FMT* that can provide step-by-step visualization.
        
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
        self.num_samples = kwargs.get('num_samples', self.num_samples)
        self.connection_radius = kwargs.get('connection_radius', self.connection_radius)
        self.collision_check_resolution = kwargs.get('collision_check_resolution', self.collision_check_resolution)
        
        # Generate samples (including start and goal)
        samples = [start, goal]
        while len(samples) < self.num_samples:
            x = random.randint(0, self.world.width - 1)
            y = random.randint(0, self.world.height - 1)
            
            if self.is_valid_point((x, y)):
                samples.append((x, y))
        
        # Create nodes
        nodes = [FMTNode(pos) for pos in samples]
        start_node = nodes[0]
        goal_node = nodes[1]
        
        # Initialize sets
        unvisited = set(nodes)
        open_set = set([start_node])
        closed_set = set()
        
        # Initialize start node
        start_node.cost = 0
        start_node.in_unvisited = False
        start_node.in_open = True
        unvisited.remove(start_node)
        
        # Precompute neighbors for all nodes
        for node in nodes:
            node.neighbors = self.find_neighbors(node, nodes, self.connection_radius)
        
        # Call callback with initial state
        if callback:
            # Create a list of all nodes and their parents for visualization
            tree_nodes = nodes
            current_path = None
            callback(tree_nodes, current_path, 0, "Initial samples")
        
        # Main loop
        iteration = 0
        while open_set and goal_node not in closed_set:
            iteration += 1
            
            # Find node in open with lowest cost
            z = min(open_set, key=lambda n: n.cost)
            
            # Move z from open to closed
            open_set.remove(z)
            closed_set.add(z)
            z.in_open = False
            z.in_closed = True
            
            # Find unvisited neighbors of z
            x_near = [n for n in z.neighbors if n.in_unvisited]
            
            # For each unvisited neighbor
            for x in x_near:
                # Find neighbors of x that are in open
                y_near = [n for n in x.neighbors if n.in_open]
                
                # Skip if no open neighbors
                if not y_near:
                    continue
                
                # Find best parent for x
                y_min = min(y_near, key=lambda n: n.cost + self.distance(n.position, x.position))
                
                # Check if the connection is collision-free
                if self.is_collision_free(y_min.position, x.position):
                    # Connect x to y_min
                    x.parent = y_min
                    x.cost = y_min.cost + self.distance(y_min.position, x.position)
                    
                    # Move x from unvisited to open
                    unvisited.remove(x)
                    open_set.add(x)
                    x.in_unvisited = False
                    x.in_open = True
            
            # Check if goal is in closed set
            current_path = None
            if goal_node in closed_set:
                # Extract path for visualization
                path = []
                current = goal_node
                
                while current is not None:
                    path.insert(0, current.position)
                    current = current.parent
                
                current_path = path
            
            # Call callback with current state
            if callback and iteration % 10 == 0:  # Call every 10 iterations to avoid slowdown
                callback(nodes, current_path, iteration, f"Iteration {iteration}")
        
        # Check if goal was reached
        final_path = None
        if goal_node in closed_set:
            # Extract path
            path = []
            current = goal_node
            
            while current is not None:
                path.insert(0, current.position)
                current = current.parent
            
            final_path = path
            
            # Call callback with final state
            if callback:
                callback(nodes, final_path, iteration, "Final path")
        else:
            # No path found
            if callback:
                callback(nodes, None, iteration, "No path found")
        
        return final_path 