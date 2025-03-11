import numpy as np
import heapq
import math
import time
from src.planning.base_planner import BasePlanner

class BITNode:
    """
    Node class for BIT* algorithm.
    Represents a vertex in the implicit random geometric graph.
    """
    def __init__(self, position):
        self.position = position  # (x, y) tuple
        self.g = float('inf')     # Cost-to-come from start
        self.h = 0.0              # Heuristic (cost-to-go to goal)
        self.f = float('inf')     # Total cost (g + h)
        self.parent = None        # Parent in the tree
        self.children = set()     # Children in the tree
        self.in_tree = False      # Whether the vertex is in the tree
        self.sample_batch = -1    # Batch number when this node was sampled
        
    def __lt__(self, other):
        """Comparison for priority queue based on f-value."""
        if abs(self.f - other.f) < 1e-6:  # If f-values are essentially equal
            return self.g < other.g  # Break ties using g-value
        return self.f < other.f
    
class Edge:
    """
    Edge class for BIT* algorithm.
    Represents a potential connection between two vertices.
    """
    def __init__(self, start, end, g_start, c_edge, h_end):
        self.start = start    # Start node
        self.end = end        # End node
        self.g = g_start      # Cost-to-come to start
        self.c = c_edge       # Cost of the edge
        self.h = h_end        # Heuristic of end
        self.f = g_start + c_edge + h_end  # Total estimated cost
        
    def __lt__(self, other):
        """Comparison for priority queue based on f-value."""
        if abs(self.f - other.f) < 1e-6:  # If f-values are essentially equal
            if abs(self.g + self.c - other.g - other.c) < 1e-6:  # If g+c values are essentially equal
                return self.g < other.g  # Break ties using g-value
            return self.g + self.c < other.g + other.c  # Break ties using g+c value
        return self.f < other.f

class BITStarPlanner(BasePlanner):
    """
    Batch Informed Trees (BIT*) algorithm for path planning.
    
    BIT* unifies graph-based and sampling-based planning by viewing samples as
    defining an implicit random geometric graph (RGG). It uses a heuristic to 
    efficiently search a series of increasingly dense RGGs while reusing previous
    information.
    
    Key features:
    - Combines benefits of graph-search and sampling-based algorithms
    - Efficiently searches implicit random geometric graphs
    - Reuses information between batches of samples
    - Focuses search towards the goal using a heuristic
    - Anytime performance with asymptotic optimality
    - Excellent performance in high-dimensional spaces
    
    Reference:
    Gammell, J. D., Srinivasa, S. S., & Barfoot, T. D. (2015).
    "Batch Informed Trees (BIT*): Sampling-based optimal planning via the
    heuristically guided search of implicit random geometric graphs."
    In IEEE International Conference on Robotics and Automation (ICRA).
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.samples = []            # List of all samples
        self.vertices = []           # List of all vertices in the graph
        self.start_node = None       # Start node
        self.goal_node = None        # Goal node
        self.queue_edges = []        # Priority queue of edges
        self.queue_vertices = []     # Priority queue of vertices
        self.batch_size = 100        # Number of samples per batch
        self.max_batches = 10        # Maximum number of batches
        self.r = float('inf')        # Connection radius
        self.r_factor = 1.1          # Factor for connection radius growth
        self.goal_sample_rate = 0.05 # Probability of sampling the goal
        self.focus_search = True     # Whether to focus the search
        self.rewire_factor = 1.1     # Factor for rewiring
        self.epsilon = 0.001         # Small value for floating-point comparison
        self.c_best = float('inf')   # Cost of best solution
        self.current_batch = 0       # Current batch number
        self.sample_ellipsoid = False # Whether to sample from an ellipsoid
        self.ellipse_center = None   # Center of sampling ellipsoid
        self.ellipse_axes = None     # Axes of sampling ellipsoid
        self.ellipse_rotation = None # Rotation matrix for ellipsoid
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def estimate_free_space_volume(self):
        """
        Estimate the volume of free space in the world.
        A simple approximation based on the world dimensions.
        """
        # Assume 70% of the world is free space (not obstacles)
        return 0.7 * self.world.width * self.world.height
    
    def calculate_connection_radius(self, n):
        """
        Calculate the connection radius based on the number of samples.
        Uses the theory of random geometric graphs to ensure asymptotic optimality.
        """
        # Constants for the RGG theory
        d = 2  # 2D space
        free_space_volume = self.estimate_free_space_volume()
        
        # Calculate radius based on RGG theory
        gamma = 2 * (1 + 1/d) * (free_space_volume / math.pi) ** (1/d)
        radius = gamma * (math.log(n) / n) ** (1/d)
        
        # Scale radius by rewire factor
        radius *= self.rewire_factor
        
        return radius
    
    def is_collision_free(self, pos1, pos2):
        """
        Check if the path between pos1 and pos2 is collision-free.
        """
        # Calculate distance and number of steps to check
        dist = self.distance(pos1, pos2)
        steps = max(int(dist * 2), 5)  # At least 5 checks along the line
        
        # Check points along the line
        for i in range(steps + 1):
            t = i / steps
            x = int(round(pos1[0] * (1 - t) + pos2[0] * t))
            y = int(round(pos1[1] * (1 - t) + pos2[1] * t))
            
            if not self.is_valid_point((x, y)):
                return False
        
        return True
    
    def get_neighbors(self, node, radius):
        """Get all vertices within radius of node."""
        return [v for v in self.vertices if self.distance(node.position, v.position) <= radius and v != node]
    
    def get_edge_cost(self, pos1, pos2):
        """Calculate the cost of an edge between two positions."""
        return self.distance(pos1, pos2)
    
    def get_heuristic(self, pos):
        """
        Calculate the heuristic (estimated cost-to-go) from a position to the goal.
        Uses Euclidean distance as an admissible heuristic.
        """
        return self.distance(pos, self.goal_node.position)
    
    def setup_ellipsoid(self):
        """
        Set up the sampling ellipsoid based on the current best solution.
        The ellipsoid is defined such that it contains all paths that could
        potentially improve the current best solution.
        """
        if self.c_best == float('inf') or not self.focus_search:
            self.sample_ellipsoid = False
            return
        
        # Set up ellipsoid for focused sampling
        self.sample_ellipsoid = True
        
        # Calculate center of the ellipsoid
        start_pos = self.start_node.position
        goal_pos = self.goal_node.position
        self.ellipse_center = ((start_pos[0] + goal_pos[0]) / 2, 
                              (start_pos[1] + goal_pos[1]) / 2)
        
        # Calculate the minimum distance between start and goal
        c_min = self.distance(start_pos, goal_pos)
        
        # Calculate the semi-major axis (half the length of the ellipsoid)
        a = self.c_best / 2
        
        # Calculate the semi-minor axis
        b = math.sqrt(self.c_best**2 - c_min**2) / 2
        
        self.ellipse_axes = (a, b)
        
        # Calculate rotation of the ellipsoid
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        theta = math.atan2(dy, dx)
        
        # Create rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        self.ellipse_rotation = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
    
    def sample_from_ellipsoid(self):
        """
        Sample a point from the informed ellipsoidal region.
        This focuses sampling to the region that could potentially improve
        the current best path.
        """
        # Sample from a unit circle
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
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
    
    def sample_free(self):
        """
        Sample a collision-free point.
        If a current solution exists and focus_search is True,
        sample from the ellipsoidal region.
        """
        if self.sample_ellipsoid:
            # Sample from the ellipsoid
            while True:
                pos = self.sample_from_ellipsoid()
                if self.is_valid_point(pos):
                    return pos
        else:
            # Regular sampling
            while True:
                x = np.random.randint(0, self.world.width)
                y = np.random.randint(0, self.world.height)
                pos = (x, y)
                if self.is_valid_point(pos):
                    return pos
    
    def sample_batch(self):
        """
        Sample a batch of configurations.
        Includes goal sampling with probability goal_sample_rate.
        """
        new_samples = []
        
        # Goal sampling
        if np.random.random() < self.goal_sample_rate and self.distance(self.goal_node.position, self.start_node.position) > self.epsilon:
            new_samples.append(self.goal_node.position)
        
        # Sample remaining points
        for _ in range(self.batch_size - len(new_samples)):
            new_samples.append(self.sample_free())
        
        return new_samples
    
    def prune(self):
        """
        Prune the graph to remove vertices that cannot improve the solution.
        This removes vertices with f-value greater than c_best.
        """
        if self.c_best == float('inf'):
            return
        
        pruned_vertices = []
        
        for v in self.vertices:
            # Keep vertices in the tree
            if v.in_tree:
                pruned_vertices.append(v)
                continue
            
            # Calculate f-value
            v.h = self.get_heuristic(v.position)
            v.f = v.g + v.h
            
            # Keep vertices that could potentially improve the solution
            if v.f < self.c_best - self.epsilon:
                pruned_vertices.append(v)
        
        self.vertices = pruned_vertices
    
    def expand_vertex(self, v):
        """
        Process a vertex by adding edges to the queue.
        """
        # Skip if vertex is not in the tree
        if not v.in_tree:
            return
        
        # Find neighbors within the connection radius
        neighbors = self.get_neighbors(v, self.r)
        
        # Process each neighbor
        for u in neighbors:
            # Skip if neighbor is in the tree and is a child of v
            if u.in_tree and u in v.children:
                continue
            
            # Skip if neighbor is in the tree and is an ancestor of v
            if u.in_tree and self.is_ancestor(u, v):
                continue
            
            # Calculate edge cost
            edge_cost = self.get_edge_cost(v.position, u.position)
            
            # Skip if the edge is too long
            if edge_cost > self.r:
                continue
            
            # Compute total cost through this vertex
            g_new = v.g + edge_cost
            
            # Skip if this would not improve the path to u
            if u.in_tree and g_new >= u.g:
                continue
            
            # Check if the path is collision-free
            if not self.is_collision_free(v.position, u.position):
                continue
            
            # Create edge and add to queue
            edge = Edge(v, u, v.g, edge_cost, u.h)
            heapq.heappush(self.queue_edges, edge)
    
    def is_ancestor(self, node, potential_descendant):
        """
        Check if node is an ancestor of potential_descendant in the tree.
        """
        current = potential_descendant.parent
        while current is not None:
            if current == node:
                return True
            current = current.parent
        return False
    
    def add_to_tree(self, parent, child, edge_cost):
        """
        Add a child node to the tree with the given parent.
        Updates the costs and parent-child relationships.
        """
        # Remove old parent-child relationship
        if child.in_tree and child.parent is not None:
            child.parent.children.remove(child)
        
        # Update child information
        child.parent = parent
        child.g = parent.g + edge_cost
        child.f = child.g + child.h
        child.in_tree = True
        
        # Update parent information
        parent.children.add(child)
        
        # Check for improved path to goal
        if child == self.goal_node and child.g < self.c_best:
            self.c_best = child.g
            self.setup_ellipsoid()
    
    def get_path(self):
        """
        Extract the path from start to goal.
        """
        if not self.goal_node.in_tree:
            return None
        
        path = []
        current = self.goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        return path[::-1]  # Reverse to get path from start to goal
    
    def initialize(self, start, goal):
        """
        Initialize the BIT* algorithm with start and goal positions.
        """
        # Clear the graph
        self.samples = []
        self.vertices = []
        self.queue_edges = []
        self.queue_vertices = []
        self.c_best = float('inf')
        self.current_batch = 0
        
        # Create start and goal nodes
        self.start_node = BITNode(start)
        self.goal_node = BITNode(goal)
        
        # Initialize start node
        self.start_node.g = 0
        self.start_node.h = self.get_heuristic(start)
        self.start_node.f = self.start_node.h
        self.start_node.in_tree = True
        self.start_node.sample_batch = 0
        
        # Initialize goal node
        self.goal_node.h = 0
        self.goal_node.sample_batch = 0
        
        # Add start and goal to vertices
        self.vertices = [self.start_node, self.goal_node]
        
        # Initialize the vertex queue with just the start node
        heapq.heappush(self.queue_vertices, self.start_node)
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using BIT*.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            batch_size: Optional number of samples per batch.
            max_batches: Optional maximum number of batches.
            rewire_factor: Optional factor for rewiring.
            focus_search: Optional boolean to focus search using an ellipsoid.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.max_batches = kwargs.get('max_batches', self.max_batches)
        self.rewire_factor = kwargs.get('rewire_factor', self.rewire_factor)
        self.focus_search = kwargs.get('focus_search', self.focus_search)
        
        # Initialize
        self.initialize(start, goal)
        
        # Main loop - process batches
        for batch in range(self.max_batches):
            self.current_batch = batch
            
            # Sample a new batch
            new_samples = self.sample_batch()
            
            # Add new samples as vertices
            for sample in new_samples:
                # Skip duplicates
                if any(self.distance(sample, v.position) < self.epsilon for v in self.vertices):
                    continue
                
                node = BITNode(sample)
                node.h = self.get_heuristic(sample)
                node.sample_batch = batch
                self.vertices.append(node)
            
            # Update connection radius
            n = len(self.vertices)
            self.r = self.calculate_connection_radius(n)
            
            # Prune the graph
            self.prune()
            
            # Clear the queues
            self.queue_edges = []
            self.queue_vertices = [self.start_node]
            
            # Process the graph
            while (self.queue_vertices or self.queue_edges) and \
                  (not self.queue_edges or self.queue_vertices and self.queue_vertices[0].f <= self.queue_edges[0].f):
                
                # Get the next vertex
                if self.queue_vertices:
                    v = heapq.heappop(self.queue_vertices)
                    self.expand_vertex(v)
                
                # Process the next edge if the queue is not empty and the best vertex has higher cost
                while self.queue_edges and (not self.queue_vertices or self.queue_edges[0].f <= self.queue_vertices[0].f):
                    # Get the next edge
                    edge = heapq.heappop(self.queue_edges)
                    
                    # Skip if this edge would not improve the current tree
                    if edge.end.in_tree and edge.g + edge.c >= edge.end.g:
                        continue
                    
                    # Add the edge to the tree
                    self.add_to_tree(edge.start, edge.end, edge.c)
                    
                    # Add the end vertex to the queue
                    heapq.heappush(self.queue_vertices, edge.end)
                    
                    # If this edge improved the solution, update c_best and setup ellipsoid
                    if edge.end == self.goal_node and edge.g + edge.c < self.c_best:
                        self.c_best = edge.g + edge.c
                        self.setup_ellipsoid()
            
            # If we found a solution and the batch did not improve it, we can stop
            if batch > 0 and self.goal_node.in_tree and self.goal_node.sample_batch < batch:
                break
        
        # Return the final path
        return self.get_path()
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of BIT* that can provide step-by-step visualization.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            callback: Function to call after each batch with current tree and path.
            **kwargs: Additional parameters.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Update parameters if provided
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.max_batches = kwargs.get('max_batches', self.max_batches)
        self.rewire_factor = kwargs.get('rewire_factor', self.rewire_factor)
        self.focus_search = kwargs.get('focus_search', self.focus_search)
        
        # Initialize
        self.initialize(start, goal)
        
        # For tracking progress and visualization
        total_edges_processed = 0
        total_iterations = 0
        
        # Call callback with initial state
        if callback:
            print("Calling initial callback")
            callback(self.vertices, None, 0, "Initial state")
            total_iterations += 1
        
        # Main loop - process batches
        for batch in range(self.max_batches):
            self.current_batch = batch
            
            # Sample a new batch
            new_samples = self.sample_batch()
            
            # Add new samples as vertices
            for sample in new_samples:
                # Skip duplicates
                if any(self.distance(sample, v.position) < self.epsilon for v in self.vertices):
                    continue
                
                node = BITNode(sample)
                node.h = self.get_heuristic(sample)
                node.sample_batch = batch
                self.vertices.append(node)
            
            # Update connection radius
            n = len(self.vertices)
            self.r = self.calculate_connection_radius(n)
            
            # Prune the graph
            self.prune()
            
            # Clear the queues
            self.queue_edges = []
            self.queue_vertices = [self.start_node]
            
            # Call callback after adding new samples
            if callback:
                print(f"Calling sample callback for batch {batch}")
                callback(self.vertices, self.get_path(), total_iterations,
                       f"Batch {batch}: Added {len(new_samples)} samples")
                total_iterations += 1
            
            # Process the graph
            edge_count = 0
            
            while (self.queue_vertices or self.queue_edges) and \
                  (not self.queue_edges or self.queue_vertices and self.queue_vertices[0].f <= self.queue_edges[0].f):
                
                # Get the next vertex
                if self.queue_vertices:
                    v = heapq.heappop(self.queue_vertices)
                    self.expand_vertex(v)
                
                # Process the next edge if the queue is not empty and the best vertex has higher cost
                edge_processing_count = 0
                max_edges_per_callback = 5  # Process fewer edges before each callback
                
                while self.queue_edges and (not self.queue_vertices or self.queue_edges[0].f <= self.queue_vertices[0].f):
                    # Get the next edge
                    edge = heapq.heappop(self.queue_edges)
                    
                    # Skip if this edge would not improve the current tree
                    if edge.end.in_tree and edge.g + edge.c >= edge.end.g:
                        continue
                    
                    # Add the edge to the tree
                    self.add_to_tree(edge.start, edge.end, edge.c)
                    
                    # Add the end vertex to the queue
                    heapq.heappush(self.queue_vertices, edge.end)
                    
                    edge_count += 1
                    total_edges_processed += 1
                    edge_processing_count += 1
                    
                    # Call callback more frequently during edge processing
                    if callback and edge_processing_count >= max_edges_per_callback:
                        print(f"Calling edge processing callback at edge {edge_count}")
                        current_path = self.get_path()
                        callback(self.vertices, current_path, total_iterations,
                               f"Batch {batch}: Processing edges ({edge_count} processed)")
                        total_iterations += 1
                        edge_processing_count = 0
                
                # Periodically call callback during vertex processing too
                if callback and edge_count % 20 == 0:
                    print(f"Calling periodic callback during vertex processing")
                    current_path = self.get_path()
                    callback(self.vertices, current_path, total_iterations, 
                           f"Batch {batch}: Vertex expansion")
                    total_iterations += 1
            
            # Call callback after processing all edges in this batch
            if callback:
                print(f"Calling batch completion callback for batch {batch}")
                current_path = self.get_path()
                callback(self.vertices, current_path, total_iterations,
                       f"Batch {batch}: Completed ({edge_count} edges processed)")
                total_iterations += 1
                
            # Check if we have a solution
            if self.goal_node.in_tree:
                # Return the solution path
                path = self.get_path()
                
                # Final callback
                if callback:
                    print("Calling solution found callback")
                    callback(self.vertices, path, total_iterations, 
                           f"Solution found after {batch+1} batches")
                    total_iterations += 1
                
                return path
        
        # No solution found
        if callback:
            print("Calling final no solution callback")
            callback(self.vertices, None, total_iterations, "No solution found")
        
        return None