import numpy as np
import random
import time
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode

class DynamicRRTPlanner(BasePlanner):
    """
    Dynamic Rapidly-exploring Random Tree (Dynamic-RRT) algorithm for path planning.
    
    Dynamic-RRT extends RRT to handle dynamic environments where obstacles may move
    or new obstacles may appear. It efficiently repairs the existing tree when changes
    occur in the environment.
    
    Key features:
    - Efficient tree repair when environment changes
    - Maintains a valid tree structure at all times
    - Can quickly replan paths when obstacles move
    - Supports incremental planning
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.nodes = []
        self.step_size = 1.0
        self.max_iterations = 1000
        self.goal_sample_rate = 0.1
        self.goal_threshold = 1.0
        self.prune_threshold = 0.5  # Threshold for pruning invalid nodes
        self.rewire_radius = 5.0    # Radius for rewiring after pruning
        self.changed_obstacles = []  # List of changed obstacle positions
    
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
    
    def prune_invalid_nodes(self):
        """
        Prune nodes that are no longer valid due to environment changes.
        Returns a list of valid nodes.
        """
        valid_nodes = []
        root_node = self.nodes[0]  # Start node is always the root
        valid_nodes.append(root_node)
        
        # BFS to check and prune invalid branches
        queue = [root_node]
        visited = {root_node.position}
        
        while queue:
            current = queue.pop(0)
            
            # Find children of current node
            children = [node for node in self.nodes if node.parent == current]
            
            for child in children:
                # Check if edge is still valid
                if self.is_collision_free(current.position, child.position):
                    valid_nodes.append(child)
                    if child.position not in visited:
                        queue.append(child)
                        visited.add(child.position)
        
        return valid_nodes
    
    def rewire_tree(self, valid_nodes):
        """
        Rewire the tree after pruning to maintain connectivity.
        Attempts to reconnect orphaned subtrees.
        """
        # Identify orphaned nodes (nodes not in valid_nodes)
        orphaned_nodes = [node for node in self.nodes if node not in valid_nodes]
        
        if not orphaned_nodes:
            return valid_nodes
        
        # Try to reconnect orphaned nodes to valid nodes
        for orphan in orphaned_nodes:
            # Find valid nodes within rewire radius
            nearby_valid = [node for node in valid_nodes 
                           if self.distance(node.position, orphan.position) <= self.rewire_radius]
            
            if nearby_valid:
                # Find closest valid node that can be connected to orphan
                valid_connections = []
                for valid_node in nearby_valid:
                    if self.is_collision_free(valid_node.position, orphan.position):
                        dist = self.distance(valid_node.position, orphan.position)
                        valid_connections.append((valid_node, dist))
                
                if valid_connections:
                    # Connect to closest valid node
                    valid_connections.sort(key=lambda x: x[1])
                    best_valid, _ = valid_connections[0]
                    
                    # Update orphan's parent
                    orphan.parent = best_valid
                    orphan.cost = best_valid.cost + self.distance(best_valid.position, orphan.position)
                    valid_nodes.append(orphan)
                    
                    # Recursively try to reconnect orphan's children
                    orphan_children = [node for node in orphaned_nodes if node.parent == orphan]
                    for child in orphan_children:
                        if self.is_collision_free(orphan.position, child.position):
                            child.cost = orphan.cost + self.distance(orphan.position, child.position)
                            valid_nodes.append(child)
        
        return valid_nodes
    
    def update_tree(self, changed_obstacles):
        """
        Update the tree when obstacles change.
        Prunes invalid branches and attempts to rewire the tree.
        """
        self.changed_obstacles = changed_obstacles
        
        # Prune invalid nodes
        valid_nodes = self.prune_invalid_nodes()
        
        # Rewire tree to reconnect valid orphaned subtrees
        self.nodes = self.rewire_tree(valid_nodes)
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Dynamic-RRT.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            step_size: Optional step size for extending the tree.
            max_iterations: Optional maximum number of iterations.
            goal_sample_rate: Optional probability of sampling the goal.
            goal_threshold: Optional distance threshold to consider goal reached.
            changed_obstacles: Optional list of changed obstacle positions.
            
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
        self.prune_threshold = kwargs.get('prune_threshold', self.prune_threshold)
        self.rewire_radius = kwargs.get('rewire_radius', self.rewire_radius)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Get changed obstacles if provided
        changed_obstacles = kwargs.get('changed_obstacles', [])
        
        # Initialize tree if empty or update existing tree
        if not self.nodes:
            # Initialize tree with start node
            start_node = RRTNode(start)
            self.nodes = [start_node]
        else:
            # Update tree based on changed obstacles
            if changed_obstacles:
                self.update_tree(changed_obstacles)
            
            # Check if start node needs to be updated
            if self.nodes[0].position != start:
                # Create new start node
                start_node = RRTNode(start)
                self.nodes.insert(0, start_node)
        
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
        Interactive version of Dynamic-RRT that can provide step-by-step visualization.
        
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
        self.prune_threshold = kwargs.get('prune_threshold', self.prune_threshold)
        self.rewire_radius = kwargs.get('rewire_radius', self.rewire_radius)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Get changed obstacles if provided
        changed_obstacles = kwargs.get('changed_obstacles', [])
        
        # Initialize tree if empty or update existing tree
        if not self.nodes:
            # Initialize tree with start node
            start_node = RRTNode(start)
            self.nodes = [start_node]
        else:
            # Update tree based on changed obstacles
            if changed_obstacles:
                self.update_tree(changed_obstacles)
                
                # Call callback after tree update
                if callback:
                    callback(self.nodes, None, 0, "Tree updated due to obstacle changes")
            
            # Check if start node needs to be updated
            if self.nodes[0].position != start:
                # Create new start node
                start_node = RRTNode(start)
                self.nodes.insert(0, start_node)
        
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
                    current_path = path
                
                # Call callback with current state
                if callback:
                    callback(self.nodes, current_path, i + 1, "Tree growing")
                
                # Return path if goal is reached
                if current_path:
                    return current_path
        
        # No path found
        if callback:
            callback(self.nodes, None, self.max_iterations, "Max iterations reached")
        
        return None 