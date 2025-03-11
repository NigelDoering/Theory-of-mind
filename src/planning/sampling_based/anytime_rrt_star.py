import numpy as np
import random
import time
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode
from src.planning.sampling_based.rrt_star import RRTStarPlanner

class AnytimeRRTStarPlanner(RRTStarPlanner):
    """
    Anytime RRT* algorithm for path planning.
    
    Anytime RRT* is a variant of RRT* that can provide a valid solution at any time
    during execution, with the solution quality improving over time. It's particularly
    useful for real-time applications where a quick initial solution is needed,
    which can then be refined as more computation time is available.
    
    Key features:
    - Provides initial solutions quickly
    - Continuously improves solution quality over time
    - Can be interrupted at any time to return the best solution found so far
    - Maintains all optimality guarantees of RRT*
    
    Reference:
    Ferguson, D., & Stentz, A. (2006, May).
    "Anytime RRTs."
    In Proceedings 2006 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 5369-5375). IEEE.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.time_limit = None  # Time limit in seconds (None for no limit)
        self.improvement_threshold = 0.05  # Minimum improvement ratio to restart
        self.restart_freq = 100  # Frequency of checking for restarts
        self.max_restarts = 5  # Maximum number of restarts
        self.solution_buffer = []  # Buffer to store solutions
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Anytime RRT*.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            step_size: Optional step size for extending the tree.
            max_iterations: Optional maximum number of iterations.
            goal_sample_rate: Optional probability of sampling the goal.
            goal_threshold: Optional distance threshold to consider goal reached.
            search_radius: Optional radius for nearest neighbors search.
            rewire_factor: Optional factor to scale search radius.
            time_limit: Optional time limit in seconds.
            improvement_threshold: Optional minimum improvement ratio to restart.
            restart_freq: Optional frequency of checking for restarts.
            max_restarts: Optional maximum number of restarts.
            
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
        self.time_limit = kwargs.get('time_limit', self.time_limit)
        self.improvement_threshold = kwargs.get('improvement_threshold', self.improvement_threshold)
        self.restart_freq = kwargs.get('restart_freq', self.restart_freq)
        self.max_restarts = kwargs.get('max_restarts', self.max_restarts)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Clear solution buffer
        self.solution_buffer = []
        
        # Start timer if time limit is set
        start_time = time.time()
        
        # Track best solution
        best_path = None
        best_cost = float('inf')
        
        # Perform multiple planning iterations with restarts
        restarts = 0
        while restarts <= self.max_restarts:
            # Check time limit
            if self.time_limit and time.time() - start_time > self.time_limit:
                break
            
            # Initialize tree with start node
            self.nodes = []
            start_node = RRTNode(start)
            start_node.cost = 0.0
            self.nodes.append(start_node)
            
            # Track best goal node in this iteration
            best_goal_node = None
            best_goal_cost = float('inf')
            
            # Main loop
            for i in range(self.max_iterations):
                # Check time limit
                if self.time_limit and time.time() - start_time > self.time_limit:
                    break
                
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
                        
                        # Construct path
                        path = [goal]
                        current = best_goal_node
                        while current.parent is not None:
                            path.append(current.position)
                            current = current.parent
                        path.append(start)
                        path.reverse()
                        
                        # Add to solution buffer
                        self.solution_buffer.append((path, goal_cost))
                        
                        # Update best solution
                        if goal_cost < best_cost:
                            best_path = path
                            best_cost = goal_cost
                
                # Check if we should restart
                if i > 0 and i % self.restart_freq == 0 and best_goal_node is not None:
                    # Calculate improvement ratio
                    if len(self.solution_buffer) >= 2:
                        prev_cost = self.solution_buffer[-2][1]
                        curr_cost = self.solution_buffer[-1][1]
                        improvement = (prev_cost - curr_cost) / prev_cost
                        
                        # If improvement is below threshold, restart
                        if improvement < self.improvement_threshold:
                            break
            
            # Increment restart counter
            restarts += 1
        
        # Return best path found
        return best_path
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of Anytime RRT* that can provide step-by-step visualization.
        
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
        self.time_limit = kwargs.get('time_limit', self.time_limit)
        self.improvement_threshold = kwargs.get('improvement_threshold', self.improvement_threshold)
        self.restart_freq = kwargs.get('restart_freq', self.restart_freq)
        self.max_restarts = kwargs.get('max_restarts', self.max_restarts)
        
        # Store goal for random sampling
        self.goal = goal
        
        # Clear solution buffer
        self.solution_buffer = []
        
        # Start timer if time limit is set
        start_time = time.time()
        
        # Track best solution
        best_path = None
        best_cost = float('inf')
        
        # Call callback with initial state
        if callback:
            callback([], None, 0, "Starting Anytime RRT*")
        
        # Perform multiple planning iterations with restarts
        restarts = 0
        while restarts <= self.max_restarts:
            # Check time limit
            if self.time_limit and time.time() - start_time > self.time_limit:
                break
            
            # Initialize tree with start node
            self.nodes = []
            start_node = RRTNode(start)
            start_node.cost = 0.0
            self.nodes.append(start_node)
            
            # Track best goal node in this iteration
            best_goal_node = None
            best_goal_cost = float('inf')
            
            # Call callback at start of restart
            if callback:
                callback(self.nodes, best_path, restarts, f"Restart {restarts}")
            
            # Main loop
            for i in range(self.max_iterations):
                # Check time limit
                if self.time_limit and time.time() - start_time > self.time_limit:
                    break
                
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
                        
                        # Construct path
                        path = [goal]
                        current = best_goal_node
                        while current.parent is not None:
                            path.append(current.position)
                            current = current.parent
                        path.append(start)
                        path.reverse()
                        
                        # Add to solution buffer
                        self.solution_buffer.append((path, goal_cost))
                        current_path = path
                        
                        # Update best solution
                        if goal_cost < best_cost:
                            best_path = path
                            best_cost = goal_cost
                
                # Call callback periodically
                if callback and i % 10 == 0:
                    callback(self.nodes, best_path, restarts * self.max_iterations + i, 
                             f"Restart {restarts}, Iteration {i}, Best cost: {best_cost:.2f}")
                
                # Check if we should restart
                if i > 0 and i % self.restart_freq == 0 and best_goal_node is not None:
                    # Calculate improvement ratio
                    if len(self.solution_buffer) >= 2:
                        prev_cost = self.solution_buffer[-2][1]
                        curr_cost = self.solution_buffer[-1][1]
                        improvement = (prev_cost - curr_cost) / prev_cost
                        
                        # If improvement is below threshold, restart
                        if improvement < self.improvement_threshold:
                            if callback:
                                callback(self.nodes, best_path, restarts * self.max_iterations + i,
                                         f"Restarting due to low improvement: {improvement:.4f}")
                            break
            
            # Increment restart counter
            restarts += 1
        
        # Call callback with final state
        if callback:
            callback(self.nodes, best_path, restarts * self.max_iterations,
                     f"Final solution, Cost: {best_cost:.2f}, Restarts: {restarts}")
        
        # Return best path found
        return best_path 