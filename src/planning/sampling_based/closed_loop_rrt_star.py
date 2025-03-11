import numpy as np
import random
import math
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode
from src.planning.sampling_based.rrt_star import RRTStarPlanner

class VehicleState:
    """
    Class representing the state of a vehicle for closed-loop planning.
    Includes position, orientation, and other dynamic properties.
    """
    def __init__(self, x, y, theta=0.0, v=0.0, omega=0.0):
        self.x = x          # x-coordinate
        self.y = y          # y-coordinate
        self.theta = theta  # orientation (radians)
        self.v = v          # linear velocity
        self.omega = omega  # angular velocity
    
    def copy(self):
        """Create a copy of this state."""
        return VehicleState(self.x, self.y, self.theta, self.v, self.omega)
    
    def to_tuple(self):
        """Convert state to tuple for position-based operations."""
        return (int(round(self.x)), int(round(self.y)))

class ControlInput:
    """
    Class representing a control input for the vehicle model.
    """
    def __init__(self, v, omega, duration=1.0):
        self.v = v              # linear velocity
        self.omega = omega      # angular velocity
        self.duration = duration  # duration to apply this control
    
    def copy(self):
        """Create a copy of this control input."""
        return ControlInput(self.v, self.omega, self.duration)

class CLRRTStarNode(RRTNode):
    """
    Node class for Closed-Loop RRT* algorithm.
    Extends RRTNode with state and control information.
    """
    def __init__(self, state):
        super().__init__((int(round(state.x)), int(round(state.y))))
        self.state = state.copy()  # Full state including orientation and velocities
        self.control = None  # Control input that led to this state
        self.trajectory = []  # Sequence of states from parent to this node

class ClosedLoopRRTStarPlanner(BasePlanner):
    """
    Closed-Loop RRT* algorithm for path planning.
    
    Closed-Loop RRT* extends RRT* by incorporating vehicle dynamics and control
    constraints. Instead of connecting nodes with straight lines, it simulates
    the vehicle's motion under various control inputs to generate feasible trajectories.
    
    Key features:
    - Incorporates vehicle dynamics and control constraints
    - Generates dynamically feasible trajectories
    - Maintains optimality guarantees of RRT*
    - Suitable for non-holonomic vehicles
    
    Reference:
    Kuwata, Y., Teo, J., Fiore, G., Karaman, S., Frazzoli, E., & How, J. P. (2009).
    "Real-time motion planning with applications to autonomous urban driving."
    IEEE Transactions on Control Systems Technology, 17(5), 1105-1118.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.nodes = []
        self.max_iterations = 1000
        self.goal_sample_rate = 0.1
        self.goal_threshold = 2.0
        self.search_radius = 10.0
        self.rewire_factor = 2.0
        self.step_size = 5.0
        
        # Vehicle parameters
        self.max_linear_velocity = 2.0
        self.max_angular_velocity = 0.5
        self.min_linear_velocity = 0.0
        self.min_angular_velocity = -0.5
        self.control_steps = 5  # Number of discrete control inputs to sample
        self.simulation_dt = 0.1  # Time step for simulation
        self.simulation_steps = 10  # Number of steps to simulate for each control
        
        # Generate control set
        self.control_set = self.generate_control_set()
    
    def generate_control_set(self):
        """
        Generate a set of control inputs to sample from.
        Returns a list of ControlInput objects.
        """
        control_set = []
        
        # Linear velocity values
        v_values = np.linspace(self.min_linear_velocity, self.max_linear_velocity, self.control_steps)
        
        # Angular velocity values
        omega_values = np.linspace(self.min_angular_velocity, self.max_angular_velocity, self.control_steps)
        
        # Create all combinations
        for v in v_values:
            for omega in omega_values:
                # Skip zero velocity if omega is not zero (can't turn in place)
                if abs(v) < 1e-6 and abs(omega) > 1e-6:
                    continue
                
                control_set.append(ControlInput(v, omega, self.simulation_dt * self.simulation_steps))
        
        return control_set
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def state_distance(self, state1, state2):
        """
        Calculate distance between two states.
        Incorporates position and orientation differences.
        """
        pos_dist = np.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
        angle_dist = abs(self.normalize_angle(state1.theta - state2.theta))
        
        # Weight position and orientation differences
        return pos_dist + 0.5 * angle_dist
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def nearest_node(self, state):
        """Find the nearest node in the tree to the given state."""
        return min(self.nodes, key=lambda node: self.state_distance(node.state, state))
    
    def near_nodes(self, state, radius):
        """Find all nodes within a certain radius of the given state."""
        return [node for node in self.nodes 
                if self.state_distance(node.state, state) <= radius]
    
    def simulate_dynamics(self, state, control):
        """
        Simulate vehicle dynamics under the given control input.
        Returns a new state after applying the control for one time step.
        """
        # Simple bicycle model
        new_state = state.copy()
        
        # Update position and orientation
        new_state.x += control.v * np.cos(state.theta) * self.simulation_dt
        new_state.y += control.v * np.sin(state.theta) * self.simulation_dt
        new_state.theta += control.omega * self.simulation_dt
        new_state.theta = self.normalize_angle(new_state.theta)
        
        # Update velocities
        new_state.v = control.v
        new_state.omega = control.omega
        
        return new_state
    
    def simulate_trajectory(self, start_state, control):
        """
        Simulate a trajectory by applying a control input for multiple time steps.
        Returns a list of states representing the trajectory.
        """
        trajectory = [start_state.copy()]
        current_state = start_state.copy()
        
        for _ in range(self.simulation_steps):
            current_state = self.simulate_dynamics(current_state, control)
            
            # Check if the new state is valid
            if not self.is_valid_point(current_state.to_tuple()):
                return None  # Collision detected
            
            trajectory.append(current_state.copy())
        
        return trajectory
    
    def steer(self, from_state, to_state):
        """
        Steer from from_state towards to_state by selecting the best control input.
        Returns a tuple of (new_state, control, trajectory) or None if no valid control is found.
        """
        best_control = None
        best_trajectory = None
        best_final_state = None
        best_distance = float('inf')
        
        # Try each control input
        for control in self.control_set:
            # Simulate trajectory
            trajectory = self.simulate_trajectory(from_state, control)
            
            # Skip if trajectory is invalid (collision)
            if trajectory is None:
                continue
            
            # Get final state
            final_state = trajectory[-1]
            
            # Calculate distance to target state
            distance = self.state_distance(final_state, to_state)
            
            # Update best if this is better
            if distance < best_distance:
                best_distance = distance
                best_control = control
                best_trajectory = trajectory
                best_final_state = final_state
        
        if best_control is None:
            return None  # No valid control found
        
        return (best_final_state, best_control, best_trajectory)
    
    def is_collision_free(self, trajectory):
        """
        Check if a trajectory is collision-free.
        """
        if trajectory is None:
            return False
        
        for state in trajectory:
            if not self.is_valid_point(state.to_tuple()):
                return False
        
        return True
    
    def random_state(self):
        """Generate a random state within the world bounds."""
        if random.random() < self.goal_sample_rate:
            # Sample the goal with probability goal_sample_rate
            return self.goal_state
        else:
            # Sample a random position
            x = random.randint(0, self.world.width - 1)
            y = random.randint(0, self.world.height - 1)
            theta = random.uniform(-np.pi, np.pi)
            return VehicleState(x, y, theta)
    
    def calculate_cost(self, node):
        """Calculate the cost of a node by summing up the trajectory lengths."""
        cost = 0.0
        current = node
        
        while current.parent is not None:
            # Add trajectory length
            for i in range(len(current.trajectory) - 1):
                state1 = current.trajectory[i]
                state2 = current.trajectory[i + 1]
                cost += self.distance(state1.to_tuple(), state2.to_tuple())
            
            current = current.parent
        
        return cost
    
    def choose_parent(self, near_nodes, new_state, new_trajectory):
        """
        Choose the best parent for a new node from a list of near nodes.
        Returns the best parent node and the associated control, trajectory, and cost.
        """
        if not near_nodes:
            return None, None, None, float('inf')
        
        # Find the node that would give the lowest cost
        min_cost = float('inf')
        best_parent = None
        best_control = None
        best_trajectory = None
        
        for node in near_nodes:
            # Try to steer from this node to the new state
            result = self.steer(node.state, new_state)
            
            if result is None:
                continue  # No valid control found
            
            final_state, control, trajectory = result
            
            # Check if the trajectory is collision-free
            if not self.is_collision_free(trajectory):
                continue
            
            # Calculate cost through this node
            trajectory_cost = 0.0
            for i in range(len(trajectory) - 1):
                state1 = trajectory[i]
                state2 = trajectory[i + 1]
                trajectory_cost += self.distance(state1.to_tuple(), state2.to_tuple())
            
            total_cost = node.cost + trajectory_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_parent = node
                best_control = control
                best_trajectory = trajectory
        
        return best_parent, best_control, best_trajectory, min_cost
    
    def rewire(self, new_node, near_nodes):
        """
        Rewire the tree by checking if paths through the new node are better.
        Updates parent pointers and costs for affected nodes.
        """
        for node in near_nodes:
            # Skip the parent of the new node
            if node == new_node.parent:
                continue
            
            # Try to steer from new_node to this node
            result = self.steer(new_node.state, node.state)
            
            if result is None:
                continue  # No valid control found
            
            final_state, control, trajectory = result
            
            # Check if the trajectory is collision-free
            if not self.is_collision_free(trajectory):
                continue
            
            # Calculate cost through new_node
            trajectory_cost = 0.0
            for i in range(len(trajectory) - 1):
                state1 = trajectory[i]
                state2 = trajectory[i + 1]
                trajectory_cost += self.distance(state1.to_tuple(), state2.to_tuple())
            
            new_cost = new_node.cost + trajectory_cost
            
            # If new path is better, rewire
            if new_cost < node.cost:
                node.parent = new_node
                node.cost = new_cost
                node.control = control
                node.trajectory = trajectory
                
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
            # Calculate trajectory cost
            trajectory_cost = 0.0
            for i in range(len(child.trajectory) - 1):
                state1 = child.trajectory[i]
                state2 = child.trajectory[i + 1]
                trajectory_cost += self.distance(state1.to_tuple(), state2.to_tuple())
            
            # Update child's cost
            child.cost = node.cost + trajectory_cost
            
            # Recursively update descendants
            self.update_descendants_cost(child)
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Closed-Loop RRT*.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            start_theta: Optional initial orientation (radians).
            goal_theta: Optional goal orientation (radians).
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
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        self.goal_threshold = kwargs.get('goal_threshold', self.goal_threshold)
        self.search_radius = kwargs.get('search_radius', self.search_radius)
        self.rewire_factor = kwargs.get('rewire_factor', self.rewire_factor)
        
        # Create start and goal states
        start_theta = kwargs.get('start_theta', 0.0)
        goal_theta = kwargs.get('goal_theta', 0.0)
        
        start_state = VehicleState(start[0], start[1], start_theta)
        self.goal_state = VehicleState(goal[0], goal[1], goal_theta)
        
        # Initialize tree with start node
        self.nodes = []
        start_node = CLRRTStarNode(start_state)
        start_node.cost = 0.0
        self.nodes.append(start_node)
        
        # Track best goal node
        best_goal_node = None
        best_goal_cost = float('inf')
        
        # Main loop
        for i in range(self.max_iterations):
            # Sample random state
            random_state = self.random_state()
            
            # Find nearest node
            nearest = self.nearest_node(random_state)
            
            # Steer towards random state
            result = self.steer(nearest.state, random_state)
            
            if result is None:
                continue  # No valid control found
            
            new_state, control, trajectory = result
            
            # Skip if trajectory is invalid
            if not self.is_collision_free(trajectory):
                continue
            
            # Calculate search radius (grows with log(n))
            n = len(self.nodes)
            radius = min(self.search_radius * (np.log(n + 1) / n) ** (1/2), self.rewire_factor * self.step_size)
            
            # Find near nodes
            near_nodes_list = self.near_nodes(new_state, radius)
            
            # Choose best parent
            best_parent, best_control, best_trajectory, min_cost = self.choose_parent(near_nodes_list, new_state, trajectory)
            
            if best_parent is None:
                # If no valid parent found, use nearest as parent
                best_parent = nearest
                best_control = control
                best_trajectory = trajectory
                
                # Calculate cost
                trajectory_cost = 0.0
                for i in range(len(trajectory) - 1):
                    state1 = trajectory[i]
                    state2 = trajectory[i + 1]
                    trajectory_cost += self.distance(state1.to_tuple(), state2.to_tuple())
                
                min_cost = best_parent.cost + trajectory_cost
            
            # Create new node
            new_node = CLRRTStarNode(new_state)
            new_node.parent = best_parent
            new_node.cost = min_cost
            new_node.control = best_control
            new_node.trajectory = best_trajectory
            self.nodes.append(new_node)
            
            # Rewire tree
            self.rewire(new_node, near_nodes_list)
            
            # Check if goal is reached
            if self.distance(new_state.to_tuple(), goal) <= self.goal_threshold:
                # Check if this path to goal is better
                goal_cost = new_node.cost
                if goal_cost < best_goal_cost:
                    best_goal_node = new_node
                    best_goal_cost = goal_cost
        
        # Construct path if goal was reached
        if best_goal_node is not None:
            # Extract full trajectory
            full_trajectory = []
            current = best_goal_node
            
            while current.parent is not None:
                # Add trajectory in reverse
                for state in reversed(current.trajectory):
                    full_trajectory.insert(0, state.to_tuple())
                
                current = current.parent
            
            # Add start position
            full_trajectory.insert(0, start)
            
            return full_trajectory
        
        # No path found
        return None
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of Closed-Loop RRT* that can provide step-by-step visualization.
        
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
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        self.goal_threshold = kwargs.get('goal_threshold', self.goal_threshold)
        self.search_radius = kwargs.get('search_radius', self.search_radius)
        self.rewire_factor = kwargs.get('rewire_factor', self.rewire_factor)
        
        # Create start and goal states
        start_theta = kwargs.get('start_theta', 0.0)
        goal_theta = kwargs.get('goal_theta', 0.0)
        
        start_state = VehicleState(start[0], start[1], start_theta)
        self.goal_state = VehicleState(goal[0], goal[1], goal_theta)
        
        # Initialize tree with start node
        self.nodes = []
        start_node = CLRRTStarNode(start_state)
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
            # Sample random state
            random_state = self.random_state()
            
            # Find nearest node
            nearest = self.nearest_node(random_state)
            
            # Steer towards random state
            result = self.steer(nearest.state, random_state)
            
            if result is None:
                continue  # No valid control found
            
            new_state, control, trajectory = result
            
            # Skip if trajectory is invalid
            if not self.is_collision_free(trajectory):
                continue
            
            # Calculate search radius (grows with log(n))
            n = len(self.nodes)
            radius = min(self.search_radius * (np.log(n + 1) / n) ** (1/2), self.rewire_factor * self.step_size)
            
            # Find near nodes
            near_nodes_list = self.near_nodes(new_state, radius)
            
            # Choose best parent
            best_parent, best_control, best_trajectory, min_cost = self.choose_parent(near_nodes_list, new_state, trajectory)
            
            if best_parent is None:
                # If no valid parent found, use nearest as parent
                best_parent = nearest
                best_control = control
                best_trajectory = trajectory
                
                # Calculate cost
                trajectory_cost = 0.0
                for i in range(len(trajectory) - 1):
                    state1 = trajectory[i]
                    state2 = trajectory[i + 1]
                    trajectory_cost += self.distance(state1.to_tuple(), state2.to_tuple())
                
                min_cost = best_parent.cost + trajectory_cost
            
            # Create new node
            new_node = CLRRTStarNode(new_state)
            new_node.parent = best_parent
            new_node.cost = min_cost
            new_node.control = best_control
            new_node.trajectory = best_trajectory
            self.nodes.append(new_node)
            
            # Rewire tree
            self.rewire(new_node, near_nodes_list)
            
            # Check if goal is reached
            current_path = None
            if self.distance(new_state.to_tuple(), goal) <= self.goal_threshold:
                # Check if this path to goal is better
                goal_cost = new_node.cost
                if goal_cost < best_goal_cost:
                    best_goal_node = new_node
                    best_goal_cost = goal_cost
                    
                    # Extract full trajectory for visualization
                    full_trajectory = []
                    current = best_goal_node
                    
                    while current.parent is not None:
                        # Add trajectory in reverse
                        for state in reversed(current.trajectory):
                            full_trajectory.insert(0, state.to_tuple())
                        
                        current = current.parent
                    
                    # Add start position
                    full_trajectory.insert(0, start)
                    current_path = full_trajectory
            
            # Call callback with current state
            if callback and i % 10 == 0:  # Call every 10 iterations to avoid slowdown
                callback(self.nodes, current_path, i + 1, f"Iteration {i+1}")
        
        # Construct final path if goal was reached
        final_path = None
        if best_goal_node is not None:
            # Extract full trajectory
            full_trajectory = []
            current = best_goal_node
            
            while current.parent is not None:
                # Add trajectory in reverse
                for state in reversed(current.trajectory):
                    full_trajectory.insert(0, state.to_tuple())
                
                current = current.parent
            
            # Add start position
            full_trajectory.insert(0, start)
            final_path = full_trajectory
            
            # Call callback with final state
            if callback:
                callback(self.nodes, final_path, self.max_iterations, "Final path")
        else:
            # No path found
            if callback:
                callback(self.nodes, None, self.max_iterations, "No path found")
        
        return final_path