import numpy as np
import random
import math
from collections import defaultdict
from src.planning.base_planner import BasePlanner
from src.planning.sampling_based.rrt import RRTNode
from src.planning.sampling_based.rrt_star import RRTStarPlanner

class GoalNode:
    """
    Node representing a potential goal for an agent.
    """
    def __init__(self, position):
        self.position = position
        self.probability = 0.0  # Probability that this is the true goal
        self.cost_to_reach = float('inf')  # Cost to reach from observed position

class GoalInferenceRRTPlanner(RRTStarPlanner):
    """
    Goal-Inference RRT (GI-RRT) algorithm for path planning and goal inference.
    
    GI-RRT extends RRT* to reason about goals and infer the intentions of
    observed agents. It is particularly useful for theory of mind applications
    where an agent needs to understand the goals of other agents based on
    their observed trajectories.
    
    Key features:
    - Infers goals from observed partial trajectories
    - Generates optimal paths to potential goals
    - Assigns probabilities to multiple goal hypotheses
    - Updates beliefs with new observations
    - Supports prediction of future agent behavior
    
    Note: This is an implementation inspired by goal inference techniques and
    optimal planning algorithms, designed specifically for theory of mind applications.
    """
    
    def __init__(self, world):
        super().__init__(world)
        self.potential_goals = []        # List of potential goals
        self.observed_trajectory = []    # Observed agent trajectory
        self.goal_prior = {}             # Prior probabilities for goals
        self.rationality_coefficient = 5.0  # Rationality coefficient (beta)
        self.prediction_horizon = 20     # Steps to predict into the future
        self.planning_horizon = 50       # Planning horizon for path generation
        self.max_planning_iterations = 500  # Maximum iterations for planning
    
    def cost_to_go(self, from_pos, to_pos):
        """
        Estimate the cost to go from from_pos to to_pos.
        Uses Euclidean distance as a heuristic.
        """
        return self.distance(from_pos, to_pos)
    
    def calculate_path_cost(self, path):
        """
        Calculate the total cost of a path.
        """
        if not path or len(path) < 2:
            return float('inf')
        
        cost = 0
        for i in range(len(path) - 1):
            cost += self.distance(path[i], path[i + 1])
        
        return cost
    
    def add_potential_goal(self, goal_position, prior_probability=None):
        """
        Add a potential goal for the observed agent.
        
        Parameters:
            goal_position: Tuple (x, y) of the goal position.
            prior_probability: Optional prior probability for this goal.
        """
        # Check if the goal is valid
        if not self.is_valid_point(goal_position):
            return False
        
        # Check if the goal already exists
        for goal in self.potential_goals:
            if self.distance(goal.position, goal_position) < 1e-6:
                return False
        
        # Create a new goal node
        goal = GoalNode(goal_position)
        
        # Set prior probability
        if prior_probability is not None:
            self.goal_prior[goal_position] = prior_probability
        else:
            # Uniform prior if not specified
            self.goal_prior[goal_position] = 1.0 / (len(self.potential_goals) + 1)
        
        # Add to list of potential goals
        self.potential_goals.append(goal)
        
        # Normalize prior probabilities
        self.normalize_goal_probabilities()
        
        return True
    
    def normalize_goal_probabilities(self):
        """
        Normalize goal probabilities to sum to 1.
        """
        total_probability = sum(self.goal_prior.values())
        
        if total_probability > 0:
            for goal_pos in self.goal_prior:
                self.goal_prior[goal_pos] /= total_probability
    
    def add_observation(self, position):
        """
        Add an observed position to the trajectory.
        
        Parameters:
            position: Tuple (x, y) of the observed position.
        """
        # Add to observed trajectory
        self.observed_trajectory.append(position)
        
        # Update goal probabilities based on new observation
        if len(self.observed_trajectory) >= 2:
            self.update_goal_probabilities()
    
    def update_goal_probabilities(self):
        """
        Update the probabilities of each goal based on the observed trajectory.
        Uses Bayesian inference with a rationality model.
        """
        if not self.observed_trajectory or not self.potential_goals:
            return
        
        # Last observed position
        current_pos = self.observed_trajectory[-1]
        
        # Calculate cost to reach each goal from current position
        for goal in self.potential_goals:
            # Plan a path to the goal
            path = self.plan(current_pos, goal.position, 
                            max_iterations=self.max_planning_iterations)
            
            if path:
                goal.cost_to_reach = self.calculate_path_cost(path)
            else:
                goal.cost_to_reach = float('inf')
        
        # Calculate probabilities using the principle of rational action
        likelihoods = {}
        for goal in self.potential_goals:
            goal_pos = goal.position
            
            # Skip goals that are unreachable
            if goal.cost_to_reach == float('inf'):
                likelihoods[goal_pos] = 0
                continue
            
            # Calculate how efficient the observed trajectory is for reaching this goal
            observed_cost = 0
            for i in range(len(self.observed_trajectory) - 1):
                observed_cost += self.distance(self.observed_trajectory[i], 
                                              self.observed_trajectory[i + 1])
            
            # Calculate optimal cost from start to current position
            optimal_cost_to_current = float('inf')
            start_pos = self.observed_trajectory[0]
            path_to_current = self.plan(start_pos, current_pos, 
                                       max_iterations=self.max_planning_iterations)
            
            if path_to_current:
                optimal_cost_to_current = self.calculate_path_cost(path_to_current)
            
            # Calculate efficiency ratio (how close to optimal is the observed trajectory)
            if optimal_cost_to_current == 0:
                efficiency = 1.0
            else:
                efficiency = min(1.0, optimal_cost_to_current / max(observed_cost, 1e-6))
            
            # Likelihood model: P(trajectory | goal) proportional to exp(-beta * (cost difference))
            # Higher beta means more rational agents (more likely to take optimal paths)
            cost_difference = observed_cost - optimal_cost_to_current
            likelihood = math.exp(-self.rationality_coefficient * max(0, cost_difference))
            
            # Scale by efficiency
            likelihood *= efficiency
            
            likelihoods[goal_pos] = likelihood
        
        # Bayesian update: P(goal | trajectory) proportional to P(trajectory | goal) * P(goal)
        posteriors = {}
        for goal_pos, likelihood in likelihoods.items():
            prior = self.goal_prior.get(goal_pos, 0)
            posteriors[goal_pos] = likelihood * prior
        
        # Normalize posteriors
        total_posterior = sum(posteriors.values())
        if total_posterior > 0:
            for goal_pos in posteriors:
                posteriors[goal_pos] /= total_posterior
        
        # Update probabilities
        self.goal_prior = posteriors
        
        # Update goal node probabilities
        for goal in self.potential_goals:
            goal.probability = self.goal_prior.get(goal.position, 0)
    
    def predict_future_trajectory(self, steps=None):
        """
        Predict the future trajectory of the observed agent.
        
        Parameters:
            steps: Number of steps to predict (default: self.prediction_horizon)
            
        Returns:
            List of (x, y) tuples representing the predicted trajectory.
        """
        if not self.observed_trajectory or not self.potential_goals:
            return None
        
        if steps is None:
            steps = self.prediction_horizon
        
        # Current position
        current_pos = self.observed_trajectory[-1]
        
        # Find the most likely goal
        most_likely_goal = max(self.potential_goals, key=lambda g: g.probability)
        
        # If the probability is too low, don't make a prediction
        if most_likely_goal.probability < 0.1:
            return None
        
        # Plan a path to the most likely goal
        path = self.plan(current_pos, most_likely_goal.position, 
                        max_iterations=self.max_planning_iterations)
        
        if not path:
            return None
        
        # Return the first 'steps' positions of the path (or the whole path if shorter)
        return path[:min(steps + 1, len(path))]
    
    def get_goal_probabilities(self):
        """
        Get the probabilities for each potential goal.
        
        Returns:
            Dictionary mapping goal positions to probabilities.
        """
        return {goal.position: goal.probability for goal in self.potential_goals}
    
    def most_likely_goal(self):
        """
        Get the most likely goal.
        
        Returns:
            GoalNode representing the most likely goal, or None if no goals exist.
        """
        if not self.potential_goals:
            return None
        
        return max(self.potential_goals, key=lambda g: g.probability)
    
    def infer_goals(self, observed_trajectory, potential_goals=None, prior_probabilities=None):
        """
        Infer goals from an observed trajectory.
        
        Parameters:
            observed_trajectory: List of (x, y) tuples representing the observed positions.
            potential_goals: Optional list of (x, y) tuples representing potential goals.
            prior_probabilities: Optional dictionary mapping goal positions to prior probabilities.
            
        Returns:
            Dictionary mapping goal positions to inferred probabilities.
        """
        # Reset state
        self.observed_trajectory = []
        self.potential_goals = []
        self.goal_prior = {}
        
        # Add potential goals
        if potential_goals:
            for i, goal_pos in enumerate(potential_goals):
                prior = prior_probabilities.get(goal_pos, None) if prior_probabilities else None
                self.add_potential_goal(goal_pos, prior)
        
        # Add observations
        for pos in observed_trajectory:
            self.add_observation(pos)
        
        # Return goal probabilities
        return self.get_goal_probabilities()
    
    def plan_with_goal_inference(self, start, goal, observed_agent_trajectory=None, 
                               potential_goals=None, **kwargs):
        """
        Plan a path from start to goal while considering observed agent behavior.
        
        This method performs both goal inference and path planning to create plans
        that take into account the predicted behavior of other agents.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            observed_agent_trajectory: Optional list of (x, y) tuples representing observed positions.
            potential_goals: Optional list of (x, y) tuples representing potential goals.
            **kwargs: Additional parameters for the planning algorithm.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        # Update parameters from kwargs
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        self.prediction_horizon = kwargs.get('prediction_horizon', self.prediction_horizon)
        self.rationality_coefficient = kwargs.get('rationality_coefficient', self.rationality_coefficient)
        
        # If observed agent trajectory is provided, perform goal inference
        predicted_trajectory = None
        if observed_agent_trajectory and potential_goals:
            # Infer goals from observed trajectory
            self.infer_goals(observed_agent_trajectory, potential_goals)
            
            # Predict future trajectory of the observed agent
            predicted_trajectory = self.predict_future_trajectory()
        
        # Plan a path to the goal avoiding predicted trajectory
        # If no predicted trajectory, just do regular planning
        if not predicted_trajectory:
            return super().plan(start, goal, **kwargs)
        
        # Modify the planning to avoid predicted trajectory
        return self.plan_avoiding_trajectory(start, goal, predicted_trajectory, **kwargs)
    
    def plan_avoiding_trajectory(self, start, goal, trajectory_to_avoid, **kwargs):
        """
        Plan a path from start to goal while avoiding a predicted trajectory.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            trajectory_to_avoid: List of (x, y) tuples representing a trajectory to avoid.
            **kwargs: Additional parameters for the planning algorithm.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        # Initialize planning parameters
        self.step_size = kwargs.get('step_size', self.step_size)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        
        # Safety distance to maintain from trajectory
        safety_distance = kwargs.get('safety_distance', 2.0)
        
        # Create time buffer around predicted trajectory points
        # Each point in trajectory_to_avoid becomes a small region to avoid
        trajectory_buffer = []
        for i, point in enumerate(trajectory_to_avoid):
            # Create a time-indexed buffer
            # Earlier points have smaller buffer (agent has already passed)
            # Later points have larger buffer (more uncertainty)
            time_factor = 0.5 + (i / len(trajectory_to_avoid))
            buffer_radius = safety_distance * time_factor
            trajectory_buffer.append((point, buffer_radius))
        
        # Store goal for sampling
        self.goal = goal
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        start_node.cost = 0.0
        self.nodes.append(start_node)
        
        # Initialize best solution
        best_goal_node = None
        best_cost = float('inf')
        
        # Main RRT* loop
        for i in range(self.max_iterations):
            # Sample random position (with goal bias)
            if random.random() < self.goal_sample_rate:
                random_pos = goal
            else:
                random_pos = self.random_position()
            
            # Find nearest node in the tree
            nearest = self.nearest_node(random_pos)
            
            # Steer towards random position
            new_pos = self.steer(nearest.position, random_pos)
            
            # Skip if position is invalid or too close to predicted trajectory
            if not self.is_valid_point(new_pos) or self.is_too_close_to_trajectory(new_pos, trajectory_buffer):
                continue
            
            # Check if the path is collision-free and doesn't cross the predicted trajectory
            if not self.is_collision_free(nearest.position, new_pos) or \
               self.path_crosses_trajectory(nearest.position, new_pos, trajectory_buffer):
                continue
            
            # Find nearby nodes for potential connection
            radius = min(self.search_radius, self.step_size * 2.0)
            near_nodes = self.near_nodes(new_pos, radius)
            
            # Choose parent with lowest cost
            min_cost = nearest.cost + self.distance(nearest.position, new_pos)
            min_parent = nearest
            
            for near_node in near_nodes:
                if self.is_collision_free(near_node.position, new_pos) and \
                   not self.path_crosses_trajectory(near_node.position, new_pos, trajectory_buffer):
                    cost = near_node.cost + self.distance(near_node.position, new_pos)
                    if cost < min_cost:
                        min_cost = cost
                        min_parent = near_node
            
            # Create new node
            new_node = RRTNode(new_pos)
            new_node.parent = min_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            
            # Rewire tree
            for near_node in near_nodes:
                if near_node != min_parent:
                    if self.is_collision_free(new_node.position, near_node.position) and \
                       not self.path_crosses_trajectory(new_node.position, near_node.position, trajectory_buffer):
                        cost = new_node.cost + self.distance(new_node.position, near_node.position)
                        if cost < near_node.cost:
                            near_node.parent = new_node
                            near_node.cost = cost
            
            # Check if the new node can reach the goal
            if self.distance(new_pos, goal) <= self.step_size and \
               self.is_collision_free(new_pos, goal) and \
               not self.path_crosses_trajectory(new_pos, goal, trajectory_buffer):
                # Create goal node
                goal_node = RRTNode(goal)
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + self.distance(new_node.position, goal)
                
                # Update best solution if better
                if best_goal_node is None or goal_node.cost < best_cost:
                    best_goal_node = goal_node
                    best_cost = goal_node.cost
        
        # Extract path if goal was reached
        if best_goal_node is not None:
            path = [best_goal_node.position]
            current = best_goal_node.parent
            
            while current is not None:
                path.append(current.position)
                current = current.parent
            
            path.reverse()
            return path
        
        # No path found
        return None
    
    def is_too_close_to_trajectory(self, position, trajectory_buffer):
        """
        Check if a position is too close to any point in the trajectory buffer.
        
        Parameters:
            position: Tuple (x, y) to check.
            trajectory_buffer: List of ((x, y), radius) tuples representing the trajectory buffer.
            
        Returns:
            True if too close, False otherwise.
        """
        for point, radius in trajectory_buffer:
            if self.distance(position, point) < radius:
                return True
        return False
    
    def path_crosses_trajectory(self, start, end, trajectory_buffer):
        """
        Check if a path from start to end crosses the trajectory buffer.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            end: Tuple (x, y) of the end position.
            trajectory_buffer: List of ((x, y), radius) tuples representing the trajectory buffer.
            
        Returns:
            True if the path crosses the trajectory, False otherwise.
        """
        # Check discrete points along the path
        path_length = self.distance(start, end)
        num_checks = max(int(path_length / 0.5), 5)  # Check at least 5 points
        
        for i in range(1, num_checks):
            t = i / num_checks
            x = int(start[0] * (1 - t) + end[0] * t)
            y = int(start[1] * (1 - t) + end[1] * t)
            check_pos = (x, y)
            
            if self.is_too_close_to_trajectory(check_pos, trajectory_buffer):
                return True
        
        return False
    
    def interactive_plan_with_goal_inference(self, start, goal, observed_agent_trajectory=None,
                                          potential_goals=None, callback=None, **kwargs):
        """
        Interactive version of plan_with_goal_inference that can provide step-by-step visualization.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            observed_agent_trajectory: Optional list of (x, y) tuples representing observed positions.
            potential_goals: Optional list of (x, y) tuples representing potential goals.
            callback: Function to call after each iteration with current tree and path.
            **kwargs: Additional parameters for the planning algorithm.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        # Update parameters from kwargs
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', self.goal_sample_rate)
        self.prediction_horizon = kwargs.get('prediction_horizon', self.prediction_horizon)
        self.rationality_coefficient = kwargs.get('rationality_coefficient', self.rationality_coefficient)
        
        # If observed agent trajectory is provided, perform goal inference
        predicted_trajectory = None
        if observed_agent_trajectory and potential_goals:
            # Infer goals from observed trajectory
            goal_probs = self.infer_goals(observed_agent_trajectory, potential_goals)
            
            # Predict future trajectory of the observed agent
            predicted_trajectory = self.predict_future_trajectory()
            
            # Call callback with initial inference results
            if callback:
                callback([], None, 0, 
                         f"Goal inference: {', '.join([f'{g}:{p:.2f}' for g, p in goal_probs.items()])}")
        
        # If no predicted trajectory, just do regular interactive planning
        if not predicted_trajectory:
            return super().interactive_plan(start, goal, callback, **kwargs)
        
        # Safety distance to maintain from trajectory
        safety_distance = kwargs.get('safety_distance', 2.0)
        
        # Create time buffer around predicted trajectory points
        trajectory_buffer = []
        for i, point in enumerate(predicted_trajectory):
            time_factor = 0.5 + (i / len(predicted_trajectory))
            buffer_radius = safety_distance * time_factor
            trajectory_buffer.append((point, buffer_radius))
        
        # Call callback with predicted trajectory
        if callback:
            callback([], None, 0, 
                     f"Predicted trajectory: {predicted_trajectory}")
        
        # Store goal for sampling
        self.goal = goal
        
        # Initialize tree with start node
        self.nodes = []
        start_node = RRTNode(start)
        start_node.cost = 0.0
        self.nodes.append(start_node)
        
        # Initialize best solution
        best_goal_node = None
        best_cost = float('inf')
        
        # Main RRT* loop
        for i in range(self.max_iterations):
            # Sample random position (with goal bias)
            if random.random() < self.goal_sample_rate:
                random_pos = goal
            else:
                random_pos = self.random_position()
            
            # Find nearest node in the tree
            nearest = self.nearest_node(random_pos)
            
            # Steer towards random position
            new_pos = self.steer(nearest.position, random_pos)
            
            # Skip if position is invalid or too close to predicted trajectory
            if not self.is_valid_point(new_pos) or self.is_too_close_to_trajectory(new_pos, trajectory_buffer):
                continue
            
            # Check if the path is collision-free and doesn't cross the predicted trajectory
            if not self.is_collision_free(nearest.position, new_pos) or \
               self.path_crosses_trajectory(nearest.position, new_pos, trajectory_buffer):
                continue
            
            # Find nearby nodes for potential connection
            radius = min(self.search_radius, self.step_size * 2.0)
            near_nodes = self.near_nodes(new_pos, radius)
            
            # Choose parent with lowest cost
            min_cost = nearest.cost + self.distance(nearest.position, new_pos)
            min_parent = nearest
            
            for near_node in near_nodes:
                if self.is_collision_free(near_node.position, new_pos) and \
                   not self.path_crosses_trajectory(near_node.position, new_pos, trajectory_buffer):
                    cost = near_node.cost + self.distance(near_node.position, new_pos)
                    if cost < min_cost:
                        min_cost = cost
                        min_parent = near_node
            
            # Create new node
            new_node = RRTNode(new_pos)
            new_node.parent = min_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            
            # Rewire tree
            for near_node in near_nodes:
                if near_node != min_parent:
                    if self.is_collision_free(new_node.position, near_node.position) and \
                       not self.path_crosses_trajectory(new_node.position, near_node.position, trajectory_buffer):
                        cost = new_node.cost + self.distance(new_node.position, near_node.position)
                        if cost < near_node.cost:
                            near_node.parent = new_node
                            near_node.cost = cost
            
            # Check if the new node can reach the goal
            current_path = None
            if self.distance(new_pos, goal) <= self.step_size and \
               self.is_collision_free(new_pos, goal) and \
               not self.path_crosses_trajectory(new_pos, goal, trajectory_buffer):
                # Create goal node
                goal_node = RRTNode(goal)
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + self.distance(new_node.position, goal)
                
                # Update best solution if better
                if best_goal_node is None or goal_node.cost < best_cost:
                    best_goal_node = goal_node
                    best_cost = goal_node.cost
                    
                    # Construct current path
                    current_path = [goal]
                    node = new_node
                    while node is not None:
                        current_path.append(node.position)
                        node = node.parent
                    current_path.reverse()
            
            # Call callback periodically
            if callback and i % 10 == 0:  # Every 10 iterations
                callback_path = None
                if best_goal_node is not None:
                    callback_path = [best_goal_node.position]
                    node = best_goal_node.parent
                    while node is not None:
                        callback_path.append(node.position)
                        node = node.parent
                    callback_path.reverse()
                
                callback(self.nodes, callback_path, i, 
                         f"Iteration {i}: {len(self.nodes)} nodes, best cost: {best_cost:.2f}")
        
        # Extract final path
        final_path = None
        if best_goal_node is not None:
            final_path = [best_goal_node.position]
            current = best_goal_node.parent
            
            while current is not None:
                final_path.append(current.position)
                current = current.parent
            
            final_path.reverse()
            
            # Call callback with final path
            if callback:
                callback(self.nodes, final_path, self.max_iterations, 
                         f"Final path found, cost: {best_cost:.2f}")
        else:
            # No path found
            if callback:
                callback(self.nodes, None, self.max_iterations, "No path found")
        
        return final_path