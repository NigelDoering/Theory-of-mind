import numpy as np

class Agent:
    """
    Represents an agent in the 2D world.

    When an Agent is initialized, it takes nominal start and goal coordinates
    (e.g., from the world's starting and goal spaces), but samples the actual
    start and goal from a 2D Gaussian around those coordinates. This introduces
    variability in both the agent's initial and final positions.

    Positions are then rounded to the nearest integer.
    """
    def __init__(
        self,
        agent_id,
        world,
        nominal_start,
        nominal_goal,
        start_std=1.0,
        goal_std=1.0, 
        random_seed=None
    ):
        """
        Parameters
        ----------
        agent_id : str or int
            Unique identifier for this agent.
        nominal_start : tuple
            (x, y) coordinates of the nominal start position.
        nominal_goal : tuple
            (x, y) coordinates of the nominal goal position.
        start_std : float, optional
            Standard deviation for sampling the actual start position.
        goal_std : float, optional
            Standard deviation for sampling the actual goal position.
        random_seed : int, optional
            Seed for reproducible sampling.
        """
        self.agent_id = agent_id
        self.nominal_start = nominal_start
        self.nominal_goal = nominal_goal
        self.start_std = start_std
        self.goal_std = goal_std
        self.world = world
        
        # Create a random generator for reproducible sampling
        self.rng = np.random.default_rng(random_seed)
        
        # Sample the actual start position and round to the nearest integer
        start_x = self.rng.normal(nominal_start[0], start_std)
        start_y = self.rng.normal(nominal_start[1], start_std)
        self.start = (
            int(np.rint(start_x)),
            int(np.rint(start_y))
        )
        
        # Sample the actual goal position and round to the nearest integer
        goal_x = self.rng.normal(nominal_goal[0], goal_std)
        goal_y = self.rng.normal(nominal_goal[1], goal_std)
        self.goal = (
            int(np.rint(goal_x)),
            int(np.rint(goal_y))
        )
    
    def __repr__(self):
        return (
            f"Agent(agent_id={self.agent_id}, "
            f"start={self.start}, "
            f"goal={self.goal})"
        )

    def get_positions(self):
        """
        Returns the actual start and goal positions for this agent.
        """
        return self.start, self.goal

    def plan_path(self, planner, **planner_kwargs):
        """
        Use a given planner to compute a path from the agent's start to its goal.
        
        Parameters:
            planner: A path planner object that implements the plan method.
            **planner_kwargs: Additional parameters to pass to the planner.
            
        Returns:
            A planned path as a list of (x, y) positions, or None if no path found.
        """
        return planner.plan(self.start, self.goal, **planner_kwargs)
    
    def initialize_agent(self):
        """Reset agent to a new starting position for a new episode"""
        # Sample the actual start position and round to the nearest integer
        start_x = self.rng.normal(self.nominal_start[0], self.start_std)
        start_y = self.rng.normal(self.nominal_start[1], self.start_std)
        self.start = (
            int(np.rint(start_x)),
            int(np.rint(start_y))
        )
        
        # Sample the actual goal position and round to the nearest integer
        goal_x = self.rng.normal(self.nominal_goal[0], self.goal_std)
        goal_y = self.rng.normal(self.nominal_goal[1], self.goal_std)
        self.goal = (
            int(np.rint(goal_x)),
            int(np.rint(goal_y))
        )
        
        # Update current position to the new start position
        self.current_position = self.start
        
        return self.start, self.goal

class UCSDAgent(Agent):
    """
    Agent class with customizable observation range and reward preferences.
    """
    def __init__(self, simulation_space, observation_radius=10, reward_preferences=None, 
                 agent_id=None, start_pos=(25, 25), goal_pos=None, start_std=1.0, goal_std=1.0):
        # Generate a unique ID if none provided
        if agent_id is None:
            agent_id = f"agent_{id(self)}"
        
        # Set goal position if not provided (we'll determine a random landmark as goal)
        if goal_pos is None and hasattr(simulation_space, 'landmark_locations') and simulation_space.landmark_locations:
            # Pick a random landmark as the goal
            import random
            landmark_name = random.choice(list(simulation_space.landmark_locations.keys()))
            goal_pos = simulation_space.landmark_locations[landmark_name]
        elif goal_pos is None:
            # Default goal if no landmarks
            goal_pos = (400, 400)
            
        # Now call the parent constructor with the correct parameters
        super().__init__(
            agent_id=agent_id,
            world=simulation_space,  # simulation_space is equivalent to world
            nominal_start=start_pos,
            nominal_goal=goal_pos,
            start_std=start_std,
            goal_std=goal_std
        )
        
        self.id = agent_id  # Add this line to ensure the id attribute exists
        self.simulation_space = simulation_space  # Store for our own reference
        self.observation_radius = observation_radius
        self.reward_preferences = reward_preferences or {}  # Default empty dict
        self.observed_cells = {}  # To track what the agent has seen
        self.beliefs = {}  # To store beliefs about other agents
        self.current_position = self.start  # Initialize current position
        
    def initialize_agent(self):
        """Reset agent to a new starting position and clear observation history"""
        # Call the parent class initialize_agent method
        start, goal = super().initialize_agent()
        
        # Reset agent-specific attributes
        self.observed_cells = {}  # Clear observation history
        self.beliefs = {}         # Clear beliefs about other agents
        
        # Return the new start and goal positions
        return start, goal
        
    def observe_environment(self):
        """Get the agent's partial observation of the environment"""
        x, y = self.current_position
        observed = {}
        
        # Observe cells within observation radius
        for i in range(max(0, x - self.observation_radius), min(self.simulation_space.width, x + self.observation_radius + 1)):
            for j in range(max(0, y - self.observation_radius), min(self.simulation_space.height, y + self.observation_radius + 1)):
                # Calculate distance to determine if cell is within observation radius
                dist = ((i - x) ** 2 + (j - y) ** 2) ** 0.5
                if dist <= self.observation_radius:
                    # Check if the simulation_space has a get_cell_content method
                    if hasattr(self.simulation_space, 'get_cell_content'):
                        cell_content = self.simulation_space.get_cell_content(i, j)
                    else:
                        # Default to assuming empty cell
                        cell_content = 'empty'
                    
                    observed[(i, j)] = cell_content
                    self.observed_cells[(i, j)] = cell_content
                    
        return observed
    
    def calculate_reward(self, position):
        """Calculate reward based on agent's preferences"""
        x, y = position
        reward = 0
        
        # Check if position corresponds to any landmark
        for landmark_name, coords in self.simulation_space.landmark_locations.items():
            if coords == (x, y):
                # If agent has preference for this landmark, apply it
                preference_value = self.reward_preferences.get(landmark_name, 0)
                reward += preference_value
                break
                
        return reward
    
    def choose_action(self):
        """Choose action based on current observation and goals"""
        # This could implement a basic planner or connect to your existing planning algorithms
        # For example, using your stochastic A* implementation
        
        # Observe current environment
        observation = self.observe_environment()
        
        # Plan path to maximize expected reward
        # (This is where you would use your planning algorithms)
        
        # Return chosen action (e.g., "up", "down", "left", "right")
        return self.plan_next_step()
    
    def plan_next_step(self):
        """
        Plan the next step for the agent based on its current position and goals.
        Returns: A direction string ('up', 'down', 'left', 'right') for the agent's next move.
        """
        # First, determine target based on reward preferences
        target_landmark = self.determine_best_landmark()
        
        if target_landmark is None:
            # If no landmark is preferred, make a random move
            import random
            return random.choice(['up', 'down', 'left', 'right'])
        
        # Get the target position
        target_position = self.simulation_space.landmark_locations.get(target_landmark)
        if not target_position:
            # Fallback if landmark isn't found
            import random
            return random.choice(['up', 'down', 'left', 'right'])
        
        # Simple greedy movement - move in the direction that gets us closer to the target
        dx = target_position[0] - self.current_position[0]
        dy = target_position[1] - self.current_position[1]
        
        # Choose the direction with the largest component
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def determine_best_landmark(self):
        """
        Determine the best landmark to target based on reward preferences.
        Returns: The name of the landmark with the highest reward.
        """
        if not self.reward_preferences:
            return None
            
        # Find the landmark with the highest reward
        best_landmark = None
        best_reward = float('-inf')
        
        for landmark_name, reward in self.reward_preferences.items():
            if reward > best_reward:
                best_reward = reward
                best_landmark = landmark_name
                
        return best_landmark