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