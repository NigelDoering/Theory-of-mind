import numpy as np

class Simulation:
    """
    Parent simulation class for collecting agent trajectories and associated information.
    
    This class is designed to store data that will later be used for modeling, such as 
    the full trajectories of agents and their assigned goals.
    """
    def __init__(self):
        # Dictionary to store trajectories, keyed by agent ID.
        # Each trajectory can be a list (or array) of (x, y) tuples.
        self.agent_trajectories = {}
        # Dictionary to store each agent's goal.
        self.agent_goals = {}
    
    def add_agent_trajectory(self, agent_id, trajectory, goal):
        """
        Add trajectory data and the corresponding goal for a specific agent.
        
        Parameters:
            agent_id (str or int): Unique identifier for the agent.
            trajectory (list or np.array): Sequence of (x, y) positions.
            goal (tuple): The goal position assigned to the agent.
        """
        self.agent_trajectories[agent_id] = trajectory
        self.agent_goals[agent_id] = goal
    
    def get_agent_trajectory(self, agent_id):
        """Return the trajectory for a given agent."""
        return self.agent_trajectories.get(agent_id)
    
    def get_all_trajectories(self):
        """Return the dictionary of all agent trajectories."""
        return self.agent_trajectories
    
    def get_all_goals(self):
        """Return the dictionary of all agent goals."""
        return self.agent_goals 