import numpy as np
import time
import os
import pickle
from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.agents.agent_species import RandomWalkAgent, GoalDirectedAgent

def collect_agent_trajectories(num_agents=20, episodes_per_agent=5, max_steps=100, save_dir='data'):
    """
    Collect trajectories from different agents in the campus environment.
    
    Args:
        num_agents: Number of agents to create
        episodes_per_agent: Number of episodes to run per agent
        max_steps: Maximum steps per episode
        save_dir: Directory to save trajectory data
        
    Returns:
        Dictionary mapping agent IDs to lists of trajectories
    """
    # Create environment
    env = CampusEnvironment()
    
    # Dictionary to store trajectories
    all_trajectories = {}
    
    # Create agents with different parameters
    for agent_id in range(num_agents):
        # Randomly choose agent type
        if np.random.random() < 0.8:
            # Create a goal-directed agent with random preferences
            # Randomly select goal preferences among available locations
            goals = env.get_points_of_interest()
            goal_preferences = {}
            
            # Sample from Dirichlet distribution as mentioned in the paper
            if goals:
                reward_samples = np.random.dirichlet([0.01] * len(goals))
                for i, goal in enumerate(goals):
                    goal_preferences[goal] = reward_samples[i]
            
            agent = GoalDirectedAgent(
                agent_id=f"agent_{agent_id}",
                goal_preferences=goal_preferences,
                rationality=np.random.uniform(0.5, 3.0)  # Random rationality parameter
            )
        else:
            # Create a random agent
            agent = RandomWalkAgent(agent_id=f"agent_{agent_id}")
        
        # Set agent's environment
        agent.environment = env
        
        # Collect trajectories for this agent
        agent_trajectories = []
        
        for episode in range(episodes_per_agent):
            # Reset environment and agent
            state = env.reset()
            agent.reset()
            
            # Run episode
            for step in range(max_steps):
                # Agent step already records the state-action pairs
                agent.step()
                
                # Check if episode is done
                if agent.at_goal():
                    break
            
            # Add trajectory to list - this is already in (state, action) format
            agent_trajectories.append(agent.trajectory)
        
        # Store agent's trajectories
        all_trajectories[agent_id] = agent_trajectories
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save trajectories
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"agent_trajectories_{timestamp}.pkl")
    
    with open(filename, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"Trajectories saved to {filename}")
    return all_trajectories

