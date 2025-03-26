import numpy as np
# import pymc3 as pm
from src.environment.world import UCSDCampus


class ToMEnvironment:
    """
    Environment wrapper for ToM experiments with meta-learning capabilities.
    """
    def __init__(self, campus=None):
        self.campus = campus or UCSDCampus()
        self.agents = []
        self.trajectories = {}  # Store agent trajectories for learning
        self.agent_beliefs = {}  # Store agent beliefs about other agents
        self.current_step = 0   # Initialize step counter
        
    def add_agent(self, agent):
        """Add an agent to the environment"""
        self.agents.append(agent)
        self.campus.add_agent(agent)
        self.trajectories[agent.id] = []
        self.agent_beliefs[agent.id] = {}
        
    def reset(self):
        """Reset the environment for a new episode"""
        # Clear trajectories
        for agent_id in self.trajectories:
            self.trajectories[agent_id] = []
        
        # Reset beliefs
        for agent_id in self.agent_beliefs:
            self.agent_beliefs[agent_id] = {}
        
        # Reset campus if it has a reset method
        if hasattr(self.campus, 'reset'):
            self.campus.reset()
        
        # Reset current step counter
        self.current_step = 0
        
        # Reinitialize agents
        for agent in self.agents:
            if hasattr(agent, 'initialize_agent'):
                agent.initialize_agent()
            # Make sure current_position is set
            if hasattr(agent, 'start'):
                agent.current_position = agent.start
    
    # Update the run_episode method to track current_step properly
    def run_episode(self, max_steps=100):
        """Run a single episode of the environment"""
        for step in range(max_steps):
            self.current_step = step  # Update the current_step
            
            for agent in self.agents:
                # Let agent observe and act
                observation = agent.observe_environment()
                action = agent.choose_action()
                
                # Execute action
                old_position = agent.current_position
                new_position = self.campus.move_agent(agent, action)
                
                # Record trajectory
                self.trajectories[agent.id].append({
                    'step': step,
                    'observation': observation,
                    'action': action,
                    'position': new_position,
                    'reward': agent.calculate_reward(new_position)
                })
                
                # Update beliefs (Theory of Mind)
                self.update_agent_beliefs(agent)
    
    # Fix the update_agent_beliefs method to use self.current_step
    def update_agent_beliefs(self, observer_agent):
        """Update agent's beliefs about other agents (ToM)"""
        observation = observer_agent.observe_environment()
        
        # Check if other agents are in observation
        for other_agent in self.agents:
            if other_agent.id == observer_agent.id:
                continue  # Skip self
                
            if other_agent.current_position in observation:
                # Agent is observed, update beliefs
                if other_agent.id not in observer_agent.beliefs:
                    observer_agent.beliefs[other_agent.id] = {
                        'observed_positions': [],
                        'inferred_goals': {},
                        'inferred_rewards': {}
                    }
                
                # Record observation using self.current_step, not campus.current_step
                observer_agent.beliefs[other_agent.id]['observed_positions'].append(
                    (self.current_step, other_agent.current_position)
                )
                
                # Inference could happen here or in a separate process
    
    def visualize_campus(self):
        """Visualize the campus and agent positions"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 10))
        
        # Draw campus (obstacles in black)
        for i in range(self.campus.height):
            for j in range(self.campus.width):
                if self.campus.is_obstacle(i, j):
                    plt.plot(j, i, 'ks', markersize=1)
        
        # Draw landmarks
        for name, (x, y) in self.campus.landmark_locations.items():
            plt.plot(y, x, 'b*', markersize=10)
            plt.text(y, x, name, fontsize=8)
        
        # Draw agents
        for agent in self.agents:
            x, y = agent.current_position
            plt.plot(y, x, 'ro', markersize=8)
            
            # Draw observation radius
            circle = plt.Circle((y, x), agent.observation_radius, 
                                fill=False, color='r', linestyle='--', alpha=0.5)
            plt.gca().add_artist(circle)
        
        plt.xlim(0, self.campus.width)
        plt.ylim(0, self.campus.height)
        plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
        plt.title('UCSD Campus Simulation')
        plt.show()


class ToMNetSimulation:
    """
    Simulation class that uses trajectories to learn and predict agent behaviors.
    """
    def __init__(self, environment):
        self.environment = environment
        self.training_data = []
        self.test_data = []
        
    def generate_training_data(self, num_episodes=100):
        """Generate training data by running episodes"""
        for episode in range(num_episodes):
            # Reset environment
            self.environment.reset()
            
            # Randomize agent preferences and initial positions
            self.randomize_agents()
            
            # Run episode
            self.environment.run_episode()
            
            # Extract training examples from trajectories
            for agent_id, trajectory in self.environment.trajectories.items():
                # Find the agent with this ID in the agents list
                agent = None
                for a in self.environment.agents:
                    if a.id == agent_id:
                        agent = a
                        break
                        
                if agent is None:
                    print(f"Warning: Could not find agent with ID {agent_id}")
                    continue
                    
                # Format as training data for ToMNet
                if len(trajectory) > 10:  # Make sure we have enough trajectory data
                    past_trajectory = trajectory[:-10]  # All but last 10 steps
                    future_trajectory = trajectory[-10:]  # Last 10 steps
                    
                    self.training_data.append({
                        'agent_id': agent_id,
                        'past_trajectory': past_trajectory,
                        'future_trajectory': future_trajectory,
                        'agent_preferences': agent.reward_preferences,  # Use the agent object directly
                        'campus': self.environment.campus
                    })
                else:
                    print(f"Warning: Trajectory for agent {agent_id} is too short ({len(trajectory)} steps)")
    
    def train_tomnet(self):
        """
        Train the ToMNet model using generated data
        
        This would connect to a machine learning framework like TensorFlow or PyTorch
        to implement the actual ToMNet architecture.
        """
        # Implementation would depend on ML framework of choice
        pass
    
    def randomize_agents(self):
        """Randomize agent preferences and starting positions"""
        # First check if campus has landmarks
        if hasattr(self.environment.campus, 'landmark_locations') and self.environment.campus.landmark_locations:
            landmarks = list(self.environment.campus.landmark_locations.keys())
            
            for agent in self.environment.agents:
                # Initialize agent position (make sure this method exists)
                if hasattr(agent, 'initialize_agent'):
                    agent.initialize_agent()
                
                # Randomize preferences
                if hasattr(agent, 'reward_preferences'):
                    preferences = {}
                    for landmark in landmarks:
                        # Each agent gets random preferences for landmarks
                        preferences[landmark] = np.random.uniform(-10, 10)
                    agent.reward_preferences = preferences