import sys
import os

# Add the project root directory to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Now import the modules
from src.models.ToMNet import ToMEnvironment, ToMNetSimulation
from src.environment.world import UCSDCampus
from src.agents.agent import UCSDAgent

# At the end of your main1.py file
if __name__ == "__main__":
    # This only runs when the module is run directly
    print("Running ToMNet simulation for UCSD campus...")
    
    # Create the UCSD campus environment
    campus = UCSDCampus(width=500, height=500)

    # Create environment wrapper for ToM experiments
    tom_env = ToMEnvironment(campus)

    # Add agents with different observation capabilities and preferences
    agent1 = UCSDAgent(
        simulation_space=campus,
        observation_radius=15,  # Can see further
        reward_preferences={
            "Geisel_Library": 8.0,
            "CSE_Building": 5.0,
            "Cognitive_Science_Building": 3.0,
            "Price_Center": 1.0,
            "RIMAC": -2.0
        }
    )

    agent2 = UCSDAgent(
        simulation_space=campus,
        observation_radius=8,  # Limited visibility
        reward_preferences={
            "Geisel_Library": 2.0,
            "CSE_Building": -1.0,
            "Cognitive_Science_Building": 7.0,
            "Price_Center": 4.0,
            "RIMAC": 9.0
        }
    )

    tom_env.add_agent(agent1)
    tom_env.add_agent(agent2)
    
    # Create the ToMNet simulation
    tom_simulation = ToMNetSimulation(tom_env)
    
    # Generate training data
    tom_simulation.generate_training_data(num_episodes=100)
    
    # Train the model
    tom_simulation.train_tomnet()
    
    # Visualize the environment
    tom_env.visualize_campus()
    
    # Test goal inference
    # inferred_goal = tom_simulation.infer_agent_goals(agent1.id, agent2.id)
    # print(f"Agent 1 infers that Agent 2 is likely heading to: {inferred_goal}")