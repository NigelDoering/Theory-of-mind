import os
import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force Agg backend for all plots
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.agents.agent_factory import AgentFactory
from real_world_src.simulation.simulator import Simulator
from real_world_src.utils.run_manager import RunManager
from real_world_src.utils.trajectory_collector import TrajectoryCollector
from real_world_src.models.tomnet import ToMNet
from real_world_src.models.tomnet_trainer import ToMNetTrainer
from real_world_src.agents.tom_agent import ToMAgent
from real_world_src.utils.visualization import (
    plot_agent_paths, visualize_agent_decision, visualize_tom_predictions,
    animate_tom_predictions, plot_species_grid
)

def main():
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Initialize the run manager
    run_manager = RunManager(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tom_results'))
    run_dir = run_manager.start_new_run()
    
    print("=" * 50)
    print("Machine Theory of Mind (ToMNet) Training")
    print("=" * 50)
    
    # Step 1: Create the UCSD campus environment
    print("\nStep 1: Initializing UCSD campus environment...")
    campus = CampusEnvironment()
    
    # Step 2: Populate with diverse agents for data collection
    print("\nStep 2: Creating diverse agent population for data collection...")
    
    # Create more agents of each type to generate sufficient training data
    training_population = {
        "shortest": 5,
        "random": 5,
        "landmark": 5,
        "social": 5,
        "explorer": 5,
        "obstacle": 5,
        "scared": 5,
        "risky": 5
    }
    
    training_agents = AgentFactory.populate_environment(campus, training_population)
    print(f"Created {len(training_agents)} agents for data collection")
    
    # Step 3: Initialize trajectory collector
    print("\nStep 3: Initializing trajectory collector...")
    trajectory_collector = TrajectoryCollector(campus, max_trajectory_length=1000)
    
    # Step 4: Run simulations to collect trajectory data
    print("\nStep 4: Running simulations to collect trajectory data...")
    num_episodes = 100  # Number of episodes to run for data collection
    
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode+1}/{num_episodes}...")
        
        # Create a simulator with the trajectory collector
        simulator = Simulator(campus, run_manager=run_manager)
        simulator.set_trajectory_collector(trajectory_collector)
        
        # Run the simulation without animation (faster)
        simulator.run_simulation(max_steps=100, animate=False, save_animation=False)
        
        # Reset agents for the next episode with new random positions
        for agent in training_agents:
            agent.reset()
    
    # Step 5: Train the ToMNet model
    print("\nStep 5: Training the ToMNet model...")
    
    # Configure model parameters
    input_dim = 5  # State encoding dimension from trajectory_collector
    state_dim = 5  # Current state encoding dimension
    hidden_dim = 64  # Hidden layer dimension
    output_dim = 5  # Prediction output dimension (same as state encoding)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Create and train the model
    trainer = ToMNetTrainer(
        input_dim=input_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        lr=0.001,
        device=device
    )
    
    # Train the model
    model = trainer.train(
        trajectory_collector=trajectory_collector,
        batch_size=32,
        epochs=50
    )
    
    # Save the trained model
    model_path = os.path.join(run_dir, "tomnet_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Step 6: Create a testing environment with new agents
    print("\nStep 6: Creating testing environment...")
    
    # Reset the environment
    campus = CampusEnvironment()
    
    # Create a smaller set of agents for testing
    test_population = {
        "shortest": 1,
        "random": 1,
        "landmark": 1,
        "social": 1,
        "explorer": 1,
        "obstacle": 1,
        "scared": 1,
        "risky": 1
    }
    
    test_agents = AgentFactory.populate_environment(campus, test_population)
    
    # Step 7: Create a ToM agent that will predict others' behavior
    print("\nStep 7: Creating ToM agent with trained model...")
    tom_agent = ToMAgent(
        agent_id="tom_agent_1",
        environment=campus,
        tomnet=model,
        start_node=None,  # Will be assigned automatically
        goal_node=None,   # Will be assigned automatically
        color="#ffffff"   # White color to distinguish it
    )
    
    # Add the ToM agent to the environment
    campus.add_agent(tom_agent)
    
    # Assign the trajectory collector to the ToM agent
    tom_agent.set_trajectory_collector(trajectory_collector)
    
    # Step 8: Run a test simulation with visualization
    print("\nStep 8: Running test simulation with ToM agent...")
    simulator = Simulator(campus, run_manager=run_manager)
    simulator.set_trajectory_collector(trajectory_collector)
    
    # Enable animation for this test run
    simulator.run_simulation(max_steps=1000, animate=True, save_animation=True)
    
    # Step 9: Visualize ToM predictions
    print("\nStep 9: Visualizing ToM predictions...")
    
    # Generate static visualization of ToM predictions
    tom_pred_fig, tom_pred_ax = visualize_tom_predictions(
        campus, tom_agent, title="ToM Agent Predictions of Other Agents"
    )
    
    # Save the visualization
    tom_pred_path = run_manager.get_plot_path("tom_predictions.png")
    tom_pred_fig.savefig(tom_pred_path, dpi=300, bbox_inches='tight')
    print(f"Saved ToM predictions visualization to {tom_pred_path}")
    plt.close(tom_pred_fig)
    
    # Create an animated visualization of ToM predictions
    tom_anim_path = run_manager.get_animation_path("tom_predictions.gif")
    animate_tom_predictions(
        campus, tom_agent, 
        title="Theory of Mind in Action", 
        max_frames=1000,
        interval=100,
        save_path=tom_anim_path,
        dpi=200
    )
    print(f"Saved ToM animation to {tom_anim_path}")
    
    # Step 10: Generate a README.md file for this run
    print("\nStep 10: Generating README.md documentation...")
    readme_path = os.path.join(run_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# ToMNet Training Run #{run_manager.current_run}\n\n")
        f.write("This folder contains results from training and testing a Machine Theory of Mind (ToMNet) neural network.\n\n")
        
        f.write("## Model Architecture\n\n")
        f.write("The ToMNet consists of three neural networks:\n\n")
        f.write("1. **Character Network**: Processes past trajectories to extract agent traits\n")
        f.write("2. **Mental State Network**: Analyzes recent observations to infer current mental state\n")
        f.write("3. **Prediction Network**: Combines outputs from both networks to predict future actions\n\n")
        
        f.write("## Training Details\n\n")
        f.write(f"- Input dimension: {input_dim}\n")
        f.write(f"- State dimension: {state_dim}\n")
        f.write(f"- Hidden dimension: {hidden_dim}\n")
        f.write(f"- Output dimension: {output_dim}\n")
        f.write(f"- Training device: {device}\n")
        f.write(f"- Training epochs: 20\n")
        f.write(f"- Batch size: 32\n")
        f.write(f"- Number of training episodes: {num_episodes}\n\n")
        
        f.write("## Results\n\n")
        f.write("### Visualizations\n\n")
        
        f.write("#### Static Predictions\n\n")
        f.write(f"![ToM Predictions](static_plots/tom_predictions.png)\n\n")
        
        f.write("#### Animated Predictions\n\n")
        f.write(f"![ToM Animation](animations/tom_predictions.gif)\n\n")
        
        f.write("## Model Files\n\n")
        f.write(f"- Trained model: [tomnet_model.pt](tomnet_model.pt)\n")
    
    print(f"README.md generated at {readme_path}")
    print("\nToMNet training and testing complete!")

if __name__ == "__main__":
    main()