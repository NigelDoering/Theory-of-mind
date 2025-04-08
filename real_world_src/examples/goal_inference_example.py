import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from real_world_src.utils.collect_trajectories import collect_agent_trajectories
from real_world_src.utils.goal_inference_helper import GoalInferenceHelper
from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.agents.agent_species import GoalDirectedAgent

def run_goal_inference_example(model_path, experiment=1, data_path=None, collect_new_data=False):
    """
    Run an example of goal inference.
    
    Args:
        model_path: Path to trained model checkpoint
        experiment: Experiment number (1 or 2)
        data_path: Path to saved trajectory data
        collect_new_data: Whether to collect new trajectory data
    """
    # Get trajectory data
    if collect_new_data:
        print("Collecting new trajectory data...")
        agent_trajectories = collect_agent_trajectories(
            num_agents=5, 
            episodes_per_agent=3, 
            max_steps=100, 
            save_dir='data'
        )
    elif data_path:
        print(f"Loading trajectory data from {data_path}...")
        with open(data_path, 'rb') as f:
            agent_trajectories = pickle.load(f)
    else:
        raise ValueError("Either collect_new_data must be True or data_path must be provided")
    
    # Create goal inference helper
    helper = GoalInferenceHelper(
        model_path=model_path,
        experiment=experiment,
        input_channels=3,
        char_dim=2,
        hidden_dim=64,
        num_actions=5,
        num_objects=4
    )
    
    # Select an agent for inference
    agent_id = list(agent_trajectories.keys())[0]
    agent_data = agent_trajectories[agent_id]
    
    print(f"Selected agent {agent_id} for inference")
    
    # Use first trajectory as past trajectory
    past_trajectory = agent_data[0]
    
    # Use first state of second trajectory as query state
    query_state = agent_data[1][0][0]
    
    # Define action and object names for visualization
    action_names = ['Up', 'Right', 'Down', 'Left', 'Stay']
    object_names = ['Library', 'Cafe', 'Dorm', 'Lab']
    
    # Perform goal inference
    results = helper.infer_goals(
        past_trajectory=past_trajectory,
        query_state=query_state,
        action_names=action_names,
        object_names=object_names
    )
    
    # Save results
    helper.save_inference_results(results, save_dir='inference_results')
    
    # Display results
    print("\nInference Results:")
    print(f"Most likely action: {action_names[results['most_likely_action']]}")
    print(f"Most likely goal: {object_names[results['most_likely_object']]}")
    
    print("\nAction probabilities:")
    for i, prob in enumerate(results['action_probs']):
        print(f"  {action_names[i]}: {prob:.4f}")
    
    print("\nGoal probabilities:")
    for i, prob in enumerate(results['consumption_probs']):
        print(f"  {object_names[i]}: {prob:.4f}")
    
    # Show visualizations
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(plt.imread('inference_results/action_prediction.png'))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(plt.imread('inference_results/consumption_prediction.png'))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(plt.imread('inference_results/successor_representation.png'))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ToMNet goal inference example")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--experiment", type=int, default=1, choices=[1, 2],
                        help="Experiment to run (1 or 2)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to saved trajectory data")
    parser.add_argument("--collect_new_data", action="store_true",
                        help="Whether to collect new trajectory data")
    
    args = parser.parse_args()
    
    run_goal_inference_example(
        model_path=args.model_path,
        experiment=args.experiment,
        data_path=args.data_path,
        collect_new_data=args.collect_new_data
    )