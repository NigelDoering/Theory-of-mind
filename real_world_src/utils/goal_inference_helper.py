import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from real_world_src.models.tomnet import GoalInferenceToMNet
from real_world_src.utils.data_processing import process_state, prepare_trajectory_data
from real_world_src.evaluation.evaluate_goal_inference import (
    visualize_action_prediction, 
    visualize_object_consumption,
    visualize_successor_representation
)

class GoalInferenceHelper:
    """Helper class for performing goal inference with trained ToMNet models."""
    
    def __init__(self, model_path, experiment=1, input_channels=3, char_dim=2, 
                 hidden_dim=64, num_actions=5, num_objects=4):
        """
        Initialize the goal inference helper.
        
        Args:
            model_path: Path to trained model checkpoint
            experiment: Experiment number (1 or 2)
            input_channels: Number of input channels
            char_dim: Character embedding dimension
            hidden_dim: Hidden dimension
            num_actions: Number of possible actions
            num_objects: Number of possible objects
        """
        self.experiment = experiment
        self.input_channels = input_channels
        
        # Create model
        self.model = GoalInferenceToMNet(
            input_dim=input_channels,
            state_dim=input_channels * input_channels,
            hidden_dim=hidden_dim,
            char_dim=char_dim,
            num_actions=num_actions,
            num_objects=num_objects,
            experiment=experiment
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def infer_goals(self, past_trajectory, query_state, action_names=None, object_names=None):
        """
        Perform goal inference.
        
        Args:
            past_trajectory: Past trajectory of an agent
            query_state: Query state for goal inference
            action_names: Optional list of action names
            object_names: Optional list of object names
            
        Returns:
            Dictionary with inference results
        """
        # Process past trajectory
        if self.experiment == 1:
            processed_past = prepare_trajectory_data(past_trajectory, self.input_channels)
            processed_past = processed_past.unsqueeze(0)  # Add batch dimension
        else:
            # For experiment 2, process as snapshots
            processed_past = []
            for state, action in past_trajectory:
                processed_state = process_state(state, self.input_channels)
                processed_past.append(processed_state)
            
            # Convert to tensor
            processed_past = torch.tensor(np.array(processed_past), dtype=torch.float32)
            processed_past = processed_past.unsqueeze(0)  # Add batch dimension
        
        # Process query state
        processed_query = process_state(query_state, self.input_channels)
        processed_query = torch.tensor(processed_query, dtype=torch.float32)
        processed_query = processed_query.unsqueeze(0)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            action_pred, consumption_pred, sr_pred = self.model(processed_past, processed_query)
        
        # Create visualizations
        action_fig = visualize_action_prediction(
            self.model, processed_past, processed_query, action_names)
        
        consumption_fig = visualize_object_consumption(
            self.model, processed_past, processed_query, object_names)
        
        sr_fig = visualize_successor_representation(
            self.model, processed_past, processed_query)
        
        # Process predictions
        action_probs = action_pred.squeeze(0).cpu().numpy()
        consumption_probs = consumption_pred.squeeze(0).cpu().numpy()
        sr_maps = sr_pred.squeeze(0).cpu().numpy()
        
        most_likely_action = np.argmax(action_probs)
        most_likely_object = np.argmax(consumption_probs)
        
        # Create results dictionary
        results = {
            'action_probs': action_probs,
            'consumption_probs': consumption_probs,
            'sr_maps': sr_maps,
            'most_likely_action': most_likely_action,
            'most_likely_object': most_likely_object,
            'action_fig': action_fig,
            'consumption_fig': consumption_fig,
            'sr_fig': sr_fig
        }
        
        return results
    
    def save_inference_results(self, results, save_dir='inference_results'):
        """
        Save inference results to files.
        
        Args:
            results: Dictionary with inference results
            save_dir: Directory to save results
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save visualizations
        results['action_fig'].savefig(os.path.join(save_dir, 'action_prediction.png'))
        results['consumption_fig'].savefig(os.path.join(save_dir, 'consumption_prediction.png'))
        results['sr_fig'].savefig(os.path.join(save_dir, 'successor_representation.png'))
        
        # Save numerical results
        np.save(os.path.join(save_dir, 'action_probs.npy'), results['action_probs'])
        np.save(os.path.join(save_dir, 'consumption_probs.npy'), results['consumption_probs'])
        np.save(os.path.join(save_dir, 'sr_maps.npy'), results['sr_maps'])
        
        # Save summary text
        with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
            f.write(f"Most likely action: {results['most_likely_action']}\n")
            f.write(f"Most likely object: {results['most_likely_object']}\n")
            f.write("\nAction probabilities:\n")
            for i, prob in enumerate(results['action_probs']):
                f.write(f"  Action {i}: {prob:.4f}\n")
            
            f.write("\nConsumption probabilities:\n")
            for i, prob in enumerate(results['consumption_probs']):
                f.write(f"  Object {i}: {prob:.4f}\n")
        
        print(f"Saved inference results to {save_dir}")