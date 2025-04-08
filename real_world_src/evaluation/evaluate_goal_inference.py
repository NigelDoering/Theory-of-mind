import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def evaluate_goal_inference(model, test_data, experiment=1):
    """
    Evaluate goal inference model on test data.
    
    Args:
        model: Trained GoalInferenceToMNet
        test_data: Test data loader
        experiment: Experiment number (1 or 2)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    action_correct = 0
    action_total = 0
    consumption_correct = 0
    consumption_total = 0
    
    # Metrics for SR prediction (depends on your evaluation criteria)
    sr_error = 0
    sr_total = 0
    
    all_action_preds = []
    all_action_targets = []
    all_consumption_preds = []
    all_consumption_targets = []
    
    with torch.no_grad():
        for batch in test_data:
            # Extract data based on experiment
            if experiment == 1:
                past_trajectory = batch['past_trajectory'].squeeze(1)
                query_state = batch['query_state']
                consumption_target = batch['consumption_target']
                action_targets = batch['action_targets']
                
                # Forward pass
                action_pred, consumption_pred, sr_pred = model(past_trajectory, query_state)
                
                # Calculate action accuracy
                action_pred_class = torch.argmax(action_pred, dim=1)
                action_target_class = torch.argmax(action_targets[:, 0], dim=1)
                
            else:  # Experiment 2
                past_snapshots = batch['past_snapshots']
                query_state = batch['query_state']
                consumption_target = batch['consumption_target']
                action_target = batch['action_target']
                
                # Forward pass
                action_pred, consumption_pred, sr_pred = model(past_snapshots, query_state)
                
                # Calculate action accuracy
                action_pred_class = torch.argmax(action_pred, dim=1)
                action_target_class = action_target
            
            # Action accuracy
            action_correct += (action_pred_class == action_target_class).sum().item()
            action_total += action_target_class.size(0)
            
            # Consumption accuracy (binary classification threshold at 0.5)
            consumption_pred_binary = (consumption_pred > 0.5).float()
            consumption_correct += (consumption_pred_binary == consumption_target).all(dim=1).sum().item()
            consumption_total += consumption_target.size(0)
            
            # Store predictions for further analysis
            all_action_preds.append(action_pred.cpu().numpy())
            all_action_targets.append(action_target_class.cpu().numpy())
            all_consumption_preds.append(consumption_pred.cpu().numpy())
            all_consumption_targets.append(consumption_target.cpu().numpy())
    
    # Calculate metrics
    action_accuracy = action_correct / action_total if action_total > 0 else 0
    consumption_accuracy = consumption_correct / consumption_total if consumption_total > 0 else 0
    
    # Combine all predictions
    all_action_preds = np.concatenate(all_action_preds, axis=0)
    all_action_targets = np.concatenate(all_action_targets, axis=0)
    all_consumption_preds = np.concatenate(all_consumption_preds, axis=0)
    all_consumption_targets = np.concatenate(all_consumption_targets, axis=0)
    
    # Calculate additional metrics like confusion matrix
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    
    action_cm = confusion_matrix(all_action_targets, np.argmax(all_action_preds, axis=1))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_action_targets, np.argmax(all_action_preds, axis=1), average='weighted'
    )
    
    metrics = {
        'action_accuracy': action_accuracy,
        'consumption_accuracy': consumption_accuracy,
        'action_confusion_matrix': action_cm,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def visualize_successor_representation(model, trajectory, query_state, grid_size=(11, 11)):
    """
    Visualize the successor representation prediction.
    
    Args:
        model: Trained GoalInferenceToMNet
        trajectory: Past trajectory tensor
        query_state: Query state tensor
        grid_size: Size of the gridworld
        
    Returns:
        Figure with SR visualization
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        action_pred, consumption_pred, sr_pred = model(trajectory.unsqueeze(0), query_state.unsqueeze(0))
        
        # Get SR prediction (3 discount factors)
        sr_pred = sr_pred.squeeze(0).cpu().numpy()
        
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define custom colormap for SR visualization
    cmap = LinearSegmentedColormap.from_list('sr_cmap', ['white', 'red'])
    
    discount_factors = [0.5, 0.9, 0.99]
    
    for i, (ax, df) in enumerate(zip(axes, discount_factors)):
        # Reshape SR prediction to grid
        sr_grid = sr_pred[i].reshape(grid_size)
        
        # Plot heatmap
        sns.heatmap(sr_grid, ax=ax, cmap=cmap, vmin=0, vmax=1, 
                   cbar_kws={'label': 'Visitation probability'})
        
        ax.set_title(f'Discount factor: {df}')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
    
    plt.tight_layout()
    return fig

def visualize_object_consumption(model, trajectory, query_state, object_names=None):
    """
    Visualize the object consumption prediction.
    
    Args:
        model: Trained GoalInferenceToMNet
        trajectory: Past trajectory tensor
        query_state: Query state tensor
        object_names: Optional list of object names
        
    Returns:
        Figure with consumption visualization
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        action_pred, consumption_pred, sr_pred = model(trajectory.unsqueeze(0), query_state.unsqueeze(0))
        
        # Get consumption prediction
        consumption_pred = consumption_pred.squeeze(0).cpu().numpy()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Use object names if provided, otherwise use indices
    if object_names is None:
        object_names = [f'Object {i}' for i in range(len(consumption_pred))]
    
    # Plot bar chart
    ax.bar(object_names, consumption_pred)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Consumption Probability')
    ax.set_title('Predicted Object Consumption Probabilities')
    
    # Add value labels
    for i, v in enumerate(consumption_pred):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    return fig

def visualize_action_prediction(model, trajectory, query_state, action_names=None):
    """
    Visualize the action prediction.
    
    Args:
        model: Trained GoalInferenceToMNet
        trajectory: Past trajectory tensor
        query_state: Query state tensor
        action_names: Optional list of action names
        
    Returns:
        Figure with action prediction visualization
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        action_pred, consumption_pred, sr_pred = model(trajectory.unsqueeze(0), query_state.unsqueeze(0))
        
        # Get action prediction
        action_pred = action_pred.squeeze(0).cpu().numpy()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Use action names if provided, otherwise use indices
    if action_names is None:
        action_names = [f'Action {i}' for i in range(len(action_pred))]
    
    # Plot bar chart
    ax.bar(action_names, action_pred)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Action Probability')
    ax.set_title('Predicted Action Probabilities')
    
    # Add value labels
    for i, v in enumerate(action_pred):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    return fig