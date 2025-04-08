import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import os
import argparse
import pickle

from real_world_src.models.tomnet import GoalInferenceToMNet
from real_world_src.utils.data_processing import prepare_experiment1_data, prepare_experiment2_data
from real_world_src.utils.collect_trajectories import collect_agent_trajectories
from real_world_src.evaluation.evaluate_goal_inference import evaluate_goal_inference

def train_goal_inference(experiment=1, input_channels=3, char_dim=2, hidden_dim=64, 
                         num_actions=5, num_objects=4, batch_size=32, max_snapshots=10,
                         learning_rate=0.001, num_epochs=100, save_dir='checkpoints',
                         data_path=None, collect_new_data=False):
    """
    Train the ToMNet model for goal inference.
    
    Args:
        experiment: Experiment number (1 or 2)
        input_channels: Number of input channels
        char_dim: Character embedding dimension
        hidden_dim: Hidden dimension
        num_actions: Number of possible actions
        num_objects: Number of possible objects
        batch_size: Batch size for training
        max_snapshots: Maximum number of snapshots for Experiment 2
        learning_rate: Learning rate
        num_epochs: Number of epochs
        save_dir: Directory to save model checkpoints
        data_path: Path to saved trajectory data
        collect_new_data: Whether to collect new trajectory data
        
    Returns:
        Trained model
    """
    # Get trajectory data
    if collect_new_data:
        print("Collecting new trajectory data...")
        agent_trajectories = collect_agent_trajectories(
            num_agents=20, 
            episodes_per_agent=5, 
            max_steps=50, 
            save_dir='data'
        )
    elif data_path:
        print(f"Loading trajectory data from {data_path}...")
        with open(data_path, 'rb') as f:
            agent_trajectories = pickle.load(f)
    else:
        raise ValueError("Either collect_new_data must be True or data_path must be provided")
    
    # Split data into train/test sets
    agent_ids = list(agent_trajectories.keys())
    np.random.shuffle(agent_ids)
    
    # Use 80% for training, 20% for testing
    train_size = int(0.8 * len(agent_ids))
    train_ids = agent_ids[:train_size]
    test_ids = agent_ids[train_size:]
    
    train_trajectories = {agent_id: agent_trajectories[agent_id] for agent_id in train_ids}
    test_trajectories = {agent_id: agent_trajectories[agent_id] for agent_id in test_ids}
    
    # Prepare data loaders
    if experiment == 1:
        print("Preparing data for Experiment 1: Single past MDP with full trajectory")
        train_loader = prepare_experiment1_data(train_trajectories, batch_size, input_channels)
        test_loader = prepare_experiment1_data(test_trajectories, batch_size, input_channels)
    else:
        print("Preparing data for Experiment 2: Many past MDPs with single snapshots")
        train_loader = prepare_experiment2_data(train_trajectories, batch_size, input_channels, max_snapshots)
        test_loader = prepare_experiment2_data(test_trajectories, batch_size, input_channels, max_snapshots)
    
    # Create model
    model = GoalInferenceToMNet(
        input_dim=input_channels,
        state_dim=input_channels * input_channels,  # Assuming square input
        hidden_dim=hidden_dim,
        char_dim=char_dim,
        num_actions=num_actions,
        num_objects=num_objects,
        experiment=experiment
    )
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    action_loss_fn = nn.CrossEntropyLoss()
    consumption_loss_fn = nn.BCELoss()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create experiment-specific directory
    exp_dir = os.path.join(save_dir, f"experiment_{experiment}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        action_losses = 0
        consumption_losses = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Extract data based on experiment
            if experiment == 1:
                past_trajectory = batch['past_trajectory'].squeeze(1)
                query_state = batch['query_state']
                consumption_target = batch['consumption_target']
                action_targets = batch['action_targets']
                
                # Forward pass
                action_pred, consumption_pred, sr_pred = model(past_trajectory, query_state)
                
                # Calculate losses
                action_loss = action_loss_fn(action_pred, torch.argmax(action_targets.squeeze(1), dim=1))
            else:
                past_snapshots = batch['past_snapshots']
                query_state = batch['query_state']
                consumption_target = batch['consumption_target']
                action_target = batch['action_target']
                
                # Forward pass
                action_pred, consumption_pred, sr_pred = model(past_snapshots, query_state)
                
                # Calculate losses
                action_loss = action_loss_fn(action_pred, action_target)
            
            # Common losses
            consumption_loss = consumption_loss_fn(consumption_pred, consumption_target)
            
            # Total loss
            loss = action_loss + consumption_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            action_losses += action_loss.item()
            consumption_losses += consumption_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'action_loss': action_losses / (progress_bar.n + 1),
                'consumption_loss': consumption_losses / (progress_bar.n + 1)
            })
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, "
              f"Action Loss: {action_losses/len(train_loader):.4f}, "
              f"Consumption Loss: {consumption_losses/len(train_loader):.4f}")
        
        # Evaluate on test set
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_goal_inference(model, test_loader, experiment)
            print(f"Test - Action Acc: {metrics['action_accuracy']:.4f}, "
                  f"Consumption Acc: {metrics['consumption_accuracy']:.4f}, "
                  f"F1: {metrics['f1']:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(exp_dir, 'best_model.pt'))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(exp_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    
    # Final evaluation
    print("Final evaluation on test set...")
    metrics = evaluate_goal_inference(model, test_loader, experiment)
    print(f"Test - Action Acc: {metrics['action_accuracy']:.4f}, "
          f"Consumption Acc: {metrics['consumption_accuracy']:.4f}, "
          f"F1: {metrics['f1']:.4f}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ToMNet for goal inference")
    
    parser.add_argument("--experiment", type=int, default=1, choices=[1, 2],
                        help="Experiment to run (1 or 2)")
    parser.add_argument("--input_channels", type=int, default=3,
                        help="Number of input channels")
    parser.add_argument("--char_dim", type=int, default=2,
                        help="Character embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension")
    parser.add_argument("--num_actions", type=int, default=5,
                        help="Number of possible actions")
    parser.add_argument("--num_objects", type=int, default=4,
                        help="Number of possible objects")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--max_snapshots", type=int, default=10,
                        help="Maximum number of snapshots for Experiment 2")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to saved trajectory data")
    parser.add_argument("--collect_new_data", action="store_true",
                        help="Whether to collect new trajectory data")
    
    args = parser.parse_args()
    
    train_goal_inference(
        experiment=args.experiment,
        input_channels=args.input_channels,
        char_dim=args.char_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
        num_objects=args.num_objects,
        batch_size=args.batch_size,
        max_snapshots=args.max_snapshots,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        data_path=args.data_path,
        collect_new_data=args.collect_new_data
    )