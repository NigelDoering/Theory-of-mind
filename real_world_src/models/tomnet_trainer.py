import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from real_world_src.models.tomnet import ToMNet

class TrajectoryDataset(Dataset):
    """Dataset for ToMnet training."""
    
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Process past_trajectories
        past_trajs = self._process_trajectories(example['past_trajectories'])
        
        # Process recent trajectory
        recent_traj = torch.tensor(example['recent_trajectory'], dtype=torch.float32)
        
        # Process current state
        current_state = torch.tensor(example['current_state'], dtype=torch.float32)
        
        # Process target (next position)
        target = torch.tensor(example['target'], dtype=torch.float32)
        
        return past_trajs, recent_traj, current_state, target
    
    def _process_trajectories(self, trajectories):
        """Convert trajectory list to fixed-length tensor."""
        # For simplicity, just use the state encodings
        # In a full implementation, we need to handle variable length sequences
        max_traj_len = 20
        max_num_trajs = 5
        
        # Fixed size tensor
        result = torch.zeros((max_num_trajs, max_traj_len, 5), dtype=torch.float32)
        
        for i, traj in enumerate(trajectories[:max_num_trajs]):
            for j, step in enumerate(traj[:max_traj_len]):
                if 'state_encoding' in step:
                    result[i, j] = torch.tensor(step['state_encoding'], dtype=torch.float32)
        
        return result

class ToMNetTrainer:
    """Trainer for the ToMNet model."""
    
    def __init__(self, input_dim, state_dim, hidden_dim, output_dim, lr=0.001, device=None):
        """
        Initialize the ToMNet trainer.
        
        Args:
            input_dim: Dimension of input features
            state_dim: Dimension of current state features 
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output predictions
            lr: Learning rate
            device: Device to train on (cpu or cuda)
        """
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the model
        self.model = ToMNet(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def train(self, trajectory_collector, batch_size=32, epochs=20):
        """
        Train the ToMNet model.
        
        Args:
            trajectory_collector: TrajectoryCollector with recorded trajectories
            batch_size: Batch size for training
            epochs: Number of epochs
            
        Returns:
            Trained ToMNet model
        """
        print(f"Starting ToMNet training on {self.device}...")
        
        # Prepare data
        all_species = list(trajectory_collector.trajectories.keys())
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # Process each species
            for species in all_species:
                trajectories = trajectory_collector.trajectories[species]
                
                if len(trajectories) < 3:  # Need at least a few trajectories
                    print(f"Skipping {species} - not enough data")
                    continue
                
                # Split trajectories into past, recent, and future
                train_samples = []
                
                for i, trajectory in enumerate(trajectories):
                    if len(trajectory) < 5:  # Skip very short trajectories
                        continue
                    
                    # Use up to 5 other trajectories as past trajectories
                    past_traj_indices = [j for j in range(len(trajectories)) if j != i]
                    past_traj_indices = past_traj_indices[:5]  # Limit to 5
                    
                    if not past_traj_indices:  # Skip if no past trajectories
                        continue
                    
                    past_trajs = [trajectories[j] for j in past_traj_indices]
                    
                    # Use first half as recent, second half as future for prediction
                    split_point = len(trajectory) // 2
                    if split_point < 2:  # Ensure at least 2 points for recent
                        continue
                        
                    recent_traj = trajectory[:split_point]
                    future_traj = trajectory[split_point:]
                    
                    train_samples.append((past_trajs, recent_traj, future_traj))
                
                # Skip if not enough samples
                if len(train_samples) < batch_size:
                    print(f"Skipping {species} - not enough valid samples")
                    continue
                
                # Train in batches
                for batch_start in range(0, len(train_samples), batch_size):
                    batch_end = min(batch_start + batch_size, len(train_samples))
                    batch = train_samples[batch_start:batch_end]
                    
                    # Prepare batch data
                    past_batch = []
                    recent_batch = []
                    future_batch = []
                    
                    for past_trajs, recent_traj, future_traj in batch:
                        # Prepare past trajectories
                        past_tensor, _ = trajectory_collector.get_trajectory_tensor(past_trajs)
                        past_batch.append(past_tensor)
                        
                        # Prepare recent trajectory
                        fixed_length = 50  # Use same fixed length as in get_trajectory_tensor
                        recent_tensor, _ = trajectory_collector.get_trajectory_tensor(
                            [recent_traj], fixed_length=fixed_length)
                        recent_batch.append(recent_tensor.squeeze(0))  # Remove batch dim
                        
                        # Prepare future trajectory (target)
                        future_tensor, _ = trajectory_collector.get_trajectory_tensor(
                            [future_traj], fixed_length=fixed_length)
                        future_batch.append(future_tensor.squeeze(0))  # Remove batch dim
                    
                    # Convert to tensors and move to device
                    past_tensor = torch.stack(past_batch).to(self.device)
                    recent_tensor = torch.stack(recent_batch).to(self.device)
                    
                    # Current state is the last state of recent trajectory
                    current_state = torch.stack([r[-1] for r in recent_batch]).to(self.device)
                    
                    # Target is the future trajectory
                    target = torch.stack(future_batch).to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(past_tensor, recent_tensor, current_state)
                    
                    # Output and target are both [batch_size, seq_len, output_dim]
                    pred_steps = min(output.shape[1], target.shape[1], 5)
                    
                    # Explicitly print shapes for debugging
                    if epoch == 0 and batch_count == 0:
                        print(f"Output shape: {output.shape}")
                        print(f"Target shape: {target.shape}")
                        print(f"Using first {pred_steps} steps for prediction")
                    
                    # Compute loss on matching dimensions
                    loss = self.criterion(output[:, :pred_steps, :], target[:, :pred_steps, :])
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
            
            # Print epoch statistics
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, No valid batches!")
        
        print("Training completed!")
        return self.model