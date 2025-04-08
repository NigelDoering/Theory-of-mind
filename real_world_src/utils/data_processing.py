import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def preprocess_trajectory(trajectory, input_dim=None):
    """
    Preprocess a single trajectory for ToMNet input.
    
    Args:
        trajectory: List of states/actions in grid world format
        input_dim: Expected input dimension
        
    Returns:
        Tensor with shape [seq_len, input_dim]
    """
    # If input is already a tensor, ensure correct shape
    if isinstance(trajectory, torch.Tensor):
        if input_dim and trajectory.shape[-1] != input_dim:
            raise ValueError(f"Trajectory has dimension {trajectory.shape[-1]}, expected {input_dim}")
        return trajectory
    
    # Convert list of observations to tensor
    traj_tensor = torch.tensor(trajectory, dtype=torch.float32)
    
    # Ensure correct shape
    if input_dim and traj_tensor.shape[-1] != input_dim:
        # If shape doesn't match, try to reshape or pad
        if len(traj_tensor.shape) == 1:
            # Single dimensional data, reshape to [seq_len, 1]
            traj_tensor = traj_tensor.unsqueeze(-1)
            # Pad if necessary
            if input_dim > 1:
                pad_size = input_dim - traj_tensor.shape[-1]
                traj_tensor = torch.nn.functional.pad(traj_tensor, (0, pad_size))
        else:
            raise ValueError(f"Cannot reshape trajectory of shape {traj_tensor.shape} to include dimension {input_dim}")
    
    return traj_tensor

def prepare_trajectory_data(trajectory, input_channels=3):
    """
    Process a trajectory from the real-world environment for ToMNet input.
    
    Args:
        trajectory: List of state-action pairs from the real-world environment
        input_channels: Number of input channels expected by the model
        
    Returns:
        Processed trajectory tensor with shape [seq_len, input_channels]
    """
    # Extract states and actions from trajectory
    states = [t[0] for t in trajectory]
    actions = [t[1] for t in trajectory[:-1]] + [None]  # Last action is None
    
    # Convert to appropriate format based on your environment
    processed_data = []
    for state, action in zip(states, actions):
        # Convert state to appropriate feature representation
        state_features = process_state(state, input_channels)
        
        # Add action if available
        if action is not None:
            action_features = process_action(action)
            combined_features = np.concatenate([state_features, action_features], axis=-1)
        else:
            combined_features = state_features
            
        processed_data.append(combined_features)
    
    # Convert to tensor
    return torch.tensor(np.array(processed_data), dtype=torch.float32)

def process_state(state, input_channels=3):
    """
    Process a state from the real-world environment.
    
    Args:
        state: State from the real-world environment
        input_channels: Number of input channels
        
    Returns:
        Processed state features
    """
    # Adapt this function based on your specific state representation
    if hasattr(state, 'position'):
        # If state has position attribute (like in campus environment)
        x, y = state.position
        
        # Create a spatial representation
        # For example, a one-hot encoding of the position in a grid
        grid_size = 11  # Adjust based on your environment
        grid = np.zeros((grid_size, grid_size, input_channels))
        
        # Normalize position to grid coordinates
        grid_x = min(int(x * grid_size), grid_size - 1)
        grid_y = min(int(y * grid_size), grid_size - 1)
        
        # Mark agent position
        grid[grid_x, grid_y, 0] = 1.0
        
        # Add additional features if available (objects, walls, etc.)
        # This depends on your specific environment
        
        return grid
    else:
        # Fallback for other state representations
        # Convert the state to a feature vector
        return np.array(state).reshape(-1, input_channels)

def process_action(action):
    """
    Process an action from the real-world environment.
    
    Args:
        action: Action from the real-world environment
        
    Returns:
        One-hot encoding of the action
    """
    # Adapt this function based on your specific action representation
    if isinstance(action, int):
        # If action is an integer (like in discrete action spaces)
        num_actions = 5  # Adjust based on your environment
        one_hot = np.zeros(num_actions)
        one_hot[action] = 1.0
        return one_hot
    elif isinstance(action, (list, tuple, np.ndarray)):
        # If action is a vector (like in continuous action spaces)
        return np.array(action)
    else:
        # For other action types, convert to an appropriate representation
        return np.array([float(action)])

class Experiment1Dataset(Dataset):
    """Dataset for Experiment 1: Single past MDP with full trajectory."""
    
    def __init__(self, agent_trajectories, input_channels=3):
        self.examples = []
        self.input_channels = input_channels
        
        # Process each agent's trajectories
        for agent_id, trajectories in agent_trajectories.items():
            if len(trajectories) < 2:
                continue  # Need at least one past and one current trajectory
                
            # Use the first trajectory as past trajectory
            past_traj = trajectories[0]
            
            # Use the remaining trajectories as current/query trajectories
            for current_traj in trajectories[1:]:
                if len(current_traj) == 0:
                    continue
                    
                # Process past trajectory
                processed_past = prepare_trajectory_data(past_traj, input_channels)
                
                # Extract query state (initial state of current trajectory)
                query_state = process_state(current_traj[0][0], input_channels)
                
                # Extract targets
                # Action target: First action of current trajectory
                action_target = process_action(current_traj[0][1])
                
                # Consumption target: Extract from final state if available
                consumption_target = np.zeros(4)  # Default: no consumption
                if len(current_traj) > 1:
                    final_state = current_traj[-1][0]
                    if hasattr(final_state, 'consumed_objects'):
                        consumption_target = np.array(final_state.consumed_objects)
                
                self.examples.append({
                    'past_trajectory': processed_past,
                    'query_state': query_state,
                    'action_target': action_target,
                    'consumption_target': consumption_target
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'past_trajectory': example['past_trajectory'].unsqueeze(0),  # Add trajectory dimension
            'query_state': torch.tensor(example['query_state'], dtype=torch.float32),
            'action_targets': torch.tensor(example['action_target'], dtype=torch.float32).unsqueeze(0),
            'consumption_target': torch.tensor(example['consumption_target'], dtype=torch.float32)
        }

class Experiment2Dataset(Dataset):
    """Dataset for Experiment 2: Many past MDPs with single snapshots each."""
    
    def __init__(self, agent_trajectories, input_channels=3, max_snapshots=10):
        self.examples = []
        self.input_channels = input_channels
        self.max_snapshots = max_snapshots
        
        # Process each agent's trajectories
        for agent_id, trajectories in agent_trajectories.items():
            if len(trajectories) < 2:
                continue  # Need at least one past trajectory and one current
                
            # Extract snapshots from past trajectories
            past_snapshots = []
            for traj in trajectories[:-1]:  # All except the last trajectory
                for step in traj:
                    state, action = step
                    snapshot = {
                        'state': process_state(state, input_channels),
                        'action': action
                    }
                    past_snapshots.append(snapshot)
                    
                    if len(past_snapshots) >= max_snapshots:
                        break
                if len(past_snapshots) >= max_snapshots:
                    break
            
            # If we have past snapshots, use the last trajectory as current/query
            if past_snapshots and trajectories[-1]:
                current_traj = trajectories[-1]
                
                # Process query state (initial state of current trajectory)
                query_state = process_state(current_traj[0][0], input_channels)
                
                # Extract targets
                # Action target: First action of current trajectory
                action_target = current_traj[0][1]
                
                # Consumption target: Extract from final state if available
                consumption_target = np.zeros(4)  # Default: no consumption
                if len(current_traj) > 1:
                    final_state = current_traj[-1][0]
                    if hasattr(final_state, 'consumed_objects'):
                        consumption_target = np.array(final_state.consumed_objects)
                
                # Process past snapshots
                processed_snapshots = np.zeros((self.max_snapshots, *past_snapshots[0]['state'].shape))
                for i, snapshot in enumerate(past_snapshots):
                    if i >= self.max_snapshots:
                        break
                    processed_snapshots[i] = snapshot['state']
                
                self.examples.append({
                    'past_snapshots': processed_snapshots,
                    'num_snapshots': len(past_snapshots),
                    'query_state': query_state,
                    'action_target': action_target,
                    'consumption_target': consumption_target
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'past_snapshots': torch.tensor(example['past_snapshots'], dtype=torch.float32),
            'num_snapshots': example['num_snapshots'],
            'query_state': torch.tensor(example['query_state'], dtype=torch.float32),
            'action_target': torch.tensor(example['action_target'], dtype=torch.long),
            'consumption_target': torch.tensor(example['consumption_target'], dtype=torch.float32)
        }

def prepare_experiment1_data(agent_trajectories, batch_size=32, input_channels=3):
    """
    Prepare data for Experiment 1.
    
    Args:
        agent_trajectories: Dictionary mapping agent IDs to list of trajectories
        batch_size: Batch size for training
        input_channels: Number of input channels
        
    Returns:
        DataLoader for Experiment 1
    """
    dataset = Experiment1Dataset(agent_trajectories, input_channels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def prepare_experiment2_data(agent_trajectories, batch_size=32, input_channels=3, max_snapshots=10):
    """
    Prepare data for Experiment 2.
    
    Args:
        agent_trajectories: Dictionary mapping agent IDs to list of trajectories
        batch_size: Batch size for training
        input_channels: Number of input channels
        max_snapshots: Maximum number of snapshots to use per agent
        
    Returns:
        DataLoader for Experiment 2
    """
    dataset = Experiment2Dataset(agent_trajectories, input_channels, max_snapshots)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)