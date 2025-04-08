import numpy as np
import collections

class TrajectoryCollector:
    """Collects and organizes agent trajectories for ToMNet training."""
    
    def __init__(self, environment, max_trajectory_length=100):
        self.environment = environment
        self.max_trajectory_length = max_trajectory_length
        self.trajectories = {}  # Map of species -> list of trajectories
        self.episode_buffers = {}  # Current episode trajectories by agent ID
        
    def reset(self):
        """Reset current episode trajectories."""
        self.episode_buffers = {}
    
    def record_agent_trajectory(self, agent):
        """
        Record an agent's full trajectory at the end of an episode.
        
        Args:
            agent: Agent that has completed an episode
        """
        # Use the agent's built-in trajectory
        if not agent.trajectory:
            return
            
        agent_id = agent.id
        species = agent.species
        
        # Initialize species in trajectories dict if needed
        if species not in self.trajectories:
            self.trajectories[species] = []
            
        # Store trajectory
        self.trajectories[species].append(agent.trajectory)
        
        # Also store in episode buffer
        self.episode_buffers[agent_id] = agent.trajectory
    
    def record_state(self, agent):
        """
        Record the current state of an agent in the environment.
        This is a legacy method - prefer using record_agent_trajectory.
        
        Args:
            agent: The agent to record
        """
        if agent.id not in self.episode_buffers:
            self.episode_buffers[agent.id] = []
            
        # Use the agent's most recent state-action pair if available
        if agent.trajectory:
            latest_state_action = agent.trajectory[-1]
            
            # Check if this is a new state
            if not self.episode_buffers[agent.id] or latest_state_action[0]['position'] != self.episode_buffers[agent.id][-1]['position']:
                # Encode the state
                state_encoding = self._encode_agent_state(agent)
                self.episode_buffers[agent.id].append(state_encoding)
                
                # Trim if needed
                if len(self.episode_buffers[agent.id]) > self.max_trajectory_length:
                    self.episode_buffers[agent.id].pop(0)
    
    def _encode_agent_state(self, agent):
        """
        Encode agent state as a feature vector.
        Returns a dictionary of features.
        """
        # Get basic position and node
        position = agent.position
        current_node = agent.current_node
        goal_node = agent.goal_node
        
        # Calculate distance to goal if possible
        dist_to_goal = 0
        if current_node and goal_node and hasattr(self.environment, 'get_path_length'):
            try:
                dist_to_goal = self.environment.get_path_length([current_node, goal_node])
            except:
                # If path finding fails, use Euclidean distance
                goal_pos = self.environment.get_node_coordinates(goal_node)
                dist_to_goal = ((position[0] - goal_pos[0])**2 + (position[1] - goal_pos[1])**2) ** 0.5
        
        # The state encoding should include relevant features for prediction
        state = {
            'position': position,
            'node': current_node,
            'goal_node': goal_node,
            'dist_to_goal': dist_to_goal,
            'species': agent.species
        }
        
        return state
        
    def finalize_episode(self):
        """
        Finalize the current episode and store completed trajectories.
        """
        # This is now simpler since we're using the agent's built-in trajectories
        for agent_id, states in self.episode_buffers.items():
            if len(states) < 2:  # Need at least start and end
                continue
                
            # Get agent species from the first state
            species = states[0]['species']
            
            # Store trajectory by species
            if species not in self.trajectories:
                self.trajectories[species] = []
                
            # Clone the states to avoid reference issues
            trajectory = [dict(state) for state in states]
            self.trajectories[species].append(trajectory)
        
        # Reset episode buffers
        self.reset()
        
    def get_trajectory_tensor(self, trajectories, normalize=True, fixed_length=50):
        """
        Convert a list of trajectories to tensor format for ToMNet.
        
        Args:
            trajectories: List of trajectories (each a list of state dicts)
            normalize: Whether to normalize features
            fixed_length: Fixed length for all trajectories (will pad or truncate)
            
        Returns:
            Tensor of shape (num_trajs, fixed_length, feature_dim)
        """
        import torch
        import numpy as np
        
        # Extract features from trajectories
        num_trajs = len(trajectories)
        
        # Feature extraction functions
        def extract_position(state):
            return list(state['position'])
            
        def extract_direction(states, i):
            if i < len(states) - 1:
                pos1 = states[i]['position']
                pos2 = states[i+1]['position']
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist > 0:
                    return [dx/dist, dy/dist]
            return [0, 0]
            
        def extract_goal_distance(state):
            return [state['dist_to_goal']]
        
        # Initialize tensor arrays with fixed length for all trajectories
        features = []
        masks = []
        
        for traj in trajectories:
            # Either truncate or pad to fixed_length
            traj_features = []
            traj_len = min(len(traj), fixed_length)  # Truncate if needed
            
            for i in range(fixed_length):
                if i < traj_len:
                    # Extract features for this position
                    position = extract_position(traj[i])
                    direction = extract_direction(traj, i)
                    goal_dist = extract_goal_distance(traj[i])
                    
                    # Combine features
                    state_features = position + direction + goal_dist
                    traj_features.append(state_features)
                else:
                    # Padding with zeros
                    traj_features.append([0] * 5)  # 5 features
                    
            features.append(traj_features)
            
            # Create mask (1 for actual data, 0 for padding)
            mask = [1] * traj_len + [0] * (fixed_length - traj_len)
            masks.append(mask)
            
        # Convert to numpy arrays
        features_array = np.array(features, dtype=np.float32)
        masks_array = np.array(masks, dtype=np.float32)
        
        # Normalize if requested
        if normalize:
            # Compute means and stds only on real data (using the masks)
            means = np.sum(features_array * masks_array[:, :, np.newaxis], axis=(0, 1)) / np.sum(masks_array)
            stds = np.sqrt(np.sum(((features_array - means) * masks_array[:, :, np.newaxis])**2, axis=(0, 1)) / np.sum(masks_array))
            stds[stds == 0] = 1.0  # Avoid division by zero
            
            # Normalize
            features_array = (features_array - means) / stds
            
        # Convert to torch tensors
        features_tensor = torch.tensor(features_array, dtype=torch.float32)
        masks_tensor = torch.tensor(masks_array, dtype=torch.float32)
        
        return features_tensor, masks_tensor