import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CharacterNetwork(nn.Module):
    """Processes past trajectories to extract persistent agent traits."""
    def __init__(self, input_dim, hidden_dim, output_dim, use_resnet=True):
        super(CharacterNetwork, self).__init__()
        self.use_resnet = use_resnet
        
        if use_resnet:
            # 5-layer ResNet as described in the paper
            self.resnet = self._create_resnet(input_dim, 32)
        else:
            self.conv = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
            
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True, num_layers=2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # For Experiment 2: direct mapping from single observation to embedding
        self.single_obs_fc = nn.Linear(32, output_dim)
        
    def _create_resnet(self, in_channels, out_channels):
        """Create a 5-layer ResNet as specified in the paper."""
        layers = []
        # Initial conv layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        # 4 residual blocks
        for _ in range(4):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            ])
            # Skip connection
            layers.append(nn.ReLU())
            
        return nn.Sequential(*layers)
    
    def forward(self, past_trajectories, single_observation_mode=False):
        """
        Args:
            past_trajectories: Tensor of shape [batch_size, num_trajs, seq_len, input_dim]
                containing past movement data for several episodes
            single_observation_mode: If True, treat each trajectory as a single observation
                (for Experiment 2)
        """
        batch_size, num_trajs, seq_len, input_dim = past_trajectories.shape
        
        if single_observation_mode:
            # Experiment 2: Process each single observation independently
            # Reshape to process all observations
            x = past_trajectories.view(batch_size * num_trajs, 1, seq_len, input_dim)
            
            if self.use_resnet:
                x = self.resnet(x)
            else:
                x = F.relu(self.conv(x))
                
            # Average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size * num_trajs, -1)
            
            # Direct mapping to character embedding
            char_embeds = self.single_obs_fc(x)
            char_embeds = char_embeds.view(batch_size, num_trajs, -1)
            
            # Sum across observations as described in Experiment 2
            character_embedding = torch.sum(char_embeds, dim=1)
            
        else:
            # Experiment 1: Process full trajectory
            # Reshape to process all trajectories
            x = past_trajectories.view(batch_size * num_trajs, seq_len, input_dim)
            
            # Process through ResNet if available
            if self.use_resnet:
                # Reshape for 2D convolution
                x_spatial = x.view(batch_size * num_trajs * seq_len, 1, -1, input_dim)
                x_spatial = self.resnet(x_spatial)
                x_spatial = F.adaptive_avg_pool2d(x_spatial, (1, 1))
                x = x_spatial.view(batch_size * num_trajs, seq_len, -1)
            
            # Process through LSTM
            _, (h_n, _) = self.lstm(x)
            
            # Get the last layer's hidden state
            h_n = h_n[-1]  # Shape: [batch_size * num_trajs, hidden_dim]
            
            # Reshape back to batch_size x num_trajs x hidden_dim
            h_n = h_n.view(batch_size, num_trajs, -1)
            
            # For Experiment 1: Use a single trajectory
            # For compatibility with existing code, we'll still average if multiple trajectories
            character_embedding = torch.mean(h_n, dim=1)
            
            # Process through fully connected layers with ReLU activation
            character_embedding = F.relu(self.fc1(character_embedding))
            character_embedding = self.fc2(character_embedding)
        
        return character_embedding

class MentalStateNetwork(nn.Module):
    """Analyzes recent observations to infer current mental state, 
    incorporating character traits."""
    def __init__(self, input_dim, hidden_dim, output_dim, char_dim):
        super(MentalStateNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Add layers to process character information with recent trajectory
        self.char_fc = nn.Linear(char_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Layers to compute final mental state
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, recent_trajectory, character):
        """
        Args:
            recent_trajectory: Tensor of shape [batch_size, seq_len, input_dim]
                containing the recent movements in the current episode
            character: Character embedding from CharacterNetwork [batch_size, char_dim]
        """
        # Process recent trajectory through LSTM
        lstm_out, (h_n, _) = self.lstm(recent_trajectory)  # lstm_out: [batch, seq, hidden]
        
        # Get the final hidden state
        trajectory_features = h_n.squeeze(0)  # [batch_size, hidden_dim]
        
        # Process character embedding
        char_features = F.relu(self.char_fc(character))  # [batch_size, hidden_dim]
        
        # Compute attention weights based on character + trajectory features
        combined = torch.cat([trajectory_features, char_features], dim=1)
        
        # Integrate character traits with trajectory understanding
        mental_state = F.relu(self.fc1(combined))
        mental_state = self.fc2(mental_state)
        
        return mental_state

class PredictionNetwork(nn.Module):
    """Combines character and mental state to predict future actions."""
    def __init__(self, char_dim, mental_dim, state_dim, hidden_dim, output_dim, seq_len=5):
        super(PredictionNetwork, self).__init__()
        self.fc1 = nn.Linear(char_dim + mental_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.seq_len = seq_len
        
    def forward(self, character, mental_state, current_state):
        """
        Args:
            character: Character embedding from CharacterNetwork
            mental_state: Mental state embedding from MentalStateNetwork
            current_state: Current environment state
        """
        # Concatenate all inputs
        combined = torch.cat([character, mental_state, current_state], dim=1)
        
        # Process through fully connected layers with ReLU activation
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # Expand to create a sequence for LSTM input [batch, seq_len, hidden_dim]
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # Process through LSTM to generate sequence
        x, _ = self.lstm(x)
        
        # Project to output dimension for each step in sequence
        predictions = self.fc3(x)  # Shape: [batch_size, seq_len, output_dim]
        
        return predictions

class PredictionHeads(nn.Module):
    """Prediction heads for different outputs as described in the paper."""
    def __init__(self, input_dim, num_actions=5, num_objects=4):
        super(PredictionHeads, self).__init__()
        self.num_actions = num_actions
        self.num_objects = num_objects
        
        # Shared torso
        self.prediction_torso = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_actions)
        )
        
        # Consumption prediction head
        self.consumption_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_objects),
            nn.Sigmoid()
        )
        
        # Successor representation head
        self.sr_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)  # 3 channels for different discount factors
        )
    
    def forward(self, x):
        """
        Forward pass through prediction heads
        
        Args:
            x: Output from the prediction torso
            
        Returns:
            action_pred: Action probability distribution [batch_size, num_actions]
            consumption_pred: Object consumption probabilities [batch_size, num_objects]
            sr_pred: Successor representation predictions [batch_size, 3, height, width]
        """
        # Shared torso
        features = self.prediction_torso(x)
        
        # Action prediction (policy)
        action_logits = self.action_head(features)
        action_pred = F.softmax(action_logits, dim=1)
        
        # Consumption prediction
        consumption_pred = self.consumption_head(features)
        
        # Successor representation prediction
        sr_pred = self.sr_head(features)
        # Apply softmax independently for each discount factor
        batch_size, channels, height, width = sr_pred.shape
        sr_pred = sr_pred.view(batch_size, channels, -1)
        sr_pred = F.softmax(sr_pred, dim=2)
        sr_pred = sr_pred.view(batch_size, channels, height, width)
        
        return action_pred, consumption_pred, sr_pred

class ToMNet(nn.Module):
    """Complete Theory of Mind neural network combining all components."""
    def __init__(self, input_dim, state_dim, hidden_dim=64, char_dim=32, mental_dim=32, 
                 output_dim=5, seq_len=5, enable_goal_inference=False, num_objects=4):
        super(ToMNet, self).__init__()
        self.enable_goal_inference = enable_goal_inference
        
        # Original ToMNet components
        self.character_net = CharacterNetwork(input_dim, hidden_dim, char_dim)
        self.mental_state_net = MentalStateNetwork(input_dim, hidden_dim, mental_dim, char_dim)
        self.prediction_net = PredictionNetwork(
            char_dim, mental_dim, state_dim, hidden_dim, output_dim, seq_len)
        
        # Goal inference components (if enabled)
        if enable_goal_inference:
            self.goal_inference_heads = PredictionHeads(
                char_dim + state_dim, num_actions=output_dim, num_objects=num_objects)
        
    def forward(self, past_trajectories, recent_trajectory, current_state, query_state=None):
        """
        Args:
            past_trajectories: Multiple past episodes [batch, num_trajs, seq_len, features]
            recent_trajectory: Recent movements [batch, seq_len, features]
            current_state: Current state [batch, state_dim]
            query_state: Optional initial state for goal inference [batch, height, width, channels]
        """
        # Extract character traits from past trajectories
        character = self.character_net(past_trajectories)
        
        # Standard ToMNet forward path
        mental_state = self.mental_state_net(recent_trajectory, character)
        predictions = self.prediction_net(character, mental_state, current_state)
        
        # Goal inference if enabled and query_state is provided
        if self.enable_goal_inference and query_state is not None:
            # Spatialize the character embedding
            batch_size, height, width, channels = query_state.shape
            char_spatial = character.view(batch_size, -1, 1, 1).expand(-1, -1, height, width)
            
            # Prepare query state
            query_state = query_state.permute(0, 3, 1, 2)  # [batch, channels, height, width]
            combined = torch.cat([query_state, char_spatial], dim=1)
            
            # Get goal inference predictions
            action_pred, consumption_pred, sr_pred = self.goal_inference_heads(combined)
            
            return predictions, (action_pred, consumption_pred, sr_pred)
        
        return predictions


# class GoalInferenceToMNet(nn.Module):
#     """ToMNet specialized for goal inference experiments."""
#     def __init__(self, input_dim, state_dim, hidden_dim=64, char_dim=2, 
#                  num_actions=5, num_objects=4, experiment=1):
#         super(GoalInferenceToMNet, self).__init__()
#         self.experiment = experiment
        
#         # Create networks based on the experiment
#         if experiment == 1:
#             # Experiment 1: Single past MDP with full trajectory
#             self.character_net = CharacterNetwork(input_dim, hidden_dim, char_dim, use_resnet=True)
#         else:
#             # Experiment 2: Many past MDPs with single snapshot each
#             self.character_net = CharacterNetwork(input_dim, hidden_dim, char_dim, use_resnet=False)
            
#         # The state embedding dimension
#         self.state_embed_dim = input_dim + char_dim
        
#         # Prediction heads
#         self.prediction_heads = PredictionHeads(
#             self.state_embed_dim, num_actions, num_objects)
    
#     def forward(self, past_trajectories, query_state):
#         """
#         Forward pass for goal inference
        
#         Args:
#             past_trajectories: Past episodes [batch_size, num_trajs, seq_len, input_dim]
#             query_state: Initial state of new MDP [batch_size, height, width, channels]
            
#         Returns:
#             action_pred: Action probability distribution
#             consumption_pred: Object consumption probabilities
#             sr_pred: Successor representation predictions
#         """
#         # Extract character embedding
#         single_observation_mode = (self.experiment == 2)
#         character = self.character_net(past_trajectories, single_observation_mode)
        
#         # Spatialize the character embedding and concatenate with query state
#         batch_size, height, width, channels = query_state.shape
#         char_spatial = character.view(batch_size, self.state_embed_dim - channels, 1, 1).expand(-1, -1, height, width)
        
#         # Concatenate query state with character embedding
#         query_state = query_state.permute(0, 3, 1, 2)  # [batch, channels, height, width]
#         combined = torch.cat([query_state, char_spatial], dim=1)
        
#         # Make predictions
#         action_pred, consumption_pred, sr_pred = self.prediction_heads(combined)
        
#         return action_pred, consumption_pred, sr_pred