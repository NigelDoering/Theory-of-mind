import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CharacterNetwork(nn.Module):
    """Processes past trajectories to extract persistent agent traits."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CharacterNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, past_trajectories):
        """
        Args:
            past_trajectories: Tensor of shape [batch_size, num_trajs, seq_len, input_dim]
                containing past movement data for several episodes
        """
        batch_size, num_trajs, seq_len, input_dim = past_trajectories.shape
        
        # Reshape to process all trajectories
        x = past_trajectories.view(batch_size * num_trajs, seq_len, input_dim)
        
        # Process through LSTM
        _, (h_n, _) = self.lstm(x)
        
        # Get the last layer's hidden state
        h_n = h_n[-1]  # Shape: [batch_size * num_trajs, hidden_dim]
        
        # Reshape back to batch_size x num_trajs x hidden_dim
        h_n = h_n.view(batch_size, num_trajs, -1)
        
        # Aggregate over trajectories (mean pooling)
        character_embedding = torch.mean(h_n, dim=1)  # Shape: [batch_size, hidden_dim]
        
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

class ToMNet(nn.Module):
    """Complete Theory of Mind neural network combining all components."""
    def __init__(self, input_dim, state_dim, hidden_dim=64, char_dim=32, mental_dim=32, 
                 output_dim=5, seq_len=5):
        super(ToMNet, self).__init__()
        self.character_net = CharacterNetwork(input_dim, hidden_dim, char_dim)
        self.mental_state_net = MentalStateNetwork(input_dim, hidden_dim, mental_dim, char_dim)
        self.prediction_net = PredictionNetwork(
            char_dim, mental_dim, state_dim, hidden_dim, output_dim, seq_len)
        
    def forward(self, past_trajectories, recent_trajectory, current_state):
        """
        Args:
            past_trajectories: Multiple past episodes of behavior [batch, num_trajs, seq_len, features]
            recent_trajectory: Recent movements in current episode [batch, seq_len, features]
            current_state: Current environment state [batch, state_dim]
        """
        # First extract character traits from past trajectories
        character = self.character_net(past_trajectories)
        
        # Then use character traits to help infer mental state from recent trajectory
        mental_state = self.mental_state_net(recent_trajectory, character)
        
        # Finally, predict future actions based on character, mental state, and current state
        predictions = self.prediction_net(character, mental_state, current_state)
        return predictions