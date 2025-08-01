import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
import json
import pickle
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.models.tomnet_causal_dataloader import CampusDataLoader

# -----------------------------
# Trajectory Encoder (TGAT/Transformer)
# -----------------------------
class TrajectoryEncoder(nn.Module):
    """
    Encodes an agent's trajectory (sequence of node indices and times) using a Transformer.
    """
    def __init__(self, num_nodes, node_feat_dim, time_emb_dim, hidden_dim, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        
        # Node embedding layer
        self.node_embedding = nn.Embedding(num_nodes, node_feat_dim)
        
        # Time embedding using sinusoidal encoding
        self.time_embedding = nn.Linear(1, time_emb_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_feat_dim + time_emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(node_feat_dim + time_emb_dim, hidden_dim)
        
    def forward(self, node_ids, timestamps, mask=None):
        """
        Args:
            node_ids: (batch_size, seq_len) - Node IDs for each trajectory
            timestamps: (batch_size, seq_len) - Timestamps for each node
            mask: (batch_size, seq_len) - Mask for padding (1 for valid, 0 for padding)
        """
        batch_size, seq_len = node_ids.shape
        
        # Embed nodes
        node_emb = self.node_embedding(node_ids)  # (batch_size, seq_len, node_feat_dim)
        
        # Embed timestamps (normalize to [0, 1] range)
        timestamps_norm = timestamps.unsqueeze(-1) / (timestamps.max() + 1e-8)
        time_emb = self.time_embedding(timestamps_norm)  # (batch_size, seq_len, time_emb_dim)
        
        # Concatenate node and time embeddings
        combined_emb = torch.cat([node_emb, time_emb], dim=-1)  # (batch_size, seq_len, node_feat_dim + time_emb_dim)
        
        # Create attention mask for padding
        if mask is None:
            mask = (node_ids != 0).float()  # Assume 0 is padding
        attention_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        
        # Apply transformer
        transformer_out = self.transformer(combined_emb, src_key_padding_mask=(mask == 0))
        
        # Global pooling (mean over valid tokens)
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = (transformer_out * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        # Project to final output dimension
        output = self.output_proj(pooled)  # (batch_size, hidden_dim)
        
        return output

# -----------------------------
# World Graph Encoder (GraphSAGE/GAT)
# -----------------------------
class WorldGraphEncoder(nn.Module):
    """
    Encodes the static world graph using Graph Neural Networks.
    """
    def __init__(self, node_feat_dim, hidden_dim, n_layers=2, use_gat=True):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        
        # Node feature projection
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_gat:
                conv = GATConv(hidden_dim if i > 0 else hidden_dim, 
                             hidden_dim // 4,  # GAT uses multiple heads
                             heads=4,
                             dropout=0.1)
            else:
                conv = SAGEConv(hidden_dim if i > 0 else hidden_dim, 
                              hidden_dim,
                              aggr='mean')
            self.conv_layers.append(conv)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: (num_nodes, node_feat_dim) - Node features
            edge_index: (2, num_edges) - Graph connectivity
            batch: (num_nodes,) - Batch assignment for nodes
        """
        # Project node features
        x = self.node_proj(x)
        
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Global pooling (mean over all nodes)
        if batch is not None:
            # If we have batch information, pool per graph
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            # Otherwise, take mean over all nodes
            x = x.mean(dim=0, keepdim=True)
        
        # Final projection
        x = self.output_proj(x)
        
        return x

# -----------------------------
# ToM Graph Encoder (Combined)
# -----------------------------
class ToMGraphEncoder(nn.Module):
    """
    Combined encoder that processes both trajectory and world graph data.
    """
    def __init__(self, num_nodes, node_feat_dim, time_emb_dim, hidden_dim, n_layers=2, n_heads=4, dropout=0.1, use_gat=True):
        super().__init__()
        
        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
            num_nodes=num_nodes,
            node_feat_dim=node_feat_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # World graph encoder
        self.world_encoder = WorldGraphEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            use_gat=use_gat
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, trajectory_data, graph_data):
        """
        Args:
            trajectory_data: Dict with 'node_ids', 'timestamps', 'mask'
            graph_data: Dict with 'x', 'edge_index', 'batch'
        """
        # Encode trajectory
        traj_encoding = self.trajectory_encoder(
            trajectory_data['node_ids'],
            trajectory_data['timestamps'],
            trajectory_data.get('mask', None)
        )
        
        # Encode world graph
        graph_encoding = self.world_encoder(
            graph_data['x'],
            graph_data['edge_index'],
            graph_data.get('batch', None)
        )
        
        # Ensure same batch size
        if traj_encoding.shape[0] != graph_encoding.shape[0]:
            # Repeat graph encoding for each trajectory
            graph_encoding = graph_encoding.expand(traj_encoding.shape[0], -1)
        
        # Fuse encodings
        combined = torch.cat([traj_encoding, graph_encoding], dim=-1)
        fused_encoding = self.fusion(combined)
        
        return fused_encoding

# -----------------------------
# Main Function for Testing
# -----------------------------
def main():
    """Test the ToM Graph Encoder with real campus data."""
    print("Testing ToM Graph Encoder with real campus data...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize data loader
    data_loader = CampusDataLoader()
    
    # Prepare sample data
    sample_agent_ids = [0, 1, 2, 3, 4]  # Sample agent IDs
    
    print("\nPreparing trajectory data...")
    trajectory_data = data_loader.prepare_trajectory_batch(sample_agent_ids, max_seq_len=30)
    
    print("\nPreparing graph data...")
    graph_data = data_loader.prepare_graph_data()
    
    # Move data to device
    for key in trajectory_data:
        trajectory_data[key] = trajectory_data[key].to(device)
    for key in graph_data:
        graph_data[key] = graph_data[key].to(device)
    
    # Initialize encoder
    encoder = ToMGraphEncoder(
        num_nodes=len(data_loader.node_id_mapping),  # Number of unique nodes
        node_feat_dim=4,  # 2 coords + 2 padding features
        time_emb_dim=16,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        use_gat=True
    ).to(device)
    
    # Update the node embedding size based on the actual number of unique nodes
    num_unique_nodes = len(data_loader.node_id_mapping)
    encoder.trajectory_encoder.node_embedding = nn.Embedding(num_unique_nodes, 4).to(device)
    print(f"Updated node embedding size to {num_unique_nodes}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = encoder(trajectory_data, graph_data)
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0][:5]}")
    
    print("\n✅ ToM Graph Encoder test completed successfully!")
    
    # Print model summary
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    main() 