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

# -----------------------------
# Trajectory Encoder (TGAT/Transformer)
# -----------------------------
class TrajectoryEncoder(nn.Module):
    """
    Encodes an agent's trajectory (sequence of node indices and times) using a Transformer.
    """
    def __init__(self, node_feat_dim, time_emb_dim, hidden_dim, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        
        # Node embedding layer
        self.node_embedding = nn.Embedding(100000, node_feat_dim)  # Large enough for OSM node IDs
        
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
    def __init__(self, node_feat_dim, time_emb_dim, hidden_dim, n_layers=2, n_heads=4, dropout=0.1, use_gat=True):
        super().__init__()
        
        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
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
# Data Loader for Real Campus Data
# -----------------------------
class CampusDataLoader:
    """
    Loads and prepares data from the campus environment and trajectory files.
    """
    def __init__(self, data_dir="./data/1k/"):

        self.data_dir = data_dir
        self.env = None
        self.agents = None
        self.path_data = None
        self.goal_data = None
        self.node_id_mapping = {}  # Map OSM node IDs to sequential indices
        self.reverse_node_mapping = {}  # Map sequential indices back to OSM node IDs
        
        # Load data
        self._load_data()
        self._build_node_mapping()
        
    def _load_data(self):
        """Load all necessary data files."""
        print("Loading campus environment...")
        self.env = CampusEnvironment()
        
        print("Loading agents...")
        with open(os.path.join(self.data_dir, "agents.pkl"), 'rb') as f:
            self.agents = pickle.load(f)
        print(f"Loaded {len(self.agents)} agents")
        
        print("Loading path data...")
        with open(os.path.join(self.data_dir, "path_data.json"), 'r') as f:
            self.path_data = json.load(f)
        print(f"Path data keys: {len(self.path_data)}")
        
        print("Loading goal data...")
        with open(os.path.join(self.data_dir, "goal_data.json"), 'r') as f:
            self.goal_data = json.load(f)
        print(f"Goal data keys: {len(self.goal_data)}")
    
    def _build_node_mapping(self):
        """Build mapping from OSM node IDs to sequential indices."""
        print("Building node ID mapping...")
        
        # Collect all unique node IDs from path data
        all_node_ids = set()
        if self.path_data is not None:
            for agent_data in self.path_data.values():
                for segment in agent_data.values():
                    if isinstance(segment, list):
                        all_node_ids.update(segment)
        
        # Also add nodes from the environment graph
        if self.env is not None:
            all_node_ids.update(self.env.G.nodes())
        
        # Create mapping
        for i, node_id in enumerate(sorted(all_node_ids)):
            self.node_id_mapping[node_id] = i
            self.reverse_node_mapping[i] = node_id
        
        print(f"Mapped {len(self.node_id_mapping)} unique nodes")
        print(f"Node ID range: {min(self.node_id_mapping.keys())} to {max(self.node_id_mapping.keys())}")
        print(f"Sequential range: 0 to {len(self.node_id_mapping) - 1}")
    
    def prepare_trajectory_batch(self, agent_ids, max_seq_len=100):
        """
        Prepare trajectory data for a batch of agents.
        
        Args:
            agent_ids: List of agent IDs to process
            max_seq_len: Maximum sequence length for trajectories
            
        Returns:
            Dict with 'node_ids', 'timestamps', 'mask' tensors
        """
        batch_size = len(agent_ids)
        
        # Initialize tensors
        node_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        timestamps = torch.zeros(batch_size, max_seq_len, dtype=torch.float)
        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float)
        
        for i, agent_id in enumerate(agent_ids):
            agent_id_str = str(agent_id)
            
            if self.path_data is not None and agent_id_str in self.path_data:
                # Get all trajectory segments for this agent
                agent_paths = self.path_data[agent_id_str]
                
                # Flatten all segments into a single trajectory
                all_nodes = []
                for segment_key in sorted(agent_paths.keys()):
                    segment_nodes = agent_paths[segment_key]
                    if isinstance(segment_nodes, list):
                        all_nodes.extend(segment_nodes)
                
                # Truncate to max_seq_len
                if len(all_nodes) > max_seq_len:
                    all_nodes = all_nodes[:max_seq_len]
                
                seq_len = len(all_nodes)
                
                # Map OSM node IDs to sequential indices
                mapped_nodes = [self.node_id_mapping.get(node_id, 0) for node_id in all_nodes]
                
                # Fill tensors
                node_ids[i, :seq_len] = torch.tensor(mapped_nodes)
                timestamps[i, :seq_len] = torch.arange(seq_len, dtype=torch.float)
                mask[i, :seq_len] = 1.0
                
                # print(f"Agent {agent_id}: {seq_len} nodes")
            else:
                print(f"Warning: Agent {agent_id} not found in path data")
        
        return {
            'node_ids': node_ids,
            'timestamps': timestamps,
            'mask': mask
        }
    
    def prepare_graph_data(self):
        """
        Prepare graph data from the campus environment.
        
        Returns:
            Dict with 'x', 'edge_index' tensors
        """
        # Get graph from environment
        if self.env is None:
            raise ValueError("Environment not loaded")
        
        G = self.env.G
        
        # Create node features (using node coordinates)
        node_features = []
        node_mapping = {}
        
        for i, node in enumerate(G.nodes()):
            # Get node coordinates
            coords = (G.nodes[node]['y'], G.nodes[node]['x'])  # lat, lon
            node_features.append([coords[0], coords[1], 0.0, 0.0])  # Add some padding features
            node_mapping[node] = i
        
        # Create edge index
        edge_list = []
        for u, v in G.edges():
            if u in node_mapping and v in node_mapping:
                edge_list.append([node_mapping[u], node_mapping[v]])
                edge_list.append([node_mapping[v], node_mapping[u]])  # Undirected graph
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return {
            'x': x,
            'edge_index': edge_index
        }

    def get_goal_distribution(self, agent_id):
        """
        Returns a goal distribution vector over all nodes for the given agent.
        For goal_data structured as {episode_num: {agent_id: goal_node, ...}},
        builds a histogram over all episodes of the goal nodes reached by this agent.
        """
        num_nodes = len(self.node_id_mapping)
        dist = torch.zeros(num_nodes, dtype=torch.float)
        agent_id_str = str(agent_id)
        # If goal_data is structured as {episode_num: {agent_id: goal_node, ...}}
        if self.goal_data is not None and isinstance(next(iter(self.goal_data.values())), dict):
            goal_counts = {}
            for episode_dict in self.goal_data.values():
                if agent_id_str in episode_dict:
                    goal_node = episode_dict[agent_id_str]
                    idx = self.node_id_mapping.get(goal_node, None)
                    if idx is not None:
                        goal_counts[idx] = goal_counts.get(idx, 0) + 1
            total = sum(goal_counts.values())
            if total > 0:
                for idx, count in goal_counts.items():
                    dist[idx] = count / total
        else:
            # Fallback to previous logic for other formats
            if self.goal_data is not None and agent_id_str in self.goal_data:
                goal = self.goal_data[agent_id_str]
                if isinstance(goal, int):
                    idx = self.node_id_mapping.get(goal, None)
                    if idx is not None:
                        dist[idx] = 1.0
                elif isinstance(goal, list):
                    valid_idxs = [self.node_id_mapping.get(g, None) for g in goal if self.node_id_mapping.get(g, None) is not None]
                    if valid_idxs:
                        for idx in valid_idxs:
                            dist[idx] = 1.0 / len(valid_idxs)
                elif isinstance(goal, dict):
                    total = 0.0
                    for k, v in goal.items():
                        idx = self.node_id_mapping.get(int(k), None)
                        if idx is not None:
                            dist[idx] = float(v)
                            total += float(v)
                    if total > 0:
                        dist /= total
        return dist

    def get_partial_trajectory(self, agent_id, max_len):
        """
        Returns a truncated trajectory (list of node indices) for the agent, up to max_len.
        """
        agent_id_str = str(agent_id)
        if self.path_data is not None and agent_id_str in self.path_data:
            agent_paths = self.path_data[agent_id_str]
            all_nodes = []
            for segment_key in sorted(agent_paths.keys()):
                segment_nodes = agent_paths[segment_key]
                if isinstance(segment_nodes, list):
                    all_nodes.extend(segment_nodes)
            if len(all_nodes) > max_len:
                all_nodes = all_nodes[:max_len]
            mapped_nodes = [self.node_id_mapping.get(node_id, 0) for node_id in all_nodes]
            return mapped_nodes
        return []

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