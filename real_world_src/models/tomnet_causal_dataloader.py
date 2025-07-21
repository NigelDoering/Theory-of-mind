import torch
import numpy as np
import json
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from real_world_src.environment.campus_env import CampusEnvironment

# -----------------------------
# Data Loader for Real Campus Data
# -----------------------------
class CampusDataLoader:
    """
    Loads and prepares data from the campus environment and trajectory files.
    """
    def __init__(self, data_dir="./data/1k/", node_mapping_path=None, save_node_mapping_path=None, mode='train'):
        self.data_dir = data_dir
        self.env = None
        self.agents = None
        self.path_data = None
        self.goal_data = None
        self.node_id_mapping = {}  # Map OSM node IDs to sequential indices
        self.reverse_node_mapping = {}  # Map sequential indices back to OSM node IDs
        self.node_mapping_path = node_mapping_path
        self.save_node_mapping_path = save_node_mapping_path
        self.mode = mode
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
        """Build or update mapping from OSM node IDs to sequential indices."""
        mapping_exists = self.node_mapping_path and os.path.exists(self.node_mapping_path)
        if mapping_exists:
            print(f"Loading node_id_mapping from {self.node_mapping_path}")
            with open(self.node_mapping_path, 'rb') as f:
                mapping = pickle.load(f)
                self.node_id_mapping = mapping['node_id_mapping']
                self.reverse_node_mapping = mapping['reverse_node_mapping']
            print(f"Loaded node_id_mapping with {len(self.node_id_mapping)} nodes.")
        else:
            print("No existing node_id_mapping found. Initializing new mapping.")
            self.node_id_mapping = {}
            self.reverse_node_mapping = {}
        # Collect all unique node IDs from path data and environment
        all_node_ids = set()
        if self.path_data is not None:
            for agent_data in self.path_data.values():
                for segment in agent_data.values():
                    if isinstance(segment, list):
                        all_node_ids.update(segment)
        if self.env is not None:
            all_node_ids.update(self.env.G.nodes())
        # In train mode, add new nodes to the mapping
        if self.mode == 'train':
            max_idx = max(self.node_id_mapping.values(), default=-1)
            for node_id in sorted(all_node_ids):
                if node_id not in self.node_id_mapping:
                    max_idx += 1
                    self.node_id_mapping[node_id] = max_idx
                    self.reverse_node_mapping[max_idx] = node_id
            print(f"[TRAIN] Updated node_id_mapping to {len(self.node_id_mapping)} nodes.")
            # Save the updated mapping
            if self.save_node_mapping_path:
                print(f"Saving node_id_mapping to {self.save_node_mapping_path}")
                with open(self.save_node_mapping_path, 'wb') as f:
                    pickle.dump({
                        'node_id_mapping': self.node_id_mapping,
                        'reverse_node_mapping': self.reverse_node_mapping
                    }, f)
        else:
            # In eval mode, do not add new nodes, just use the loaded mapping
            print(f"[EVAL] Using fixed node_id_mapping with {len(self.node_id_mapping)} nodes.")
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
    
    def prepare_graph_data(self, save_cuda_path=None, device=None):
        """
        Prepare graph data from the campus environment.
        Returns:
            Dict with 'x', 'edge_index' tensors (pinned memory)
        If save_cuda_path is provided, saves the dict with tensors on device to that path.
        """
        # Get graph from environment
        if self.env is None:
            raise ValueError("Environment not loaded")
        G = self.env.G
        # Create node features (using node coordinates)
        node_features = []
        node_mapping = {}
        for i, node in enumerate(G.nodes()):
            coords = (G.nodes[node]['y'], G.nodes[node]['x'])  # lat, lon
            node_features.append([coords[0], coords[1], 0.0, 0.0])
            node_mapping[node] = i
        # Create edge index
        edge_list = []
        for u, v in G.edges():
            if u in node_mapping and v in node_mapping:
                edge_list.append([node_mapping[u], node_mapping[v]])
                edge_list.append([node_mapping[v], node_mapping[u]])  # Undirected graph
        # Convert to tensors with pin_memory
        x = torch.tensor(node_features, dtype=torch.float).pin_memory()
        edge_index = torch.tensor(edge_list, dtype=torch.long).pin_memory().t().contiguous()
        graph_data = {'x': x, 'edge_index': edge_index}
        # Optionally move to device and save
        if save_cuda_path and device is not None:
            graph_data_cuda = {k: v.to(device, non_blocking=True) for k, v in graph_data.items()}
            torch.save(graph_data_cuda, save_cuda_path)
        return graph_data

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
