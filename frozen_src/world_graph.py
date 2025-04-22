import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
from IPython.display import display, clear_output
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import time
from collections import defaultdict, deque

class WorldGraph:
    """Represents the environment as a graph, excluding holes."""
    
    def __init__(self, goal_generator):
        """Initialize with a goal space generator."""
        self.goal_generator = goal_generator
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build a graph representation of the environment."""
        G = nx.Graph()
        
        # Get dimensions
        map_array = self.goal_generator.map
        nrow, ncol = len(map_array), len(map_array[0])
        
        # Add nodes for all non-hole positions
        for i in range(nrow):
            for j in range(ncol):
                if map_array[i][j] != b'H':  # Skip holes
                    node_id = i * ncol + j
                    
                    # Calculate node properties
                    pos = (i, j)
                    reward = 0.0
                    node_type = map_array[i][j].decode('utf-8')
                    
                    # Check if this is a goal position
                    if pos in self.goal_generator.goal_rewards:
                        reward = self.goal_generator.goal_rewards[pos]
                    
                    # Check if this is the start position
                    is_start = (node_type == 'S')
                    
                    # Check if this is the final goal
                    is_final = (pos == self.goal_generator.final_goal if self.goal_generator.final_goal else False)
                    
                    # Add the node with all its properties
                    G.add_node(node_id, 
                              pos=pos, 
                              reward=reward, 
                              type=node_type,
                              is_start=is_start,
                              is_final=is_final,
                              goal_id=self.goal_generator.goal_ids.get(pos, None))
        
        # Add edges between adjacent nodes
        for node in list(G.nodes()):
            i, j = G.nodes[node]['pos']
            
            # Check all 4 directions (up, right, down, left)
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                
                # Check if the adjacent position is valid and not a hole
                if (0 <= ni < nrow and 0 <= nj < ncol and 
                    map_array[ni][nj] != b'H'):
                    neighbor_id = ni * ncol + nj
                    
                    # Add edge with weight 1
                    if neighbor_id in G:
                        G.add_edge(node, neighbor_id, weight=1)
        
        return G
    
    def get_neighbors(self, node):
        """Get all neighbors of a node."""
        return list(self.graph.neighbors(node))
    
    def get_node_info(self, node):
        """Get information about a node."""
        return self.graph.nodes[node]
    
    def get_start_node(self):
        """Get the start node."""
        for node in self.graph.nodes:
            if self.graph.nodes[node]['is_start']:
                return node
        return None
    
    def get_final_node(self):
        """Get the final goal node."""
        for node in self.graph.nodes:
            if self.graph.nodes[node]['is_final']:
                return node
        return None