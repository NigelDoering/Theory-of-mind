import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from ..utils.config import VISUAL_CONFIG
from real_world_src.utils.config import VISUAL_CONFIG
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time

class CampusEnvironment:
    """Environment class for UCSD campus using OSMnx."""
    
    def __init__(self, place_name="University of California, San Diego, La Jolla, CA, USA"):
        # Configure OSMnx
        ox.settings.log_console = True
        ox.settings.use_cache = True
        
        # Load the map data
        self.place_name = place_name
        ##### PATH to graphml file #####
        # This file should be in the same directory as this script or provide a full path
        # If the file does not exist, it will be downloaded and saved
        graphml_path = "ucsd_campus.graphml"
        if os.path.exists(graphml_path):
            print(f"Loading graph from {graphml_path}...")
            self.G = ox.load_graphml(graphml_path)
        else:
            print(f"GraphML file not found. Downloading map data for {place_name}...")
            self.G = ox.graph_from_place(place_name, network_type="all")
            ox.save_graphml(self.G, filepath=graphml_path)
        self.G_undirected = nx.Graph(self.G)

        #print(f"Loading map data for {place_name}...")
        #self.G = ox.graph_from_place(place_name, network_type="all")
        #self.G_undirected = nx.Graph(self.G)
        
        
        # Save the graph for future use
        ox.save_graphml(self.G, filepath="ucsd_campus.graphml")
        
        # Store the node positions (for visualization)
        self.node_coords = {node: (data['x'], data['y']) for node, data in self.G.nodes(data=True)}
        
        # Store buildings data
        self.buildings = ox.features_from_place(place_name, tags={'building': True})
        
        # Precompute useful attributes
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges())
        
        # Landmark nodes (high centrality nodes)
        self.landmarks = self.get_landmark_nodes(15)
        
        # Agent tracking
        self.agents = []
        
        print(f"Environment loaded with {len(self.nodes)} nodes and {len(self.edges)} edges")

    def set_goals(self, goals):
        self.goals = goals
    
    def get_random_node(self):
        """Return a random node from the graph."""
        return random.choice(self.nodes)
        
    def get_landmark_nodes(self, num_landmarks=15):
        """Select notable nodes as landmarks (using degree centrality)."""
        centrality = nx.degree_centrality(self.G_undirected)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:num_landmarks]]
    
    def get_nearest_node(self, point):
        """Find the nearest node to a given point (x, y)."""
        return ox.distance.nearest_nodes(self.G, X=[point[0]], Y=[point[1]])[0]

    def get_node_coordinates(self, node):
        """Get coordinates of a specific node."""
        return self.node_coords[node]
    
    def get_path_length(self, path):
        """Calculate the total length of a path."""
        length = 0
        for i in range(len(path) - 1):
            try:
                edge_data = self.G.get_edge_data(path[i], path[i+1])
                if edge_data and 'length' in edge_data[0]:
                    length += edge_data[0]['length']
            except:
                pass  # Yet to handle edges that might not exist
        return length
    
    def visualize_map(self, show_landmarks=True, ax=None):
        """Visualize the campus map.
        
        Args:
            show_landmarks: Whether to highlight landmark nodes
            ax: Optional matplotlib axis to plot on
            
        Returns:
            fig, ax: The figure and axis objects
        """
        if ax is None:
            fig, ax = ox.plot_graph(self.G, figsize=(15, 15), node_size=5, 
                                 edge_color=VISUAL_CONFIG["edge_color"], show=False)
        else:
            # Plot graph on existing axis
            fig = ax.figure
            ox.plot_graph(self.G, ax=ax, node_size=5, 
                        edge_color=VISUAL_CONFIG["edge_color"], show=False)
        
        # Add buildings
        if self.buildings is not None:
            self.buildings.plot(ax=ax, color=VISUAL_CONFIG["building_color"], 
                             alpha=0.7, edgecolor=VISUAL_CONFIG["building_edge"])
        
        # Highlight landmark nodes
        if show_landmarks:
            landmark_x = [self.node_coords[node][0] for node in self.landmarks]
            landmark_y = [self.node_coords[node][1] for node in self.landmarks]
            ax.scatter(landmark_x, landmark_y, 
                     c=VISUAL_CONFIG["landmark_color"], 
                     s=VISUAL_CONFIG["sizes"]["landmark"], 
                     marker=VISUAL_CONFIG["markers"]["landmark"],
                     zorder=5, label="Landmarks")
        
        # Only add title and tight_layout if we created a new figure
        if ax is None:
            plt.title("UCSD Campus Environment")
            plt.tight_layout()
        
        return fig, ax
    
    def plot_map(self, ax=None, show_landmarks=True):
        """Alias for visualize_map to maintain compatibility with visualization functions."""
        return self.visualize_map(show_landmarks=show_landmarks, ax=ax)
    
    def add_agent(self, agent):
        """Add an agent to the environment."""
        self.agents.append(agent)
        agent.environment = self
    
    def reset_agents(self):
        """Reset all agents."""
        for agent in self.agents:
            agent.reset(self.goals)

    def reset_agents_ex1(self):
        """Reset all agents for experiment 1."""
        print("Using correct reset")
        for agent in self.agents:
            agent.reset_ex1()
    
    
    def step(self):
        """Update all agents by one step."""
        for agent in self.agents:
            agent.step()
            
    def all_agents_done(self):
        """Check if all agents have reached their goals."""
        return all(agent.at_goal() for agent in self.agents)
    

if __name__ == "__main__":
    campus = CampusEnvironment()
    campus.visualize_map()

    node_1 = campus.get_random_node()
    node_2 = campus.get_random_node()
    node_c_1 = campus.get_node_coordinates(node_1)
    node_c_2 = campus.get_node_coordinates(node_2)

    print(f"Node 1: {node_1} at {node_c_1}")
    print(f"Node 2: {node_2} at {node_c_2}")

    path = nx.shortest_path(campus.G, source=node_1, target=node_2)
    path_length = campus.get_path_length(path)

    print(f"Shortest path from {node_1} to {node_2}: {path}")
    print(f"Path length: {path_length}")