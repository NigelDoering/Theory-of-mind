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
        print(f"Loading map data for {place_name}...")
        self.G = ox.graph_from_place(place_name, network_type="all")
        self.G_undirected = nx.Graph(self.G)
        
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
                pass  # Handle edges that might not exist
        return length
    
    def visualize_map(self, show_landmarks=True):
        """Visualize the campus map."""
        fig, ax = ox.plot_graph(self.G, figsize=(15, 15), node_size=5, 
                               edge_color="#444444", show=False)
        
        # Add buildings
        if self.buildings is not None:
            self.buildings.plot(ax=ax, color='lightgrey', alpha=0.7, edgecolor='dimgrey')
        
        # Highlight landmark nodes
        if show_landmarks:
            landmark_x = [self.node_coords[node][0] for node in self.landmarks]
            landmark_y = [self.node_coords[node][1] for node in self.landmarks]
            ax.scatter(landmark_x, landmark_y, c='red', s=100, zorder=5, label="Landmarks")
            
        plt.title("UCSD Campus Environment")
        plt.tight_layout()
        return fig, ax
    
    def add_agent(self, agent):
        """Add an agent to the environment."""
        self.agents.append(agent)
        agent.environment = self
    
    def reset_agents(self):
        """Reset all agents."""
        for agent in self.agents:
            agent.reset()
    
    def step(self):
        """Update all agents by one step."""
        for agent in self.agents:
            agent.step()
            
    def all_agents_done(self):
        """Check if all agents have reached their goals."""
        return all(agent.at_goal() for agent in self.agents)