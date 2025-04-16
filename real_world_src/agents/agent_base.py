import networkx as nx
import numpy as np
import random

class Agent:
    """Base class for all agents navigating the UCSD campus."""
    
    def __init__(self, id=None, color='blue', speed=1.0):
        self.id = id or f"Agent-{random.randint(1000, 9999)}"
        self.color = color
        self.environment = None
        self.species = "Generic"
        self.speed = speed
        
        # Navigation
        self.start_node = None
        self.goal_node = None
        self.current_node = None
        self.next_node = None
        self.path = []
        self.path_index = 0
        self.visited_nodes = []
        
        # For position interpolation
        self.position = None  # This will be set during reset
        self.progress = 0.0
    
    def reset(self, randomize_position=True):
        """Reset the agent to a random start and goal.
        
        Args:
            randomize_position: Boolean, whether to randomize the start/goal positions.
                                Always True in the current implementation.
        """
        if self.environment is None:
            raise ValueError("Agent must be added to an environment before reset")
        
        # Random start and goal nodes
        self.start_node = self.environment.get_random_node()
        
        # Pick a goal node that's reasonably far from start
        while True:
            self.goal_node = self.environment.get_random_node()
            try:
                path = nx.shortest_path(self.environment.G_undirected, 
                                       source=self.start_node, 
                                       target=self.goal_node, 
                                       weight='length')
                if len(path) > 10:  # Make sure path is non-trivial
                    break
            except:
                continue
                
        self.current_node = self.start_node
        self.next_node = None
        self.path = []
        self.path_index = 0
        self.visited_nodes = [self.current_node]
        self.position = self.environment.get_node_coordinates(self.current_node)
        self.progress = 0.0
        
        # Plan initial path
        self.plan_path()

    def reset_ex1(self):
        """ Resets the agent, but keeps the start and goal nodes that were previously set for experiment 1.
        """
        self.current_node = self.start_node
        self.next_node = None
        self.path = []
        self.path_index = 0
        self.visited_nodes = [self.current_node]
        self.position = self.environment.get_node_coordinates(self.current_node)
        self.progress = 0.0

        # Plan initial path
        self.plan_path()
        
    def plan_path(self):
        """Plan a path from current node to goal. Override in subclasses."""
        try:
            self.path = nx.shortest_path(self.environment.G_undirected, 
                                       source=self.current_node, 
                                       target=self.goal_node, 
                                       weight='length')
            self.path_index = 0
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]
            
    def step(self):
        """Take a step toward the goal."""
        if self.at_goal():
            return
            
        # If we're between nodes, continue the interpolation
        if self.next_node is not None:
            # Update progress along the edge
            self.progress += self.speed * 0.1
            
            # If we've reached the next node
            if self.progress >= 1.0:
                self.current_node = self.next_node
                self.visited_nodes.append(self.current_node)
                self.path_index += 1
                
                # Check if we reached the end of our planned path
                if self.path_index >= len(self.path) - 1:
                    self.next_node = None
                    self.position = self.environment.get_node_coordinates(self.current_node)
                    
                    # Replan if we're not at the goal
                    if not self.at_goal():
                        self.plan_path()
                        
                        # Set up next movement if path exists
                        if len(self.path) > 1:
                            self.next_node = self.path[1]
                            self.progress = 0.0
                else:
                    # Continue to the next node in our path
                    self.next_node = self.path[self.path_index + 1]
                    self.progress = 0.0
            
            # Update position by interpolation
            if self.next_node is not None:
                start_pos = self.environment.get_node_coordinates(self.current_node)
                end_pos = self.environment.get_node_coordinates(self.next_node)
                self.position = (
                    start_pos[0] + self.progress * (end_pos[0] - start_pos[0]),
                    start_pos[1] + self.progress * (end_pos[1] - start_pos[1])
                )
        
        # If we don't have a next node, initialize movement to the next node
        elif len(self.path) > 1 and self.path_index < len(self.path) - 1:
            self.next_node = self.path[self.path_index + 1]
            self.progress = 0.0
            
    def at_goal(self):
        """Check if agent has reached its goal."""
        return self.current_node == self.goal_node
        
    def get_position(self, node=None):
        """Get current position for visualization."""
        if node is not None:
            return self.environment.get_node_coordinates(node)
        
        # Ensure we always return a valid position
        if self.position is None and self.current_node is not None:
            self.position = self.environment.get_node_coordinates(self.current_node)
            
        return self.position
        
    def get_path_coordinates(self):
        """Get the coordinates of the path for visualization."""
        if not self.visited_nodes:
            return [], []
            
        x_coords = [self.environment.get_node_coordinates(node)[0] for node in self.visited_nodes]
        y_coords = [self.environment.get_node_coordinates(node)[1] for node in self.visited_nodes]
        
        # Add current position
        if self.position:
            x_coords.append(self.position[0])
            y_coords.append(self.position[1])
            
        return x_coords, y_coords