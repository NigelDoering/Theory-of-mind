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
        
        # For trajectory recording
        self.trajectory = []  # List of (state, action) pairs
    
    def reset(self, randomize_position=True):
        """Reset the agent to a random start and goal."""
        if self.environment is None:
            raise ValueError("Agent must be added to an environment before reset")
        
        # Random start and goal nodes
        max_attempts = 10
        for _ in range(max_attempts):
            self.start_node = self.environment.get_random_node()
            
            # Pick a goal node that's reasonably far from start
            for __ in range(max_attempts):
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
                    
            # If we have both valid start and goal nodes, break
            if self.goal_node and self.start_node:
                break
                
        # If we couldn't find valid nodes, use defaults
        if not self.start_node or not self.goal_node:
            self.start_node = list(self.environment.nodes)[0]
            self.goal_node = list(self.environment.nodes)[-1]
            print(f"Warning: {self.id} using default nodes due to initialization failure")
                    
        self.current_node = self.start_node
        self.next_node = None
        self.path = []
        self.path_index = 0
        self.visited_nodes = [self.current_node]
        self.position = self.environment.get_node_coordinates(self.current_node)
        self.progress = 0.0
        self.trajectory = []  # Reset trajectory
        
        # Plan initial path
        self.plan_path()
        
        # Record initial state
        initial_state = self._get_current_state()
        initial_action = self.get_action(initial_state)
        self.trajectory.append((initial_state, initial_action))
        
    def _get_current_state(self):
        """Create a state representation of the agent's current situation."""
        state = {
            'position': self.position,
            'current_node': self.current_node,
            'goal_node': self.goal_node,
            'visited_nodes': list(self.visited_nodes),  # Copy to avoid reference issues
        }
        return state
        
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
        """Take a step toward the goal and update trajectory."""
        if self.at_goal():
            return
            
        # Get current state
        current_state = self._get_current_state()
        
        # Determine action using get_action method
        action = self.get_action(current_state)
        
        # Add to trajectory
        self.trajectory.append((current_state, action))
        
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

    def get_action(self, state=None):
        """
        Get the action that leads to the next node in the agent's path.
        
        Args:
            state: Current state (optional, used by some agent types)
            
        Returns:
            action: Integer representing the action to take
                0: up
                1: right
                2: down
                3: left
                4: stay
        """
        # If we've reached the goal, stay in place
        if self.at_goal():
            return 4  # Stay action
            
        # If we don't have a path or next node, plan one
        if not self.path or len(self.path) <= 1 or self.path_index >= len(self.path) - 1:
            self.plan_path()
            # If we still don't have a valid path, stay in place
            if not self.path or len(self.path) <= 1 or self.path_index >= len(self.path) - 1:
                return 4  # Stay action
        
        # Get current position and next position from path
        current_node = self.current_node
        next_node = self.path[self.path_index + 1]
        
        # Get coordinates
        current_pos = self.environment.get_node_coordinates(current_node)
        next_pos = self.environment.get_node_coordinates(next_node)
        
        # Calculate direction vector
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        # Convert to discrete action
        # Priority: larger component determines the action
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3  # Right or Left
        elif abs(dy) > abs(dx):
            return 0 if dy < 0 else 2  # Up or Down (y-axis may be inverted)
        else:
            # If equal components, choose based on sign
            if dx > 0:
                return 1  # Right
            elif dx < 0:
                return 3  # Left
            elif dy < 0:
                return 0  # Up
            elif dy > 0:
                return 2  # Down
            else:
                return 4  # Stay (should never happen in normal operation)