import networkx as nx
import random
from .agent_base import Agent

class ShortestPathAgent(Agent):
    """Agent that always takes the shortest path."""
    
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or 'blue')
        self.species = "ShortestPath"
        
    def plan_path(self):
        """Plan shortest path to goal."""
        try:
            self.path = nx.shortest_path(self.environment.G_undirected, 
                                       source=self.current_node, 
                                       target=self.goal_node, 
                                       weight='length')
            self.path_index = 0
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]


class RandomWalkAgent(Agent):
    """Agent that performs a biased random walk toward the goal."""
    
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or 'red')
        self.species = "RandomWalk"
        self.planning_horizon = 3  # Steps to look ahead
        
    def plan_path(self):
        """Plan a biased random path that generally moves toward the goal."""
        # Check for valid nodes first
        if self.current_node is None or self.goal_node is None:
            print(f"{self.id}: Cannot plan path - current_node or goal_node is None")
            self.path = []
            self.path_index = 0
            return
            
        try:
            # First get the shortest path as a reference
            shortest_path = nx.shortest_path(self.environment.G_undirected, 
                                          source=self.current_node, 
                                          target=self.goal_node, 
                                          weight='length')
            
            # Start building a random path
            random_path = [self.current_node]
            current = self.current_node
            
            # Generate a few steps of random path with bias toward the goal
            for _ in range(self.planning_horizon):
                if current is None:
                    break
                    
                neighbors = list(self.environment.G_undirected.neighbors(current))
                if not neighbors:
                    break
                    
                # With probability 0.7, move toward the goal
                if random.random() < 0.7 and len(shortest_path) > 1:
                    next_node_in_shortest = shortest_path[1]
                    if next_node_in_shortest in neighbors:
                        current = next_node_in_shortest
                        random_path.append(current)
                        shortest_path = shortest_path[1:]
                        continue
                
                # Otherwise, pick a random neighbor, with preference for unexplored nodes
                unvisited = [n for n in neighbors if n not in self.visited_nodes]
                if unvisited and random.random() < 0.8:
                    current = random.choice(unvisited)
                else:
                    current = random.choice(neighbors)
                random_path.append(current)
            
            self.path = random_path
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]


class LandmarkAgent(Agent):
    """Agent that navigates via landmarks."""
    
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or 'green')
        self.species = "Landmark"
        self.current_landmark = None
        
    def plan_path(self):
        """Plan path via landmarks."""
        try:
            # If we're already heading to a landmark
            if self.current_landmark and self.current_node != self.current_landmark:
                # Continue to the landmark
                landmark_path = nx.shortest_path(self.environment.G_undirected, 
                                               source=self.current_node, 
                                               target=self.current_landmark, 
                                               weight='length')
                self.path = landmark_path
                self.path_index = 0
                return
                
            # If we reached a landmark or don't have one yet, pick the next landmark
            # that gets us closer to the goal
            goal_path = nx.shortest_path(self.environment.G_undirected, 
                                        source=self.current_node, 
                                        target=self.goal_node, 
                                        weight='length')
            
            # Get distances from each landmark to the goal
            landmark_distances = {}
            for landmark in self.environment.landmarks:
                if landmark == self.current_node:
                    continue
                    
                try:
                    path = nx.shortest_path(self.environment.G_undirected, 
                                          source=landmark, 
                                          target=self.goal_node, 
                                          weight='length')
                    landmark_distances[landmark] = self.environment.get_path_length(path)
                except:
                    continue
            
            # Pick a landmark that gets us closer to the goal
            if landmark_distances:
                # Either go straight to goal if it's close, or via a landmark
                direct_distance = self.environment.get_path_length(goal_path)
                
                if direct_distance < 500 or random.random() < 0.3:  # 30% chance to go direct when far
                    # Go directly to goal
                    self.current_landmark = None
                    self.path = goal_path
                else:
                    # Go via a landmark (pick one of the closest 3 to the goal)
                    closest_landmarks = sorted(landmark_distances.keys(), 
                                            key=lambda l: landmark_distances[l])[:3]
                    self.current_landmark = random.choice(closest_landmarks)
                    
                    # Path to the landmark
                    landmark_path = nx.shortest_path(self.environment.G_undirected, 
                                                  source=self.current_node, 
                                                  target=self.current_landmark, 
                                                  weight='length')
                    self.path = landmark_path
            else:
                # No good landmarks, go direct
                self.current_landmark = None
                self.path = goal_path
                
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]


class SocialAgent(Agent):
    """Agent that follows other agents sometimes."""
    
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or 'purple')
        self.species = "Social"
        self.follow_target = None
        self.follow_duration = 0
        
    def plan_path(self):
        """Plan path considering other agents."""
        # Check if we should stop following
        if self.follow_target and self.follow_duration <= 0:
            self.follow_target = None
        
        # Decrement follow duration
        if self.follow_target:
            self.follow_duration -= 1
        
        # 30% chance to follow another agent if we're not already following
        if not self.follow_target and random.random() < 0.3 and self.environment.agents:
            # Find potential agents to follow
            potential_targets = [
                agent for agent in self.environment.agents
                if agent != self and 
                agent.current_node and 
                nx.has_path(self.environment.G_undirected, self.current_node, agent.current_node)
            ]
            
            if potential_targets:
                self.follow_target = random.choice(potential_targets)
                self.follow_duration = random.randint(5, 15)  # Follow for 5-15 steps
        
        try:
            if self.follow_target and self.follow_target.current_node:
                # Follow the target
                target_node = self.follow_target.current_node
                self.path = nx.shortest_path(self.environment.G_undirected, 
                                           source=self.current_node, 
                                           target=target_node, 
                                           weight='length')
            else:
                # If not following, take the shortest path
                self.path = nx.shortest_path(self.environment.G_undirected, 
                                           source=self.current_node, 
                                           target=self.goal_node, 
                                           weight='length')
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]


class ExplorerAgent(Agent):
    """Agent that prefers to explore less-traveled paths."""
    
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or 'orange')
        self.species = "Explorer"
        self.global_visited_edges = {}  # Track visited edges across all resets
        
    def plan_path(self):
        """Plan path preferring less-traveled edges."""
        try:
            # Create a weighted graph copy where weights are adjusted by visit count
            G_weighted = self.environment.G_undirected.copy()
            
            # Adjust edge weights based on visit counts
            for u, v, data in G_weighted.edges(data=True):
                edge_key = (min(u, v), max(u, v))  # Undirected edge key
                visit_count = self.global_visited_edges.get(edge_key, 0)
                
                # Original weight is length
                original_weight = data.get('length', 1.0)
                
                # New weight penalizes previously visited edges
                new_weight = original_weight * (1.0 + 0.5 * visit_count)

                # Update weight
                G_weighted[u][v]['weight'] = new_weight
            
            # Find path with adjusted weights
            self.path = nx.shortest_path(G_weighted, 
                                       source=self.current_node, 
                                       target=self.goal_node, 
                                       weight='weight')
            self.path_index = 0
            
            # Record the edges we're going to use
            for i in range(len(self.path) - 1):
                u, v = self.path[i], self.path[i+1]
                edge_key = (min(u, v), max(u, v))
                self.global_visited_edges[edge_key] = self.global_visited_edges.get(edge_key, 0) + 1
                
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]
    
    def reset(self):
        """Reset the agent but keep the edge history."""
        super().reset()
        # Note: We don't reset global_visited_edges to maintain exploration history


class ObstacleAvoidingAgent(Agent):
    """Agent that avoids obstacles and congested areas."""
    
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or '#8B008B')  # Dark magenta
        self.species = "ObstacleAvoider"
        self.obstacle_threshold = 3  # Consider an area congested if several agents recently visited
        self.memory_time = 20  # Remember obstacles for this many steps
        self.obstacle_memory = {}  # Map of node to time last seen congested
        
    def plan_path(self):
        """Plan path that avoids congested areas."""
        current_step = len(self.visited_nodes)
        
        try:
            # Update obstacle memory by observing agent locations
            if self.environment.agents:
                # Find clusters of agents
                agent_positions = {}
                for agent in self.environment.agents:
                    if agent != self and agent.current_node:
                        agent_positions[agent.current_node] = agent_positions.get(agent.current_node, 0) + 1
                
                # Mark congested nodes as obstacles
                for node, count in agent_positions.items():
                    if count >= self.obstacle_threshold:
                        self.obstacle_memory[node] = current_step
            
            # Clean up old obstacle memories
            nodes_to_remove = []
            for node, last_seen in self.obstacle_memory.items():
                if current_step - last_seen > self.memory_time:
                    nodes_to_remove.append(node)
            for node in nodes_to_remove:
                del self.obstacle_memory[node]
            
            # Create a graph with penalized obstacle areas
            G_weighted = self.environment.G_undirected.copy()
            
            # Add high weights to obstacle areas
            for node in self.obstacle_memory:
                for u, v, data in G_weighted.edges(data=True):
                    if u == node or v == node:
                        data['weight'] = data.get('length', 1.0) * 5.0  # 5x penalty
            
            # Find path with adjusted weights
            self.path = nx.shortest_path(G_weighted, 
                                       source=self.current_node, 
                                       target=self.goal_node, 
                                       weight='weight')
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]


class ScaredAgent(Agent):
    """Agent that is afraid of high-traffic areas and seeks quiet routes."""
    
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or '#9370DB')  # Medium purple
        self.species = "Scared"
        self.fear_radius = 3  # Distance within which other agents cause fear
        self.fear_factor = 2.0  # How much to weight fear vs. distance
        
    def plan_path(self):
        """Plan path avoiding high-traffic areas."""
        try:
            # Create a weighted graph with fear-based costs
            G_weighted = self.environment.G_undirected.copy()
            
            # Calculate fear map - how many agents are near each node
            fear_map = {}
            for node in self.environment.nodes:
                nearby_agents = 0
                node_pos = self.environment.get_node_coordinates(node)
                
                for agent in self.environment.agents:
                    if agent == self:
                        continue
                    
                    # Add safety check for agent position
                    agent_pos = agent.get_position()
                    if agent_pos is None:
                        continue
                    
                    # Now calculate distance safely
                    distance = ((node_pos[0] - agent_pos[0])**2 + 
                               (node_pos[1] - agent_pos[1])**2) ** 0.5
                    
                    if distance < self.fear_radius:
                        nearby_agents += 1
                
                fear_map[node] = nearby_agents
            
            # Adjust weights based on fear
            for u, v, data in G_weighted.edges(data=True):
                fear_weight = (fear_map.get(u, 0) + fear_map.get(v, 0)) * self.fear_factor
                data['weight'] = data.get('length', 1.0) * (1.0 + fear_weight)
            
            # Find path with adjusted weights
            self.path = nx.shortest_path(G_weighted, 
                                       source=self.current_node, 
                                       target=self.goal_node, 
                                       weight='weight')
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]


class RiskyAgent(Agent):
    """Agent that takes risky shortcuts even through congested areas."""
    
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or '#FF4500')  # OrangeRed
        self.species = "Risky"
        self.shortcut_probability = 0.4  # Probability of trying a shortcut
        self.speed = 1.5  # Moves faster than other agents
        
    def plan_path(self):
        """Plan a path that might include risky shortcuts."""
        try:
            # First get the standard shortest path
            shortest_path = nx.shortest_path(self.environment.G_undirected, 
                                           source=self.current_node, 
                                           target=self.goal_node, 
                                           weight='length')
            
            # Should we try a shortcut?
            if random.random() < self.shortcut_probability and len(shortest_path) >= 3:
                # Try to find a riskier but potentially shorter path
                # We'll use Euclidean distance rather than road length
                G_euclid = self.environment.G_undirected.copy()
                
                for u, v, data in G_euclid.edges(data=True):
                    u_pos = self.environment.get_node_coordinates(u)
                    v_pos = self.environment.get_node_coordinates(v)
                    euclid_dist = ((u_pos[0] - v_pos[0])**2 + (u_pos[1] - v_pos[1])**2) ** 0.5
                    data['weight'] = euclid_dist * 0.8  # Favor straighter lines
                
                try:
                    shortcut_path = nx.shortest_path(G_euclid, 
                                                  source=self.current_node, 
                                                  target=self.goal_node, 
                                                  weight='weight')
                    self.path = shortcut_path
                except:
                    self.path = shortest_path
            else:
                self.path = shortest_path
                
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]


class GoalDirectedAgent(Agent):
    """Agent that moves toward goals based on preferences."""
    
    def __init__(self, agent_id=None, goal_preferences=None, rationality=1.0, color=None):
        super().__init__(id=agent_id, color=color or '#1E90FF')  # Dodger blue
        self.species = "GoalDirected"
        self.goal_preferences = goal_preferences or {}
        self.rationality = rationality  # Higher values mean more optimal behavior
        
    def plan_path(self):
        """Plan path toward preferred goals."""
        try:
            # Default to shortest path if no preferences
            self.path = nx.shortest_path(self.environment.G_undirected, 
                                       source=self.current_node, 
                                       target=self.goal_node, 
                                       weight='length')
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]
    
    def get_action(self, state):
        """
        Get action based on current state and goal preferences.
        
        Args:
            state: Current state from environment
            
        Returns:
            Action to take (integer)
        """
        # First try to use the goal preferences
        if self.goal_preferences and hasattr(state, 'available_actions'):
            actions = state.available_actions
            
            # If no available actions, use default implementation
            if not actions:
                return super().get_action(state)
                
            # Choose action based on rationality
            if random.random() < self.rationality:
                # Find action that moves toward goal
                best_action = actions[0]
                best_distance = float('inf')
                
                for action in actions:
                    # Estimate distance to goal after taking this action
                    next_state = state.get_next_state(action)
                    if hasattr(next_state, 'distance_to_goal'):
                        distance = next_state.distance_to_goal()
                        if distance < best_distance:
                            best_distance = distance
                            best_action = action
                
                return best_action
            else:
                # Random action
                return random.choice(actions)
        
        # Fall back to standard path-based action selection
        return super().get_action(state)