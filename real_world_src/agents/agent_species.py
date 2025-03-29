import networkx as nx
import random
from .agent_base import Agent

class ShortestPathAgent(Agent):
    """Agent that always takes the shortest path."""
    
    def __init__(self, id=None):
        super().__init__(id, color='blue')
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
    
    def __init__(self, id=None):
        super().__init__(id, color='red')
        self.species = "RandomWalk"
        self.planning_horizon = 3  # Steps to look ahead
        
    def plan_path(self):
        """Plan a biased random path that generally moves toward the goal."""
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
    
    def __init__(self, id=None):
        super().__init__(id, color='green')
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
    
    def __init__(self, id=None):
        super().__init__(id, color='purple')
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