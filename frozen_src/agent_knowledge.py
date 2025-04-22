import networkx as nx


class AgentKnowledge:
    """Represents the agent's knowledge of the environment."""
    
    def __init__(self, world_graph, perception_radius=2):
        """Initialize with the world graph (for structure only)."""
        self.world_graph = world_graph
        self.perception_radius = perception_radius
        
        # Initial knowledge - empty graph
        self.graph = nx.Graph()
        
        # Keep track of observed but unvisited nodes
        self.frontier = set()
        
        # Track visited nodes
        self.visited = set()
        
        # Track discovered goal nodes
        self.discovered_goals = {}
        
        # Track collected rewards
        self.collected_rewards = {}
        
        # Track the final goal
        self.final_goal = None
        
    def update(self, current_node):
        """Update knowledge based on current position and perception."""
        # Mark current node as visited
        self.visited.add(current_node)
        
        # Remove from frontier if present
        if current_node in self.frontier:
            self.frontier.remove(current_node)
            
        # Get current node info
        node_info = self.world_graph.get_node_info(current_node)
        
        # Add current node to knowledge graph if not already there
        if current_node not in self.graph:
            self.graph.add_node(current_node, **node_info)
        
        # Check if this is a goal and mark as collected
        if node_info['reward'] > 0 and current_node not in self.collected_rewards:
            self.collected_rewards[current_node] = node_info['reward']
            
            # If it's the final goal, mark it
            if node_info['is_final']:
                self.final_goal = current_node
        
        # Get nodes within perception radius
        observable_nodes = self._get_nodes_in_radius(current_node, self.perception_radius)
        
        # Update knowledge with observed nodes
        for node in observable_nodes:
            node_info = self.world_graph.get_node_info(node)
            
            # Add to knowledge graph
            if node not in self.graph:
                self.graph.add_node(node, **node_info)
                
                # If it has a reward and we haven't discovered it yet, track it
                if node_info['reward'] > 0:
                    self.discovered_goals[node] = node_info['reward']
                    
                    # If it's the final goal, mark it
                    if node_info['is_final']:
                        self.final_goal = node
                
                # Add to frontier if not visited
                if node not in self.visited:
                    self.frontier.add(node)
            
            # Update edges for all observed nodes to maintain connectivity
            self._update_edges_for_node(node)
        
    def _get_nodes_in_radius(self, center_node, radius):
        """Get all nodes within the perception radius of the center node."""
        center_i, center_j = self.world_graph.get_node_info(center_node)['pos']
        
        # Get dimensions from the world graph
        map_array = self.world_graph.goal_generator.map
        nrow, ncol = len(map_array), len(map_array[0])
        
        # Find all nodes within radius (Manhattan distance)
        observable_nodes = set()
        
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                # Check Manhattan distance
                if abs(di) + abs(dj) <= radius:
                    ni, nj = center_i + di, center_j + dj
                    
                    # Check if position is valid
                    if 0 <= ni < nrow and 0 <= nj < ncol:
                        node_id = ni * ncol + nj
                        
                        # Check if node exists in world graph (not a hole)
                        if node_id in self.world_graph.graph:
                            observable_nodes.add(node_id)
        
        return observable_nodes
    
    def _update_edges_for_node(self, node):
        """Update edges for a node based on the world graph structure."""
        # Get the adjacent nodes in the world graph
        for neighbor in self.world_graph.get_neighbors(node):
            # If both nodes are in our knowledge graph, add the edge
            if neighbor in self.graph:
                # Add edge with weight 1
                self.graph.add_edge(node, neighbor, weight=1)
    
    def has_full_knowledge(self):
        """Check if the agent has discovered the entire graph."""
        return len(self.graph) == len(self.world_graph.graph)