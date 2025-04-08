from real_world_src.agents.agent_base import Agent
import numpy as np
import torch

class ToMAgent(Agent):
    """Agent that uses Theory of Mind to predict other agents' movements."""
    
    def __init__(self, agent_id, environment, tomnet, start_node=None, goal_node=None, 
                 color=None, speed=1.0, radius=10):
        super(ToMAgent, self).__init__(agent_id, color=color or '#FFFFFF', speed=speed)
        self.species = "ToM"
        self.tomnet = tomnet
        self.trajectory_collector = None  # Will be set by simulator
        self.other_agents_predictions = {}  # Store predictions for other agents
        self.prediction_horizon = 5  # How many steps ahead to predict
        
    def set_trajectory_collector(self, collector):
        """Set the trajectory collector for gathering agent data."""
        self.trajectory_collector = collector
        
    def plan_path(self):
        """Plan path considering predicted movements of other agents."""
        # First, predict where other agents will go
        self._predict_other_agents()
        
        # Find shortest path while avoiding predicted high-traffic areas
        try:
            # Get all nodes and edges from environment graph
            G = self.environment.G_undirected
            
            # Create a copy of the graph to modify edge weights
            import networkx as nx
            G_planning = G.copy()
            
            # Adjust edge weights based on predicted agent locations
            for u, v, data in G_planning.edges(data=True):
                # Default weight is length
                weight = data.get('length', 1.0)
                
                # Check for predicted agent traffic on this edge
                traffic = self._get_predicted_traffic(u, v)
                
                # Increase weight based on traffic (avoidance)
                traffic_factor = 1.0 + traffic * 0.5  # Tunable parameter
                
                # Update edge weight
                G_planning[u][v]['weight'] = weight * traffic_factor
            
            # Find path using updated weights
            self.path = nx.shortest_path(G_planning, 
                                         source=self.current_node, 
                                         target=self.goal_node, 
                                         weight='weight')
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]
            
    def _predict_other_agents(self):
        """Use ToMnet to predict future positions of other agents."""
        if not self.trajectory_collector:
            return
            
        # Get all other agents in environment
        other_agents = [agent for agent in self.environment.agents if agent.id != self.id]
        
        for agent in other_agents:
            # Get past trajectories for this agent
            agent_id = agent.id
            if agent_id not in self.trajectory_collector.episode_buffers:
                continue
                
            # Get recent trajectory
            recent_traj = self.trajectory_collector.episode_buffers[agent_id]
            if len(recent_traj) < 2:  # Need at least a couple of points
                continue
                
            # Prepare data for ToMnet
            past_trajs = []
            for species, trajs in self.trajectory_collector.trajectories.items():
                if species == agent.species:
                    past_trajs.extend(trajs[:5])  # Use up to 5 past trajectories
            
            if not past_trajs:
                continue
                
            # Convert to tensors for ToMnet
            past_tensor = self._prepare_past_trajectories(past_trajs)
            recent_tensor = self._prepare_recent_trajectory(recent_traj)
            current_state = self._prepare_current_state(agent)
            
            # Get prediction from ToMnet
            with torch.no_grad():
                prediction = self.tomnet(past_tensor, recent_tensor, current_state)
                
            # Convert prediction to likely path
            predicted_path = self._convert_prediction_to_path(agent, prediction)
            
            # Store prediction
            self.other_agents_predictions[agent_id] = predicted_path
    
    def _prepare_past_trajectories(self, past_trajs):
        """Convert past trajectories to tensor format for ToMnet."""
        # Implementation depends on exact ToMnet input format
        # Simplified version:
        return torch.tensor([0.0])  # Placeholder
        
    def _prepare_recent_trajectory(self, recent_traj):
        """Convert recent trajectory to tensor format for ToMnet."""
        # Implementation depends on exact ToMnet input format
        # Simplified version:
        return torch.tensor([0.0])  # Placeholder
        
    def _prepare_current_state(self, agent):
        """Convert current state to tensor format for ToMnet."""
        # Implementation depends on exact ToMnet input format
        # Simplified version:
        return torch.tensor([0.0])  # Placeholder
        
    def _convert_prediction_to_path(self, agent, prediction):
        """Convert ToMnet prediction to a likely path."""
        # Implementation depends on exact ToMnet output format
        # Simplified version returns current agent path:
        return agent.path if hasattr(agent, 'path') else []
        
    def _get_predicted_traffic(self, node1, node2):
        """Calculate predicted traffic on edge between node1 and node2."""
        traffic = 0
        
        for predicted_path in self.other_agents_predictions.values():
            # Check if this edge is in the predicted path
            for i in range(len(predicted_path) - 1):
                if (predicted_path[i] == node1 and predicted_path[i+1] == node2) or \
                   (predicted_path[i] == node2 and predicted_path[i+1] == node1):
                    traffic += 1
                    break
                    
        return traffic

    def infer_goal_with_tomnet(self, observed_agent_id, model_path, experiment=1):
        """
        Infer the goal of an observed agent using the trained ToMNet model.
        
        Args:
            observed_agent_id: ID of the agent to observe
            model_path: Path to trained model checkpoint
            experiment: Experiment number (1 or 2)
            
        Returns:
            Dictionary with inference results
        """
        from real_world_src.utils.goal_inference_helper import GoalInferenceHelper
        
        # Check if the observed agent exists in our memory
        if observed_agent_id not in self.observed_agents:
            print(f"Agent {observed_agent_id} not observed yet")
            return None
        
        # Get observed trajectory
        observed_trajectory = self.observed_agents[observed_agent_id]['trajectory']
        
        if not observed_trajectory:
            print(f"No trajectory observed for agent {observed_agent_id}")
            return None
        
        # Get current state
        current_state = self.environment.get_agent_state(observed_agent_id)
        
        if current_state is None:
            print(f"Could not get current state for agent {observed_agent_id}")
            return None
        
        # Create goal inference helper
        helper = GoalInferenceHelper(
            model_path=model_path,
            experiment=experiment
        )
        
        # Perform goal inference
        results = helper.infer_goals(
            past_trajectory=observed_trajectory,
            query_state=current_state
        )
        
        # Update our belief about the agent's goal
        goal_idx = results['most_likely_object']
        object_names = ['Library', 'Cafe', 'Dorm', 'Lab']  # Update based on your environment
        inferred_goal = object_names[goal_idx]
        
        self.observed_agents[observed_agent_id]['inferred_goal'] = inferred_goal
        self.observed_agents[observed_agent_id]['goal_probabilities'] = results['consumption_probs']
        
        return results