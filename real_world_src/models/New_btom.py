import numpy as np
import networkx as nx

class BToM:
    def __init__(self, campus, agents, goals, beta=1.0):
        """
        Initializes the BToM model.
        
        Parameters:
        -----------
        campus : object
            Contains the NetworkX graph (campus.G).
        agents : list
            List of agent objects (each with attributes 'id', 'path', and 'goal_node').
        beta : float, optional
            Inverse temperature parameter for the softmax.
        """
        self.campus = campus
        self.beta = beta
        self.agents = agents
        
        print("Computing shortest paths...")
        self.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(campus.G_undirected))
        print("Done")
        
        # Candidate goals are derived from the agents' goal nodes; duplicates assumed not to occur.
        self.candidate_goals = goals
        prior_prob = 1 / len(self.candidate_goals)
        self.uniform_prior = {g: prior_prob for g in self.candidate_goals}
        
        self.reset_posteriors()

    def reset_posteriors(self):
        # Initialize posterior for each agent
        self.agent_posteriors = {agent.id: self.uniform_prior.copy() for agent in self.agents}

    def get_agent_prior(self, agent):
        """
        Returns the prior (initial posterior) for the given agent.
        If agent has a posterior, return it; otherwise, return uniform prior.
        """
        return self.agent_posteriors.get(agent.id, self.uniform_prior.copy())

    def Q_value(self, s, a, g):
        """
        Computes the Q-value for transitioning from state s to state a given goal g.
        Since transitions are deterministic, this is defined as the negative shortest
        path distance from node a to goal g.
        """
        return -self.all_shortest_paths[a][g]

    def update_posterior_step(self, current_posterior, s, a):
        """
        Updates the current posterior over candidate goals based on one state-action pair.
        
        Parameters:
        -----------
        current_posterior : dict
            The current posterior over candidate goals.
        s : node
            The current state (node ID).
        a : node
            The action taken (next node).
        
        Returns:
        --------
        new_posterior : dict
            The updated posterior.
        """
        new_posterior = {}
        
        for g in self.candidate_goals:
            prob = current_posterior[g]
            
            # Check if node exists and has neighbors
            if s not in self.campus.G:
                # If state doesn't exist, maintain uniform action probability
                action_prob = 1.0 / len(self.candidate_goals)
            else:
                try:
                    actions_possible = list(self.campus.G.neighbors(s))
                    if a not in actions_possible:
                        action_prob = 0.0
                    else:
                        action_prob = self.action_probability(s, a, g)
                except:
                    # Fallback if there's any issue getting neighbors
                    action_prob = 1.0 / len(self.candidate_goals)
        
            new_posterior[g] = prob * action_prob
    
        # Normalize the posterior
        total = sum(new_posterior.values())
        if total > 0:
            for g in new_posterior:
                new_posterior[g] /= total
        else:
            # If all probabilities are 0, reset to uniform
            for g in new_posterior:
                new_posterior[g] = 1.0 / len(self.candidate_goals)
    
        return new_posterior

    def update_agent_posterior_over_path(self, agent, path):
        """
        Updates and returns the list of posterior distributions over candidate goals for each step 
        along the agent's path.
        
        Parameters:
        -----------
        agent : object
            An agent with attributes 'id' and 'path' (a list of node IDs).
        
        Returns:
        --------
        posterior_list : list of dicts
            A list where each element is the posterior distribution (dict) at that step.
        """
        current_posterior = self.get_agent_prior(agent)
        posterior_list = [current_posterior.copy()]
        
        for i in range(len(path) - 1):
            s = path[i]
            a = path[i+1]
            
            # Check if nodes exist in the graph
            if s not in self.campus.G or a not in self.campus.G:
                print(f"Warning: Node {s} or {a} not found in graph, skipping step")
                # Keep the same posterior and continue
                posterior_list.append(current_posterior.copy())
                continue
                
            current_posterior = self.update_posterior_step(current_posterior, s, a)
            posterior_list.append(current_posterior)
        
        # Optionally, update the stored posterior for the agent with the last posterior
        self.update_agent_posterior(agent, current_posterior)
        
        return posterior_list

    def infer_goal(self, agent):
        """
        Infers the most likely goal for an agent based on its entire observed trajectory.
        
        Parameters:
        -----------
        agent : object
            An agent with attributes 'id' and 'path'.
        
        Returns:
        --------
        inferred_goal : 
            The candidate goal with the highest posterior probability at the final step.
        """
        posterior_list = self.update_agent_posterior_over_path(agent)
        final_posterior = posterior_list[-1]
        inferred_goal = max(final_posterior, key=final_posterior.get)
        return inferred_goal
    
    def safe_update_posterior_step(self, current_posterior, s, a):
        """Safe version that handles missing nodes"""
        new_posterior = {}
        
        for g in self.candidate_goals:
            prob = current_posterior[g]
            
            # Check if nodes exist in the graph
            if s not in self.campus.G or a not in self.campus.G:
                # Use uniform probability if nodes don't exist
                action_prob = 1.0 / len(self.candidate_goals)
            else:
                try:
                    actions_possible = list(self.campus.G.neighbors(s))
                    if a not in actions_possible:
                        action_prob = 0.0
                    else:
                        action_prob = self.action_probability(s, a, g)
                except Exception as e:
                    print(f"Error getting neighbors for node {s}: {e}")
                    action_prob = 1.0 / len(self.candidate_goals)
            
            new_posterior[g] = prob * action_prob
        
        # Normalize
        total = sum(new_posterior.values())
        if total > 0:
            for g in new_posterior:
                new_posterior[g] /= total
        else:
            for g in new_posterior:
                new_posterior[g] = 1.0 / len(self.candidate_goals)
        
        return new_posterior

    def safe_update_agent_posterior_over_path(self, agent, path):
        """Safe version that handles missing nodes"""
        current_posterior = self.get_agent_prior(agent)
        posterior_list = [current_posterior.copy()]
        
        for i in range(len(path) - 1):
            s = path[i]
            a = path[i+1]
            
            # Check if nodes exist
            if s not in self.campus.G or a not in self.campus.G:
                print(f"Warning: Node {s} or {a} not found in graph, skipping step")
                posterior_list.append(current_posterior.copy())
                continue
                
            current_posterior = self.safe_update_posterior_step(current_posterior, s, a)
            posterior_list.append(current_posterior)
        
        # Update stored posterior
        self.update_agent_posterior(agent, current_posterior)
        return posterior_list

    def update_agent_posterior(self, agent, posterior):
        """
        Updates the stored posterior for the given agent.
        """
        self.agent_posteriors[agent.id] = posterior.copy()