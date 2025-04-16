import numpy as np
import networkx as nx

class BToM:
    def __init__(self, campus, agents, beta=1.0):
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
        
        print("Computing shortest paths...")
        self.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(campus.G))
        print("Done")
        
        # Candidate goals are derived from the agents' goal nodes; duplicates assumed not to occur.
        self.candidate_goals = [agent.goal_node for agent in agents]
        prior_prob = 1 / len(self.candidate_goals)
        uniform_prior = {g: prior_prob for g in self.candidate_goals}
        
        # Initialize posterior for each agent
        self.agent_posteriors = {agent.id: uniform_prior.copy() for agent in agents}

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
        # For each candidate goal, update the probability based on the observed action.
        for g in self.candidate_goals:
            prob = current_posterior[g]
            actions_possible = list(self.campus.G.neighbors(s))
            if a not in actions_possible:
                action_prob = 0.0
            else:
                # Compute Q-values and likelihoods for possible actions
                q_vals = np.array([self.Q_value(s, a_prime, g) for a_prime in actions_possible])
                likelihoods = np.exp(self.beta * q_vals)
                likelihoods /= likelihoods.sum()
                action_prob = likelihoods[actions_possible.index(a)]
            new_posterior[g] = prob * action_prob
        
        # Normalize the new posterior
        total = sum(new_posterior.values())
        if total > 0:
            for g in new_posterior:
                new_posterior[g] /= total
        else:
            # Fallback to uniform if total probability is zero
            new_posterior = {g: 1 / len(self.candidate_goals) for g in self.candidate_goals}
        return new_posterior

    def update_agent_posterior_over_path(self, agent):
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
        path = agent.path
        posterior_list = []
        
        # Initialize with the agent's current posterior (uniform)
        current_posterior = self.agent_posteriors[agent.id].copy()
        posterior_list.append(current_posterior)
        
        # Loop over each step along the trajectory
        for i in range(len(path)-1):
            s = path[i]
            a = path[i+1]
            current_posterior = self.update_posterior_step(current_posterior, s, a)
            posterior_list.append(current_posterior)
            
        # Optionally, update the stored posterior for the agent with the last posterior
        self.agent_posteriors[agent.id] = current_posterior
        
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