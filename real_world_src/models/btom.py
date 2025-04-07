import networkx as nx
import numpy as np

class BToM:
    def __init__(self, campus, agents, beta=1.0):
        """
        Initializes the BToM model.
        
        Parameters:
        -----------
        campus : object
            An object that contains a NetworkX graph (campus.G).
        agents : list
            A list of agent objects. Each agent must have attributes:
              - id: a unique identifier,
              - goal_node: the agent's candidate goal node,
              - path: a list of nodes representing the agent's trajectory.
        beta : float, optional
            The inverse temperature parameter for the softmax.
        """
        self.campus = campus
        self.beta = beta
        
        print("Computing shortest paths...")
        # Compute shortest paths from all nodes to all nodes
        self.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(campus.G))
        print("Done")
        
        # Set up candidate goals from agents (assume no duplicates)
        self.candidate_goals = [agent.goal_node for agent in agents]
        prior_prob = 1 / len(self.candidate_goals)
        # Create a uniform prior for candidate goals
        uniform_prior = {g: prior_prob for g in self.candidate_goals}
        
        # Initialize agent posteriors (each agent gets its own copy of the uniform prior)
        self.agent_posteriors = {agent.id: uniform_prior.copy() for agent in agents}
    
    def Q_value(self, s, a, g):
        """
        Computes the Q-value for transitioning from state s to next state a,
        given candidate goal g.
        
        Since transitions are deterministic in our graph, the Q-value is 
        defined as the negative shortest path length from a to g.
        """
        return -self.all_shortest_paths[a][g]
    
    def update_agent_posterior(self, agent, path_proportion=0.5):
        """
        Updates the posterior over candidate goals for a given agent using its trajectory.
        
        Parameters:
        -----------
        agent : object
            An agent with attributes 'id' and 'path' (a list of node IDs).
        path_proportion : float, optional
            Fraction of the trajectory to use in inference.
        
        Returns:
        --------
        posterior : dict
            The updated posterior distribution over candidate goals.
        """
        print(f"Updating posterior for agent {agent.id}")
        path = agent.path
        n_pairs = int(len(path) * path_proportion)
        # Convert the agent's path into state-action pairs:
        observed_trajectory = [(path[i], path[i+1]) for i in range(n_pairs)]
        
        # Retrieve the current posterior (initialized uniformly)
        posterior = self.agent_posteriors[agent.id].copy()
        for g in self.candidate_goals:
            prob = posterior[g]
            for (s, a) in observed_trajectory:
                # Determine possible actions (neighbors) from state s.
                actions_possible = list(self.campus.G.neighbors(s))
                if a not in actions_possible:
                    action_prob = 0.0
                else:
                    # Compute Q-values for all possible actions given goal g.
                    q_vals = np.array([self.Q_value(s, a_prime, g) for a_prime in actions_possible])
                    # Compute likelihoods using a Boltzmann (softmax) model.
                    likelihoods = np.exp(self.beta * q_vals)
                    likelihoods /= likelihoods.sum()
                    action_prob = likelihoods[actions_possible.index(a)]
                prob *= action_prob
            posterior[g] = prob
        
        # Normalize the posterior.
        total = sum(posterior.values())
        if total > 0:
            for g in posterior:
                posterior[g] /= total
        else:
            # If total probability is zero, reset to uniform.
            posterior = {g: 1 / len(self.candidate_goals) for g in self.candidate_goals}
        
        self.agent_posteriors[agent.id] = posterior
        return posterior

    def infer_goal(self, agent):
        """
        Infers the most likely goal for an agent by updating its posterior
        (if necessary) and selecting the goal with the highest probability.
        
        Parameters:
        -----------
        agent : object
            An agent with attributes 'id' and 'path'.
        
        Returns:
        --------
        inferred_goal : The candidate goal with the highest posterior probability.
        """
        posterior = self.update_agent_posterior(agent)
        inferred_goal = max(posterior, key=posterior.get)
        return inferred_goal