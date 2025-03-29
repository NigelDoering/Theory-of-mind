from .agent_species import (
    ShortestPathAgent, 
    RandomWalkAgent, 
    LandmarkAgent, 
    SocialAgent
)

class AgentFactory:
    """Factory for creating and managing agent populations."""
    
    @staticmethod
    def create_agent(species_type, agent_id=None):
        """Create an agent of the specified species type."""
        if species_type.lower() == "shortest":
            return ShortestPathAgent(agent_id)
        elif species_type.lower() == "random":
            return RandomWalkAgent(agent_id)
        elif species_type.lower() == "landmark":
            return LandmarkAgent(agent_id)
        elif species_type.lower() == "social":
            return SocialAgent(agent_id)
        else:
            raise ValueError(f"Unknown agent species: {species_type}")
    
    @staticmethod
    def populate_environment(environment, agent_counts):
        """
        Populate an environment with specified numbers of agent species.
        
        Args:
            environment: The environment to populate
            agent_counts: Dict mapping species names to counts, e.g. {"shortest": 5, "random": 3}
        """
        agent_id = 1
        for species, count in agent_counts.items():
            for _ in range(count):
                agent = AgentFactory.create_agent(species, f"{species}_{agent_id}")
                environment.add_agent(agent)
                agent_id += 1
        
        return environment.agents