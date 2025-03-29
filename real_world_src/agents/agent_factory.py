from .agent_species import (
    ShortestPathAgent, 
    RandomWalkAgent, 
    LandmarkAgent, 
    SocialAgent,
    ExplorerAgent,
    ObstacleAvoidingAgent,
    ScaredAgent,
    RiskyAgent
)
from ..utils.config import get_agent_color

class AgentFactory:
    """Factory for creating and managing agent populations."""
    
    @staticmethod
    def create_agent(species_type, agent_id=None, color=None):
        """Create an agent of the specified species type."""
        # Extract index from ID if possible
        index = 0
        if agent_id and '_' in agent_id:
            try:
                index = int(agent_id.split('_')[-1])
            except ValueError:
                pass
        
        # Use provided color or get from config
        if not color:
            color = get_agent_color(species_type, index)
            
        # Create the agent with proper species name capitalization
        if species_type.lower() == "shortest":
            return ShortestPathAgent(agent_id, color=color)
        elif species_type.lower() == "random":
            return RandomWalkAgent(agent_id, color=color)
        elif species_type.lower() == "landmark":
            return LandmarkAgent(agent_id, color=color)
        elif species_type.lower() == "social":
            return SocialAgent(agent_id, color=color)
        elif species_type.lower() == "explorer":
            return ExplorerAgent(agent_id, color=color)
        elif species_type.lower() == "obstacle":
            return ObstacleAvoidingAgent(agent_id, color=color)
        elif species_type.lower() == "scared":
            return ScaredAgent(agent_id, color=color)
        elif species_type.lower() == "risky":
            return RiskyAgent(agent_id, color=color)
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
        all_agents = []
        
        # Use clearer names for agent IDs
        species_display_names = {
            "shortest": "ShortestPath",
            "random": "RandomWalk",
            "landmark": "Landmark",
            "social": "Social",
            "explorer": "Explorer",
            "obstacle": "ObstacleAvoider",
            "scared": "Scared",
            "risky": "Risky"
        }
        
        for species, count in agent_counts.items():
            # Get proper display name for the species
            display_name = species_display_names.get(species.lower(), species.capitalize())
            
            for i in range(count):
                # Create agent ID in the format "SpeciesName_Number"
                agent_id = f"{display_name}_{i+1}"
                agent = AgentFactory.create_agent(species, agent_id)
                environment.add_agent(agent)
                all_agents.append(agent)
                print(f"  Added agent: {agent_id}")
        
        return all_agents