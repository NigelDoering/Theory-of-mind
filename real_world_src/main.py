import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.agents.agent_factory import AgentFactory
from real_world_src.simulation.simulator import Simulator
from real_world_src.utils.visualization import plot_agent_paths, visualize_agent_decision

def main():
    # Create the UCSD campus environment
    print("Initializing UCSD campus environment...")
    campus = CampusEnvironment()
    
    # Populate with agents of different species
    print("Creating agents...")
    agent_populations = {
        "shortest": 3,
        "random": 3,
        "landmark": 3,
        "social": 3
    }
    
    agents = AgentFactory.populate_environment(campus, agent_populations)
    print(f"Created {len(agents)} agents")
    
    # Create and run the simulation
    print("Starting simulation...")
    simulator = Simulator(campus)
    simulator.run_simulation(max_steps=200, animate=True, save_animation=True)
    
    # After simulation, visualize the results
    plot_agent_paths(campus, agents, "UCSD Campus Navigation Paths")
    
    # Pick one agent of each type to visualize its decision process
    for species in ["shortest", "random", "landmark", "social"]:
        agent = next((a for a in agents if a.species.lower().startswith(species.lower())), None)
        if agent:
            visualize_agent_decision(campus, agent, f"{agent.species} Agent Decision Process")

if __name__ == "__main__":
    main()