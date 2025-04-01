import os
import sys
import time
import matplotlib
matplotlib.use('Agg')  # Force Agg backend for all plots
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.agents.agent_factory import AgentFactory
from real_world_src.simulation.simulator import Simulator
from real_world_src.utils.visualization import (
    plot_agent_paths, visualize_agent_decision, plot_species_grid,
    animate_species_grid, animate_single_species
)
from real_world_src.utils.run_manager import RunManager

def main():
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Initialize the run manager
    run_manager = RunManager(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visuals'))
    run_dir = run_manager.start_new_run()
    
    # Create the UCSD campus environment
    print("Initializing UCSD campus environment...")
    campus = CampusEnvironment()
    
    # Populate with agents of different species
    print("Creating agents...")
    agent_populations = {
        "shortest": 2,  # Green palette
        "random": 2,    # Blue palette
        "landmark": 2,  # Yellow palette
        "social": 2,    # Red palette
        "explorer": 2,  # Orange palette
        "obstacle": 2,  # Purple palette
        "scared": 2,    # Lavender palette
        "risky": 2      # OrangeRed palette
    }
    
    agents = AgentFactory.populate_environment(campus, agent_populations)
    print(f"Created {len(agents)} agents")
    
    # Create and run the simulation until all agents reach their goals
    print("Starting simulation...")
    simulator = Simulator(campus, run_manager=run_manager)  # Pass the run_manager here
    simulator.run_simulation(max_steps=5000, animate=True, save_animation=True)
    
    # After simulation, visualize the results
    print("Creating visualization of all agent paths...")
    fig, ax = plot_agent_paths(campus, agents, "UCSD Campus Navigation Paths")
    all_paths_file = run_manager.get_plot_path("all_agents_paths.png")  # Saved to static_plots
    fig.savefig(all_paths_file, dpi=300, bbox_inches='tight')
    print(f"Saved all paths visualization to {all_paths_file}")
    plt.close(fig)
    
    # Create the species grid visualization (static version)
    print("Creating species grid visualization...")
    grid_fig, grid_axs = plot_species_grid(campus, agents, "Species Behavior Comparison")
    species_grid_file = run_manager.get_plot_path("species_grid.png")  # Saved to static_plots
    grid_fig.savefig(species_grid_file, dpi=300, bbox_inches='tight')
    print(f"Saved species grid visualization to {species_grid_file}")
    plt.close(grid_fig)
    
    # Pick one agent of each type to visualize its decision process
    print("Creating individual agent visualizations...")
    for species in ["shortest", "random", "landmark", "social", "explorer", "obstacle", "scared", "risky"]:
        agent = next((a for a in agents if a.species.lower().startswith(species.lower())), None)
        if agent:
            print(f"Visualizing {agent.species} agent decision process...")
            # Decision visualization - save to agent_decisions folder
            decision_fig, decision_ax = visualize_agent_decision(campus, agent, f"{agent.species} Agent Decision Process")
            decision_file = run_manager.get_agent_decision_path(f"{species}_decision.png")
            decision_fig.savefig(decision_file, dpi=300, bbox_inches='tight')
            print(f"Saved {species} decision visualization to {decision_file}")
            plt.close(decision_fig)
            
            # Also save individual agent path to species_plots folder
            species_fig, species_ax = plot_agent_paths(campus, [agent], f"{agent.species} Navigation Path")
            species_path_file = run_manager.get_species_plot_path(f"{species}_path.png")
            species_fig.savefig(species_path_file, dpi=300, bbox_inches='tight')
            print(f"Saved {species} path visualization to {species_path_file}")
            plt.close(species_fig)
    
    # Group agents by species for visualizations
    species_groups = {}
    for agent in agents:
        species = agent.species
        if species not in species_groups:
            species_groups[species] = []
        species_groups[species].append(agent)
    
    # Create a static visualization of each species group and save to species_plots
    print("Creating static species group visualizations...")
    for species, species_agents in species_groups.items():
        if species_agents:
            # Create static species group visualization
            species_group_fig, species_group_ax = plot_agent_paths(
                campus, species_agents, f"{species} Group Navigation Paths")
            species_group_file = run_manager.get_species_plot_path(f"{species}_group.png")
            species_group_fig.savefig(species_group_file, dpi=300, bbox_inches='tight')
            print(f"Saved {species} group visualization to {species_group_file}")
            plt.close(species_group_fig)
    
    # Create an animated grid visualization showing all species
    print("Creating animated species grid visualization...")
    grid_anim_file = run_manager.get_animation_path("species_grid_animation.gif")
    animate_species_grid(campus, agents, 
                        title="UCSD Campus Navigation by Species", 
                        max_frames=1000,
                        interval=50, 
                        save_path=grid_anim_file, 
                        dpi=300)
    print(f"Saved animated species grid to {grid_anim_file}")
    
    # Create individual animated GIFs for each species
    print("Creating individual species animations...")
    for species, species_agents in species_groups.items():
        if species_agents:
            print(f"Creating animation for {species} agents...")
            species_anim_file = run_manager.get_animation_path(f"{species.lower()}_animation.gif")
            # Use a specialized animation function for single species
            animate_single_species(campus, species_agents, 
                                title=f"{species} Navigation Patterns",
                                max_frames=1000,
                                interval=50,
                                save_path=species_anim_file,
                                dpi=300, enhanced=True)
            print(f"Saved {species} animation to {species_anim_file}")
    
    # Generate a README.md file for this run with all the visualizations listed
    readme_path = os.path.join(run_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Simulation Run #{run_manager.current_run}\n\n")
        f.write(f"This folder contains visualizations from simulation run #{run_manager.current_run} ")
        f.write(f"with {len(agents)} agents of {len(species_groups)} different species.\n\n")
        
        f.write("## Animations\n\n")
        animations = os.listdir(os.path.join(run_dir, "animations"))
        for anim in animations:
            f.write(f"- [{anim}](animations/{anim})\n")
        
        f.write("\n## Static Plots\n\n")
        plots = os.listdir(os.path.join(run_dir, "static_plots"))
        for plot in plots:
            f.write(f"- [{plot}](static_plots/{plot})\n")
            
        f.write("\n## Agent Decisions\n\n")
        decisions = os.listdir(os.path.join(run_dir, "agent_decisions"))
        for decision in decisions:
            f.write(f"- [{decision}](agent_decisions/{decision})\n")
    
    print(f"All visualizations for run #{run_manager.current_run} saved to {run_dir}")
    print(f"README.md created at {readme_path}")

if __name__ == "__main__":
    main()