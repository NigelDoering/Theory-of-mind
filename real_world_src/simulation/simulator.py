import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time

class Simulator:
    """Simulation manager for running multi-agent simulations."""
    
    def __init__(self, environment):
        self.environment = environment
    
    def run_simulation(self, max_steps=300, animate=True, save_animation=False):
        """Run the simulation with all agents in the environment."""
        # Reset all agents
        self.environment.reset_agents()
        
        if animate:
            # Set up visualization
            fig, ax = self.environment.visualize_map()
            
            # Set up animation
            def update(frame):
                if frame > 0:  # Skip the first frame
                    self.environment.step()
                    
                # Update agent positions
                agent_positions = np.array([agent.get_position() for agent in self.environment.agents])
                scatter.set_offsets(agent_positions)
                
                # Update paths
                for i, agent in enumerate(self.environment.agents):
                    paths[i].set_data(*agent.get_path_coordinates())
                    
                # Check if we're done
                if self.environment.all_agents_done() or frame >= max_steps:
                    anim.event_source.stop()
                    print(f"Simulation completed at step {frame}")
                    
                return [scatter] + paths
            
            # Initial positions for agents
            agent_positions = np.array([agent.get_position() for agent in self.environment.agents])
            scatter = ax.scatter(agent_positions[:, 0], agent_positions[:, 1],
                                c=[agent.color for agent in self.environment.agents],
                                s=80, zorder=10, label="Agents")
            
            # Initialize paths
            paths = []
            for agent in self.environment.agents:
                line, = ax.plot([], [], color=agent.color, alpha=0.7, linewidth=2)
                paths.append(line)
            
            # Add legend for agent types
            agent_types = set(agent.species for agent in self.environment.agents)
            for species in agent_types:
                color = next(agent.color for agent in self.environment.agents if agent.species == species)
                ax.scatter([], [], c=color, s=80, label=species)
            ax.legend()
            
            # Create animation
            anim = FuncAnimation(fig, update, frames=max_steps+1, interval=200, blit=True)
            
            # Save animation if requested
            if save_animation:
                anim.save('ucsd_path.gif', writer='pillow', fps=10)
            
            plt.show()
            
        else:
            # Non-animated version
            step = 0
            while not self.environment.all_agents_done() and step < max_steps:
                self.environment.step()
                step += 1
                
            print(f"Simulation completed in {step} steps")
            
            # Visualize final state
            self.visualize_final_state()
    
    def visualize_final_state(self):
        """Visualize the final state of all agents."""
        fig, ax = self.environment.visualize_map()
        
        # Plot all agent paths
        for agent in self.environment.agents:
            path_x, path_y = agent.get_path_coordinates()
            ax.plot(path_x, path_y, color=agent.color, alpha=0.7, linewidth=2, label=f"{agent.species} path")
            
            # Plot start and goal
            start_x, start_y = agent.get_position(agent.start_node)
            goal_x, goal_y = agent.get_position(agent.goal_node)
            ax.scatter([start_x], [start_y], color=agent.color, marker='o', s=100, zorder=10)
            ax.scatter([goal_x], [goal_y], color=agent.color, marker='*', s=150, zorder=10)
        
        plt.title("Agent Paths in UCSD Campus")
        plt.tight_layout()
        plt.show()