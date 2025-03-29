import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import os
from ..utils.config import VISUAL_CONFIG
from ..utils.run_manager import RunManager

class Simulator:
    """Simulation manager for running multi-agent simulations."""
    
    def __init__(self, environment, run_manager=None):
        self.environment = environment
        # Use the provided run_manager or create a new one if none is provided
        if run_manager:
            self.run_manager = run_manager
        else:
            self.run_manager = RunManager(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visuals'))
        
    def _get_animation_filename(self, name, species=None):
        """
        Generate a filename for an animation within the current run folder.
        
        Args:
            name: Base name for the animation
            species: Optional species name to include in filename
            
        Returns:
            Full path to the animation file
        """
        if species:
            filename = f"{name}_{species.lower()}.gif"
        else:
            filename = f"{name}.gif"
            
        return self.run_manager.get_animation_path(filename)
    
    def run_simulation(self, max_steps=300, animate=True, save_animation=False):
        """Run the simulation with all agents in the environment."""
        
        # Reset all agents
        self.environment.reset_agents()
        
        if animate:
            # Set up visualization with enhanced styling
            plt.style.use('dark_background')  # Use dark background for more vibrant colors
            
            fig, ax = plt.subplots(figsize=(15, 15), dpi=100, facecolor='#1f1f1f')
            ax.set_facecolor('#1f1f1f')
            
            # Add a subtle grid
            ax.grid(color='#333333', linestyle='-', linewidth=0.3, alpha=0.3)
            
            # Call the environment's visualization function but customize it
            self.environment.visualize_map(ax=ax)
            
            # Style improvements for the map
            plt.title("UCSD Campus Navigation Simulation", 
                     fontsize=18, fontweight='bold', color='white',
                     fontfamily='sans-serif', pad=20)
            
            # Add a timestamp display
            timestamp_text = ax.text(0.02, 0.02, "Step: 0", transform=ax.transAxes,
                                  fontsize=14, fontweight='bold', color='white',
                                  bbox=dict(facecolor='#333333', alpha=0.7, 
                                          boxstyle='round,pad=0.5', 
                                          edgecolor='#555555'))
            
            # Add goal markers to the map first (so they're always visible)
            goal_markers = []
            for agent in self.environment.agents:
                goal_x, goal_y = agent.get_position(agent.goal_node)
                goal_marker = ax.scatter([goal_x], [goal_y], 
                            color=agent.color,
                            marker=VISUAL_CONFIG["markers"]["goal"], 
                            s=VISUAL_CONFIG["sizes"]["goal"]*2.5,  # Make them bigger
                            edgecolor='white',  # White edge for visibility
                            linewidth=1.5,
                            zorder=20,  # Higher zorder to ensure visibility
                            label=f"{agent.species} Goal")
                goal_markers.append(goal_marker)
            
            # Add start markers too
            start_markers = []
            for agent in self.environment.agents:
                start_x, start_y = agent.get_position(agent.start_node)
                start_marker = ax.scatter([start_x], [start_y], 
                            color=agent.color,
                            marker=VISUAL_CONFIG["markers"]["start"], 
                            s=VISUAL_CONFIG["sizes"]["start"]*2,  # Make them bigger
                            edgecolor='white',  # White edge for visibility
                            linewidth=1.5,
                            zorder=15,  # Higher zorder to ensure visibility
                            label=f"{agent.species} Start")
                start_markers.append(start_marker)
            
            # Add a completion flag and counter to prevent multiple calls
            completion_reported = False
            animation_stopped = False
            
            # Set up animation
            def update(frame):
                nonlocal completion_reported, animation_stopped
                
                # If animation is already stopped, just return the current state
                if animation_stopped:
                    return [scatter] + paths + [timestamp_text]
                    
                if frame > 0:  # Skip the first frame
                    self.environment.step()
                    
                # Update agent positions
                agent_positions = np.array([agent.get_position() for agent in self.environment.agents])
                scatter.set_offsets(agent_positions)
                
                # Update paths with gradient effect
                for i, agent in enumerate(self.environment.agents):
                    path_x, path_y = agent.get_path_coordinates()
                    paths[i].set_data(path_x, path_y)
                
                # Update timestamp
                timestamp_text.set_text(f"Step: {frame}")
                    
                # Check if we're done - only report once
                if self.environment.all_agents_done():
                    if not completion_reported:
                        print("All agents have reached their goals!")
                        completion_reported = True
                        animation_stopped = True  # Mark animation as stopped
                        anim.event_source.stop()
                elif frame >= max_steps:
                    print(f"Reached maximum steps ({max_steps}). Not all agents reached their goals.")
                    animation_stopped = True  # Mark animation as stopped
                    anim.event_source.stop()
                    
                return [scatter] + paths + [timestamp_text]
            
            # Initial positions for agents
            agent_positions = np.array([agent.get_position() for agent in self.environment.agents])
            scatter = ax.scatter(agent_positions[:, 0], agent_positions[:, 1],
                                c=[agent.color for agent in self.environment.agents],
                                s=VISUAL_CONFIG["sizes"]["position"]*1.5, 
                                zorder=10, 
                                marker=VISUAL_CONFIG["markers"]["position"],
                                edgecolor='white', linewidth=0.5,
                                label="Agents")
            
            # Initialize paths with enhanced styling
            paths = []
            for agent in self.environment.agents:
                line, = ax.plot([], [], color=agent.color, 
                               alpha=VISUAL_CONFIG["path_alpha"]+0.2,  # More visible 
                               linewidth=VISUAL_CONFIG["sizes"]["path_line"]*1.5,  # Thicker
                               solid_capstyle='round')  # Rounded line ends
                paths.append(line)
            
            # Create a custom legend outside the plot
            handles = []
            labels = []
            
            agent_types = set(agent.species for agent in self.environment.agents)
            for species in agent_types:
                color = next(agent.color for agent in self.environment.agents if agent.species == species)
                # Path line for legend
                path_line = plt.Line2D([0], [0], color=color, 
                                     lw=VISUAL_CONFIG["sizes"]["path_line"]*1.5, 
                                     label=species)
                # Position marker for legend
                pos_marker = plt.Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["position"], 
                                      color=color, linestyle='', 
                                      markersize=10, label=f"{species} Agent")
                # Start marker for legend
                start_marker = plt.Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["start"], 
                                        color=color, linestyle='', markeredgecolor='white',
                                        markersize=10, label=f"{species} Start")
                # Goal marker for legend
                goal_marker = plt.Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["goal"], 
                                       color=color, linestyle='', markeredgecolor='white',
                                       markersize=12, label=f"{species} Goal")
                
                handles.extend([path_line, pos_marker, start_marker, goal_marker])
                labels.extend([f"{species} Path", f"{species} Agent", f"{species} Start", f"{species} Goal"])
            
            # Add landmark marker to legend
            landmark_marker = plt.Line2D([0], [0], 
                                       marker=VISUAL_CONFIG["markers"]["landmark"],
                                       color=VISUAL_CONFIG["landmark_color"], 
                                       linestyle='', markersize=10, 
                                       label="Landmark")
            handles.append(landmark_marker)
            labels.append("Landmark")
            
            # Position the legend outside the plot to avoid obscuring the visualization
            legend = ax.legend(handles, labels, 
                              loc='upper left', bbox_to_anchor=(1.01, 1.0),
                              fontsize=10, framealpha=0.8, facecolor='#333333',
                              edgecolor='#555555')
            # Set legend text color to white for dark background
            for text in legend.get_texts():
                text.set_color('white')
            
            # Adjust layout to make room for legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            
            # Create animation with a shorter interval to make it more responsive
            anim = FuncAnimation(fig, update, frames=max_steps+1, interval=200, blit=True, cache_frame_data=False)
            
            # Save animation if requested
            if save_animation:
                animation_path = self._get_animation_filename("campus_navigation")
                print(f"Saving animation to {animation_path}...")
                anim.save(animation_path, writer='pillow', fps=10, dpi=120)
                print(f"Animation saved successfully to {animation_path}")
            
            plt.show(block=True)  # Use block=True to ensure the plot window blocks until closed
            
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
        """Visualize the final state of all agents with enhanced styling."""
        # Set up figure with modern styling
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 15), dpi=100, facecolor='#1f1f1f')
        ax.set_facecolor('#1f1f1f')
        
        # Add a subtle grid
        ax.grid(color='#333333', linestyle='-', linewidth=0.3, alpha=0.3)
        
        # Call the environment's visualization function
        self.environment.visualize_map(ax=ax)
        
        # Plot all agent paths with enhanced styling
        for agent in self.environment.agents:
            path_x, path_y = agent.get_path_coordinates()
            ax.plot(path_x, path_y, color=agent.color, 
                   alpha=VISUAL_CONFIG["path_alpha"]+0.2,  # More visible 
                   linewidth=VISUAL_CONFIG["sizes"]["path_line"]*1.5,  # Thicker 
                   solid_capstyle='round',  # Rounded line ends
                   label=f"{agent.species} path")
            
            # Plot start and goal with enhanced visibility
            start_x, start_y = agent.get_position(agent.start_node)
            goal_x, goal_y = agent.get_position(agent.goal_node)
            
            # Enhanced start marker
            ax.scatter([start_x], [start_y], color=agent.color, 
                      marker=VISUAL_CONFIG["markers"]["start"], 
                      s=VISUAL_CONFIG["sizes"]["start"]*2,
                      edgecolor='white',
                      linewidth=1.5,
                      zorder=15)
            
            # Enhanced goal marker
            ax.scatter([goal_x], [goal_y], color=agent.color, 
                      marker=VISUAL_CONFIG["markers"]["goal"], 
                      s=VISUAL_CONFIG["sizes"]["goal"]*2.5,
                      edgecolor='white',
                      linewidth=1.5,
                      zorder=20)
        
        # Add improved title
        plt.title("Agent Paths in UCSD Campus", 
                 fontsize=18, fontweight='bold', color='white',
                 fontfamily='sans-serif', pad=20)
        
        # Create a custom legend outside the plot
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, 
                          loc='upper left', bbox_to_anchor=(1.01, 1.0),
                          fontsize=10, framealpha=0.8, facecolor='#333333',
                          edgecolor='#555555')
        
        # Set legend text color to white for dark background
        for text in legend.get_texts():
            text.set_color('white')
        
        # Adjust layout to make room for legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # Save the visualization
        final_state_path = self.run_manager.get_plot_path("final_state.png")
        plt.savefig(final_state_path, dpi=150, bbox_inches='tight')
        print(f"Saved final state visualization to {final_state_path}")
        
        plt.show()