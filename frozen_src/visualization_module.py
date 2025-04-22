import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import imageio
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

class InteractiveVisualizer:
    """Handles interactive visualization of the environment, knowledge, and planning."""
    
    def __init__(self, world_graph, save_dir="orienteering_visuals"):
        """Initialize the interactive visualizer."""
        self.world_graph = world_graph
        self.save_dir = save_dir
        self.step = 0
        self.episode = 0
        
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Colors and visual style
        self.colors = {
            'S': (0.0, 0.5, 0.0),    # Start: Dark Green
            'F': (0.9, 0.9, 1.0),    # Frozen: Light Blue
            'G': (1.0, 0.9, 0.0),    # Goal: Gold
            'H': (0.2, 0.2, 0.8),    # Hole: Dark Blue (not used but for completeness)
            'unknown': (0.7, 0.7, 0.7),  # Unknown: Gray
            'frontier': (0.3, 0.7, 0.3),  # Frontier: Light Green
            'visited': (0.9, 0.9, 0.9),   # Visited: Very Light Gray
            'current': (1.0, 0.2, 0.2),   # Current: Red
            'path': (1.0, 0.5, 0.0),      # Planned Path: Orange
            'reached_goal': (0.5, 0.0, 0.5)  # Reached Goal: Purple
        }
        
        # Set up the interactive plot
        matplotlib.use('TkAgg')  # Use TkAgg backend for interactivity
        plt.ion()  # Enable interactive mode
        
        # Create figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))
        self.fig.canvas.draw()  # Initial draw
        
        # Keep track of frames for animation
        self.frames = []
        
    def update(self, agent_knowledge, current_node, planned_path=None, 
               total_reward=0, budget=0, step=None, save=True):
        """
        Update the visualization with current state.
        """
        if step is not None:
            self.step = step
        
        # Clear the axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Draw the environment and agent knowledge
        self._draw_environment(self.ax1, current_node)
        self._draw_agent_knowledge(self.ax2, agent_knowledge, current_node, planned_path)
        
        # Set titles
        self.ax1.set_title("Ground Truth Environment", fontsize=14)
        self.ax2.set_title("Agent Knowledge and Plan", fontsize=14)
        
        # Update the title with status information
        self.fig.suptitle(f"Step: {self.step} | Reward: {total_reward:.2f} | Budget: {budget} | " +
                     f"Known: {len(agent_knowledge.graph)} / {len(self.world_graph.graph)} nodes | " +
                     f"Goals: {len(agent_knowledge.collected_rewards)} / {len(agent_knowledge.discovered_goals)}", 
                     fontsize=16)
        
        # Adjust layout and redraw
        plt.tight_layout()
        self.fig.canvas.draw()
        
        # Save the visualization if requested
        if save:
            save_path = f"{self.save_dir}/step_{self.step:04d}.png"
            plt.savefig(save_path)
            
            # Save frame for animation - FIXED APPROACH
            # Instead of trying to reshape the canvas buffer (which is failing),
            # we'll directly use the saved image for our animation
            if os.path.exists(save_path):
                frame = imageio.imread(save_path)
                self.frames.append(frame)
        
        # Update display but don't block
        plt.pause(0.01)
        
        # Increment step counter for next time
        self.step += 1
        
    def _draw_environment(self, ax, current_node=None):
        """Draw the ground truth environment."""
        map_array = self.world_graph.goal_generator.map
        nrow, ncol = len(map_array), len(map_array[0])
        
        # Create a grid for visualization
        grid = np.zeros((nrow, ncol, 3), dtype=float)
        
        # Fill the grid with colors
        for i in range(nrow):
            for j in range(ncol):
                node_id = i * ncol + j
                if map_array[i][j] == b'H':
                    # Holes are dark blue
                    grid[i, j] = self.colors['H']
                else:
                    # For nodes in the graph, use their type color
                    if node_id in self.world_graph.graph:
                        node_type = self.world_graph.graph.nodes[node_id]['type']
                        grid[i, j] = self.colors[node_type]
        
        # Plot the grid
        ax.imshow(grid)
        
        # Add text annotations for all cells
        for i in range(nrow):
            for j in range(ncol):
                node_id = i * ncol + j
                
                # Skip holes
                if map_array[i][j] == b'H':
                    ax.text(j, i, 'H', ha="center", va="center", color="white", fontsize=10)
                    continue
                
                # Basic tile label
                if node_id in self.world_graph.graph:
                    node_info = self.world_graph.graph.nodes[node_id]
                    label = node_info['type']
                    
                    # Add goal ID and reward if applicable
                    if node_info['reward'] > 0:
                        goal_id = node_info['goal_id']
                        reward = node_info['reward']
                        
                        if node_info['is_final']:
                            label = f"{label}\nID:{goal_id}\nFINAL\n{reward:.2f}"
                        else:
                            label = f"{label}\nID:{goal_id}\n{reward:.2f}"
                    
                    ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)
        
        # Highlight current node if provided
        if current_node is not None:
            i, j = self.world_graph.graph.nodes[current_node]['pos']
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=3))
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    def _draw_agent_knowledge(self, ax, agent_knowledge, current_node, planned_path=None):
        """Draw the agent's knowledge and planning."""
        map_array = self.world_graph.goal_generator.map
        nrow, ncol = len(map_array), len(map_array[0])
        
        # Create a grid for visualization
        grid = np.zeros((nrow, ncol, 3), dtype=float)
        
        # Fill with unknown color by default
        grid[:, :] = self.colors['unknown']
        
        # Fill known nodes
        for node in agent_knowledge.graph.nodes:
            i, j = agent_knowledge.graph.nodes[node]['pos']
            
            if node in agent_knowledge.visited:
                # Visited nodes
                grid[i, j] = self.colors['visited']
            elif node in agent_knowledge.frontier:
                # Frontier nodes
                grid[i, j] = self.colors['frontier']
            else:
                # Known but not visited or frontier
                grid[i, j] = self.colors['F']
            
            # Goals are gold
            if agent_knowledge.graph.nodes[node]['type'] == 'G':
                grid[i, j] = self.colors['G']
            
            # Start is green
            if agent_knowledge.graph.nodes[node]['type'] == 'S':
                grid[i, j] = self.colors['S']
        
        # Plot the grid
        ax.imshow(grid)
        
        # Add text annotations for known cells
        for node in agent_knowledge.graph.nodes:
            i, j = agent_knowledge.graph.nodes[node]['pos']
            node_info = agent_knowledge.graph.nodes[node]
            label = node_info['type']
            
            # Add goal ID and reward if applicable
            if node_info['reward'] > 0:
                goal_id = node_info['goal_id']
                reward = node_info['reward']
                
                if node_info['is_final']:
                    label = f"{label}\nID:{goal_id}\nFINAL\n{reward:.2f}"
                else:
                    label = f"{label}\nID:{goal_id}\n{reward:.2f}"
            
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)
        
        # Highlight the current node
        if current_node is not None:
            i, j = agent_knowledge.graph.nodes[current_node]['pos']
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=3))
        
        # Highlight collected rewards
        for node in agent_knowledge.collected_rewards:
            i, j = agent_knowledge.graph.nodes[node]['pos']
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor=self.colors['reached_goal'], linewidth=2))
        
        # Draw the planned path
        if planned_path and len(planned_path) > 1:
            xs, ys = [], []
            for node in planned_path:
                if node in agent_knowledge.graph:
                    i, j = agent_knowledge.graph.nodes[node]['pos']
                    ys.append(i)
                    xs.append(j)
            
            # Plot path
            if xs and ys:
                ax.plot(xs, ys, color=self.colors['path'], linewidth=2, marker='o', markersize=8)
                
                # Add arrows to show direction
                for k in range(len(xs) - 1):
                    dx = xs[k+1] - xs[k]
                    dy = ys[k+1] - ys[k]
                    ax.arrow(xs[k], ys[k], dx * 0.6, dy * 0.6, 
                            head_width=0.2, head_length=0.2, fc=self.colors['path'], ec=self.colors['path'])
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    def create_animation(self, episode=None):
        """
        Create an animation from the saved visualizations.
        
        Args:
            episode: Episode number to use in the filename
        """
        if episode is not None:
            self.episode = episode
            
        # Create the animation from collected frames
        if self.frames:
            try:
                # Ensure all frames have the same shape by resizing if necessary
                if len(self.frames) > 0:
                    # Get the shape of the first frame
                    target_shape = self.frames[0].shape
                    
                    # Resize any frames that don't match
                    resized_frames = []
                    for i, frame in enumerate(self.frames):
                        if frame.shape != target_shape:
                            # Resize to match the first frame
                            from skimage.transform import resize
                            frame = resize(frame, target_shape, preserve_range=True).astype(np.uint8)
                        resized_frames.append(frame)
                    
                    # Save the animation with consistent frames
                    animation_path = os.path.join(self.save_dir, f"episode_{self.episode}_animation.gif")
                    imageio.mimsave(animation_path, resized_frames, fps=2)
                    print(f"Animation saved to {animation_path}")
            except Exception as e:
                print(f"Error creating animation: {e}")
                # Fallback: create animation directly from saved image files
                try:
                    image_files = sorted([f for f in os.listdir(self.save_dir) if f.startswith('step_') and f.endswith('.png')])
                    if image_files:
                        frames = [imageio.imread(os.path.join(self.save_dir, f)) for f in image_files]
                        animation_path = os.path.join(self.save_dir, f"episode_{self.episode}_animation.gif")
                        imageio.mimsave(animation_path, frames, fps=2)
                        print(f"Animation saved using image files to {animation_path}")
                except Exception as e2:
                    print(f"Failed to create animation from files: {e2}")
        else:
            print("No frames collected for animation")
        
        # Reset frames for next episode
        self.frames = []
        
    def close(self):
        """Close the interactive plot."""
        plt.close(self.fig)