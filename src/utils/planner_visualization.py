import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

class PlannerVisualizer:
    """
    Utility class for visualizing path planning algorithms and their results.
    Supports both static visualization and animation of the planning process.
    """
    
    def __init__(self, world):
        """
        Initialize the visualizer with a world.
        
        Parameters:
            world: The World object containing the grid.
        """
        self.world = world
        self.fig = None
        self.ax = None
        self.frames = []
        self.cmap = ListedColormap(['white', 'darkblue'])
    
    def plot_grid(self, ax=None):
        """
        Plot the world grid.
        
        Parameters:
            ax: Optional matplotlib axis to plot on.
            
        Returns:
            The matplotlib axis.
        """
        if ax is None:
            if self.ax is None:
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
            ax = self.ax
        
        # Display the grid
        ax.imshow(self.world.grid, origin='lower', cmap=self.cmap, interpolation='spline16')
        
        # Remove axis ticks for clarity
        ax.set_xticks([])
        ax.set_yticks([])
        
        return ax
    
    def plot_path(self, path, start=None, goal=None, ax=None, color='orange', 
                 linewidth=2, markersize=5, title="Path"):
        """
        Plot a path on the world grid.
        
        Parameters:
            path: List of (x, y) tuples representing the path.
            start: Optional tuple (x, y) for the start position.
            goal: Optional tuple (x, y) for the goal position.
            ax: Optional matplotlib axis to plot on.
            color: Color for the path line.
            linewidth: Width of the path line.
            markersize: Size of path markers.
            title: Title for the plot.
            
        Returns:
            The matplotlib axis.
        """
        if ax is None:
            ax = self.plot_grid()
        
        # Plot the path if provided
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, '-o', color=color, linewidth=linewidth, markersize=markersize)
        
        # Show start position
        if start:
            ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        
        # Show goal position
        if goal:
            ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
        
        # Add legend
        if start or goal:
            ax.legend(loc='upper right')
        
        # Set title
        ax.set_title(title)
        
        return ax
    
    def plot_tree(self, nodes, ax=None, node_color='blue', edge_color='green', 
                 node_size=3, title="Tree"):
        """
        Plot a tree of nodes on the world grid.
        
        Parameters:
            nodes: List of nodes representing the tree.
            ax: Optional matplotlib axis to plot on.
            node_color: Color for the nodes.
            edge_color: Color for the edges.
            node_size: Size of node markers.
            title: Title for the plot.
            
        Returns:
            The matplotlib axis.
        """
        if ax is None:
            ax = self.plot_grid()
        
        # Plot nodes
        for node in nodes:
            x, y = node.position
            ax.plot(x, y, 'o', color=node_color, markersize=node_size)
            
            # Plot edge to parent
            if hasattr(node, 'parent') and node.parent is not None:
                px, py = node.parent.position
                ax.plot([x, px], [y, py], '-', color=edge_color, linewidth=0.5)
        
        # Set title
        ax.set_title(title)
        
        return ax
    
    def plot_ellipse(self, start, goal, c_best, ax=None, color='red', alpha=0.3, title=None):
        """
        Plot an elliptical sampling region.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            c_best: Best cost found so far.
            ax: Optional matplotlib axis to plot on.
            color: Color for the ellipse.
            alpha: Transparency of the ellipse.
            title: Title for the plot.
            
        Returns:
            The matplotlib axis.
        """
        if ax is None:
            ax = self.plot_grid()
        
        # Calculate ellipse parameters
        center_x = (start[0] + goal[0]) / 2
        center_y = (start[1] + goal[1]) / 2
        
        # Distance between start and goal
        c_min = np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
        
        # Semi-major axis
        a = c_best / 2
        
        # Semi-minor axis (if possible to calculate)
        if c_best > c_min:
            b = np.sqrt(c_best**2 - c_min**2) / 2
        else:
            b = 0.1  # Small value if c_best is too small
        
        # Angle of the ellipse
        if goal[0] != start[0]:
            angle = np.arctan2(goal[1] - start[1], goal[0] - start[0])
            angle_deg = np.degrees(angle)
        else:
            angle_deg = 90  # Vertical line
        
        # Create and add the ellipse patch
        ellipse = Ellipse(
            xy=(center_x, center_y),
            width=2*a,
            height=2*b,
            angle=angle_deg,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=2
        )
        ax.add_patch(ellipse)
        
        # Set title if provided
        if title:
            ax.set_title(title)
        
        return ax
    
    def visualize_planner(self, planner, start, goal, path=None, title=None):
        """
        Visualize the results of a planner.
        
        Parameters:
            planner: The planner object.
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            path: Optional path result from the planner.
            title: Optional title for the plot.
            
        Returns:
            The matplotlib figure.
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Plot grid
        self.plot_grid(self.ax)
        
        # Plot path
        if path:
            self.plot_path(path, start, goal, self.ax, title=title or f"Path from {planner.__class__.__name__}")
        else:
            # Just plot start and goal
            self.ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
            self.ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
            self.ax.legend(loc='upper right')
            self.ax.set_title(title or f"No path found - {planner.__class__.__name__}")
        
        # Plot the tree if the planner has nodes
        if hasattr(planner, 'nodes') and planner.nodes:
            self.plot_tree(planner.nodes, self.ax)
        
        # Plot ellipse for informed planners if they have c_best
        if hasattr(planner, 'c_best') and planner.c_best != float('inf'):
            self.plot_ellipse(start, goal, planner.c_best, self.ax)
        
        plt.tight_layout()
        return self.fig
    
    def callback_for_animation(self, nodes, path, iteration, message=None):
        """
        Callback function to use with interactive planning methods.
        Captures the state for animation.
        
        Parameters:
            nodes: List of nodes in the current state.
            path: Current best path if found.
            iteration: Current iteration number.
            message: Optional message to display.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot grid
        self.plot_grid(ax)
        
        # Plot tree
        if nodes:
            self.plot_tree(nodes, ax)
        
        # Plot current best path
        if path:
            self.plot_path(path, None, None, ax, color='red', linewidth=2)
        
        # Show message
        title = f"Iteration: {iteration}"
        if message:
            title += f" - {message}"
        ax.set_title(title)
        
        # Capture figure for animation
        self.frames.append((fig, ax))
        
        # Close figure to avoid memory issues
        plt.close(fig)
    
    def create_animation(self, filename=None, fps=5):
        """
        Create an animation from captured frames.
        
        Parameters:
            filename: Optional filename to save the animation (e.g., 'animation.mp4').
                      If None, the animation is just displayed.
            fps: Frames per second for the animation.
            
        Returns:
            The animation object.
        """
        if not self.frames:
            print("No frames to animate!")
            return None
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Function to update the animation
        def update(frame_idx):
            ax.clear()
            frame_fig, frame_ax = self.frames[frame_idx]
            
            # Copy content from the frame's axis
            frame_ax.figure.canvas.draw()
            img = np.array(frame_ax.figure.canvas.renderer.buffer_rgba())
            ax.imshow(img)
            ax.set_axis_off()
            
            return [ax]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.frames), 
            blit=True, repeat=True, interval=1000/fps
        )
        
        # Save animation if filename is provided
        if filename:
            ani.save(filename)
        
        plt.close()
        return ani
    
    def visualize_multiple_planners(self, planners, start, goal, paths=None, titles=None):
        """
        Visualize the results of multiple planners side by side.
        
        Parameters:
            planners: List of planner objects.
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            paths: Optional list of path results.
            titles: Optional list of titles.
            
        Returns:
            The matplotlib figure.
        """
        n_planners = len(planners)
        
        # Determine grid layout
        if n_planners <= 2:
            n_rows, n_cols = 1, n_planners
        else:
            n_cols = min(3, n_planners)
            n_rows = (n_planners + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_planners == 1:
            axes = [axes]  # Make it indexable
        else:
            axes = axes.flatten()
        
        for i, planner in enumerate(planners):
            if i < len(axes):
                ax = axes[i]
                
                # Plot grid
                self.plot_grid(ax)
                
                # Get path
                path = paths[i] if paths and i < len(paths) else None
                
                # Get title
                title = titles[i] if titles and i < len(titles) else f"{planner.__class__.__name__}"
                
                # Plot path
                if path:
                    self.plot_path(path, start, goal, ax, title=title)
                else:
                    # Just plot start and goal
                    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
                    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
                    ax.legend(loc='upper right')
                    ax.set_title(f"No path - {title}")
                
                # Plot the tree if the planner has nodes
                if hasattr(planner, 'nodes') and planner.nodes:
                    self.plot_tree(planner.nodes, ax)
        
        # Hide any unused subplots
        for i in range(n_planners, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig 