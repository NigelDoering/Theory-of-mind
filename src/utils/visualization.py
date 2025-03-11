import matplotlib.pyplot as plt
import numpy as np

def plot_path(world, path, agent_start=None, agent_goal=None, title="Path in World"):
    """
    Visualize a path in the world grid.
    
    Parameters:
        world: The World object containing the grid.
        path: List of (x, y) tuples representing the path.
        agent_start: Optional tuple (x, y) for the agent's start position.
        agent_goal: Optional tuple (x, y) for the agent's goal position.
        title: Title for the plot.
    """
    from matplotlib.colors import ListedColormap
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define a custom colormap for the grid
    cmap_custom = ListedColormap(['white', 'darkblue'])
    
    # Display the grid
    ax.imshow(world.grid, origin='lower', cmap=cmap_custom, interpolation='spline16')
    
    # Plot the path if provided
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, '-o', color='orange', linewidth=2, markersize=5)
    
    # Show start position
    if agent_start:
        ax.plot(agent_start[0], agent_start[1], 'go', markersize=10, label='Start')
    
    # Show goal position
    if agent_goal:
        ax.plot(agent_goal[0], agent_goal[1], 'r*', markersize=15, label='Goal')
    
    # Add legend
    if agent_start or agent_goal:
        ax.legend(loc='upper right')
    
    # Set title and hide ticks
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show() 