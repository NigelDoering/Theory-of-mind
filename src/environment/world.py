import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.simulation.base import Simulation

class World(Simulation):
    """
    World class represents the 2D grid environment.
    
    This class builds on the Simulation parent class by creating and managing a 
    2D grid (using a NumPy array) where each cell represents a position in the world.
    A 0 indicates a free/traversable cell, and a 1 indicates an obstacle.
    
    It also maintains the goal space and starting space.
    """
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        # Create a grid: 0 for free space, 1 for obstacles.
        self.grid = np.zeros((height, width), dtype=int)
        # Initialize goal and starting spaces as empty lists.
        self.goal_space = []       # List of (x, y) positions for potential goals.
        self.starting_space = []   # List of (x, y) positions for agent start locations.
    
    def set_obstacle(self, i, j):
        """
        Place an obstacle at position (i, j) on the grid.
        
        Parameters:
            i (int): Row index.
            j (int): Column index.
        """
        if 0 <= i < self.height and 0 <= j < self.width:
            self.grid[i, j] = 1
    
    def clear_obstacle(self, i, j):
        """
        Remove an obstacle from position (i, j) on the grid.
        
        Parameters:
            i (int): Row index.
            j (int): Column index.
        """
        if 0 <= i < self.height and 0 <= j < self.width:
            self.grid[i, j] = 0
    
    def is_traversable(self, i, j):
        """
        Check if the cell at (i, j) is free (traversable).
        
        Returns:
            bool: True if the cell is free, False if it is an obstacle.
        """
        if 0 <= i < self.height and 0 <= j < self.width:
            return self.grid[i, j] == 0
        return False
    
    def add_goal(self, position):
        """
        Add a new goal to the goal space.
        
        Parameters:
            position (tuple): (x, y) coordinate of the goal.
        """
        self.goal_space.append(position)
    
    def add_starting_position(self, position):
        """
        Add a new starting position to the starting space.
        
        Parameters:
            position (tuple): (x, y) coordinate of the starting location.
        """
        self.starting_space.append(position)

    def display_world(self):
        """
        Visualizes the world as a 2D grid with obstacles, starting positions, and goals.
        
        The grid is displayed with a border, and obstacles are colored.
        Starting positions are marked in green and goal positions in red.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Define a custom colormap:
        # 0 (free space) -> white, 1 (obstacle) -> dark blue.
        cmap_custom = ListedColormap(['white', 'darkblue'])
        
        # Display the grid.
        # We set origin='lower' so that (0,0) is at the bottom-left.
        # Use an interpolation (e.g., "spline16") for a smoother appearance.
        ax.imshow(self.grid, origin='lower', cmap=cmap_custom, interpolation='spline16')
        
        # Overlay starting positions.
        for pos in self.starting_space:
            # pos is assumed to be (x, y)
            ax.plot(pos[0], pos[1], marker='o', markersize=8, color='green', 
                    linestyle='None', label='Start')
        
        # Overlay goal positions.
        for pos in self.goal_space:
            ax.plot(pos[0], pos[1], marker='*', markersize=12, color='red', 
                    linestyle='None', label='Goal')
        
        # Remove duplicate labels in the legend.
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Remove axis ticks for clarity.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("2D World Representation")
        plt.savefig("world.png")
        plt.show()
        