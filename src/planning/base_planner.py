from abc import ABC, abstractmethod

class BasePlanner(ABC):
    """
    Abstract base class for all path planning algorithms.
    
    This interface ensures that all planning algorithms have a consistent API,
    making them interchangeable in the simulation.
    """
    
    def __init__(self, world):
        """
        Initialize the planner with a world object.
        
        Parameters:
            world: The World object containing the grid and obstacles.
        """
        self.world = world
    
    @abstractmethod
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal.
        
        Parameters:
            start: Tuple (x, y) representing the start position.
            goal: Tuple (x, y) representing the goal position.
            **kwargs: Additional algorithm-specific parameters.
            
        Returns:
            A list of (x, y) tuples representing the path from start to goal,
            or None if no path is found.
        """
        pass
    
    def is_valid_point(self, point):
        """
        Check if a point is valid (within bounds and not an obstacle).
        
        Parameters:
            point: Tuple (x, y) representing a position.
            
        Returns:
            Boolean indicating if the point is valid.
        """
        x, y = point
        if 0 <= x < self.world.width and 0 <= y < self.world.height:
            return self.world.is_traversable(y, x)  # Note: world.is_traversable takes (row, col)
        return False 