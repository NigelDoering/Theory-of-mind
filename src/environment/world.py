import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.simulation.base import Simulation
import pickle
import cv2
from pathlib import Path

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

class UCSDCampus(World):
    """
    A specialized World class that models the UCSD campus layout.
    """
    def __init__(self, width=1000, height=1000, resolution=1.0, load_from_cache=True):
        super().__init__(width=width, height=height)
        self.resolution = resolution
        self.landmark_locations = {}  # Initialize this dictionary
        self.agents = []  # Initialize agents list
        self.walkable_grid = np.ones((height, width), dtype=bool)
        self.buildings = []
        
        # Try to load campus data from cache
        cache_path = Path('resources/campus_data_cache.pkl')
        if load_from_cache and cache_path.exists():
            self.load_campus_data(str(cache_path))
        else:
            self.extract_and_process_campus()
            
    def extract_and_process_campus(self):
        """Extract campus data from PDF and process it to create the world."""
        from src.environment.ucsd_campus_extractor import UCSDCampusExtractor
        
        print("Extracting UCSD campus layout from PDF...")
        extractor = UCSDCampusExtractor('resources/UCSD_Campus_Detailed.pdf')
        campus_data = extractor.extract_campus_layout('resources/campus_data_cache.pkl')
        
        # Process the extracted data
        self._process_buildings(campus_data['buildings'])
        self._process_walkable_areas(campus_data['walkable_areas'])
        self._process_landmarks(campus_data['landmarks'])
        self._process_paths(campus_data['paths'])
        
    def _process_buildings(self, buildings):
        """Process building data to set obstacles in the grid."""
        # Scale factor to convert from PDF coordinates to grid coordinates
        scale_x = self.width / buildings[0]['rect'][2]  # Assuming buildings[0] exists
        scale_y = self.height / buildings[0]['rect'][3]
        
        for building in buildings:
            x, y, w, h = building['rect']
            
            # Scale to our grid size
            grid_x = int(x * scale_x)
            grid_y = int(y * scale_y)
            grid_w = int(w * scale_x)
            grid_h = int(h * scale_y)
            
            # Mark as obstacles in the grid
            for i in range(max(0, grid_y), min(self.height, grid_y + grid_h)):
                for j in range(max(0, grid_x), min(self.width, grid_x + grid_w)):
                    self.set_obstacle(i, j)
                    self.walkable_grid[i, j] = False
                    
            # Store building information
            self.buildings.append({
                'rect': (grid_x, grid_y, grid_w, grid_h),
                'center': (grid_x + grid_w//2, grid_y + grid_h//2)
            })
            
    def _process_walkable_areas(self, walkable_areas):
        """Process walkable area data."""
        # Resize to match our grid
        walkable_resized = cv2.resize(walkable_areas, (self.width, self.height))
        
        # Update walkable grid
        self.walkable_grid = walkable_resized > 0
        
    def _process_landmarks(self, landmarks):
        """Process landmark data."""
        # Scale factor to convert from PDF coordinates to grid coordinates
        if not self.buildings:  # Ensure we have buildings to calculate scale
            return
            
        scale_x = self.width / self.buildings[0]['rect'][2]
        scale_y = self.height / self.buildings[0]['rect'][3]
        
        # Process each landmark
        for name, info in landmarks.items():
            if info['position']:
                # Scale to our grid
                grid_x = int(info['position'][0] * scale_x)
                grid_y = int(info['position'][1] * scale_y)
                
                # Store landmark location
                self.landmark_locations[name] = (grid_x, grid_y)
                
                # Add a description for each landmark
                self.landmarks[name] = {
                    'x': grid_x, 
                    'y': grid_y, 
                    'description': info['description']
                }
                
    def _process_paths(self, paths):
        """Process path data to create walkable paths."""
        # Scale paths to our grid
        scale_x = self.width / 2048  # Assuming PDF width of 2048 (adjust as needed)
        scale_y = self.height / 2048
        
        # Mark paths as walkable
        for (x1, y1), (x2, y2) in paths:
            # Scale to our grid
            grid_x1 = int(x1 * scale_x)
            grid_y1 = int(y1 * scale_y)
            grid_x2 = int(x2 * scale_x)
            grid_y2 = int(y2 * scale_y)
            
            # Draw line on walkable grid
            cv2.line(self.walkable_grid, 
                   (grid_x1, grid_y1), 
                   (grid_x2, grid_y2), 
                   1, thickness=3)
            
            # Also ensure obstacles are not set on this path
            points = self._get_points_on_line(grid_x1, grid_y1, grid_x2, grid_y2)
            for x, y in points:
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Remove obstacle if it exists
                    self.grid[y, x] = 0
                    
    def _get_points_on_line(self, x1, y1, x2, y2):
        """Get all points on a line from (x1,y1) to (x2,y2) using Bresenham's algorithm."""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while x1 != x2 or y1 != y2:
            points.append((x1, y1))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
        points.append((x2, y2))  # Add the last point
        return points
        
    def load_campus_data(self, file_path):
        """Load campus data from a pickle file."""
        try:
            with open(file_path, 'rb') as f:
                campus_data = pickle.load(f)
                
            # Process the loaded data
            self._process_buildings(campus_data['buildings'])
            self._process_walkable_areas(campus_data['walkable_areas'])
            self._process_landmarks(campus_data['landmarks'])
            self._process_paths(campus_data['paths'])
            
            print(f"Loaded campus data from {file_path}")
            
        except Exception as e:
            print(f"Error loading campus data: {e}")
            print("Falling back to manual campus generation...")
            self.initialize_manual_campus()
                
    def initialize_manual_campus(self):
        """Manually initialize campus layout if data loading fails."""
        print("Creating manual campus layout...")
        self.landmark_locations = {}
        
        # Define key locations on campus (precise coordinates based on UCSD map)
        self.landmarks = {
            "Geisel_Library": {"x": 500, "y": 250, "description": "Main library"},
            "Price_Center": {"x": 550, "y": 300, "description": "Student center"},
            "RIMAC": {"x": 300, "y": 180, "description": "Sports facility"},
            "CSE_Building": {"x": 650, "y": 220, "description": "Computer Science dept"},
            "Cognitive_Science_Building": {"x": 600, "y": 240, "description": "Cognitive Science dept"},
            "Warren_College": {"x": 480, "y": 350, "description": "Warren College"},
            "Revelle_College": {"x": 420, "y": 280, "description": "Revelle College"},
            "Muir_College": {"x": 460, "y": 310, "description": "Muir College"},
            "Marshall_College": {"x": 530, "y": 340, "description": "Marshall College"},
            "ERC_College": {"x": 570, "y": 370, "description": "Eleanor Roosevelt College"},
            "Sixth_College": {"x": 620, "y": 280, "description": "Sixth College"},
            "Seventh_College": {"x": 350, "y": 400, "description": "Seventh College"},
            "Mandeville_Center": {"x": 480, "y": 220, "description": "Mandeville Center"},
            "Bioengineering_Building": {"x": 600, "y": 180, "description": "Bioengineering Building"},
            "Hopkins_Parking": {"x": 400, "y": 350, "description": "Hopkins Parking Structure"},
            "Library_Walk": {"x": 520, "y": 280, "description": "Library Walk"},
            "La_Jolla_Shores": {"x": 150, "y": 500, "description": "Beach area"}
        }
        
        # Mark landmark locations in the grid
        for name, info in self.landmarks.items():
            self.landmark_locations[name] = (info["x"], info["y"])
            
            # Create building shapes around landmarks
            self._create_building_shape(info["x"], info["y"], name)
            
        # Create paths between landmarks
        self._create_campus_paths()
            
    def _create_building_shape(self, x, y, name):
        """Create a building shape in the grid around the given coordinate."""
        # Building size depends on the landmark type
        if "Library" in name:
            width, height = 60, 60
        elif "College" in name:
            width, height = 80, 80
        else:
            width, height = 40, 40
            
        # Mark as obstacles in the grid
        for i in range(max(0, y - height//2), min(self.height, y + height//2)):
            for j in range(max(0, x - width//2), min(self.width, x + width//2)):
                self.set_obstacle(i, j)
                self.walkable_grid[i, j] = False
                
    def _create_campus_paths(self):
        """Create paths between landmarks."""
        # Example paths (connect key landmarks)
        paths = [
            (self.landmark_locations["Geisel_Library"], self.landmark_locations["Price_Center"]),
            (self.landmark_locations["Price_Center"], self.landmark_locations["RIMAC"]),
            (self.landmark_locations["RIMAC"], self.landmark_locations["CSE_Building"]),
            (self.landmark_locations["CSE_Building"], self.landmark_locations["Cognitive_Science_Building"]),
            (self.landmark_locations["Cognitive_Science_Building"], self.landmark_locations["Warren_College"]),
            (self.landmark_locations["Warren_College"], self.landmark_locations["Revelle_College"]),
            (self.landmark_locations["Revelle_College"], self.landmark_locations["Muir_College"]),
            (self.landmark_locations["Muir_College"], self.landmark_locations["Marshall_College"]),
            (self.landmark_locations["Marshall_College"], self.landmark_locations["ERC_College"]),
            (self.landmark_locations["ERC_College"], self.landmark_locations["Sixth_College"]),
            (self.landmark_locations["Sixth_College"], self.landmark_locations["Seventh_College"]),
            (self.landmark_locations["Seventh_College"], self.landmark_locations["Mandeville_Center"]),
            (self.landmark_locations["Mandeville_Center"], self.landmark_locations["Bioengineering_Building"]),
            (self.landmark_locations["Bioengineering_Building"], self.landmark_locations["Hopkins_Parking"]),
            (self.landmark_locations["Hopkins_Parking"], self.landmark_locations["Library_Walk"]),
            (self.landmark_locations["Library_Walk"], self.landmark_locations["La_Jolla_Shores"])
        ]
        
        # Mark paths as walkable
        for (start, end) in paths:
            self._create_path(start, end)
            
    def _create_path(self, start, end):
        """Create a path between two points."""
        points = self._get_points_on_line(start[0], start[1], end[0], end[1])
        for x, y in points:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 0
                self.walkable_grid[y, x] = True
