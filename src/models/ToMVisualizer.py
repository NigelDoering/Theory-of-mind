import sys
import os
import pygame
import numpy as np
import math
from pathlib import Path

# Add the project root to the path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.environment.ucsd_beautiful_campus import UCSDBeautifulCampus

class ToMVisualizer:
    """
    A class to connect the ToM simulation with the beautiful UCSD campus visualization.
    """
    
    def __init__(self, environment, width=1200, height=800):
        """
        Initialize the visualizer with an environment.
        
        Args:
            environment: The ToMEnvironment instance
            width: Width of the visualization window
            height: Height of the visualization window
        """
        self.environment = environment
        self.visualizer = UCSDBeautifulCampus(width=width, height=height)
        
        # Map agents from the environment to the visualizer
        self._sync_agents()
        
    def _sync_agents(self):
        """Add all agents from the environment to the visualizer with visual properties"""
        for agent in self.environment.agents:
            # Create visual properties based on agent characteristics
            observation_radius = getattr(agent, "observation_radius", 10)
            
            # Determine color based on highest reward preference
            color = (0, 0, 255)  # Default blue
            if hasattr(agent, "reward_preferences") and agent.reward_preferences:
                # Find landmark with highest preference
                max_pref = -float('inf')
                max_landmark = None
                
                for landmark, value in agent.reward_preferences.items():
                    if value > max_pref:
                        max_pref = value
                        max_landmark = landmark
                
                # Use a unique color for each agent based on preference
                if max_landmark:
                    # Create a color with some variation
                    import random
                    r = random.randint(100, 255)
                    g = random.randint(100, 255)
                    b = random.randint(100, 255)
                    color = (r, g, b)
        
            # Make sure all required visual properties are included
            visual_properties = {
                "color": color,
                "size": 10,
                "shape": "circle",
                "view_radius_color": (*color, 40),
                "trail": []  # Always initialize the trail as an empty list
            }
            
            self.visualizer.add_agent(agent, visual_properties)
    
    def run(self, steps=1000, step_delay=100):
        """
        Run the visualization for a specified number of steps.
        
        Args:
            steps: Maximum number of steps to run
            step_delay: Delay between steps in milliseconds
        """
        self.visualizer.run_visualization(self.environment, steps, step_delay)
        
    def save_screenshot(self, filename="ucsd_tom_simulation.png"):
        """Save a screenshot of the current visualization state"""
        pygame.image.save(self.visualizer.screen, filename)