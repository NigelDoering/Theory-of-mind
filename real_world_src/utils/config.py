import matplotlib.pyplot as plt
import numpy as np

# Visual configuration for different agent species
VISUAL_CONFIG = {
    # Color palettes for different agent species (lighter to darker shades)
    "color_palettes": {
        "shortest": ["#00ff00", "#00cc00", "#009900", "#006600", "#003300"],  # Green palette
        "random": ["#1e90ff", "#0000ff", "#0000cd", "#00008b", "#191970"],    # Blue palette
        "social": ["#ff9999", "#ff6666", "#ff3333", "#ff0000", "#cc0000"],    # Red palette
        "landmark": ["#ffff99", "#ffff66", "#ffff33", "#ffff00", "#cccc00"],  # Yellow palette
        "explorer": ["#ffa500", "#ff8c00", "#ff7f50", "#ff6347", "#ff4500"],  # Orange palette
        "obstacle": ["#8b008b", "#9932cc", "#9400d3", "#800080", "#4b0082"],  # Purple palette
        "scared": ["#9370db", "#7b68ee", "#6a5acd", "#483d8b", "#4169e1"],    # Lavender palette
        "risky": ["#ff4500", "#ff6347", "#ff7f50", "#ff8c00", "#ffa500"],     # OrangeRed palette
    },
    
    # Marker styles
    "markers": {
        "start": "s",       # Square for start
        "goal": "*",        # Star for goal
        "position": "o",    # Circle for current position
        "landmark": "D",    # Diamond for landmarks
    },
    
    # Marker sizes
    "sizes": {
        "start": 100,
        "goal": 200,
        "position": 120,
        "landmark": 80,
        "path_line": 2.5,
    },
    
    # Other styling
    "edge_color": "#444444",
    "node_color": "#aaaaaa",
    "building_color": "lightgrey", 
    "building_edge": "dimgrey",
    "landmark_color": "#ff0000",
    "landmark_alpha": 0.8,
    "path_alpha": 0.8,
    "grid_alpha": 0.4,
    
    # Legend properties
    "legend_loc": "upper right", 
    "legend_fontsize": 10,
}

def get_agent_color(species, index=0):
    """Get a color for an agent based on its species and index"""
    palette = VISUAL_CONFIG["color_palettes"].get(species.lower(), ["#000000"])
    return palette[index % len(palette)]

def get_agent_colors(species_counts):
    """
    Generate a set of colors for a group of agents.
    
    Args:
        species_counts: Dictionary mapping species names to counts
        
    Returns:
        Dictionary mapping species and index to color
    """
    colors = {}
    for species, count in species_counts.items():
        palette = VISUAL_CONFIG["color_palettes"].get(species.lower(), ["#000000"])
        for i in range(count):
            colors[f"{species}_{i}"] = palette[i % len(palette)]
    return colors