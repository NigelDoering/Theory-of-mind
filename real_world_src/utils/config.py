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
        "obstacle": ["#9932cc", "#8b008b", "#800080", "#4b0082", "#483d8b"],  # Purple palette
        "scared": ["#d8bfd8", "#dda0dd", "#ee82ee", "#da70d6", "#ff00ff"],    # Lavender palette
        "risky": ["#ff7f50", "#ff6347", "#ff4500", "#ff0000", "#8b0000"]      # OrangeRed palette
    },
    
    # Markers for different elements
    "markers": {
        "start": "o",
        "goal": "s",
        "position": "D",
        "landmark": "*"
    },
    
    # Sizes for different elements
    "sizes": {
        "start": 100,
        "goal": 100,
        "position": 150,
        "landmark": 120,
        "path_line": 2.5,
    },
    
    # Other styling
    "edge_color": "none",  # Set to none at root level to remove borders
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
    species = species.lower()
    if species in VISUAL_CONFIG["color_palettes"]:
        palette = VISUAL_CONFIG["color_palettes"][species]
        color_index = index % len(palette)
        return palette[color_index]
    else:
        # Default color
        return "#cccccc"

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