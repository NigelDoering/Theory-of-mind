import matplotlib.pyplot as plt
import numpy as np

def remove_borders_from_plot(ax):
    """
    Remove all borders from scatter plots in the given axes.
    This fixes the white/light grey borders around agents and goals.
    
    Args:
        ax: Matplotlib axes object
    """
    # Find and fix all scatter plots (PathCollection objects)
    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.collections.PathCollection):
            child.set_edgecolor('none')  # Remove edge color
            
    return ax