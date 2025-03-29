import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib import cm, colors
import networkx as nx
import numpy as np
from .config import VISUAL_CONFIG
import os
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Create a custom dark background style
plt.style.use('dark_background')
sns.set_context("notebook", font_scale=1.2)

cool_cmap = LinearSegmentedColormap.from_list('cool_cmap', 
                                              ['#2980b9', '#3498db', '#9b59b6', '#8e44ad'], N=100)
warm_cmap = LinearSegmentedColormap.from_list('warm_cmap', 
                                             ['#e74c3c', '#c0392b', '#d35400', '#e67e22'], N=100)

def plot_agent_paths(environment, agents, title="Agent Paths", show_buildings=True, figsize=(15, 15)):
    """Plot the paths taken by agents on the campus map."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Disable grid
    ax.grid(False)
    
    # Plot the graph
    edges = list(environment.G.edges())
    edge_coords = []
    for u, v in edges:
        u_x, u_y = environment.get_node_coordinates(u)
        v_x, v_y = environment.get_node_coordinates(v)
        edge_coords.append([(u_x, u_y), (v_x, v_y)])
    
    # Plot edges
    for edge in edge_coords:
        (x1, y1), (x2, y2) = edge
        ax.plot([x1, x2], [y1, y2], color=VISUAL_CONFIG["edge_color"], 
                alpha=VISUAL_CONFIG["grid_alpha"], linewidth=0.5)
    
    # Plot buildings if available
    if environment.buildings is not None and show_buildings:
        environment.buildings.plot(ax=ax, color=VISUAL_CONFIG["building_color"], 
                                alpha=0.5, edgecolor=VISUAL_CONFIG["building_edge"])
    
    # Plot agent paths
    for agent in agents:
        path_x, path_y = agent.get_path_coordinates()
        if path_x and path_y:
            ax.plot(path_x, path_y, color=agent.color, 
                   alpha=VISUAL_CONFIG["path_alpha"], 
                   linewidth=VISUAL_CONFIG["sizes"]["path_line"], 
                   label=f"{agent.species}")
            
            # Plot start and goal
            start_x, start_y = agent.get_position(agent.start_node)
            goal_x, goal_y = agent.get_position(agent.goal_node)
            ax.scatter([start_x], [start_y], color=agent.color, 
                      marker=VISUAL_CONFIG["markers"]["start"], 
                      s=VISUAL_CONFIG["sizes"]["start"], zorder=10)
            ax.scatter([goal_x], [goal_y], color=agent.color, 
                      marker=VISUAL_CONFIG["markers"]["goal"], 
                      s=VISUAL_CONFIG["sizes"]["goal"], zorder=10)
    
    # Plot landmarks
    landmark_x = [environment.node_coords[node][0] for node in environment.landmarks]
    landmark_y = [environment.node_coords[node][1] for node in environment.landmarks]
    ax.scatter(landmark_x, landmark_y, c=VISUAL_CONFIG["landmark_color"], 
               s=VISUAL_CONFIG["sizes"]["landmark"], 
               alpha=VISUAL_CONFIG["landmark_alpha"], 
               marker=VISUAL_CONFIG["markers"]["landmark"], 
               zorder=5, label="Landmarks")
    
    # Add legend (only include one entry per agent species)
    handles, labels = ax.get_legend_handles_labels()
    
    # Create additional entries for start and goal markers
    if agents:
        for species in set(agent.species for agent in agents):
            # Find an agent of this species
            agent = next((a for a in agents if a.species == species), None)
            if agent:
                # Add start marker to legend
                start_marker = plt.Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["start"], 
                                          color=agent.color, linestyle='', 
                                          markersize=10, label=f"{species} Start")
                # Add goal marker to legend
                goal_marker = plt.Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["goal"], 
                                         color=agent.color, linestyle='', 
                                         markersize=10, label=f"{species} Goal")
                handles.extend([start_marker, goal_marker])
                labels.extend([f"{species} Start", f"{species} Goal"])
    
    # Create a legend with unique entries
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
              loc=VISUAL_CONFIG["legend_loc"], 
              fontsize=VISUAL_CONFIG["legend_fontsize"])
    
    plt.title(title)
    plt.tight_layout()
    
    return fig, ax

def visualize_agent_decision(environment, agent, title="Agent Decision Process"):
    """Visualize the decision-making process of an agent."""
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Disable grid
    ax.grid(False)
    
    # Show the environment background
    edges = list(environment.G.edges())
    edge_coords = []
    for u, v in edges:
        u_x, u_y = environment.get_node_coordinates(u)
        v_x, v_y = environment.get_node_coordinates(v)
        edge_coords.append([(u_x, u_y), (v_x, v_y)])
    
    # Plot edges
    for edge in edge_coords:
        (x1, y1), (x2, y2) = edge
        ax.plot([x1, x2], [y1, y2], color=VISUAL_CONFIG["edge_color"], 
                alpha=VISUAL_CONFIG["grid_alpha"], linewidth=0.5)
    
    # Plot agent's current position
    current_x, current_y = agent.get_position()
    ax.scatter([current_x], [current_y], color=agent.color, 
               s=VISUAL_CONFIG["sizes"]["position"], zorder=10, 
               marker=VISUAL_CONFIG["markers"]["position"], 
               label="Current Position")
    
    # Plot goal
    goal_x, goal_y = agent.get_position(agent.goal_node)
    ax.scatter([goal_x], [goal_y], color=agent.color, 
               marker=VISUAL_CONFIG["markers"]["goal"], 
               s=VISUAL_CONFIG["sizes"]["goal"], zorder=10, label="Goal")
    
    # Plot start
    start_x, start_y = agent.get_position(agent.start_node)
    ax.scatter([start_x], [start_y], color=agent.color, 
               marker=VISUAL_CONFIG["markers"]["start"], 
               s=VISUAL_CONFIG["sizes"]["start"], zorder=10, label="Start")
    
    # Plot the planned path
    if len(agent.path) > 0:
        path_coords = [environment.get_node_coordinates(node) for node in agent.path]
        path_x = [coord[0] for coord in path_coords]
        path_y = [coord[1] for coord in path_coords]
        ax.plot(path_x, path_y, color=agent.color, 
                alpha=VISUAL_CONFIG["path_alpha"], 
                linewidth=VISUAL_CONFIG["sizes"]["path_line"], 
                linestyle='--', label="Planned Path")
    
    # If it's a landmark agent, also show landmarks
    if agent.species == "Landmark" and agent.current_landmark:
        landmark_x, landmark_y = environment.get_node_coordinates(agent.current_landmark)
        ax.scatter([landmark_x], [landmark_y], color='green', 
                   marker=VISUAL_CONFIG["markers"]["landmark"], 
                   s=VISUAL_CONFIG["sizes"]["landmark"], zorder=9, 
                   label="Current Landmark")
    
    # If it's a social agent, show the follow target
    if agent.species == "Social" and agent.follow_target:
        target_x, target_y = agent.follow_target.get_position()
        ax.scatter([target_x], [target_y], color='purple', 
                   marker=VISUAL_CONFIG["markers"]["position"], 
                   s=VISUAL_CONFIG["sizes"]["position"], zorder=9, 
                   label="Follow Target")
    
    plt.title(title)
    ax.legend(loc=VISUAL_CONFIG["legend_loc"], 
              fontsize=VISUAL_CONFIG["legend_fontsize"])
    plt.tight_layout()
    
    return fig, ax

def plot_species_grid(environment, agents, title="Species Comparison", save_path=None):
    """
    Plot each species' trajectories in a 4x2 grid layout.
    
    Args:
        environment: The CampusEnvironment
        agents: List of agents to visualize
        title: Main title for the plot
        save_path: Optional path to save the visualization
    """
    # Group agents by species
    species_groups = {}
    for agent in agents:
        species = agent.species
        if species not in species_groups:
            species_groups[species] = []
        species_groups[species].append(agent)
    
    # Create a 4x2 grid of subplots
    fig, axs = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle(title, fontsize=16)
    
    # Flatten the axs array for easier indexing
    axs = axs.flatten()
    
    # Get all species sorted alphabetically
    all_species = sorted(species_groups.keys())
    
    # Plot each species in its own subplot
    for i, species in enumerate(all_species):
        if i >= 8:  # Only plot up to 8 species (4x2 grid)
            break
            
        ax = axs[i]
        species_agents = species_groups[species]
        
        # Disable grid
        ax.grid(False)
        
        # Plot the graph background
        edges = list(environment.G.edges())
        edge_coords = []
        for u, v in edges:
            u_x, u_y = environment.get_node_coordinates(u)
            v_x, v_y = environment.get_node_coordinates(v)
            edge_coords.append([(u_x, u_y), (v_x, v_y)])
        
        # Plot edges
        for edge in edge_coords:
            (x1, y1), (x2, y2) = edge
            ax.plot([x1, x2], [y1, y2], color=VISUAL_CONFIG["edge_color"], 
                   alpha=VISUAL_CONFIG["grid_alpha"], linewidth=0.3)
        
        # Plot agent paths for this species
        for agent in species_agents:
            path_x, path_y = agent.get_path_coordinates()
            if path_x and path_y:
                ax.plot(path_x, path_y, color=agent.color, 
                       alpha=VISUAL_CONFIG["path_alpha"]+0.2,  # Make paths more visible
                       linewidth=VISUAL_CONFIG["sizes"]["path_line"]*1.2, 
                       label=f"{agent.id}")
                
                # Plot start and goal with enhanced visibility
                start_x, start_y = agent.get_position(agent.start_node)
                goal_x, goal_y = agent.get_position(agent.goal_node)
                
                ax.scatter([start_x], [start_y], color=agent.color, 
                          marker=VISUAL_CONFIG["markers"]["start"], 
                          s=VISUAL_CONFIG["sizes"]["start"]*1.2,
                          edgecolor='black', linewidth=1.0, zorder=15)
                          
                ax.scatter([goal_x], [goal_y], color=agent.color, 
                          marker=VISUAL_CONFIG["markers"]["goal"], 
                          s=VISUAL_CONFIG["sizes"]["goal"]*1.2,
                          edgecolor='black', linewidth=1.0, zorder=15)
                          
        # Customize subplot
        ax.set_title(f"{species} Agents", fontsize=14)
        ax.tick_params(left=False, right=False, labelleft=False, 
                      labelbottom=False, bottom=False, top=False)
        
        # Create legend outside plot
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Add start/goal markers to legend
            start_marker = plt.Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["start"], 
                                     color=species_agents[0].color, linestyle='',
                                     markeredgecolor='black', markersize=10, label="Start")
            goal_marker = plt.Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["goal"], 
                                    color=species_agents[0].color, linestyle='',
                                    markeredgecolor='black', markersize=10, label="Goal")
            handles.extend([start_marker, goal_marker])
            labels.extend(["Start", "Goal"])
            
            # Place legend outside plot
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1), 
                     fontsize=9, framealpha=0.9)
    
    # Hide any unused subplots
    for i in range(len(all_species), 8):
        axs[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Make room for legends
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axs

def animate_species_grid(environment, agents, title="Species Navigation Patterns", 
                        max_frames=300, interval=100, save_path=None, dpi=100):
    """
    Create an animated grid layout showing all species navigating simultaneously.
    """
    # Import required libraries first to check availability
    try:
        import imageio
        HAS_IMAGEIO = True
        print("Using imageio for high-quality animation generation")
    except ImportError:
        HAS_IMAGEIO = False
        print("Warning: imageio not found. Install with 'pip install imageio' for better animations")
    
    # Group agents by species
    species_groups = {}
    for agent in agents:
        species = agent.species
        if species not in species_groups:
            species_groups[species] = []
        species_groups[species].append(agent)
    
    # Calculate grid dimensions based on number of species
    n_species = len(species_groups)
    if n_species <= 3:
        n_cols, n_rows = n_species, 1
    elif n_species <= 6:
        n_cols, n_rows = 3, 2
    elif n_species <= 9:
        n_cols, n_rows = 3, 3
    else:
        n_cols, n_rows = 4, 3  # Maximum 12 species
    
    print(f"Creating grid animation with {n_species} species in a {n_cols}×{n_rows} grid layout")
    
    # Set up a larger figure for better detail
    fig = plt.figure(figsize=(n_cols*5, n_rows*5))
    
    # Use GridSpec for better control over subplot spacing
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.2, hspace=0.3)
    
    # Add an overall title with styling
    fig.suptitle(title, fontsize=20, fontweight='bold', 
                y=0.98, color='white', fontfamily='sans-serif')
    
    # Create a timestamp display
    timestamp_text = fig.text(0.5, 0.01, "Step: 0", ha='center', 
                            fontsize=14, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, 
                                     boxstyle='round,pad=0.5', 
                                     edgecolor='gray'))
    
    # Create subplot axes for each species
    axs = []
    species_names = sorted(species_groups.keys())
    
    # Get the edge coordinates for background (common to all subplots)
    edge_coords_collection = []
    for u, v in environment.G.edges():
        u_x, u_y = environment.get_node_coordinates(u)
        v_x, v_y = environment.get_node_coordinates(v)
        edge_coords_collection.append([(u_x, u_y), (v_x, v_y)])
    
    # Set up subplots for each species
    for i, species in enumerate(species_names[:n_rows*n_cols]):  # Limit to grid size
        row, col = i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Disable grid
        ax.grid(False)
        
        # Plot edges with semi-transparent styling
        for edge in edge_coords_collection:
            (x1, y1), (x2, y2) = edge
            ax.plot([x1, x2], [y1, y2], color=VISUAL_CONFIG["edge_color"], 
                   alpha=0.3, linewidth=0.4, zorder=1)
        
        # Add buildings if available, with a nicer semi-transparent style
        if environment.buildings is not None:
            environment.buildings.plot(ax=ax, color=VISUAL_CONFIG["building_color"], 
                                     alpha=0.3, edgecolor=VISUAL_CONFIG["building_edge"])
        
        # Customize subplot appearance
        ax.set_title(f"{species} Agents", fontsize=16, fontweight='bold', 
                    pad=10, color='white')
        ax.set_aspect('equal')
        ax.axis('off')  # Hide axes for cleaner look
        
        # Plot landmarks with a more attractive style
        landmark_x = [environment.node_coords[node][0] for node in environment.landmarks]
        landmark_y = [environment.node_coords[node][1] for node in environment.landmarks]
        ax.scatter(landmark_x, landmark_y, c=VISUAL_CONFIG["landmark_color"], 
                 s=VISUAL_CONFIG["sizes"]["landmark"], 
                 alpha=0.8, marker=VISUAL_CONFIG["markers"]["landmark"], 
                 edgecolor='white', linewidth=0.8, zorder=5)
        
        # Initialize agent markers, path lines, and labels
        species_agents = species_groups[species]
        agent_scatters = []
        agent_paths = []
        goal_markers = []
        start_markers = []
        
        # Add goal markers (fixed throughout animation)
        for agent in species_agents:
            goal_x, goal_y = agent.get_position(agent.goal_node)
            goal = ax.scatter([goal_x], [goal_y], color=agent.color, 
                           marker=VISUAL_CONFIG["markers"]["goal"], 
                           s=VISUAL_CONFIG["sizes"]["goal"]*1.8,
                           edgecolor='white', linewidth=1.0, zorder=8)
            goal_markers.append(goal)
            
            # Add start markers 
            start_x, start_y = agent.get_position(agent.start_node)
            start = ax.scatter([start_x], [start_y], color=agent.color, 
                            marker=VISUAL_CONFIG["markers"]["start"], 
                            s=VISUAL_CONFIG["sizes"]["start"]*1.5,
                            edgecolor='white', linewidth=1.0, zorder=7)
            start_markers.append(start)
            
            # Initialize empty path line with gradient coloring for visual appeal
            path_line, = ax.plot([], [], color=agent.color, 
                              alpha=0.8, linewidth=3.0, zorder=6, 
                              solid_capstyle='round')
            agent_paths.append(path_line)
            
            # Add agent position marker
            agent_pos = ax.scatter([], [], color=agent.color, 
                               s=VISUAL_CONFIG["sizes"]["position"]*1.5,
                               marker=VISUAL_CONFIG["markers"]["position"], 
                               edgecolor='white', linewidth=1.0, zorder=10)
            agent_scatters.append(agent_pos)
        
        # Store all plot elements
        axs.append({
            'ax': ax, 
            'agents': species_agents,
            'scatters': agent_scatters,
            'paths': agent_paths,
            'goals': goal_markers,
            'starts': start_markers
        })
    
    # Create a custom legend below the plot
    legend_elements = []
    
    # Add a legend element for each species
    for species in species_names:
        # Get a representative agent color
        color = species_groups[species][0].color
        
        # Add species marker to legend
        legend_elements.append(Line2D([0], [0], color=color, linewidth=3, 
                                    label=f"{species}"))
    
    # Add path and markers explanation
    legend_elements.extend([
        Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["start"], color='gray', 
              linestyle='', markersize=8, markerfacecolor='gray',
              markeredgecolor='white', label="Start"),
        Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["goal"], color='gray', 
              linestyle='', markersize=10, markerfacecolor='gray',
              markeredgecolor='white', label="Goal"),
        Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["position"], color='gray', 
              linestyle='', markersize=8, markerfacecolor='gray',
              markeredgecolor='white', label="Current Position"),
        Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["landmark"], color=VISUAL_CONFIG["landmark_color"], 
              linestyle='', markersize=8, markerfacecolor=VISUAL_CONFIG["landmark_color"],
              markeredgecolor='white', label="Landmark")
    ])
    
    # Place the legend below the subplots
    legend = fig.legend(handles=legend_elements, loc='lower center', 
                      bbox_to_anchor=(0.5, 0.02),
                      ncol=min(6, len(legend_elements)), fontsize=12,
                      framealpha=0.8, edgecolor='gray')
    
    if save_path:
        # Manual frame-by-frame generation and saving
        print(f"\nGenerating and saving grid animation to {save_path}...")
        print("This might take a while for complex animations.")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                   exist_ok=True)
        
        # Reset all agents before starting
        for agent in agents:
            agent.reset()
        
        # Use fewer frames for complex grid animations to avoid memory issues
        actual_max_frames = min(max_frames, 60)  # Limit to 60 frames for grid animation
        
        # Create a list to store frames
        frames = []
        
        try:
            # Generate frames with explicit progress reporting
            print(f"Generating {actual_max_frames} frames:")
            print("[", end="", flush=True)
            
            for frame in range(actual_max_frames):
                # Print progress
                if frame % (actual_max_frames // 20) == 0:
                    print("#", end="", flush=True)
                
                # Step the simulation (except for frame 0)
                if frame > 0:
                    environment.step()
                    
                # Update all subplots
                for subplot in axs:
                    for i, agent in enumerate(subplot['agents']):
                        # Update agent positions
                        pos = agent.get_position()
                        subplot['scatters'][i].set_offsets([pos])
                        
                        # Update path lines
                        path_x, path_y = agent.get_path_coordinates()
                        subplot['paths'][i].set_data(path_x, path_y)
                
                # Update timestamp
                timestamp_text.set_text(f"Step: {frame}")
                
                # Draw the figure
                fig.canvas.draw()
                
                # Convert to image array
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(img)
                
                # Check if all agents have reached their goals
                if environment.all_agents_done():
                    print(f"\nAll agents reached their goals at step {frame}")
                    break
            
            print("] Done!")
            
            # Save frames as GIF
            if HAS_IMAGEIO and frames:
                print(f"Saving {len(frames)} frames as GIF...")
                imageio.mimsave(save_path, frames, fps=10, loop=0)
                print(f"✓ Animation saved successfully to {save_path}")
            else:
                # Use PIL as fallback
                try:
                    from PIL import Image
                    print("Using PIL for GIF creation...")
                    images = [Image.fromarray(frame) for frame in frames]
                    images[0].save(
                        save_path,
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=interval,
                        loop=0
                    )
                    print(f"✓ Animation saved successfully to {save_path}")
                except Exception as e:
                    print(f"Error saving animation with PIL: {e}")
                    # Last resort: save at least one frame
                    fig.savefig(save_path.replace('.gif', '_static.png'), dpi=dpi)
                    print(f"✓ Saved static visualization to {save_path.replace('.gif', '_static.png')}")
            
        except Exception as e:
            print(f"\nError during animation generation: {e}")
            # Save a static image as fallback
            fig.savefig(save_path.replace('.gif', '_static.png'), dpi=dpi)
            print(f"✓ Saved static visualization to {save_path.replace('.gif', '_static.png')}")
    
    # For interactive display, return a simple animation
    anim = animation.FuncAnimation(
        fig, lambda frame: None, frames=1, 
        interval=interval, blit=False
    )
    
    return anim

def animate_single_species(environment, agents, title="Species Navigation", 
                         max_frames=300, interval=100, save_path=None, 
                         dpi=100, enhanced=False):
    """
    Create an animated visualization of a single species.
    
    Args:
        environment: The CampusEnvironment
        agents: List of agents of the same species
        title: Title for the animation
        max_frames: Maximum number of frames
        interval: Time between frames in milliseconds
        save_path: Path to save the animation
        dpi: Resolution for the saved animation
        enhanced: Whether to apply enhanced visual styling
        
    Returns:
        The animation object
    """
    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Disable grid
    ax.grid(False)
    
    # Add a more attractive title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98, 
                color='#333333', fontfamily='sans-serif')
    
    # Get the species name from the first agent
    species = agents[0].species if agents else "Unknown"
    
    # Add a subtitle with more information
    subtitle_text = f"Tracking {len(agents)} {species} agents navigating UCSD campus"
    ax.set_title(subtitle_text, fontsize=14, pad=10, color='#555555')
    
    # Plot the background map with enhanced styling if requested
    if enhanced:
        # Plot edges with gradient color based on distance from center
        edges = list(environment.G.edges())
        edge_coords = []
        edge_colors = []
        
        # Calculate the center of the map
        center_x = np.mean([environment.node_coords[node][0] for node in environment.nodes])
        center_y = np.mean([environment.node_coords[node][1] for node in environment.nodes])
        
        for u, v in edges:
            u_x, u_y = environment.get_node_coordinates(u)
            v_x, v_y = environment.get_node_coordinates(v)
            edge_coords.append([(u_x, u_y), (v_x, v_y)])
            
            # Calculate distance from center
            edge_center_x = (u_x + v_x) / 2
            edge_center_y = (u_y + v_y) / 2
            dist = np.sqrt((edge_center_x - center_x)**2 + (edge_center_y - center_y)**2)
            
            # Normalize the distance to [0, 1]
            max_dist = np.sqrt(center_x**2 + center_y**2)
            norm_dist = min(dist / max_dist, 1.0)
            
            # Create a color based on distance (farther = lighter)
            edge_colors.append((0.5, 0.5, 0.5, 0.3 + 0.5 * (1 - norm_dist)))
        
        # Plot edges with varying colors
        edge_collection = LineCollection(edge_coords, colors=edge_colors, linewidths=0.8, zorder=1)
        ax.add_collection(edge_collection)
        
        # Plot buildings with a nicer style
        if environment.buildings is not None:
            environment.buildings.plot(ax=ax, color='#f0f0f0', 
                                     alpha=0.4, edgecolor='#cccccc', linewidth=0.5)
    else:
        # Standard styling
        edges = list(environment.G.edges())
        for u, v in edges:
            u_x, u_y = environment.get_node_coordinates(u)
            v_x, v_y = environment.get_node_coordinates(v)
            ax.plot([u_x, v_x], [u_y, v_y], color=VISUAL_CONFIG["edge_color"], 
                  alpha=0.3, linewidth=0.8, zorder=1)
        
        # Plot buildings if available
        if environment.buildings is not None:
            environment.buildings.plot(ax=ax, color=VISUAL_CONFIG["building_color"], 
                                     alpha=0.3, edgecolor=VISUAL_CONFIG["building_edge"])
    
    # Plot landmarks with enhanced styling
    landmark_x = [environment.node_coords[node][0] for node in environment.landmarks]
    landmark_y = [environment.node_coords[node][1] for node in environment.landmarks]
    
    if enhanced:
        # Create a glow effect for landmarks
        for x, y in zip(landmark_x, landmark_y):
            # Add larger circle with low alpha for glow effect
            ax.scatter([x], [y], 
                      c=VISUAL_CONFIG["landmark_color"], 
                      s=VISUAL_CONFIG["sizes"]["landmark"] * 2, 
                      alpha=0.2, marker='o', zorder=3)
    
    landmark_scatter = ax.scatter(landmark_x, landmark_y, 
                               c=VISUAL_CONFIG["landmark_color"], 
                               s=VISUAL_CONFIG["sizes"]["landmark"], 
                               alpha=0.8, marker=VISUAL_CONFIG["markers"]["landmark"], 
                               edgecolor='white', linewidth=0.8, zorder=5,
                               label="Landmarks")
    
    # Add goal markers (fixed throughout animation)
    for agent in agents:
        goal_x, goal_y = agent.get_position(agent.goal_node)
        ax.scatter([goal_x], [goal_y], color=agent.color, 
                 marker=VISUAL_CONFIG["markers"]["goal"], 
                 s=VISUAL_CONFIG["sizes"]["goal"] * 1.8,
                 edgecolor='white', linewidth=1.0, zorder=8)
        
        # Add start markers
        start_x, start_y = agent.get_position(agent.start_node)
        ax.scatter([start_x], [start_y], color=agent.color, 
                 marker=VISUAL_CONFIG["markers"]["start"], 
                 s=VISUAL_CONFIG["sizes"]["start"] * 1.5,
                 edgecolor='white', linewidth=1.0, zorder=7)
    
    # Initialize agent path lines and position markers
    agent_paths = []
    agent_positions = []
    
    for agent in agents:
        # Initialize empty path line
        path_line, = ax.plot([], [], color=agent.color, 
                           alpha=0.8, linewidth=3.0, zorder=6, 
                           solid_capstyle='round')
        agent_paths.append(path_line)
        
        # Initialize agent marker
        agent_marker = ax.scatter([], [], color=agent.color, 
                               s=VISUAL_CONFIG["sizes"]["position"] * 1.5,
                               marker=VISUAL_CONFIG["markers"]["position"], 
                               edgecolor='white', linewidth=1.0, zorder=10)
        agent_positions.append(agent_marker)
    
    # Add a timestamp display
    timestamp_text = ax.text(0.02, 0.02, "Step: 0", transform=ax.transAxes,
                           fontsize=14, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, 
                                   boxstyle='round,pad=0.5', 
                                   edgecolor='gray'))
    
    # Turn off axis for cleaner look
    ax.axis('off')
    
    # Add a legend with enhanced styling
    legend_elements = [
        Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["start"], color='gray', 
             linestyle='', markersize=8, markerfacecolor='gray',
             markeredgecolor='white', label="Start"),
        Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["goal"], color='gray', 
             linestyle='', markersize=10, markerfacecolor='gray',
             markeredgecolor='white', label="Goal"),
        Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["position"], color='gray', 
             linestyle='', markersize=8, markerfacecolor='gray',
             markeredgecolor='white', label="Current Position"),
        Line2D([0], [0], marker=VISUAL_CONFIG["markers"]["landmark"], 
             color=VISUAL_CONFIG["landmark_color"], 
             linestyle='', markersize=8, markerfacecolor=VISUAL_CONFIG["landmark_color"],
             markeredgecolor='white', label="Landmark"),
        Line2D([0], [0], color=agents[0].color, linewidth=3, label=f"{species} Path")
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper right',
                     fontsize=12, framealpha=0.8, edgecolor='gray')
    
    # Add a completion flag 
    completion_reported = False
    
    # Function to initialize the animation
    def init():
        for path in agent_paths:
            path.set_data([], [])
        for position in agent_positions:
            position.set_offsets(np.empty((0, 2)))
        timestamp_text.set_text("Step: 0")
        return agent_paths + agent_positions + [timestamp_text]
    
    # Function to update each frame of the animation
    def update(frame):
        nonlocal completion_reported
        
        # Reset all agents to their initial state
        if frame == 0:
            for agent in agents:
                agent.reset()
            completion_reported = False  # Reset the flag
            return init()
        
        # Otherwise, step the simulation forward
        for agent in agents:
            agent.step()
        
        # Update all agent visualizations
        for i, agent in enumerate(agents):
            # Update the agent position marker
            pos = agent.get_position()
            agent_positions[i].set_offsets([pos])
            
            # Update the path line
            path_x, path_y = agent.get_path_coordinates()
            agent_paths[i].set_data(path_x, path_y)
        
        # Update timestamp
        timestamp_text.set_text(f"Step: {frame}")
        
        # Check if all agents have reached their goals
        if all(agent.at_goal() for agent in agents) and not completion_reported:
            print(f"All {species} agents reached their goals at step {frame}")
            completion_reported = True
            anim.event_source.stop()
        
        return agent_paths + agent_positions + [timestamp_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, init_func=init,
        interval=interval, blit=True
    )
    
    # Save the animation if path provided
    if save_path:
        print(f"Saving {species} animation to {save_path}...")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                   exist_ok=True)
        
        # Save with higher quality
        anim.save(save_path, writer='pillow', fps=10, dpi=dpi)
        print(f"Animation saved to {save_path}")
    
    return anim