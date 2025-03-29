import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_agent_paths(environment, agents, title="Agent Paths", show_buildings=True, figsize=(15, 15)):
    """Plot the paths taken by agents on the campus map."""
    fig, ax = plt.subplots(figsize=figsize)
    
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
        ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.6, linewidth=0.5)
    
    # Plot buildings if available
    if environment.buildings is not None and show_buildings:
        environment.buildings.plot(ax=ax, color='lightgrey', alpha=0.5, edgecolor='dimgrey')
    
    # Plot agent paths
    for agent in agents:
        path_x, path_y = agent.get_path_coordinates()
        if path_x and path_y:
            ax.plot(path_x, path_y, color=agent.color, alpha=0.8, linewidth=2.5, label=f"{agent.species}")
            
            # Plot start and goal
            start_x, start_y = agent.get_position(agent.start_node)
            goal_x, goal_y = agent.get_position(agent.goal_node)
            ax.scatter([start_x], [start_y], color=agent.color, marker='o', s=100, zorder=10)
            ax.scatter([goal_x], [goal_y], color=agent.color, marker='*', s=150, zorder=10)
    
    # Plot landmarks
    landmark_x = [environment.node_coords[node][0] for node in environment.landmarks]
    landmark_y = [environment.node_coords[node][1] for node in environment.landmarks]
    ax.scatter(landmark_x, landmark_y, c='red', s=50, alpha=0.6, zorder=5, label="Landmarks")
    
    # Add legend (only include one entry per agent species)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    label_dict = {}
    
    for handle, label in zip(handles, labels):
        if label not in label_dict:
            label_dict[label] = handle
            unique_labels.append(label)
            unique_handles.append(handle)
            
    ax.legend(unique_handles, unique_labels, loc='best')
    
    plt.title(title)
    plt.tight_layout()
    
    return fig, ax

def visualize_agent_decision(environment, agent, title="Agent Decision Process"):
    """Visualize the decision-making process of an agent."""
    fig, ax = plt.subplots(figsize=(15, 15))
    
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
        ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.4, linewidth=0.5)
    
    # Plot agent's current position
    current_x, current_y = agent.get_position()
    ax.scatter([current_x], [current_y], color=agent.color, s=150, zorder=10, label="Current Position")
    
    # Plot goal
    goal_x, goal_y = agent.get_position(agent.goal_node)
    ax.scatter([goal_x], [goal_y], color='red', marker='*', s=200, zorder=10, label="Goal")
    
    # Plot the planned path
    if len(agent.path) > 0:
        path_coords = [environment.get_node_coordinates(node) for node in agent.path]
        path_x = [coord[0] for coord in path_coords]
        path_y = [coord[1] for coord in path_coords]
        ax.plot(path_x, path_y, color=agent.color, alpha=0.8, linewidth=2, linestyle='--', label="Planned Path")
    
    # If it's a landmark agent, also show landmarks
    if agent.species == "Landmark" and agent.current_landmark:
        landmark_x, landmark_y = environment.get_node_coordinates(agent.current_landmark)
        ax.scatter([landmark_x], [landmark_y], color='green', marker='s', s=150, zorder=9, label="Current Landmark")
    
    # If it's a social agent, show the follow target
    if agent.species == "Social" and agent.follow_target:
        target_x, target_y = agent.follow_target.get_position()
        ax.scatter([target_x], [target_y], color='purple', marker='P', s=150, zorder=9, label="Follow Target")
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    return fig, ax