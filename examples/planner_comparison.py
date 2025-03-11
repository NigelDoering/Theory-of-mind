import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.environment.world import World
from src.agents.agent import Agent
from src.planning.planner_zone import PlannerZone
from src.utils.planner_visualization import PlannerVisualizer

def create_sample_world():
    """Create a sample world with obstacles for testing planners."""
    # Create a 100 x 100 world
    world = World(width=100, height=100)

    # Add a rectangular obstacle
    for i in range(20, 31):
        for j in range(40, 51):
            world.set_obstacle(i, j)

    # Add a horizontal bar obstacle
    for j in range(10, 91):
        world.set_obstacle(60, j)

    # Add a vertical bar obstacle
    for i in range(20, 91):
        world.set_obstacle(i, 30)

    # Add a square obstacle
    for i in range(10, 16):
        for j in range(10, 16):
            world.set_obstacle(i, j)

    # Add a maze-like structure
    for i in range(70, 90):
        for j in range(70, 75):
            world.set_obstacle(i, j)
    
    for i in range(70, 75):
        for j in range(70, 90):
            world.set_obstacle(i, j)
    
    for i in range(85, 90):
        for j in range(75, 90):
            world.set_obstacle(i, j)
    
    for i in range(75, 85):
        for j in range(85, 90):
            world.set_obstacle(i, j)

    # Add starting and goal positions
    world.add_starting_position((5, 5))
    world.add_goal((95, 95))
    
    return world

def compare_search_based_planners(world, start, goal):
    """Compare different search-based planning algorithms."""
    print("Comparing search-based planners...")
    
    # List of search-based planners to compare
    planner_types = [
        "a_star", "dijkstra", "bfs", "dfs", "best_first", 
        "bidirectional_a_star", "ara_star", "lpa_star", 
        "lrta_star", "rtaa_star", "d_star_lite", "d_star"
    ]
    
    planners = []
    paths = []
    times = []
    
    for planner_type in planner_types:
        print(f"Testing {planner_type}...")
        planner = PlannerZone.create_planner(planner_type, world)
        
        # Measure planning time
        start_time = time.time()
        path = planner.plan(start, goal)
        end_time = time.time()
        
        planners.append(planner)
        paths.append(path)
        times.append(end_time - start_time)
        
        print(f"  {'Path found' if path else 'No path found'}, time: {times[-1]:.3f}s")
    
    # Create titles with time information
    titles = [f"{p_type} ({t:.3f}s)" for p_type, t in zip(planner_types, times)]
    
    # Visualize results
    visualizer = PlannerVisualizer(world)
    fig = visualizer.visualize_multiple_planners(planners, start, goal, paths, titles)
    fig.suptitle("Search-Based Planners Comparison", fontsize=16)
    plt.savefig("search_based_comparison.png")
    plt.close(fig)
    
    return planners, paths, times

def compare_sampling_based_planners(world, start, goal):
    """Compare different sampling-based planning algorithms."""
    print("Comparing sampling-based planners...")
    
    # List of sampling-based planners to compare
    planner_types = [
        "rrt", "rrt_star", "rrt_connect", "informed_rrt_star", 
        "rrt_star_smart", "fmt_star", "bit_star", "dynamic_rrt",
        "anytime_rrt_star", "closed_loop_rrt_star", "spline_rrt_star"
    ]
    
    planners = []
    paths = []
    times = []
    
    for planner_type in planner_types:
        print(f"Testing {planner_type}...")
        planner = PlannerZone.create_planner(planner_type, world)
        
        # Common parameters for fair comparison
        kwargs = {
            'max_iterations': 5000,
            'goal_sample_rate': 0.2,
            'step_size': 5.0,
        }
        
        # Add planner-specific parameters
        if planner_type in ["rrt_star", "informed_rrt_star", "rrt_star_smart", "anytime_rrt_star"]:
            kwargs['rewire_factor'] = 3.0
        
        if planner_type == "bit_star":
            kwargs['sample_batch_size'] = 100
        
        if planner_type == "fmt_star":
            kwargs['num_samples'] = 1000
            
        # Measure planning time
        start_time = time.time()
        path = planner.plan(start, goal, **kwargs)
        end_time = time.time()
        
        planners.append(planner)
        paths.append(path)
        times.append(end_time - start_time)
        
        print(f"  {'Path found' if path else 'No path found'}, time: {times[-1]:.3f}s")
    
    # Create titles with time and path length information
    titles = []
    for i, (p_type, path, t) in enumerate(zip(planner_types, paths, times)):
        if path:
            path_length = sum(np.sqrt((path[i+1][0]-path[i][0])**2 + (path[i+1][1]-path[i][1])**2) 
                             for i in range(len(path)-1))
            titles.append(f"{p_type} ({t:.2f}s, len: {path_length:.1f})")
        else:
            titles.append(f"{p_type} ({t:.2f}s, no path)")
    
    # Visualize results
    visualizer = PlannerVisualizer(world)
    fig = visualizer.visualize_multiple_planners(planners, start, goal, paths, titles)
    fig.suptitle("Sampling-Based Planners Comparison", fontsize=16)
    plt.savefig("sampling_based_comparison.png")
    plt.close(fig)
    
    return planners, paths, times

def demonstrate_interactive_planning(world, start, goal):
    """Demonstrate interactive planning with visualization."""
    print("Demonstrating interactive planning...")
    
    # Create visualizer
    visualizer = PlannerVisualizer(world)
    
    # Create BIT* planner for demonstration
    planner = PlannerZone.create_planner("bit_star", world)
    
    # Set up callback
    callback = visualizer.callback_for_animation
    
    # Run interactive planning
    path = planner.interactive_plan(
        start, goal, 
        callback=callback,
        batch_size=50,
        max_batches=5
    )
    
    # Create animation
    animation = visualizer.create_animation(filename="bit_star_animation.mp4", fps=5)
    
    print(f"Interactive planning complete. Animation saved to bit_star_animation.mp4")
    
    return path

def demonstrate_goal_inference(world, start, goal):
    """Demonstrate goal inference capabilities."""
    print("Demonstrating goal inference...")
    
    # Create GI-RRT planner
    planner = PlannerZone.create_planner("gi_rrt", world)
    
    # Define multiple potential goals
    potential_goals = [
        (95, 95),  # Original goal
        (95, 5),   # Lower right
        (50, 95),  # Upper middle
    ]
    
    # Create simulated observed trajectory towards the true goal
    observed_trajectory = [
        (5, 5),    # Start
        (10, 10),  # Moving towards upper right (true goal)
        (15, 15),
        (20, 20),
        (25, 25)
    ]
    
    # Perform goal inference
    goal_probabilities = planner.infer_goals(
        observed_trajectory, 
        potential_goals
    )
    
    print("Goal inference results:")
    for goal_pos, probability in goal_probabilities.items():
        print(f"  Goal {goal_pos}: {probability:.4f}")
    
    # Predict future trajectory
    predicted_trajectory = planner.predict_future_trajectory()
    
    # Plan a path considering the predicted trajectory
    path = planner.plan_with_goal_inference(
        start, goal,
        observed_agent_trajectory=observed_trajectory,
        potential_goals=potential_goals,
        max_iterations=1000
    )
    
    # Visualize results
    visualizer = PlannerVisualizer(world)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot grid
    visualizer.plot_grid(ax)
    
    # Plot observed trajectory
    if observed_trajectory:
        o_x, o_y = zip(*observed_trajectory)
        ax.plot(o_x, o_y, '-o', color='blue', linewidth=2, label='Observed')
    
    # Plot predicted trajectory
    if predicted_trajectory:
        p_x, p_y = zip(*predicted_trajectory)
        ax.plot(p_x, p_y, '--o', color='purple', linewidth=2, label='Predicted')
    
    # Plot potential goals with size representing probability
    for goal_pos in potential_goals:
        prob = goal_probabilities.get(goal_pos, 0)
        ax.plot(goal_pos[0], goal_pos[1], '*', color='red', 
                markersize=15 + 20 * prob,  # Size based on probability
                alpha=0.5 + 0.5 * prob)     # Alpha based on probability
        ax.text(goal_pos[0] + 3, goal_pos[1] + 3, f"{prob:.2f}", fontsize=12)
    
    # Plot planned path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, '-', color='green', linewidth=2, label='Planned path')
    
    ax.legend()
    ax.set_title("Goal Inference and Path Planning")
    plt.savefig("goal_inference.png")
    plt.close(fig)
    
    return path, goal_probabilities, predicted_trajectory

def main():
    # Create world
    world = create_sample_world()
    
    # Display the world
    world.display_world()
    plt.savefig("world.png")
    
    # Define start and goal positions
    start = (5, 5)
    goal = (95, 95)
    
    # Compare search-based planners
    search_planners, search_paths, search_times = compare_search_based_planners(world, start, goal)
    
    # Compare sampling-based planners
    sampling_planners, sampling_paths, sampling_times = compare_sampling_based_planners(world, start, goal)
    
    # Demonstrate interactive planning
    interactive_path = demonstrate_interactive_planning(world, start, goal)
    
    # Demonstrate goal inference
    gi_path, goal_probs, predicted_traj = demonstrate_goal_inference(world, start, goal)
    
    print("All demonstrations completed. Check the generated images and animations.")

if __name__ == "__main__":
    main() 