import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frozen_src.world_graph import WorldGraph
from frozen_src.orienteering_agent import OrienteeringAgent
from frozen_src.goal_space_generator import GoalSpaceGenerator

def run_orienteering_simulation(goal_generator=None, map_size=8, n_goals=None, seed=42,
                              perception_radius=2, budget_scale=2.0, alpha=0.4,
                              goal_bonus=2.5, visualize_every=1):
    """
    Run an orienteering simulation with the given parameters.
    
    Args:
        goal_generator: Existing goal generator (optional)
        map_size: Size of the map
        n_goals: Number of goals
        seed: Random seed
        perception_radius: Agent's perception radius
        budget_scale: How much budget to give relative to map size (n^2)
        alpha: Weight for distance to final goal in node evaluation
        goal_bonus: Bonus multiplier for intermediate goals
        visualize_every: How often to update visualization
        
    Returns:
        Agent and statistics
    """
    # Generate goal space if not provided
    if goal_generator is None:
        # If n_goals is not specified, use map_size + 1
        if n_goals is None:
            n_goals = map_size + 1
            
        print(f"Generating goal space (size: {map_size}x{map_size}, goals: {n_goals})")
        goal_generator = GoalSpaceGenerator(map_size=map_size, n_goals=n_goals, seed=seed)
        goal_generator.generate_goal_space()
        
        # Visualize the goal space
        goal_generator.visualize_goal_space()
    
    # Create the world graph
    world_graph = WorldGraph(goal_generator)
    
    # Calculate budget - increased for better coverage
    budget = int(map_size ** 2 * budget_scale)
    print(f"Setting budget to {budget} steps")
    
    # Create the agent
    agent = OrienteeringAgent(
        world_graph=world_graph,
        budget=budget,
        perception_radius=perception_radius,
        alpha=alpha,
        goal_bonus=goal_bonus
    )
    
    # Run an episode
    agent.run_episode(visualize_every=visualize_every, delay=0.1)  # Fast updates with minimal delay
    
    return agent

# Example usage
if __name__ == "__main__":
    # Create a goal space generator
    goal_generator = GoalSpaceGenerator(map_size=8, n_goals=9, seed=42)
    goal_space = goal_generator.generate_goal_space()
    
    # Run the simulation with optimized parameters
    agent = run_orienteering_simulation(
        goal_generator=goal_generator, 
        perception_radius=2,
        alpha=1.0,
        goal_bonus=2.5,
        budget_scale=1.0,
        visualize_every=1
    )