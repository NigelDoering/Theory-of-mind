import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.environment.world import World
from src.agents.agent import Agent
from src.planning.path_planner import PathPlanner
from src.utils.visualization import plot_path

def create_sample_world():
    """
    Create a sample world with obstacles, starting positions, and goals.
    """
    # Create a 100 x 100 world
    world = World(width=100, height=100)

    # --- Add Obstacles ---
    # Add a rectangular obstacle from row 20 to 30 and column 40 to 50
    for i in range(20, 31):
        for j in range(40, 51):
            world.set_obstacle(i, j)

    # Add a horizontal bar obstacle at row 60 from column 10 to 30
    for j in range(10, 31):
        world.set_obstacle(60, j)

    # Add a vertical bar obstacle at column 70 from row 60 to 90
    for i in range(60, 91):
        world.set_obstacle(i, 70)

    # Add a square obstacle from (10, 10) to (15, 15)
    for i in range(10, 16):
        for j in range(10, 16):
            world.set_obstacle(i, j)

    # Add a diagonal obstacle from (50, 50) to (60, 60)
    for offset in range(0, 11):
        i = 50 + offset
        j = 50 + offset
        world.set_obstacle(i, j)

    # --- Define Starting and Goal Spaces ---
    # Add some starting positions
    world.add_starting_position((5, 5))
    world.add_starting_position((95, 75))
    world.add_starting_position((5, 95))

    # Add some goal positions
    world.add_goal((80, 20))
    world.add_goal((20, 80))
    world.add_goal((50, 50))
    
    return world

def main():
    # Create the world
    world = create_sample_world()
    
    # Display the world
    print("World created with dimensions:", world.width, "x", world.height)
    world.display_world()
    
    # Create an agent
    agent = Agent(
        agent_id=1,
        world=world,
        nominal_start=(5, 5),
        nominal_goal=(80, 20),
        start_std=2.0,
        goal_std=2.0,
        random_seed=42
    )
    print(f"Created agent: {agent}")
    
    # Create a path planner
    planner = PathPlanner(world, temperature=0.5, top_n=5)
    
    # Plan a path for the agent
    path = agent.plan_path(planner)
    
    if path:
        print(f"Found path with {len(path)} steps")
        plot_path(world, path, agent.start, agent.goal, "Agent's Planned Path")
    else:
        print("No path found!")

if __name__ == "__main__":
    main() 