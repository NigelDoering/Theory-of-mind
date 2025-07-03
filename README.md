# Theory of Mind (ToM) Simulation Framework 🧠

> A comprehensive framework for simulating and analyzing agent behaviors with Theory of Mind capabilities in both abstract and realistic environments. This project is designed to support research on Theory of Mind models by MOSAIC LAB, UCSD.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![OSMnx](https://img.shields.io/badge/OSMnx-enabled-green.svg)](https://github.com/gboeing/osmnx)
[![NetworkX](https://img.shields.io/badge/NetworkX-powered-orange.svg)](https://networkx.org/)

---

## 📋 Table of Contents

- Overview
- Repository Structure
- Installation
- Quick Start
- Using the 2D Grid World (src)
- Using the UCSD Campus Environment (real_world_src)
- Agent Types
- Visualization Tools
- Advanced Features
- Troubleshooting

---

## 🔍 Overview

This project implements two distinct simulation environments:

1. **2D Grid World (src)**: A simple grid-based environment for basic path planning and agent simulation
2. **UCSD Campus Environment (real_world_src)**: A realistic geographic environment using OpenStreetMap data of the UC San Diego campus

Both environments allow for simulating agents with different navigation behaviors and decision-making processes, supporting research on Theory of Mind - the ability to attribute mental states to oneself and others.

---

## 📂 Repository Structure

```
Theory-of-mind/
│
├── src/                         # Simple 2D grid world simulation
│   ├── simulation/              # Base simulation classes
│   ├── environment/             # Grid world environment
│   ├── agents/                  # Basic agent classes
│   ├── planning/                # Path planning algorithms
│   └── utils/                   # Utility functions
│
├── real_world_src/              # UCSD campus environment simulation
│   ├── agents/                  # 8 different agent types
│   ├── environment/             # Campus environment using OSMnx
│   ├── simulation/              # Simulation engine
│   ├── utils/                   # Configuration and visualization tools
│   ├── visuals/                 # Output visualizations by run
│   └── main.py                  # Main entry point
│
├── notebooks/                   # Analysis notebooks
├── examples/                    # Example simulations
├── cache/                       # Cached map data
└── requirements.txt             # Dependencies
```

---

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/NigelDoering/Theory-of-mind.git
   cd Theory-of-mind
   ```

2. **Create a virtual environment**
   ```bash
   # Using conda (Recommended)
   conda create -n tom python==3.12 -y
   conda activate tom
   
   # Using venv (Windows)
   python -m venv venv
   venv\Scripts\activate

   # Using venv (macOS/Linux)
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Core dependencies (Recommended)
   pip install -r requirements.txt 
   
   # Core dependencies (Experimental)
   pip install -r requirements-strict.txt 

   # Visualization enhancements
   pip install imageio pillow
   
   # For Bayesian modeling (optional)
   pip install pymc pytensor arviz
   ```

---

## 🚀 Quick Start

### Run the UCSD Campus Simulation

```bash
python real_world_src/main.py
```

This will:
1. Create a timestamped simulation run folder
2. Initialize the UCSD campus environment
3. Create agents of 8 different types (default)
4. Run the simulation with visualization
5. Save all outputs (animations, plots, etc.)

### Run a Simple 2D Grid World Simulation

```bash
python examples/simple_simulation.py
```

---

## 🌐 Using the 2D Grid World (src)

The src directory contains a simple grid-based environment for basic simulations.

### Creating a Grid World

```python
from src.environment.world import World
from src.agents.agent import Agent
from src.planning.path_planner import PathPlanner

# Create a 100x100 grid world
world = World(width=100, height=100)

# Add some obstacles
for x in range(30, 70):
    world.set_obstacle(x, 50)

# Add a start and goal
world.add_starting_position((10, 10))
world.add_goal((90, 90))

# Create an agent
agent = Agent(world)

# Plan a path
planner = PathPlanner(world)
path = planner.find_path(agent.position, world.goals[0])

# Visualize
world.visualize(path=path)
```

### Grid World Agent Types
- `Agent`: Basic agent following planned paths
- Custom agents can be created by extending the `Agent` class

---

## 🏫 Using the UCSD Campus Environment (real_world_src)

The real_world_src directory contains a realistic UCSD campus environment simulation.

### Creating a Custom Simulation

```python
from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.agents.agent_factory import AgentFactory
from real_world_src.simulation.simulator import Simulator
from real_world_src.utils.run_manager import RunManager

# Create a run manager
run_manager = RunManager('visuals')
run_dir = run_manager.start_new_run()

# Initialize campus environment
campus = CampusEnvironment()

# Specify agent populations
agent_populations = {
    "shortest": 2,  # 2 shortest path agents
    "random": 2,    # 2 random walk agents
    "landmark": 2,  # 2 landmark-based agents
    "social": 2,    # 2 social agents
    "explorer": 2,  # 2 explorer agents
    "obstacle": 2,  # 2 obstacle-avoiding agents
    "scared": 2,    # 2 scared agents
    "risky": 2      # 2 risky agents
}

# Create agents
agents = AgentFactory.populate_environment(campus, agent_populations)

# Create simulator and run
simulator = Simulator(campus, run_manager=run_manager)
simulator.run_simulation(
    max_steps=100,
    animate=True,
    save_animation=True
)
```

### Customizing Simulation Parameters

```python
# Run for more steps
simulator.run_simulation(max_steps=200)

# Run without animation (faster)
simulator.run_simulation(animate=False)

# Only create specific agent types
agent_populations = {
    "shortest": 5,  # Only shortest path agents
    "landmark": 3   # Only landmark agents
}
```

### Accessing Environment Features

```python
# Get graph nodes and edges
nodes = campus.G.nodes()
edges = campus.G.edges()

# Find nearest node to coordinates
nearest_node = campus.get_nearest_node(x, y)

# Get node coordinates
coords = campus.get_node_coordinates(node_id)

# Access landmark and building data
landmarks = campus.landmarks
buildings = campus.buildings
```

---

## 🤖 Agent Types

The UCSD campus environment includes eight agent types with distinct navigation behaviors:

| Agent Type | Description | Example |
|------------|-------------|---------|
| **ShortestPathAgent** | Uses A* algorithm to find and follow the shortest path | `ShortestPathAgent("Agent1")` |
| **RandomWalkAgent** | Takes a biased random walk toward the goal | `RandomWalkAgent("Agent2", randomness=0.7)` |
| **LandmarkAgent** | Navigates via recognizable landmarks | `LandmarkAgent("Agent3")` |
| **SocialAgent** | Follows other agents or well-traveled paths | `SocialAgent("Agent4")` |
| **ExplorerAgent** | Prefers less-traveled paths | `ExplorerAgent("Agent5")` |
| **ObstacleAvoidingAgent** | Takes safer routes around obstacles | `ObstacleAvoidingAgent("Agent6", avoidance_strength=0.8)` |
| **ScaredAgent** | Avoids obstacles and other agents | `ScaredAgent("Agent7", fear_factor=0.7)` |
| **RiskyAgent** | Takes shortcuts through difficult terrain | `RiskyAgent("Agent8", risk_tolerance=0.9)` |

---

## 📊 Visualization Tools

The framework provides extensive visualization capabilities:

### Static Visualizations

```python
from real_world_src.utils.visualization import plot_agent_paths, plot_species_grid

# Plot all agent paths
fig, ax = plot_agent_paths(campus, agents, "Agent Paths")
fig.savefig("agent_paths.png")

# Plot species grid (all types in a grid layout)
fig, axs = plot_species_grid(campus, agents, "Species Comparison")
fig.savefig("species_grid.png")
```

### Animations

```python
from real_world_src.utils.visualization import animate_species_grid, animate_single_species

# Create a grid animation of all species
animate_species_grid(
    campus, 
    agents, 
    title="Species Navigation",
    max_frames=50, 
    save_path="species_grid.gif"
)

# Animate just one species
social_agents = [a for a in agents if a.species == "Social"]
animate_single_species(
    campus, 
    social_agents,
    title="Social Agent Navigation",
    max_frames=50,
    save_path="social_agents.gif"
)
```

### Decision Visualization

```python
from real_world_src.utils.visualization import visualize_agent_decision

# Visualize an agent's decision-making process
agent = next(a for a in agents if a.species == "Landmark")
fig, ax = visualize_agent_decision(campus, agent, "Landmark Agent Decisions")
fig.savefig("landmark_decisions.png")
```

---

## 🔬 Advanced Features

### Output Organization with RunManager

The `RunManager` class organizes simulation outputs into structured folders:

```python
from real_world_src.utils.run_manager import RunManager

run_manager = RunManager('visuals')
run_dir = run_manager.start_new_run()  # Creates a timestamped folder

# Get paths for different outputs
animation_path = run_manager.get_animation_path("my_animation.gif")
plot_path = run_manager.get_plot_path("my_plot.png")
decision_path = run_manager.get_agent_decision_path("decision.png")
```

Each run folder contains:
- `animations/`: Animation GIFs
- `static_plots/`: Static visualizations
- `agent_decisions/`: Decision process visualizations
- `species_plots/`: Species-specific visualizations
- README.md: Auto-generated documentation of all outputs

### Creating Custom Agents

Create your own agent type by extending the base Agent class:

```python
from real_world_src.agents.base_agent import Agent
import networkx as nx
import random

class MyCustomAgent(Agent):
    def __init__(self, id=None, color=None):
        super().__init__(id, color=color or '#AA5500')
        self.species = "Custom"
        
    def plan_path(self):
        """Custom path planning logic."""
        try:
            # Example: Add some custom behavior
            if random.random() < 0.5:
                # Take shortest path
                self.path = nx.shortest_path(
                    self.environment.G_undirected, 
                    source=self.current_node, 
                    target=self.goal_node, 
                    weight='length'
                )
            else:
                # Take a detour through a random landmark
                landmark = random.choice(self.environment.landmarks)
                path1 = nx.shortest_path(
                    self.environment.G_undirected, 
                    source=self.current_node, 
                    target=landmark, 
                    weight='length'
                )
                path2 = nx.shortest_path(
                    self.environment.G_undirected, 
                    source=landmark, 
                    target=self.goal_node, 
                    weight='length'
                )
                # Combine paths excluding duplicate landmark
                self.path = path1[:-1] + path2
                
            self.path_index = 0
            
        except nx.NetworkXNoPath:
            print(f"{self.id}: No path found!")
            self.path = [self.current_node]
```

### Bayesian Modeling for Theory of Mind

The framework includes Bayesian modeling capabilities for inferring agent goals and beliefs:

```python
# Example from notebooks/Bayesian Modeling.ipynb
import pymc as pm
import numpy as np

# Create a model to infer agent goals from observed trajectory
with pm.Model() as model:
    # Prior on goals (uniform over possibilities)
    goal_idx = pm.Categorical("goal_idx", 
                            p=np.ones(len(possible_goals)) / len(possible_goals))
    
    # Get coordinates for the selected goal
    goal = pm.Deterministic("goal", possible_goals[goal_idx])
    
    # Agent-specific parameters
    alpha = pm.HalfNormal("alpha", sigma=1.0)  # Directness parameter
    sigma = pm.HalfNormal("sigma", sigma=1.0)  # Observation noise
    
    # Model agent behavior with observations
    for t, pos in enumerate(observed_trajectory):
        # Compute distance to goal
        dist_to_goal = pm.math.sqrt(pm.math.sum((pos - goal)**2))
        
        # Expected next position (closer to goal)
        expected_pos = pos - alpha * (pos - goal) / dist_to_goal
        
        # Observe with noise
        pm.Normal(f"obs_{t}_x", mu=expected_pos[0], sigma=sigma, observed=pos[0])
        pm.Normal(f"obs_{t}_y", mu=expected_pos[1], sigma=sigma, observed=pos[1])
    
    # Sample from posterior
    trace = pm.sample(1000, tune=500)
```

### Command-line Arguments

The main script supports command-line arguments for easy customization:

```bash
# Run with more steps and specific agent counts
python real_world_src/main.py --steps 100 --shortest 5 --random 3 --landmark 2
```

### Using the Cache System

```python
# Use cached data if available, otherwise download
campus = CampusEnvironment(use_cache=True)
```


### 🚢 AIS Data Cleaning and Processing

1. **Download Data**  
   AIS data is downloaded from [https://web.ais.dk/aisdata/](https://web.ais.dk/aisdata/) — use the **daily** CSV files (not monthly). Save each file to:

   
# AIS Data Cleaning and Processing

This repository contains utilities for cleaning and processing AIS (Automatic Identification System) data downloaded from the Danish Maritime Authority.

---

## 📥 1. Download Daily AIS Data

AIS data should be downloaded from the Danish Maritime Authority website:

👉 [https://web.ais.dk/aisdata/](https://web.ais.dk/aisdata/)

Download the **daily CSV files** (not the monthly files) and save each file to the following directory:

```
./notebooks/data/AIS_data/
```

---

## 🧹 2. Clean a Single Day of AIS Data

Run the following script to clean and process a single day’s AIS data:

```bash
python utils/clean_ais_single_day.py <filename.csv>
```

This script:
- Filters out invalid or noisy data points
- Groups the AIS messages by unique IMO number
- Saves a cleaned CSV file for each ship in the following directory structure:

```
./notebooks/data/trips/<YYYY_MM_DD>/IMO_<imo>.csv
```

---

## 🧭 3. Segment Full IMO Tracks Into Distinct Voyages

Once multiple days of data have been cleaned, run the following script to segment voyages:

```bash
python utils/process_imo_trips.py
```

This script:
- Loads all per-IMO files across dates
- Concatenates them into a full track per IMO
- Segments the full track into distinct voyages using a **4-hour gap threshold**
- Saves each segmented voyage into:

```
./notebooks/data/processed_trips/IMO_<imo>_trip_<n>.csv
```

---

## 🎯 4. Output

Each file in:

```
./notebooks/data/processed_trips/
```

now represents a **single trajectory (voyage)** that can be directly used for model training or downstream analysis.

---

## 🛠 Notes

- Ensure filenames follow the convention `aisdk-YYYY-MM-DD.csv` when passed to the cleaning script.
- All scripts assume that IMO numbers are consistent across dates.
- Output directories will be created automatically if they do not exist.
---

## ⚠️ Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **OSMnx Download Errors** | • Use cached version: `CampusEnvironment(load_cached=True)`<br>• Check your internet connection<br>• Try different network type: `CampusEnvironment(network_type='walk')`<br>• Verify place name: `CampusEnvironment(place_name="University of California San Diego")` |
| **Animation Generation Problems** | • Install required packages: `pip install imageio pillow`<br>• Reduce frame count: `max_frames=30`<br>• Use lower DPI: `dpi=80` |
| **Memory Errors** | • Reduce number of agents<br>• Reduce maximum steps<br>• Use smaller region of interest<br>• Run without animation: `animate=False` |
| **Agent Path Issues** | • Check graph connectivity: `print(nx.is_connected(campus.G_undirected))`<br>• Verify start/goal nodes are valid<br>• Print diagnostic information |

### **_Let Your Mind explore and build its own Theories!_**