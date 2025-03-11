# Theory of Mind Simulation

A framework for simulating agents with theory of mind capabilities in a 2D grid world environment (For the time being). This project is designed to support research on Theory of Mind models by MOSAIC LAB, UCSD.

## Repo Structure (To be updated soon though!)

- `src/simulation/`: Base simulation classes
- `src/environment/`: World environment classes
- `src/agents/`: Agent classes for the simulation
- `src/planning/`: Path planning algorithms and node classes
- `src/utils/`: Utility functions, including visualization
- `examples/`: Example simulations and demonstrations

## Getting Started

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run an example simulation: `python examples/simple_simulation.py`

## Features

- 2D grid world environment with obstacles
- Agents with stochastic start and goal positions
- Path planning using stochastic A* algorithm
- Visualization utilities for the environment and agent paths

## Future Development

- Enhancing the simulation world to build the envisioned environment
- Implementing agent perception and beliefs
- Adding more complex agent behaviors
- Supporting inference of agent goals and beliefs
- Implementing Bayesian inference for theory of mind modeling 
- Implementing the ANN version of the theory of mind modeling
- Novelty!