# Frozen RL Project

## Overview
The Frozen RL project implements a multi-goal reinforcement learning environment based on the classic Frozen Lake problem. It features a hierarchical Q-learning agent capable of navigating through a grid world with multiple goals, random hole positions, and customizable reward structures.

## Project Structure
```
frozen_rl
├── __init__.py
├── main.py
├── agents
│   ├── __init__.py
│   └── hierarchical_agent.py
├── environments
│   ├── __init__.py
│   └── multi_goal_frozen_lake.py
├── utils
│   ├── __init__.py
│   ├── experiment.py
│   └── visualization.py
├── config.py
├── experiments
│   └── config_samples
│       └── example_config.json
└── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can use pip to install the necessary packages:

```bash
pip install -r requirements.txt
```

## Usage
To run the project, execute the `main.py` file. This will initialize the environment and agent, and start the training process.

```bash
python main.py
```

## Components
- **Agents**: The `agents/hierarchical_agent.py` file contains the implementation of the `HierarchicalQLearningAgent` class, which is responsible for learning and decision-making in the environment.
  
- **Environments**: The `environments/multi_goal_frozen_lake.py` file defines the `MultiGoalFrozenLakeEnv` class, extending the standard Frozen Lake environment to support multiple goals and custom reward structures.

- **Utilities**: The `utils` directory contains helper functions for running experiments and visualizing results.

- **Configuration**: The `config.py` file holds configuration settings for the project, including default parameters for the environment and agent.

- **Experiments**: The `experiments/config_samples/example_config.json` file serves as a template for experiment configurations, storing goal IDs, rewards, alpha values, and other parameters.

## Experiment Replication
To replicate experiments, modify the `example_config.json` file with your desired parameters, including goal IDs, rewards, alpha values, Dirichlet probabilities, the final goal, and the seed. Load this configuration in your experiments to ensure consistent results.

## Visualization
The project includes visualization tools to render the training process live in the Frozen Lake environment, allowing you to observe the agent's learning progress in real-time.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.