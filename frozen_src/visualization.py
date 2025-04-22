import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
from pathlib import Path

sns.set_theme()

from frozen_src.environment import FrozenLakeEnv
from frozen_src.agent import QLearningAgent

def visualize_environment(desc, save_path: Path = None) -> None:
    """Visualize the FrozenLake environment using matplotlib"""
    plt.figure(figsize=(7, 7))
    
    # Create a grid of the same size as the environment
    nrow, ncol = desc.shape
    grid = np.zeros((nrow, ncol, 3), dtype=float)  # RGB array
    
    # Define colors for different tiles
    colors = {
        b'S': [0.0, 0.5, 0.0],    # Start: Dark Green
        b'F': [0.9, 0.9, 1.0],    # Frozen: Light Blue
        b'H': [0.2, 0.2, 0.8],    # Hole: Dark Blue
        b'G': [1.0, 0.9, 0.0]     # Goal: Gold
    }
    
    # Fill the grid with colors based on the environment
    for i in range(nrow):
        for j in range(ncol):
            grid[i, j] = colors[desc[i][j]]
    
    # Plot the grid
    plt.imshow(grid)
    
    # Add text annotations
    tile_dict = {b'S': 'S', b'F': 'F', b'H': 'H', b'G': 'G'}
    for i in range(nrow):
        for j in range(ncol):
            plt.text(j, i, tile_dict[desc[i][j]], 
                     ha="center", va="center", color="black", fontsize=15)
    
    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    
    # Add a title with map size
    plt.title(f"FrozenLake Environment ({nrow}x{ncol})")
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(save_path)  # Save the visualization
    plt.show()

def print_text_map(desc):
    """Print a text-based visualization of the map to the console"""
    nrow, ncol = desc.shape
    print("FrozenLake Map:")
    print("┌" + "─" * (ncol * 2 - 1) + "┐")
    for i in range(nrow):
        row_str = "│"
        for j in range(ncol):
            tile = desc[i][j]
            if tile == b'S':
                symbol = "S"  # Start
            elif tile == b'F':
                symbol = "·"  # Frozen tile
            elif tile == b'H':
                symbol = "O"  # Hole
            elif tile == b'G':
                symbol = "G"  # Goal
            row_str += symbol
            if j < ncol - 1:
                row_str += " "
        row_str += "│"
        print(row_str)
    print("└" + "─" * (ncol * 2 - 1) + "┘")
    print("S: Start, ·: Frozen, O: Hole, G: Goal")

def run_experiment(
    env: FrozenLakeEnv,
    agent: QLearningAgent,
    total_episodes: int,
    n_runs: int = 1,
    max_steps: int = None,
    render: bool = False,
    render_delay: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """Run Q-learning via agent.train over n_runs and collect results."""
    episodes = np.arange(total_episodes)
    rewards = np.zeros((total_episodes, n_runs))
    steps = np.zeros((total_episodes, n_runs))
    qtables = np.zeros((n_runs, env.observation_space.n, env.action_space.n))
    # central train
    for run in range(n_runs):
        agent.reset()
        rp, sp = agent.train(env, num_episodes=total_episodes, max_steps=max_steps,
                             render=render, render_delay=render_delay)
        rewards[:, run] = rp
        steps[:, run] = sp
        qtables[run] = agent.q_table.copy()
    return rewards, steps, episodes, qtables, [], []


def postprocess(
    episodes: np.ndarray,
    rewards: np.ndarray,
    steps: np.ndarray,
    map_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert experiment outputs into tidy DataFrames for plotting."""
    n_runs = rewards.shape[1]
    df = pd.DataFrame({
        'Episode': np.tile(episodes, n_runs),
        'Reward': rewards.flatten(order='F'),
        'Step': steps.flatten(order='F')
    })
    df['CumReward'] = df['Reward'].cumsum()
    df['MapSize'] = f"{map_size}x{map_size}"
    df_steps = pd.DataFrame({'Episode': episodes, 'Steps': steps.mean(axis=1)})
    df_steps['MapSize'] = f"{map_size}x{map_size}"
    return df, df_steps


def plot_states_actions_distribution(
    states: List[int], actions: List[int], map_size: int, save_path: Path = None
) -> None:
    """Plot histograms of visited states and taken actions."""
    labels = {0:'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'}
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.histplot(states, ax=ax[0], kde=True)
    ax[0].set_title('States')
    sns.histplot(actions, ax=ax[1])
    ax[1].set_xticks(list(labels.keys()), labels=list(labels.values()))
    ax[1].set_title('Actions')
    plt.suptitle(f'Distribution for map {map_size}x{map_size}')
    if save_path:
        plt.savefig(save_path)
    # plt.show(block=False)


def qtable_directions_map(
    qtable: np.ndarray, map_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return max Q-values and best-action arrow map."""
    q_max = qtable.max(axis=1).reshape(map_size, map_size)
    q_best = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    arrows = {0:'←',1:'↓',2:'→',3:'↑'}
    direction_map = np.full(q_best.shape, '', dtype='<U1')
    eps = np.finfo(float).eps
    flat_max = q_max.flatten()
    for i, a in enumerate(q_best.flatten()):
        if flat_max[i] > eps:
            direction_map.flat[i] = arrows[a]
    return q_max, direction_map


def plot_q_values_map(
    qtable: np.ndarray, env: FrozenLakeEnv, map_size: int, save_path: Path = None
) -> None:
    """Display last env frame and Q-value heatmap with arrows."""
    q_vals, q_dirs = qtable_directions_map(qtable, map_size)
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    img = env.env.render()
    ax[0].imshow(img)
    ax[0].axis('off'); ax[0].set_title('Last Frame')
    sns.heatmap(
        q_vals, annot=q_dirs, fmt='', cmap='Blues', ax=ax[1],
        cbar_kws={'label':'Q-value'}, linewidths=0.5, linecolor='black'
    )
    ax[1].set_title('Learned Q-values (arrows = policy)')
    if save_path:
        plt.savefig(save_path)
    # plt.show(block=False)


def plot_steps_and_rewards(
    df_rewards: pd.DataFrame, df_steps: pd.DataFrame, save_path: Path = None
) -> None:
    """Plot cumulative rewards and average steps over episodes."""
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.lineplot(data=df_rewards, x='Episode', y='CumReward', hue='MapSize', ax=ax[0])
    ax[0].set_ylabel('Cumulated Reward')
    sns.lineplot(data=df_steps, x='Episode', y='Steps', hue='MapSize', ax=ax[1])
    ax[1].set_ylabel('Average Steps')
    plt.suptitle('Learning Curves')
    if save_path:
        plt.savefig(save_path)
    # plt.show(block=False)


def train_with_render(
    env: FrozenLakeEnv,
    agent: QLearningAgent,
    episodes: int,
    delay: float = 0.1
) -> QLearningAgent:
    """Train and render the agent live in the env window."""
    # delegate to central train
    _, _ = agent.train(env, num_episodes=episodes, render=True, render_delay=delay)
    return agent