import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import sys
import os
import time

class HierarchicalQLearningAgent:
    """
    Hierarchical Q-learning agent for multi-goal environments.
    
    This agent uses a two-level hierarchy:
    1. High level: Choose which goal to pursue next
    2. Low level: Learn how to reach the chosen goal using movement actions
    """
    
    def __init__(
        self,
        env,
        high_alpha: float = 0.2,
        low_alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        """Initialize the agent."""
        self.env = env
        self.high_alpha = high_alpha
        self.low_alpha = low_alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Store Dirichlet distribution parameters for tracking
        self.dirichlet_probabilities = env.dirichlet_probs if hasattr(env, 'dirichlet_probs') else None
        
        # Action space (MODIFIED: only movement actions - no collect)
        self.n_actions = 4  # Only up, down, left, right
        
        # State space
        self.n_states = env.observation_space.n
        
        # Initialize Q-tables
        self._init_q_tables()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.goals_reached = []
        
        # Current goal tracking
        self.current_goal = None
        
    def _init_q_tables(self):
        """Initialize Q-tables for high and low level policies."""
        # High-level Q-table: state -> goal -> value
        self.high_q_table = defaultdict(lambda: defaultdict(float))
        
        # Low-level Q-tables: goal -> (state, action) -> value
        self.low_q_tables = defaultdict(lambda: defaultdict(float))
        
        # Find all goal positions and convert to state IDs
        self.goal_states = []
        for pos in self.env.goal_positions:
            state = self.env.pos_to_state[pos]
            self.goal_states.append(state)
        
        # Initialize Q-values with small random values to break ties
        for state in range(self.n_states):
            for goal in self.goal_states:
                # Initialize high-level Q-values
                self.high_q_table[state][goal] = np.random.uniform(0, 0.1)
                
                # Initialize low-level Q-values (MODIFIED: only 4 actions)
                for action in range(self.n_actions):
                    self.low_q_tables[goal][(state, action)] = np.random.uniform(0, 0.1)
        
        # Bias movement actions based on proximity to goal
        for goal in self.goal_states:
            goal_pos = self.env.state_to_pos[goal]
            for state in range(self.n_states):
                pos = self.env.state_to_pos[state]
                
                # If at goal position, prefer exploring elsewhere
                if pos == goal_pos:
                    # At goal, bias toward movement in all directions
                    for action in range(4):
                        self.low_q_tables[goal][(state, action)] = 0.2
        
    def _select_goal(self, state, eval_mode=False):
        """
        Select the next goal to pursue using epsilon-greedy.
        
        Args:
            state: Current state
            eval_mode: If True, use greedy selection
            
        Returns:
            Selected goal state
        """
        # MODIFIED: All goals are always available - no collected goals concept
        available_goals = self.goal_states
            
        # Epsilon-greedy selection with value bias
        if eval_mode or random.random() > self.epsilon:
            # Get Q-values and rewards for goals
            q_values = {}
            for g in available_goals:
                goal_pos = self.env.state_to_pos[g]
                goal_reward = self.env.goal_rewards.get(goal_pos, 0)
                # Weight by both Q-value and actual reward
                q_values[g] = self.high_q_table[state][g] + (0.1 * goal_reward)
                
            max_value = max(q_values.values()) if q_values else 0
            best_goals = [g for g, v in q_values.items() if v == max_value]
            return random.choice(best_goals) if best_goals else random.choice(available_goals)
        else:
            # Random selection, but with bias toward higher reward goals
            if random.random() < 0.3 and available_goals:  # 30% chance of reward-biased selection
                goal_rewards = [(g, self.env.goal_rewards.get(self.env.state_to_pos[g], 0)) 
                               for g in available_goals]
                # Normalize probabilities to ensure they sum to 1
                total_reward = sum(r for _, r in goal_rewards) + 1e-9
                probabilities = [r / total_reward for _, r in goal_rewards]
                return np.random.choice(available_goals, p=probabilities)
            return random.choice(available_goals)
    
    def _select_action(self, state, goal, eval_mode=False):
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            goal: Current goal
            eval_mode: If True, use greedy selection
            
        Returns:
            Selected action
        """
        # MODIFIED: No special handling for being at a goal - just select movement actions
            
        # Epsilon-greedy selection
        if eval_mode or random.random() > self.epsilon:
            # Greedy selection
            q_values = [self.low_q_tables[goal][(state, a)] for a in range(self.n_actions)]
            max_value = max(q_values)
            best_actions = [a for a, v in enumerate(q_values) if v == max_value]
            return random.choice(best_actions)
        else:
            # Random selection (uniform)
            return random.randint(0, self.n_actions - 1)
    
    def train(self, n_episodes=5000, max_steps=100, eval_interval=100, render=False):
        """Train the agent for specified number of episodes."""
        # Reset performance tracking if this is a new training session
        if not hasattr(self, 'training_started') or not self.training_started:
            self.episode_rewards = []
            self.episode_steps = []
            self.goals_reached = []
            self.training_started = True
            if not hasattr(self, 'max_goals_per_episode'):
                self.max_goals_per_episode = 0
        
        # Use tqdm for progress bar without showing every detail
        pbar = tqdm(range(n_episodes), desc="Training")
        
        for episode in pbar:
            # Reset environment
            state, _ = self.env.reset()
            
            # Track visited goals for this episode
            visited_goals = set()
            
            # Track episode performance
            total_reward = 0
            steps = 0
            goals_reached = 0
            
            # Episode loop
            for step in range(max_steps):
                # Select goal
                goal = self._select_goal(state)
                self.current_goal = goal
                
                # Select action for current state and goal
                action = self._select_action(state, goal)
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                # MODIFIED: Check if we're at a goal position
                agent_pos = self.env.state_to_pos[next_state]
                if agent_pos in self.env.goal_positions and next_state not in visited_goals:
                    visited_goals.add(next_state)
                    goals_reached += 1
                
                # Update Q-tables (Q-learning update)
                # Low-level Q-table (action selection)
                old_value = self.low_q_tables[goal][(state, action)]
                next_best = max([self.low_q_tables[goal][(next_state, a)] for a in range(self.n_actions)])
                new_value = old_value + self.low_alpha * (reward + self.gamma * next_best - old_value)
                self.low_q_tables[goal][(state, action)] = new_value
                
                # High-level Q-table update if we reached the goal
                if next_state == goal:
                    # Update goal value
                    old_value = self.high_q_table[state][goal]
                    # Choose best next goal from next state
                    next_goals = [g for g in self.goal_states]
                    if next_goals:
                        next_best = max([self.high_q_table[next_state][g] for g in next_goals])
                    else:
                        next_best = 0
                    # Update high-level Q-value
                    new_value = old_value + self.high_alpha * (reward + self.gamma * next_best - old_value)
                    self.high_q_table[state][goal] = new_value
                
                # Update state
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Update metrics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.goals_reached.append(goals_reached)
            
            # Track maximum goals found in any episode
            self.max_goals_per_episode = max(self.max_goals_per_episode if hasattr(self, 'max_goals_per_episode') else 0, 
                                            goals_reached)
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Update progress bar with metrics - only every 10 episodes
            if episode % 10 == 0:
                pbar.set_postfix({
                    'reward': f"{np.mean(self.episode_rewards[-10:]):.2f}",
                    'steps': f"{np.mean(self.episode_steps[-10:]):.1f}",
                    'goals': f"{np.mean(self.goals_reached[-10:]):.1f}",
                    'epsilon': f"{self.epsilon:.3f}"
                })
            
            # Run evaluation
            if eval_interval > 0 and episode % eval_interval == 0:
                eval_reward, eval_steps, eval_goals = self.evaluate(5, max_steps)
                print(f"\nEval: reward={eval_reward:.2f}, steps={eval_steps:.1f}, goals={eval_goals:.1f}")
        
        return {
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'goals_reached': self.goals_reached
        }
    
    def evaluate(self, n_episodes=5, max_steps=100, render=False):
        """
        Evaluate the agent.
        
        Args:
            n_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            
        Returns:
            Average reward, steps and goals reached
        """
        eval_rewards = []
        eval_steps = []
        eval_goals_reached = []
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            
            # Track visited goals
            visited_goals = set()
            
            goal = self._select_goal(state, eval_mode=True)
            
            total_reward = 0
            steps = 0
            goals_reached = 0
            
            for step in range(max_steps):
                # Select best action (no exploration)
                action = self._select_action(state, goal, eval_mode=True)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update metrics
                total_reward += reward
                steps += 1
                
                # Render if requested
                if render:
                    self.env.render()
                    time.sleep(0.1)  # Slow down for visualization
                
                # MODIFIED: Check if at goal position
                agent_pos = self.env.state_to_pos[next_state]
                if agent_pos in self.env.goal_positions and next_state not in visited_goals:
                    visited_goals.add(next_state)
                    goals_reached += 1
                
                # Select new goal if reached current one
                if next_state == goal:
                    goal = self._select_goal(next_state, eval_mode=True)
                
                # Update state
                state = next_state
                
                # Break if done
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_steps.append(steps)
            eval_goals_reached.append(goals_reached)
        
        return np.mean(eval_rewards), np.mean(eval_steps), np.mean(eval_goals_reached)

    def plot_training_progress(self, save_path=None):
        """
        Plot the training progress.
        
        Args:
            save_path: Optional path to save the figure
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot rewards
        axs[0].plot(self.episode_rewards)
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        
        # Plot steps
        axs[1].plot(self.episode_steps)
        axs[1].set_title('Episode Steps')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Steps')
        
        # Plot goals reached
        axs[2].plot(self.goals_reached)
        axs[2].set_title('Goals Reached per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Goals Reached')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Training progress saved to {save_path}")
        
        plt.show()
    
    def save(self, filename='hierarchical_agent.pkl'):
        """Save the agent to a file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        data = {
            'high_q_table': dict(self.high_q_table),
            'low_q_tables': {k: dict(v) for k, v in self.low_q_tables.items()},
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'goals_reached': self.goals_reached,
            'dirichlet_probabilities': self.dirichlet_probabilities
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Agent saved to {filename}")
    
    def load(self, filename='hierarchical_agent.pkl'):
        """Load the agent from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Convert dictionaries back to defaultdicts
        self.high_q_table = defaultdict(lambda: defaultdict(float))
        for s, goals in data['high_q_table'].items():
            for g, v in goals.items():
                self.high_q_table[s][g] = v
                
        self.low_q_tables = defaultdict(lambda: defaultdict(float))
        for g, states in data['low_q_tables'].items():
            for sa, v in states.items():
                self.low_q_tables[g][sa] = v
        
        self.epsilon = data['epsilon']
        self.episode_rewards = data['episode_rewards']
        self.episode_steps = data['episode_steps']
        self.goals_reached = data['goals_reached']
        self.dirichlet_probabilities = data.get('dirichlet_probabilities', None)
        self.training_started = True
        
        print(f"Agent loaded from {filename}")

    def visualize_q_tables(self, save_path=None):
        """
        Visualize high and low Q-tables for the current state.
        
        Args:
            save_path: Optional path to save the figure
        """
        import matplotlib.pyplot as plt
        
        # Get current state (for visualization purposes)
        state, _ = self.env.reset()
        
        # Setup the figure
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        
        # High Q-table visualization (goal selection)
        size = self.env.size
        high_q_grid = np.zeros((size, size))
        
        # Fill the grid with high-level Q-values
        for goal_state in self.goal_states:
            goal_pos = self.env.state_to_pos[goal_state]
            high_q_grid[goal_pos[0], goal_pos[1]] = self.high_q_table[state][goal_state]
        
        # Plot high-level Q-values
        im1 = axs[0].imshow(high_q_grid, cmap='viridis')
        axs[0].set_title(f'High-Level Q-Values (Goal Selection) from State {state}')
        axs[0].set_xlabel('Column')
        axs[0].set_ylabel('Row')
        fig.colorbar(im1, ax=axs[0], label='Goal Value')
        
        # Add text annotations for goal values
        for goal_state in self.goal_states:
            goal_pos = self.env.state_to_pos[goal_state]
            i, j = goal_pos
            value = self.high_q_table[state][goal_state]
            axs[0].text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color='white' if value > np.mean(high_q_grid) else 'black')
        
        # Low Q-table visualization (action selection)
        # Choose a goal for low-level visualization
        if hasattr(self, 'current_goal') and self.current_goal is not None:
            goal = self.current_goal
        else:
            goal = list(self.goal_states)[0]  # First goal
        
        # Get best action for each state
        action_names = ['Left', 'Down', 'Right', 'Up']
        best_action_grid = np.zeros((size, size))
        max_q_grid = np.zeros((size, size))  # For tracking max Q-values
        
        # Fill grids with best actions and their Q-values
        for i in range(size):
            for j in range(size):
                cell_state = self.env.pos_to_state.get((i, j))
                if cell_state is not None:
                    q_values = [self.low_q_tables[goal].get((cell_state, a), 0) for a in range(self.n_actions)]
                    best_action = np.argmax(q_values)
                    best_action_grid[i, j] = best_action
                    max_q_grid[i, j] = max(q_values)
        
        # Plot low-level Q-values as best actions
        im2 = axs[1].imshow(best_action_grid, cmap='tab10', vmin=0, vmax=3)
        axs[1].set_title(f'Low-Level Best Actions for Goal at {self.env.state_to_pos[goal]}')
        axs[1].set_xlabel('Column')
        axs[1].set_ylabel('Row')
        
        # Colorbar for action values
        cbar = fig.colorbar(im2, ax=axs[1], label='Action', ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(action_names)
        
        # Add text annotations showing action and Q-value
        for i in range(size):
            for j in range(size):
                cell_state = self.env.pos_to_state.get((i, j))
                if cell_state is not None:
                    action = int(best_action_grid[i, j])
                    q_val = max_q_grid[i, j]
                    text = f"{action_names[action][0]}\n{q_val:.1f}"
                    axs[1].text(j, i, text, ha='center', va='center', 
                              color='black', fontsize=8)
        
        # Plot the goal position
        goal_pos = self.env.state_to_pos[goal]
        axs[1].plot(goal_pos[1], goal_pos[0], 'yo', markersize=12, alpha=0.7)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Q-tables visualization saved to {save_path}")
        
        # Display but don't block
        plt.draw()
        plt.pause(0.1)
        plt.close()