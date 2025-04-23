import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import sys
import os

class HierarchicalQLearningAgent:
    """
    Hierarchical Q-learning agent for multi-goal environments.
    
    This agent uses a two-level hierarchy:
    1. High level: Choose which goal to pursue next
    2. Low level: Learn how to reach the chosen goal and when to collect rewards
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
        """
        Initialize the agent.
        
        Args:
            env: The environment (must be MultiGoalFrozenLakeEnv)
            high_alpha: Learning rate for high-level policy
            low_alpha: Learning rate for low-level policy
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
        """
        self.env = env
        self.high_alpha = high_alpha
        self.low_alpha = low_alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Store Dirichlet distribution parameters for tracking
        self.dirichlet_probabilities = env.dirichlet_probs
        
        # Action space (including collect)
        self.n_actions = env.action_space.n
        
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
        self.goal_reached = False
        
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
        
        # Add final goal
        final_goal_state = self.env.pos_to_state[self.env.final_goal_pos]
        if final_goal_state not in self.goal_states:
            self.goal_states.append(final_goal_state)
        
        self.final_goal_state = final_goal_state
        
    def _select_goal(self, state, eval_mode=False):
        """
        Select the next goal to pursue using epsilon-greedy.
        
        Args:
            state: Current state
            eval_mode: If True, use greedy selection
            
        Returns:
            Selected goal state
        """
        # Find available goals (not collected yet)
        available_goals = []
        for goal_state in self.goal_states:
            goal_pos = self.env.state_to_pos[goal_state]
            if goal_pos not in self.env.collected_goals:
                available_goals.append(goal_state)
        
        # Always include final goal
        if self.final_goal_state not in available_goals:
            available_goals.append(self.final_goal_state)
            
        # If no goals available, return final goal
        if not available_goals:
            return self.final_goal_state
        
        # Epsilon-greedy selection
        if eval_mode or random.random() > self.epsilon:
            # Greedy selection based on high-level Q-values
            q_values = {g: self.high_q_table[state][g] for g in available_goals}
            max_value = max(q_values.values()) if q_values else 0
            best_goals = [g for g, v in q_values.items() if v == max_value]
            return random.choice(best_goals) if best_goals else random.choice(available_goals)
        else:
            # Random selection
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
        # Epsilon-greedy selection
        if eval_mode or random.random() > self.epsilon:
            # Greedy selection
            q_values = [self.low_q_tables[goal][(state, a)] for a in range(self.n_actions)]
            max_value = max(q_values)
            best_actions = [a for a, v in enumerate(q_values) if v == max_value]
            return random.choice(best_actions)
        else:
            # Random selection
            return random.randint(0, self.n_actions - 1)
    
    def train(self, n_episodes=5000, max_steps=100, eval_interval=100, render=False):
        """
        Train the agent.
        
        Args:
            n_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            eval_interval: How often to evaluate the agent
            render: Whether to render the environment during training
            
        Returns:
            Training metrics
        """
        # Reset performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.goals_reached = []
        
        # Progress bar
        pbar = tqdm(range(n_episodes), desc="Training")
        
        for episode in pbar:
            # Reset environment
            state, _ = self.env.reset()
            
            # Select initial goal
            goal = self._select_goal(state)
            self.current_goal = goal
            
            # Track episode performance
            total_reward = 0
            steps = 0
            goals_reached = 0
            
            # Episode loop
            for step in range(max_steps):
                # Select action
                action = self._select_action(state, goal)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update metrics
                total_reward += reward
                steps += 1
                
                # Render if requested
                if render and episode % (n_episodes // 10) == 0:
                    self.env.render()
                
                # Check if goal reached (agent moved to goal location)
                goal_pos = self.env.state_to_pos[goal]
                agent_pos = self.env.state_to_pos[next_state]
                
                # Update low-level Q-table
                old_value = self.low_q_tables[goal][(state, action)]
                next_best = max([self.low_q_tables[goal][(next_state, a)] for a in range(self.n_actions)])
                new_value = old_value + self.low_alpha * (reward + self.gamma * next_best - old_value)
                self.low_q_tables[goal][(state, action)] = new_value
                
                # Check if we've reached the goal and need to choose a new one
                if agent_pos == goal_pos and action == 4:  # Collect action
                    # Update high-level Q-table
                    old_value = self.high_q_table[state][goal]
                    next_goals = [g for g in self.goal_states if g != goal]
                    if next_goals:
                        best_next_goal = max(next_goals, key=lambda g: self.high_q_table[next_state][g])
                        next_best = self.high_q_table[next_state][best_next_goal]
                    else:
                        next_best = 0
                    new_value = old_value + self.high_alpha * (reward + self.gamma * next_best - old_value)
                    self.high_q_table[state][goal] = new_value
                    
                    # Select new goal
                    goal = self._select_goal(next_state)
                    goals_reached += 1
                
                # Update state
                state = next_state
                
                # Break if done
                if done:
                    break
            
            # Record episode performance
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.goals_reached.append(goals_reached)
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Update progress bar
            if episode % 10 == 0:
                pbar.set_postfix({
                    'reward': f"{np.mean(self.episode_rewards[-10:]):.2f}",
                    'steps': f"{np.mean(self.episode_steps[-10:]):.1f}",
                    'goals': f"{np.mean(self.goals_reached[-10:]):.1f}",
                    'epsilon': f"{self.epsilon:.3f}"
                })
            
            # Run evaluation
            if episode % eval_interval == 0:
                eval_reward, eval_steps, eval_goals = self.evaluate(5)
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
            goal = self._select_goal(state, eval_mode=True)
            
            total_reward = 0
            steps = 0
            goals_reached = 0
            
            for step in range(max_steps):
                # Select best action
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
                
                # Check if goal reached (agent moved to goal location)
                goal_pos = self.env.state_to_pos[goal]
                agent_pos = self.env.state_to_pos[next_state]
                
                # If reached goal and collected, select new goal
                if agent_pos == goal_pos and action == 4:  # Collect action
                    goals_reached += 1
                    if not done:
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

    def plot_training_progress(self):
        """Plot the training progress."""
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
        plt.savefig("training_progress.png")
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
        
        print(f"Agent loaded from {filename}")