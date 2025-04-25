import numpy as np
import random
import math
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import time

class FlatQLearningAgent:
    """
    A flat Q-learning agent for navigating environments with multiple goals.
    Modified to prioritize visiting all goals rather than fixating on high-value ones.
    """
    
    def __init__(
        self,
        env,
        alpha=0.7,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        min_epsilon=0.02
    ):
        """Initialize the agent with learning parameters."""
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Action space (0: Left, 1: Down, 2: Right, 3: Up)
        self.n_actions = 4
        
        # State space
        self.n_states = env.observation_space.n
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        self._init_q_values()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.goals_reached = []
        self.training_started = False
        self.max_goals_per_episode = 0
        
        # Add goal tracking
        self.last_visited_goal = None
        self.goal_visit_counts = defaultdict(int)
    
    def _init_q_values(self):
        """Initialize Q-values with small random values."""
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.q_table[state][action] = np.random.uniform(0, 0.1)
                
        # Add bias toward all goals to encourage initial exploration
        for goal_pos in self.env.goal_positions:
            goal_state = self.env.pos_to_state[goal_pos]
            # Find adjacent states and add bias toward the goal
            size = self.env.size
            row, col = goal_pos
            
            # Add bonus to actions that lead toward the goal
            for dr, dc, action in [(0, -1, 0), (1, 0, 1), (0, 1, 2), (-1, 0, 3)]:
                new_r, new_c = row + dr, col + dc
                if 0 <= new_r < size and 0 <= new_c < size:
                    if (new_r, new_c) in self.env.pos_to_state:
                        adj_state = self.env.pos_to_state[(new_r, new_c)]
                        # Opposite action leads to the goal
                        opposite_action = (action + 2) % 4
                        self.q_table[adj_state][opposite_action] += 0.5
    
    def select_action(self, state, visited_goals=None, eval_mode=False):
        """
        Select action with goal-directed exploration.
        
        Args:
            state: Current state
            visited_goals: Set of already visited goals (to inform exploration)
            eval_mode: If True, use greedy selection
        """
        if visited_goals is None:
            visited_goals = set()
            
        # Get current position
        pos = self.env.state_to_pos[state]
        
        # Check if at a goal position
        if pos in self.env.goal_positions and pos not in visited_goals:
            # At unvisited goal - record it
            self.last_visited_goal = pos
            self.goal_visit_counts[pos] += 1
        
        # Goal-directed exploration - give higher probability to actions leading toward unvisited goals
        if not eval_mode and random.random() < self.epsilon:
            # Find unvisited goals
            unvisited_goals = [g for g in self.env.goal_positions if g not in visited_goals]
            
            if unvisited_goals:
                # Calculate distances to unvisited goals
                goal_distances = []
                goal_rewards = []
                
                for goal_pos in unvisited_goals:
                    # Manhattan distance
                    dist = abs(goal_pos[0] - pos[0]) + abs(goal_pos[1] - pos[1])
                    reward = self.env.goal_rewards.get(goal_pos, 0)
                    goal_distances.append(dist)
                    goal_rewards.append(reward)
                
                # Normalize distances (closer is better)
                max_dist = max(goal_distances) if goal_distances else 1
                norm_distances = [1 - (d/max_dist) for d in goal_distances]
                
                # Normalize rewards
                max_reward = max(goal_rewards) if goal_rewards else 1
                norm_rewards = [r/max_reward for r in goal_rewards]
                
                # Combine distance and reward (higher value = better target)
                goal_scores = [0.7*r + 0.3*(1-d) for r, d in zip(norm_rewards, norm_distances)]
                
                # Select a goal based on scores
                total_score = sum(goal_scores)
                if total_score > 0:
                    goal_probs = [s/total_score for s in goal_scores]
                    target_idx = np.random.choice(len(unvisited_goals), p=goal_probs)
                    target_goal = unvisited_goals[target_idx]
                    
                    # Choose action that moves toward the target goal
                    action_scores = [0, 0, 0, 0]  # Left, Down, Right, Up
                    
                    # Horizontal movement
                    if pos[1] < target_goal[1]:  # Goal is to the right
                        action_scores[2] += 1  # Right
                    elif pos[1] > target_goal[1]:  # Goal is to the left
                        action_scores[0] += 1  # Left
                        
                    # Vertical movement
                    if pos[0] < target_goal[0]:  # Goal is below
                        action_scores[1] += 1  # Down
                    elif pos[0] > target_goal[0]:  # Goal is above
                        action_scores[3] += 1  # Up
                    
                    # Add some noise to prevent getting stuck
                    for i in range(4):
                        action_scores[i] += random.uniform(0, 0.5)
                    
                    # Choose best action
                    return action_scores.index(max(action_scores))
            
            # If no unvisited goals or didn't select target goal action
            return random.randint(0, self.n_actions - 1)
        else:
            # Greedy selection based on Q-values
            q_values = [self.q_table[state][a] for a in range(self.n_actions)]
            max_value = max(q_values)
            best_actions = [a for a, v in enumerate(q_values) if v == max_value]
            
            # Break ties randomly to prevent loops
            return random.choice(best_actions)
    
    def _goal_directed_action_selection(self, state, visited_goals):
        """Select actions with bias toward unvisited goals"""
        if random.random() < self.epsilon:
            # Explore with bias toward unvisited goals
            unvisited_goals = [g for g in self.env.goal_positions if g not in visited_goals]
            
            if unvisited_goals and random.random() < 0.7:  # 70% chance of goal-directed exploration
                # Select an unvisited goal based on expected reward and distance
                state_pos = self.env.state_to_pos[state]
                goal_values = []
                
                for goal in unvisited_goals:
                    # Compute Manhattan distance
                    distance = abs(state_pos[0] - goal[0]) + abs(state_pos[1] - goal[1])
                    # Get reward for this goal
                    reward = self.env.goal_rewards.get(goal, 0)
                    # Value is a combination of reward and proximity
                    value = reward / (1 + 0.1 * distance)  # Discount by distance
                    goal_values.append((goal, value))
                
                # Softmax selection of goal
                total = sum(math.exp(v) for _, v in goal_values)
                probs = [math.exp(v)/total for _, v in goal_values]
                target_idx = np.random.choice(len(unvisited_goals), p=probs)
                target_goal = unvisited_goals[target_idx]
                
                # Choose action that moves toward the selected goal
                best_action = self._get_direction_to_goal(state, target_goal)
                return best_action
            
            # Regular random exploration
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation but with bonus for actions leading to unvisited goals
            q_values = np.array([self.q_table[state][a] for a in range(self.n_actions)])
            
            # Add bonus for directions leading toward unvisited goals
            for goal_pos in self.env.goal_positions:
                if goal_pos not in visited_goals:
                    direction = self._get_direction_to_goal(state, goal_pos)
                    if direction is not None:
                        q_values[direction] += 0.5  # Small bonus
            
            return np.argmax(q_values)
    
    def _get_direction_to_goal(self, state, goal_pos):
        """Determine best direction to move toward a goal"""
        state_pos = self.env.state_to_pos[state]
        row_diff = goal_pos[0] - state_pos[0] 
        col_diff = goal_pos[1] - state_pos[1]
        
        # Prioritize the larger difference
        if abs(row_diff) > abs(col_diff):
            return 1 if row_diff > 0 else 3  # Down or Up
        else:
            return 2 if col_diff > 0 else 0  # Right or Left

    def _update_q_value(self, state, action, reward, next_state, visited_goals):
        """Update Q-values with bonus for actions leading to unvisited goals"""
        # Standard Q-learning update
        old_value = self.q_table[state][action]
        next_q_values = [self.q_table[next_state][a] for a in range(self.n_actions)]
        next_best = max(next_q_values)
        
        # Calculate bonus based on whether next_state is closer to unvisited goals
        bonus = 0
        next_pos = self.env.state_to_pos[next_state]
        
        # Add bonus for being in a position to reach unvisited goals
        unvisited_goals = [g for g in self.env.goal_positions if g not in visited_goals]
        if unvisited_goals:
            # Calculate current minimum distance to any unvisited goal
            current_pos = self.env.state_to_pos[state]
            current_min_dist = min(abs(current_pos[0] - g[0]) + abs(current_pos[1] - g[1]) for g in unvisited_goals)
            next_min_dist = min(abs(next_pos[0] - g[0]) + abs(next_pos[1] - g[1]) for g in unvisited_goals)
            
            # Bonus for getting closer to an unvisited goal
            if next_min_dist < current_min_dist:
                bonus = 0.2
        
        # Update Q-value with bonus
        new_value = old_value + self.alpha * (reward + bonus + self.gamma * next_best - old_value)
        self.q_table[state][action] = new_value

    def train(self, n_episodes=5000, max_steps=100, eval_interval=100, render=False):
        """Train with goal-directed exploration and diminishing returns for revisits"""
        # Reset performance tracking for a new training session
        if not self.training_started:
            self.episode_rewards = []
            self.episode_steps = []
            self.goals_reached = []
            self.training_started = True
        
        # Progress bar
        pbar = tqdm(range(n_episodes), desc="Training")
        
        for episode in pbar:
            # Reset environment
            state, _ = self.env.reset()
            visited_goals = set()
            
            # Episode variables
            total_reward = 0
            steps = 0
            goals_reached = 0
            
            for step in range(max_steps):
                # Modify action selection to consider visited goals
                action = self._goal_directed_action_selection(state, visited_goals)
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Check for goal collection
                next_pos = self.env.state_to_pos[next_state]
                if next_pos in self.env.goal_positions and next_pos not in visited_goals:
                    visited_goals.add(next_pos)
                    goals_reached += 1
                
                # Dynamic Q-value update with diminishing returns
                self._update_q_value(state, action, reward, next_state, visited_goals)
                
                # Update tracking
                total_reward += reward
                steps += 1
                
                # Render if requested
                if render:
                    self.env.render()
                    time.sleep(0.1)
                
                # Update state
                state = next_state
                
                # Break if episode is done
                if done:
                    break
            
            # Update metrics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.goals_reached.append(goals_reached)
            self.max_goals_per_episode = max(self.max_goals_per_episode, goals_reached)
            
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
            if eval_interval > 0 and episode % eval_interval == 0:
                eval_reward, eval_steps, eval_goals = self.evaluate(5, max_steps)
                print(f"\nEval: reward={eval_reward:.2f}, steps={eval_steps:.1f}, goals={eval_goals:.1f}")
        
        return {
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'goals_reached': self.goals_reached
        }
    
    def evaluate(self, n_episodes=5, max_steps=100, render=False):
        """Evaluate agent with loop prevention."""
        eval_rewards = []
        eval_steps = []
        eval_goals_reached = []
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            visited_goals = set()
            
            total_reward = 0
            steps = 0
            goals_reached = 0
            
            # Loop detection
            state_sequence = []
            loop_counter = 0
            
            for step in range(max_steps):
                # Select action with loop detection
                if loop_counter >= 3:
                    # Take random action to break potential loop
                    action = random.randint(0, self.n_actions - 1)
                    loop_counter = 0
                else:
                    # Use goal-directed policy
                    action = self.select_action(state, visited_goals, eval_mode=True)
                
                # Track state visitation
                state_sequence.append(state)
                if len(state_sequence) > 10:
                    state_sequence.pop(0)
                    
                    # Check for loops (same 2-3 states repeating)
                    if len(set(state_sequence[-6:])) <= 2:
                        loop_counter += 1
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update metrics
                total_reward += reward
                steps += 1
                
                # Check if at a goal
                agent_pos = self.env.state_to_pos[next_state]
                if agent_pos in self.env.goal_positions and agent_pos not in visited_goals:
                    visited_goals.add(agent_pos)
                    goals_reached += 1
                    loop_counter = 0  # Reset loop counter when goal found
                    print(f"Reached goal at {agent_pos} with reward {self.env.goal_rewards.get(agent_pos, 0):.2f}")
                
                # Render if requested
                if render:
                    self.env.render()
                    time.sleep(0.1)
                
                # Update state
                state = next_state
                
                # Break if done
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_steps.append(steps)
            eval_goals_reached.append(goals_reached)
        
        return np.mean(eval_rewards), np.mean(eval_steps), np.mean(eval_goals_reached)
    
    def get_policy(self):
        """
        Get the current greedy policy.
        
        Returns:
            Dictionary mapping states to actions
        """
        policy = {}
        for state in range(self.n_states):
            q_values = [self.q_table[state][a] for a in range(self.n_actions)]
            policy[state] = np.argmax(q_values)
        return policy
    
    def save(self, filepath):
        """Save agent to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'episode_steps': self.episode_steps,
                'goals_reached': self.goals_reached
            }, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in data['q_table'].items():
                for action, value in actions.items():
                    self.q_table[state][action] = value
            self.alpha = data['alpha']
            self.gamma = data['gamma']
            self.epsilon = data['epsilon']
            self.episode_rewards = data['episode_rewards']
            self.episode_steps = data['episode_steps']
            self.goals_reached = data['goals_reached']
        print(f"Agent loaded from {filepath}")
    
    def plot_training_progress(self, window_size=100, save_path=None):
        """
        Plot enhanced training progress metrics with smoothing.
        
        Args:
            window_size: Size of the window for smoothing
            save_path: Optional path to save the figure
        """
        if not self.episode_rewards:
            print("No training data to plot")
            return
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Calculate cumulative rewards
        cum_rewards = np.cumsum(self.episode_rewards)
        
        # Episode rewards
        axs[0].plot(self.episode_rewards, alpha=0.3, color='blue', label='Per-episode')
        # Add smoothed line
        if len(self.episode_rewards) > window_size:
            smoothed = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            axs[0].plot(range(window_size-1, len(self.episode_rewards)), smoothed, 'r-', 
                       linewidth=2, label=f'Smoothed (window={window_size})')
        axs[0].set_ylabel('Reward')
        axs[0].set_title('Episode Rewards')
        axs[0].grid(True)
        axs[0].legend()
        
        # Cumulative rewards
        axs[1].plot(cum_rewards, color='green')
        axs[1].set_ylabel('Cumulative Reward')
        axs[1].set_title('Cumulative Rewards')
        axs[1].grid(True)
        
        # Episode steps and goals in dual axis
        ax2 = axs[2].twinx()
        
        # Plot steps
        p1 = axs[2].plot(self.episode_steps, alpha=0.3, color='purple', label='Steps')
        if len(self.episode_steps) > window_size:
            smoothed_steps = np.convolve(self.episode_steps, np.ones(window_size)/window_size, mode='valid')
            axs[2].plot(range(window_size-1, len(self.episode_steps)), smoothed_steps, 
                       color='darkviolet', linewidth=2, label=f'Smoothed Steps')
        axs[2].set_ylabel('Steps')
        axs[2].set_xlabel('Episode')
        
        # Plot goals on secondary y-axis
        p2 = ax2.plot(self.goals_reached, alpha=0.3, color='orange', label='Goals Reached')
        if len(self.goals_reached) > window_size:
            smoothed_goals = np.convolve(self.goals_reached, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(self.goals_reached)), smoothed_goals, 
                    color='red', linewidth=2, label=f'Smoothed Goals')
        ax2.set_ylabel('Goals Reached')
        
        # Combine legends from both axes
        lines1, labels1 = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        axs[2].set_title('Episode Length and Goals Reached')
        axs[2].grid(True)
        
        # Set up common xlabel
        fig.text(0.5, 0.04, 'Episode', ha='center', fontsize=12)
        
        # Add an overall title
        plt.suptitle('Agent Training Progress', fontsize=16)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Enhanced training progress plot saved to {save_path}")
        plt.show()
    
    def visualize_q_values(self, save_path=None):
        """
        Create an enhanced visualization of Q-values as heatmaps with policy arrows.
        
        Args:
            save_path: Optional path to save the figure
        """
        # Get environment dimensions
        size = self.env.size
        
        # Create figure with subplots for each action
        fig, axes = plt.subplots(2, 2, figsize=(15, 13))
        axes = axes.flatten()
        
        # Action names for titles
        action_names = ['Left', 'Down', 'Right', 'Up']
        
        # Create Q-value grids for each action
        for action in range(self.n_actions):
            # Initialize grid for this action
            q_grid = np.zeros((size, size))
            
            # Fill with Q-values
            for i in range(size):
                for j in range(size):
                    if (i, j) in self.env.pos_to_state:
                        state = self.env.pos_to_state[(i, j)]
                        q_grid[i, j] = self.q_table[state][action]
            
            # Create heatmap for this action
            im = axes[action].imshow(q_grid, cmap='viridis', vmin=q_grid.min(), vmax=q_grid.max())
            axes[action].set_title(f'Q-Values for {action_names[action]}')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=axes[action])
            cbar.set_label('Q-value')
            
            # Add text annotations showing actual Q-values
            for i in range(size):
                for j in range(size):
                    if (i, j) in self.env.pos_to_state:
                        val = q_grid[i, j]
                        # Skip text for very small values
                        if abs(val) > 0.01:
                            # Choose text color based on background darkness
                            text_color = 'white' if val > (q_grid.max() + q_grid.min())/2 else 'black'
                            axes[action].text(j, i, f'{val:.2f}', ha='center', va='center', 
                                             color=text_color, fontsize=8)
            
            # Mark goals and holes
            for pos in self.env.goal_positions:
                y, x = pos
                axes[action].plot(x, y, 'o', markersize=12, markeredgecolor='white', 
                                 markerfacecolor='none', markeredgewidth=2)
                
            for pos in self.env.hole_positions:
                y, x = pos
                axes[action].plot(x, y, 'X', markersize=12, markeredgecolor='red',
                                 markerfacecolor='none', markeredgewidth=2)
            
            # Add grid lines
            axes[action].grid(True, color='black', alpha=0.2)
            
            # Set ticks to integer positions
            axes[action].set_xticks(np.arange(size))
            axes[action].set_yticks(np.arange(size))
            
        plt.tight_layout()
        
        # Create a second figure for the policy heatmap
        plt.figure(figsize=(10, 8))
        
        # Get the best action for each state
        policy = self.get_policy()
        q_max = np.zeros((size, size))
        policy_grid = np.zeros((size, size), dtype=int)
        
        # Fill grid with best actions and max Q-values
        for i in range(size):
            for j in range(size):
                if (i, j) in self.env.pos_to_state:
                    state = self.env.pos_to_state[(i, j)]
                    policy_grid[i, j] = policy[state]
                    q_max[i, j] = max([self.q_table[state][a] for a in range(self.n_actions)])
        
        # Action symbols for arrows
        action_symbols = ['←', '↓', '→', '↑']
        
        # Create heatmap of max Q-values
        im = plt.imshow(q_max, cmap='Blues')
        plt.colorbar(im, label='Max Q-value')
        
        # Overlay policy arrows
        for i in range(size):
            for j in range(size):
                if (i, j) in self.env.pos_to_state:
                    # Skip goals and holes
                    if (i, j) in self.env.goal_positions or (i, j) in self.env.hole_positions:
                        continue
                    action = policy_grid[i, j]
                    plt.text(j, i, action_symbols[action], ha='center', va='center', 
                            color='black', fontsize=15, fontweight='bold')
        
        # Mark goals with their rewards
        for pos in self.env.goal_positions:
            y, x = pos
            reward = self.env.goal_rewards.get(pos, 0)
            plt.plot(x, y, 'o', markersize=15, markeredgecolor='gold', 
                    markerfacecolor='green', alpha=0.7)
            plt.text(x, y, f'{reward:.1f}', ha='center', va='center', 
                    color='white', fontsize=9, fontweight='bold')
        
        # Mark holes
        for pos in self.env.hole_positions:
            y, x = pos
            plt.plot(x, y, 'X', markersize=15, markeredgecolor='black',
                    markerfacecolor='red', alpha=0.7)
        
        plt.grid(True, color='black', alpha=0.2)
        plt.title('Policy with Max Q-values Heatmap')
        plt.xticks(np.arange(size))
        plt.yticks(np.arange(size))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Q-value and policy visualization saved to {save_path}")
        
        plt.tight_layout()
        plt.show()