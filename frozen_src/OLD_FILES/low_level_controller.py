import numpy as np
import heapq
from collections import deque

class LowLevelController:
    """Low-level controller that navigates to sub-goals using A* on a local belief map."""
    
    def __init__(self, env, learning_rate=0.8, gamma=0.95, epsilon=0.2):
        """
        Initialize the low-level controller.
        
        Args:
            env: FrozenLake environment
            learning_rate: Learning rate for Q-learning
            gamma: Discount factor
            epsilon: Exploration probability
        """
        self.env = env
        self.env_unwrapped = env.unwrapped
        self.nrow = env.unwrapped.nrow
        self.ncol = env.unwrapped.ncol
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.n
        self.q_table = np.random.uniform(low=0.01, high=0.05, size=(self.state_size, self.action_size))
        self.local_map = None
        self.current_path = None
        
    def initialize_local_map(self, size):
        """Initialize the local belief map."""
        # 0: unknown, 1: free, 2: hole, 3: goal
        self.local_map = np.zeros((size, size), dtype=int)
        
    def update_local_map(self, observation, position):
        """Update the local belief map based on observation."""
        # In a real implementation, we would convert the observation to a local map update
        # For simplicity, we're assuming the observation directly gives us the map state
        row, col = position // self.env.ncol, position % self.env.ncol
        self.local_map[row, col] = 1  # Mark as free space
        
        # Check adjacent cells if they're holes (this is simplified)
        # In a real implementation, you'd use the actual observation
        for dr, dc, action in [(0, 1, 2), (1, 0, 1), (0, -1, 0), (-1, 0, 3)]:  # right, down, left, up
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.env.nrow and 0 <= nc < self.env.ncol:
                # This is a simplification - in reality would use observation
                state = nr * self.env.ncol + nc
                if self.env.unwrapped.desc[nr][nc] == b'H':  # Hole
                    self.local_map[nr, nc] = 2
                elif self.env.unwrapped.desc[nr][nc] == b'G':  # Goal
                    self.local_map[nr, nc] = 3
                else:
                    # Only mark as free if we directly observe it
                    if abs(dr) + abs(dc) == 1:  # Adjacent cell
                        self.local_map[nr, nc] = 1
        
    def plan_path(self, start, goal):
        """Plan a path from start to goal using A* on the local map."""
        if self.local_map is None:
            self.initialize_local_map(self.env.nrow)
        
        # A* algorithm for path planning
        open_set = [(0, start)]
        heapq.heapify(open_set)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        start_row, start_col = start // self.env.ncol, start % self.env.ncol
        goal_row, goal_col = goal // self.env.ncol, goal % self.env.ncol
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = self.reconstruct_path(came_from, current)
                self.current_path = path
                return path
            
            row, col = current // self.env.ncol, current % self.env.ncol
            
            # Explore neighbors
            for dr, dc, action in [(0, 1, 2), (1, 0, 1), (0, -1, 0), (-1, 0, 3)]:  # right, down, left, up
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.env.nrow and 0 <= nc < self.env.ncol:
                    neighbor = nr * self.env.ncol + nc
                    
                    # Skip holes
                    if self.local_map[nr, nc] == 2:
                        continue
                    
                    # Higher cost for unknown cells
                    if self.local_map[nr, nc] == 0:
                        tentative_g_score = g_score[current] + 2.0
                    else:
                        tentative_g_score = g_score[current] + 1.0
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        if (f_score[neighbor], neighbor) not in open_set:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None
    
    def heuristic(self, state, goal):
        """Manhattan distance heuristic."""
        state_row, state_col = state // self.env.ncol, state % self.env.ncol
        goal_row, goal_col = goal // self.env.ncol, goal % self.env.ncol
        return abs(state_row - goal_row) + abs(state_col - goal_col)
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def select_action(self, state, known_holes=None):
        """
        Select an action based on Q-values, but never choose actions leading to known holes.
        
        Args:
            state: Current state
            known_holes: Set of states identified as holes
            
        Returns:
            Selected action
        """
        if known_holes is None:
            known_holes = set()
        
        # Get row and column of current state
        row, col = state // self.ncol, state % self.ncol
        
        # Get available actions (exclude actions that lead to holes)
        available_actions = []
        for action in range(self.env.action_space.n):
            # Check where this action would lead
            dr, dc = 0, 0
            if action == 0:  # LEFT
                dc = -1
            elif action == 1:  # DOWN
                dr = 1
            elif action == 2:  # RIGHT
                dc = 1
            elif action == 3:  # UP
                dr = -1
            
            # Calculate next position
            next_row, next_col = row + dr, col + dc
            
            # Skip if out of bounds
            if not (0 <= next_row < self.nrow and 0 <= next_col < self.ncol):
                continue
                
            next_state = next_row * self.ncol + next_col
            
            # Skip if it's a known hole
            if next_state in known_holes:
                continue
                
            available_actions.append(action)
        
        # If no safe actions are available, just use all actions 
        # (this should be rare, but prevents crashes)
        if not available_actions:
            available_actions = list(range(self.env.action_space.n))
        
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            # Exploration: choose randomly from available actions
            return np.random.choice(available_actions)
        else:
            # Exploitation: choose best action from available actions
            q_values = [self.get_q_value(state, a) for a in available_actions]
            best_idx = np.argmax(q_values)
            return available_actions[best_idx]
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning."""
        # Q-learning update formula
        best_next_action = np.argmax(self.q_table[next_state])
        
        # Calculate TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        
        # Update Q-value
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def random_action(self):
        """Return a random action for exploration."""
        return np.random.randint(0, self.action_size)
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        return self.q_table[state, action]