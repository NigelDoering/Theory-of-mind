import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
from tqdm import tqdm
from environment import FrozenLakeEnv
import time
import random

@dataclass
class QLearningConfig:
    learning_rate: float = 0.8
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01

class QLearningAgent:
    """An efficient Q-learning agent with configurable hyperparameters and training loop."""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[QLearningConfig] = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or QLearningConfig()
        self.reset()

    def reset(self) -> None:
        """Reset Q-table and exploration rate."""
        self.q_table = np.random.uniform(low=0.0, high=0.01, size=(self.state_size, self.action_size))
        self.epsilon = self.config.epsilon

    def select_action(self, state: int) -> int:
        """Epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        max_value = np.max(self.q_table[state])
        max_indices = np.where(self.q_table[state] == max_value)[0]
        return np.random.choice(max_indices)

    def learn(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> float:
        """Update Q-value using the Q-learning rule and return TD error."""
        best_next_action = np.argmax(self.q_table[next_state])
        if done:
            td_target = reward
        else:
            td_target = reward + self.config.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.config.learning_rate * td_error
        return abs(td_error)

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

    def train(
        self,
        env,
        num_episodes: int = 1000,
        max_steps: Optional[int] = None,
        callback: Optional[Callable[..., None]] = None,
        render: bool = False,
        render_delay: float = 0.0,
    ) -> Tuple[List[float], List[int], List[float]]:
        """
        Train the agent on the environment with optional live rendering.
        Returns:
          - rewards_per_episode: total reward per episode
          - steps_per_episode: number of steps taken per episode
          - errors_per_episode: total TD error per episode
        """
        rewards_per_episode: List[float] = []
        steps_per_episode: List[int] = []
        errors_per_episode: List[float] = []

        for ep in tqdm(range(num_episodes), desc=f"Training episodes"):
            # unpack reset returning (state, info)
            result = env.reset()
            if isinstance(result, tuple) and len(result) == 2:
                state, _ = result
            else:
                state = result
            total_reward = 0.0
            step = 0
            done = False
            error_sum = 0.0

            while not done:
                action = self.select_action(state)
                # support env.step returning either 5-tuple or 4-tuple
                result = env.step(action)
                if len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = result
                err = self.learn(state, action, reward, next_state, done)
                error_sum += err
                state = next_state
                total_reward += reward
                step += 1
                if render:
                    env.env.render()
                    time.sleep(render_delay)
                if max_steps and step >= max_steps:
                    break
            # record and decay
            rewards_per_episode.append(total_reward)
            steps_per_episode.append(step)
            errors_per_episode.append(error_sum)
            self.decay_epsilon()
            if callback:
                # callback may accept error arg
                try:
                    callback(ep, total_reward, error_sum)
                except TypeError:
                    callback(ep, total_reward)
        return rewards_per_episode, steps_per_episode, errors_per_episode

    def evaluate(
        self,
        env,
        num_episodes: int = 100,
        max_steps: Optional[int] = None,
        render: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate the learned policy without exploration.
        Returns:
          - avg_reward: average reward per episode
          - avg_steps: average steps per episode
        """
        total_rewards = []
        total_steps = []
        orig_epsilon = self.epsilon
        self.epsilon = 0.0  # greedy policy

        for _ in range(num_episodes):
            # unpack reset returning (state, info)
            result = env.reset()
            if isinstance(result, tuple) and len(result) == 2:
                state, _ = result
            else:
                state = result
            episode_reward = 0.0
            steps = 0
            done = False
            while not done:
                action = self.select_action(state)
                # support env.step returning either 5-tuple or 4-tuple
                result = env.step(action)
                if len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = result
                state = next_state
                episode_reward += reward
                steps += 1
                if max_steps and steps >= max_steps:
                    break
                if render:
                    env.render()
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        self.epsilon = orig_epsilon
        return float(np.mean(total_rewards)), float(np.mean(total_steps))

    def get_policy(self) -> np.ndarray:
        """Return the best action for each state based on current Q-table."""
        return np.argmax(self.q_table, axis=1)

if __name__ == "__main__":
    # simple interaction test
    env = FrozenLakeEnv(size=4)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    state = env.reset()
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    # support env.step returning either 5-tuple or 4-tuple
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        next_state, reward, done, _ = result
    print(f"Next: state={next_state}, reward={reward}, done={done}")