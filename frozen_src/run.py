import os
import sys
import shutil
import wandb
import numpy as np
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frozen_src.environment import FrozenLakeEnv
from frozen_src.agent import QLearningAgent
from frozen_src.visualization import postprocess, plot_steps_and_rewards, plot_q_values_map

# Setup local output directory inside frozen_src
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize WandB
wandb.init(
    project="frozenlake-ql",
    config={
        "n_runs": 50,
        "episodes": 5000,  # increased episodes to match notebook and support larger grids
    }
)

map_sizes = [4, 7, 9, 11, 15]
for map_size in map_sizes:
    # Create environment and agent
    env = FrozenLakeEnv(size=map_size, is_slippery=False)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    n_runs = wandb.config.n_runs
    total_episodes = wandb.config.episodes
    # Define a prefix to group metrics by grid size
    prefix = f"Grid_{map_size}x{map_size}"  # consistent prefix as in notebook

    # Containers for rewards and steps
    all_rewards = []
    all_steps = []
    for run in range(n_runs):
        agent.reset()
        # track cumulative reward for this run
        cum_reward = [0]
        # callback for per-episode logging (receives ep, reward, error)
        def _log(ep, reward, error=0.0):
            cum_reward[0] += reward
            wandb.log({
                f"{prefix}/run": run,
                f"{prefix}/episode": ep,
                f"{prefix}/reward": reward,
                f"{prefix}/cum_reward": cum_reward[0],
                f"{prefix}/loss": error,
                f"{prefix}/epsilon": agent.epsilon,
            })
        # train (errors returned and passed into callback)
        rpe, spe, erre = agent.train(env, num_episodes=total_episodes, callback=_log)
        all_rewards.append(rpe)
        all_steps.append(spe)
        # record last episode video only for last run
        if run == n_runs - 1:
            video_dir = OUTPUT_DIR / "videos" / f"map_{map_size}"
            if video_dir.exists(): shutil.rmtree(video_dir)
            # ensure video directory exists before recording
            video_dir.mkdir(parents=True, exist_ok=True)
            # create a fresh Gymnasium FrozenLake-v1 for recording (ensures it's a Gymnasium Env)
            video_base = gym.make(
                "FrozenLake-v1",
                render_mode="rgb_array",
                desc=env.desc,
                is_slippery=False,
            )
            video_env = RecordVideo(video_base, str(video_dir), name_prefix=f"fl{map_size}")
            # unpack reset returning (obs, info)
            reset_result = video_env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            done = False
            while not done:
                action = agent.select_action(state)
                result = video_env.step(action)
                # unpack either 5-tuple or 4-tuple
                if len(result) == 5:
                    state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = result
            try: video_env.close()
            except: pass
            # log last episode video
            mp4s = list(video_dir.glob("*.mp4"))
            if mp4s:
                wandb.log({f"{prefix}/last_episode_video": wandb.Video(str(mp4s[0]), format="mp4")})

    # aggregate arrays
    rewards = np.array(all_rewards).T  # shape (episodes, runs)
    steps = np.array(all_steps).T
    episodes = np.arange(total_episodes)

    # Postprocess results
    df_rewards, df_steps = postprocess(episodes, rewards, steps, map_size)

    # Save and log learning curves
    img_path = OUTPUT_DIR / f"steps_rewards_{map_size}x{map_size}.png"
    plot_steps_and_rewards(df_rewards, df_steps, save_path=img_path)
    wandb.log({f"{prefix}/learning_curve": wandb.Image(str(img_path))})

    # Log average policy heatmap
    avg_qtable = np.mean([agent.q_table for _ in range(n_runs)], axis=0)
    heatmap_path = OUTPUT_DIR / f"heatmap_{map_size}x{map_size}.png"
    plot_q_values_map(avg_qtable, env, map_size, save_path=heatmap_path)
    wandb.log({f"{prefix}/heatmap": wandb.Image(str(heatmap_path))})

wandb.finish()