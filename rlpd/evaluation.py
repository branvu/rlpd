from typing import Dict

import gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=2)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    steps, episode, total_reward, episode_success = 0, 0, 0, 0.0
    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
            info = env.unwrapped.get_info()
            if "sort_success" in info and info['sort_success']:
                episode_success += 1
            steps += 1
            
    return {"success": episode_success / num_episodes, "avg ep length": steps / num_episodes}
    #return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
