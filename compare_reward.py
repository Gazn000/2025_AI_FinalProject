import gym
import gym_cityflow
import numpy as np
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

import csv
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

class Logger(BaseCallback):
    def __init__(self, log_freq, log_path, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_path = log_path
        self.records = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get('infos', [])
            if infos:
                info = infos[0]
                if 'avg_waiting_time' in info and 'max_waiting_time' in info and 'throughput' in info:
                    self.records.append((
                        self.num_timesteps,
                        info['avg_waiting_time'],
                        info['max_waiting_time'],
                        info['throughput']
                    ))
        return True

    def _on_training_end(self):
        df = pd.DataFrame(self.records, columns=["timesteps", "avg_waiting_time", "max_waiting_time", "throughput"])
        df.to_csv(self.log_path, index=False)
        if self.verbose > 0:
            print(f"Saved waiting time log to {self.log_path}")

def get_next_model_path(base_dir="Result", base_name="deepq_1x1"):
    i = 1
    while True:
        path = os.path.join(base_dir, f"{base_name}_{i}")
        if not os.path.exists(path):
            return path
        i += 1

if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    model = DQN("MlpPolicy", env, verbose=1)
    log_interval = 10
    total_episodes = 101
    #model.learn(total_timesteps=env.steps_per_episode*total_episodes, log_interval=log_interval)
    # model.save("Result/deepq_1x1")
    save_path = get_next_model_path()
    callback = Logger(
        log_freq=log_interval,
        log_path= "Result/DQN_reward.csv",
        verbose=1
    )

    model.learn(
        total_timesteps=env.steps_per_episode * total_episodes,
        log_interval=log_interval,
        callback=callback
    )
    model.save(save_path)


    model = DQN.load(save_path)
    env.set_save_replay(True)
    env.set_replay_path("/home/fanyi/AI/gym_cityflow/gym_cityflow/envs/1x1_config")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
