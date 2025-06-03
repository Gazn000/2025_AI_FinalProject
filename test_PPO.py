import gym
import gym_cityflow
import numpy as np
#from stable_baselines3.deepq.policies import MlpPolicy
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
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

if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    model = PPO("MlpPolicy", env, verbose=1)
    log_interval = 10
    total_episodes = 101
    logger = Logger(log_freq=log_interval, log_path="ppo_log.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)
    model.save("ppo_1x1_625")

    model = PPO.load("ppo_1x1_625")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
