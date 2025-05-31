import gym
import gym_cityflow
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    # 建立環境，並用 DummyVecEnv 包裝
    def make_env():
        return gym.make('gym_cityflow:CityFlow-1x3-LowTraffic-v0')

    env = DummyVecEnv([make_env])

    model = DQN("MlpPolicy", env, verbose=1)
    log_interval = 10
    total_episodes = 100

    # 總訓練步數，假設 env.envs[0].steps_per_episode 可取得
    total_timesteps = env.envs[0].steps_per_episode * total_episodes

    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    model.save("deepq_1x1")

    # 載入模型
    model = DQN.load("deepq_1x1")

    # 載入後設定環境
    model.set_env(env)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
