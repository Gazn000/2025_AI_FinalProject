import gym
import gym_cityflow
import numpy as np
from stable_baselines3 import DQN,PPO, A2C
from sb3_contrib import QRDQN
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
                if 'avg_waiting_time' in info and 'max_waiting_time' in info and 'throughput' and 'reward' in info:
                    self.records.append((
                        self.num_timesteps,
                        info['avg_waiting_time'],
                        info['max_waiting_time'],
                        info['throughput'],
                        info['reward']
                    ))
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.records, columns=["timesteps", "avg_waiting_time", "max_waiting_time", "throughput", "reward"])
        df.to_csv(self.log_path, index=False)
        if self.verbose > 0:
            print(f"Saved training logs to {self.log_path}")

if __name__ == "__main__":
    # learning rate 0.0625
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    
    log_interval = 10
    total_episodes = 101
    
    # dqn
    model = DQN("MlpPolicy", env, learning_rate=0.0000625, verbose=1)
    callback = Logger(
        log_freq=log_interval,
        log_path= "Self_result/DQN_625.csv",
        verbose=1
    )

    model.learn(
        total_timesteps=env.steps_per_episode * total_episodes,
        log_interval=log_interval,
        callback=callback
    )
    model.save("Self_result/dqn_1x1_625")
    model = DQN.load("Self_result/dqn_1x1_625")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_dqn_625.txt")
    obs = env.reset()

    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1

        if step_count > env.steps_per_episode:
            break
    '''   
    # ppo
    model = PPO("MlpPolicy", env, learning_rate=0.0000625,verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/ppo_log_625.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)
    model.save("Self_result/ppo_1x1_625")

    model = PPO.load("Self_result/ppo_1x1_625")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_ppo_625.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1

        if step_count > env.steps_per_episode:
            break

    # qr dqn
    model = QRDQN("MlpPolicy", env, learning_rate=0.0000625,verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/QR_DQN_log_625.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)

    model.save("Self_result/qrdqn_1x1_625")
    model = QRDQN.load("Self_result/qrdqn_1x1_625")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_qrdqn_625.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1

        if step_count > env.steps_per_episode:
            break
    
    # a2c
    model = A2C("MlpPolicy", env, learning_rate=0.0000625, verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/a2c_log_625.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)
    model.save("Self_result/a2c_1x1_625")
    model = A2C.load("Self_result/a2c_1x1_625")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_a2c_625.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1

        if step_count > env.steps_per_episode:
            break

    # learning rate 0.001
    # dqn
    model = DQN("MlpPolicy", env, learning_rate=0.001, verbose=1)
    callback = Logger(
        log_freq=log_interval,
        log_path= "Self_result/DQN_001.csv",
        verbose=1
    )

    model.learn(
        total_timesteps=env.steps_per_episode * total_episodes,
        log_interval=log_interval,
        callback=callback
    )
    model.save("Self_result/dqn_1x1_001")
    model = DQN.load("Self_result/dqn_1x1_001")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_dqn_001.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1
        if step_count > env.steps_per_episode:
            break
    # ppo
    model = PPO("MlpPolicy",env, learning_rate=0.001, verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/ppo_log_001.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)
    model.save("Self_result/ppo_1x1_001")

    model = PPO.load("Self_result/ppo_1x1_001")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_ppo_001.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1
        if step_count > env.steps_per_episode:
            break
    # qr dqn
    model = QRDQN("MlpPolicy", env, learning_rate=0.001,verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/QR_DQN_log_001.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)

    model.save("Self_result/qrdqn_1x1_001")
    model = QRDQN.load("Self_result/qrdqn_1x1_001")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_qrdqn_001.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1
        if step_count > env.steps_per_episode:
            break
    
    # a2c
    model = A2C("MlpPolicy", env, learning_rate=0.001, verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/a2c_log_001.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)
    model.save("Self_result/a2c_1x1_001")
    model = A2C.load("Self_result/a2c_1x1_001")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_a2c_001.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1
        if step_count > env.steps_per_episode:
            break
    # learning rate 0.0001
    # dqn
    model = DQN("MlpPolicy", env, learning_rate=0.0001, verbose=1)
    callback = Logger(
        log_freq=log_interval,
        log_path= "Self_result/DQN_0001.csv",
        verbose=1
    )

    model.learn(
        total_timesteps=env.steps_per_episode * total_episodes,
        log_interval=log_interval,
        callback=callback
    )
    model.save("Self_result/dqn_1x1_0001")
    model = DQN.load("Self_result/dqn_1x1_0001")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_dqn_0001.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1
        if step_count > env.steps_per_episode:
            break
    # ppo
    model = PPO("MlpPolicy", env, learning_rate=0.0001,verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/ppo_log_0001.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)
    model.save("Self_result/ppo_1x1_0001")

    model = PPO.load("Self_result/ppo_1x1_0001")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_ppo_0001.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1
        if step_count > env.steps_per_episode:
            break
    # qr dqn
    model = QRDQN("MlpPolicy", env, learning_rate=0.0001,verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/QR_DQN_log_0001.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)

    model.save("Self_result/qrdqn_1x1_0001")
    model = QRDQN.load("Self_result/qrdqn_1x1_0001")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_qrdqn_0001.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        step_count += 1
        if step_count > env.steps_per_episode:
            break
    
    # a2c
    model = A2C("MlpPolicy", env, learning_rate=0.0001, verbose=1)
    logger = Logger(log_freq=log_interval, log_path="Self_result/a2c_log_0001.csv", verbose=1)
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, callback=logger)
    model.save("Self_result/a2c_1x1_0001")
    model = A2C.load("Self_result/a2c_1x1_0001")
    env.set_save_replay(True)
    env.set_replay_path("gym_cityflow/gym_cityflow/envs/1x1_config/replay_a2c_0001.txt")
    obs = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        step_count += 1
        if step_count > env.steps_per_episode:
            break
    '''