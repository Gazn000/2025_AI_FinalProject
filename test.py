import gym
import gym_cityflow
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    model = DQN("MlpPolicy", env, verbose=1)

    total_timesteps = 10000
    model.learn(total_timesteps=total_timesteps)
    model.save("dqn_cityflow")

    model = DQN.load("dqn_cityflow")

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
