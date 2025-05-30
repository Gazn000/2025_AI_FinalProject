import gym
import gym_cityflow  # Make sure your CityFlow environment is registered
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    # Create the CityFlow traffic environment
    env = gym.make("gym_cityflow:CityFlow-1x1-LowTraffic-v0")
    env = Monitor(env)  # Wrap with Monitor to record training statistics

    # Initialize the QRDQN model with MLP policy
    model = QRDQN("MlpPolicy", env, verbose=1)

    # Training settings
    log_interval = 10
    total_episodes = 100
    steps_per_episode = env.unwrapped.steps_per_episode  # Custom field in your env

    # Start training the model
    model.learn(
        total_timesteps=steps_per_episode * total_episodes,
        log_interval=log_interval
    )

    # Save the trained model
    model.save("qrdqn_1x1")

    # Load the model for inference
    model = QRDQN.load("qrdqn_1x1")

    # Run the trained model in an infinite loop for testing
    obs = env.reset()
    while True:
        # Predict the best action deterministically
        action, _ = model.predict(obs, deterministic=True)

        # Apply the action to the environment
        obs, reward, done, truncated, info = env.step(action)

        # If episode ends, reset the environment
        if done or truncated:
            obs = env.reset()
