from gym.envs.registration import register

# source: https://github.com/openai/gym/blob/master/gym/envs/__init__.py

register(
    id='CityFlow-1x1-LowTraffic-v0', # use id to pass to gym.make(id)
    entry_point='gym_cityflow.envs:CityFlow_1x1_LowTraffic'
    # max_episode_steps =
    # reward_threshold =
)

register(
    id='CityFlow-2x2-LowTraffic-v0',
    entry_point='gym_cityflow.envs:CityFlow_2x2_LowTraffic',
)


register(
    id='CityFlow-1x3-LowTraffic-v0',
    entry_point='gym_cityflow.envs:CityFlow_1x3_LowTraffic',
)