import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import cityflow
import numpy as np
import os

class CityFlow_1x1_LowTraffic(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self):
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1x1_config")
        self.cityflow = cityflow.Engine(os.path.join(self.config_dir, "config.json"), thread_num=1)
        self.intersection_id = "intersection_1_1"

        self.vehicle_waiting_time = {}
        self.sec_per_step = 1.0

        self.steps_per_episode = 100
        self.current_step = 0
        self.is_done = False
        self.reward_range = (-float('inf'), float('inf'))
        self.start_lane_ids = [
             "road_0_1_0_0", "road_0_1_0_1",
             "road_1_0_1_0", "road_1_0_1_1",
             "road_2_1_2_0", "road_2_1_2_1",
             "road_1_2_3_0", "road_1_2_3_1"
        ]

        self.all_lane_ids = [
             "road_0_1_0_0", "road_0_1_0_1",
             "road_1_0_1_0", "road_1_0_1_1",
             "road_2_1_2_0", "road_2_1_2_1",
             "road_1_2_3_0", "road_1_2_3_1",
             "road_1_1_0_0", "road_1_1_0_1",
             "road_1_1_1_0", "road_1_1_1_1",
             "road_1_1_2_0", "road_1_1_2_1",
             "road_1_1_3_0", "road_1_1_3_1"
        ]

        self.mode = "start_waiting"
        assert self.mode in ["all_all", "start_waiting"], "mode must be one of 'all_all' or 'start_waiting'"

        self.action_space = spaces.Discrete(9)
        if self.mode == "all_all":
            self.observation_space = spaces.MultiDiscrete([100]*16)
        else:
            self.observation_space = spaces.MultiDiscrete([100]*8)

        self.prev_speeds = {} # used for acceleration calculation
        self.comfort_lambda = 0.1 # Comfort penalty weight

    def step(self, action):
        assert self.action_space.contains(action), f"{action} invalid ({type(action)})"
        self.cityflow.set_tl_phase(self.intersection_id, action)
        self.cityflow.next_step()

        state = self._get_state()
        self._update_waiting_time()
        reward, info = self._get_reward()

        self.current_step += 1

        if self.is_done:
            logger.warn("Calling 'step()' after done=True; call 'reset()' first.")
            reward = 0.0

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        return state, reward, self.is_done, info

    def reset(self):
        self.cityflow.reset()
        self.is_done = False
        self.current_step = 0

        self.vehicle_waiting_time = {}
        self.prev_speeds.clear()  # reset previous speeds
        return self._get_state()

    def _get_state(self):
        lane_vehicles_dict = self.cityflow.get_lane_vehicle_count()
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()

        if self.mode=="all_all":
            state = np.zeros(len(self.all_lane_ids) * 2, dtype=np.float32)
            for i in range(len(self.all_lane_ids)):
                state[i*2] = lane_vehicles_dict.get(self.all_lane_ids[i], 0)
                state[i*2 + 1] = lane_waiting_vehicles_dict.get(self.all_lane_ids[i], 0)
            return state

        if self.mode=="start_waiting":
            state = np.zeros(len(self.start_lane_ids), dtype=np.float32)
            for i in range(len(self.start_lane_ids)):
                state[i] = lane_waiting_vehicles_dict.get(self.start_lane_ids[i], 0)
            return state

    def _update_waiting_time(self):
        all_lanes_vehicles = self.cityflow.get_lane_vehicles()
        vehicle_speeds = self.cityflow.get_vehicle_speed()
        waiting_vehicles_current = set()

        for lane_id, vehicle_ids in all_lanes_vehicles.items():
            for v_id in vehicle_ids:
                speed = float(vehicle_speeds.get(v_id, 0))
                if speed < 0.1:
                    waiting_vehicles_current.add(v_id)

        for v_id in waiting_vehicles_current:
            if v_id in self.vehicle_waiting_time:
                self.vehicle_waiting_time[v_id] += self.sec_per_step
            else:
                self.vehicle_waiting_time[v_id] = self.sec_per_step

        for v_id in list(self.vehicle_waiting_time.keys()):
            if v_id not in waiting_vehicles_current:
                del self.vehicle_waiting_time[v_id]

    def _get_reward(self):
        total_weighted_waiting_time = 0.0
        num_waiting_vehicles = 0
        max_waiting_time = 0.0
        lane_to_road = {lane_id: "_".join(lane_id.split("_")[:-1]) for lane_id in self.all_lane_ids}

        if self.mode == "all_all":
            target_roads = set(lane_to_road[lane_id] for lane_id in self.all_lane_ids)
            for vehicle_id, waiting_time in self.vehicle_waiting_time.items():
                vehicle_info = self.cityflow.get_vehicle_info(vehicle_id)
                current_road = vehicle_info.get("road", None)
                if current_road in target_roads:
                    total_weighted_waiting_time += waiting_time ** 2
                    num_waiting_vehicles += 1
                    max_waiting_time = max(max_waiting_time, waiting_time)
        elif self.mode == "start_waiting":
            target_roads = set(lane_to_road[lane_id] for lane_id in self.start_lane_ids)
            for vehicle_id, waiting_time in self.vehicle_waiting_time.items():
                vehicle_info = self.cityflow.get_vehicle_info(vehicle_id)
                current_road = vehicle_info.get("road", None)
                if current_road in target_roads:
                    total_weighted_waiting_time += waiting_time ** 2
                    num_waiting_vehicles += 1
                    max_waiting_time = max(max_waiting_time, waiting_time)

        # Calculate acceleration penalty
        vehicle_ids = self.cityflow.get_vehicles()
        acc_penalty = 0.0
        time_step = self.sec_per_step

        for vid in vehicle_ids:
            curr_speed = self.cityflow.get_vehicle_speed(vid)
            prev_speed = self.prev_speeds.get(vid, curr_speed)  # avoid zero division
            acceleration = (curr_speed - prev_speed) / time_step
            acc_penalty += acceleration ** 2
            self.prev_speeds[vid] = curr_speed

        # Caculate total reward, comfort penalty weight can be adjusted
        reward = -total_weighted_waiting_time / 10000.0 - self.comfort_lambda * acc_penalty

        avg_waiting_time = (sum(self.vehicle_waiting_time.values()) / num_waiting_vehicles) if num_waiting_vehicles > 0 else 0
        throughput = self.cityflow.get_vehicle_count() if hasattr(self.cityflow, "get_vehicle_count") else 0
        info = {
            "avg_waiting_time": avg_waiting_time,
            "max_waiting_time": max_waiting_time,
            "throughput": throughput,
            "acc_penalty": acc_penalty
        }

        return reward, info

    def set_replay_path(self, path):
        self.cityflow.set_replay_file(path)

    def seed(self, seed=None):
        self.cityflow.set_random_seed(seed)

    def get_path_to_config(self):
        return self.config_dir

    def set_save_replay(self, save_replay):
        self.cityflow.set_save_replay(save_replay)
