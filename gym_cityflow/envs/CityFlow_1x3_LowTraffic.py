import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import cityflow
import numpy as np
import os

class CityFlow_1x3_LowTraffic(gym.Env):
    """
    Description:
        A single intersection with low traffic.
        8 roads, 1 intersection (plus 4 virtual intersections).

    State:
        Type: array[20 * 3]
        The number of vehicless and waiting vehicles on each lane.

    Actions:
        Type: Discrete(9 ** 3)
        index of one of 9 light phases.

        Note:
            Below is a snippet from "roadnet.json" file which defines lightphases for "intersection_1_1".

            "lightphases": [
              {"time": 5, "availableRoadLinks": []},
              {"time": 30, "availableRoadLinks": [ 0, 4 ] },
              {"time": 30, "availableRoadLinks": [ 2, 7 ] },
              {"time": 30, "availableRoadLinks": [ 1, 5 ] },
              {"time": 30,"availableRoadLinks": [3,6]},
              {"time": 30,"availableRoadLinks": [0,1]},
              {"time": 30,"availableRoadLinks": [4,5]},
              {"time": 30,"availableRoadLinks": [2,3]},
              {"time": 30,"availableRoadLinks": [6,7]}]

    Reward:
        The total amount of time -- in seconds -- that all the vehicles in the intersection
        waitied for.

        Todo: as a way to ensure fairness -- i.e. not a single lane gets green lights for too long --
        instead of simply summing up the waiting time, we could weigh the waiting time of each car by how
        much it had waited so far.
    """

    metadata = {'render.modes':['human']}
    def __init__(self):
        #super(CityFlow_1x1_LowTraffic, self).__init__()
        # hardcoded settings from "config.json" file
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1x3_config")
        self.cityflow = cityflow.Engine(os.path.join(self.config_dir, "config.json"), thread_num=1)
        self.intersection_id = \
        [
            "intersection_1_1",
            "intersection_2_1",
            "intersection_3_1"
        ]

        self.vehicle_waiting_time = {}
        self.sec_per_step = 1.0

        self.steps_per_episode = 200
        self.current_step = 0
        self.is_done = False
        self.reward_range = (-float('inf'), float('inf'))
        self.start_lane_ids = \
            [
            "road_0_1_0_0",
            "road_0_1_0_1",
            "road_0_1_0_2",
            "road_1_0_1_0",
            "road_1_0_1_1",
            "road_1_0_1_2",
            "road_1_2_3_0",
            "road_1_2_3_1",
            "road_1_2_3_2",
            "road_2_2_3_0",
            "road_2_2_3_1",
            "road_2_2_3_2",
            "road_2_0_1_0",
            "road_2_0_1_1",
            "road_2_0_1_2",
            "road_3_2_3_0",
            "road_3_2_3_1",
            "road_3_2_3_2",
            "road_3_0_1_0",
            "road_3_0_1_1",
            "road_3_0_1_2",
            "road_4_1_2_0",
            "road_4_1_2_1",
            "road_4_1_2_2"
            ]


        self.all_lane_ids = \
           [
            "road_0_1_0_0",
            "road_0_1_0_1",
            "road_0_1_0_2",
            "road_1_0_1_0",
            "road_1_0_1_1",
            "road_1_0_1_2",
            "road_2_1_2_0",
            "road_2_1_2_1",
            "road_2_1_2_2",
            "road_1_2_3_0",
            "road_1_2_3_1",
            "road_1_2_3_2",
            "road_1_1_0_0",
            "road_1_1_0_1",
            "road_1_1_0_2",
            "road_1_1_1_0",
            "road_1_1_1_1",
            "road_1_1_1_2",
            "road_1_1_2_0",
            "road_1_1_2_1",
            "road_1_1_2_2",
            "road_1_1_3_0",
            "road_1_1_3_1",
            "road_1_1_3_2",
            "road_2_1_3_0",
            "road_2_1_3_1",
            "road_2_1_3_2",
            "road_2_0_1_0",
            "road_2_0_1_1",
            "road_2_0_1_2",
            "road_3_1_2_0",
            "road_3_1_2_1",
            "road_3_1_2_2",
            "road_2_1_0_0",
            "road_2_1_0_1",
            "road_2_1_0_2",
            "road_2_1_1_0",
            "road_2_1_1_1",
            "road_2_1_1_2",
            "road_2_2_3_0",
            "road_2_2_3_1",
            "road_2_2_3_2",
            "road_3_1_3_0",
            "road_3_1_3_1",
            "road_3_1_3_2",
            "road_3_0_1_0",
            "road_3_0_1_1",
            "road_3_0_1_2",
            "road_3_1_0_0",
            "road_3_1_0_1",
            "road_3_1_0_2",
            "road_4_1_2_0",
            "road_4_1_2_1",
            "road_4_1_2_2",
            "road_3_1_1_0",
            "road_3_1_1_1",
            "road_3_1_1_2",
            "road_3_2_3_0",
            "road_3_2_3_1",
            "road_3_2_3_2"
            ]


        """
        road id:
        ["road_0_1_0",
         "road_1_0_1",
         "road_2_1_2",
         "road_1_2_3",
         "road_1_1_0",
         "road_1_1_1",
         "road_1_1_2",
         "road_1_1_3",
         "road_2_1_3",
         "road_2_0_1",
         "road_3_1_2",
         "road_2_1_0",
         "road_2_1_1",
         "road_2_2_3",
         "road_3_1_3",
         "road_3_0_1",
         "road_3_1_0",
         "road_4_1_2",
         "road_3_1_1",
         "road_3_2_3"]
         
        start road id:
        ["road_0_1_0",
        "road_1_0_1",
        "road_1_2_3",
        "road_2_2_3",
        "road_2_0_1",
        "road_3_2_3",
        "road_3_0_1",
        "road_4_1_2"]
        
        lane id:
        [
        "road_0_1_0_0",
        "road_0_1_0_1",
        "road_0_1_0_2",
        "road_1_0_1_0",
        "road_1_0_1_1",
        "road_1_0_1_2",
        "road_2_1_2_0",
        "road_2_1_2_1",
        "road_2_1_2_2",
        "road_1_2_3_0",
        "road_1_2_3_1",
        "road_1_2_3_2",
        "road_1_1_0_0",
        "road_1_1_0_1",
        "road_1_1_0_2",
        "road_1_1_1_0",
        "road_1_1_1_1",
        "road_1_1_1_2",
        "road_1_1_2_0",
        "road_1_1_2_1",
        "road_1_1_2_2",
        "road_1_1_3_0",
        "road_1_1_3_1",
        "road_1_1_3_2",
        "road_2_1_3_0",
        "road_2_1_3_1",
        "road_2_1_3_2",
        "road_2_0_1_0",
        "road_2_0_1_1",
        "road_2_0_1_2",
        "road_3_1_2_0",
        "road_3_1_2_1",
        "road_3_1_2_2",
        "road_2_1_0_0",
        "road_2_1_0_1",
        "road_2_1_0_2",
        "road_2_1_1_0",
        "road_2_1_1_1",
        "road_2_1_1_2",
        "road_2_2_3_0",
        "road_2_2_3_1",
        "road_2_2_3_2",
        "road_3_1_3_0",
        "road_3_1_3_1",
        "road_3_1_3_2",
        "road_3_0_1_0",
        "road_3_0_1_1",
        "road_3_0_1_2",
        "road_3_1_0_0",
        "road_3_1_0_1",
        "road_3_1_0_2",
        "road_4_1_2_0",
        "road_4_1_2_1",
        "road_4_1_2_2",
        "road_3_1_1_0",
        "road_3_1_1_1",
        "road_3_1_1_2",
        "road_3_2_3_0",
        "road_3_2_3_1",
        "road_3_2_3_2"
        ]

         
         start lane id:
         [
        "road_0_1_0_0",
        "road_0_1_0_1",
        "road_0_1_0_2",
        "road_1_0_1_0",
        "road_1_0_1_1",
        "road_1_0_1_2",
        "road_1_2_3_0",
        "road_1_2_3_1",
        "road_1_2_3_2",
        "road_2_2_3_0",
        "road_2_2_3_1",
        "road_2_2_3_2",
        "road_2_0_1_0",
        "road_2_0_1_1",
        "road_2_0_1_2",
        "road_3_2_3_0",
        "road_3_2_3_1",
        "road_3_2_3_2",
        "road_3_0_1_0",
        "road_3_0_1_1",
        "road_3_0_1_2",
        "road_4_1_2_0",
        "road_4_1_2_1",
        "road_4_1_2_2"
        ]

        """

        self.mode = "start_waiting"
        assert self.mode == "all_all" or self.mode == "start_waiting", "mode must be one of 'all_all' or 'start_waiting'"
        """
        `mode` variable changes both reward and state.
        
        "all_all":
            - state: waiting & running vehicle count from all lanes (incoming & outgoing)
            - reward: waiting vehicle count from all lanes
            
        "start_waiting" - 
            - state: only waiting vehicle count from only start lanes (only incoming)
            - reward: waiting vehicle count from start lanes
        """
        """
        if self.mode == "all_all":
            self.state_space = len(self.all_lane_ids) * 2

        if self.mode == "start_waiting":
            self.state_space = len(self.start_lane_ids)
        """
        
        self.action_space = spaces.Discrete(9 ** 3)
        if self.mode == "all_all":
            self.observation_space = spaces.MultiDiscrete([100]*96)
        else:
            self.observation_space = spaces.MultiDiscrete([100]*24)

    def decode_action(self, action):
        num_intersections = len(self.intersection_id)
        num_phases = 9
        phases = []
        for _ in range(num_intersections):
            phases.append(action % num_phases)
            action //= num_phases
        phases.reverse()
        return phases


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        phases = self.decode_action(action)
        for i, inter_id in enumerate(self.intersection_id):
            self.cityflow.set_tl_phase(inter_id, phases[i])

        self.cityflow.next_step()

        state = self._get_state()
        self._update_waiting_time()
        reward, info = self._get_reward()

        self.current_step += 1

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                        "You should always call 'reset()' once you receive 'done = True' "
                        "-- any further steps are undefined behavior.")
            reward = 0.0

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        return state, reward, self.is_done, info


    def reset(self):
        self.cityflow.reset()
        self.is_done = False
        self.current_step = 0

        self.vehicle_waiting_time = {}
        return self._get_state()

    def render(self, mode='human'):
        print("Current time: " + self.cityflow.get_current_time())

    def _get_state(self):
        lane_vehicles_dict = self.cityflow.get_lane_vehicle_count()
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()

        state = None

        if self.mode=="all_all":
            state = np.zeros(len(self.all_lane_ids) * 2, dtype=np.float32)
            for i in range(len(self.all_lane_ids)):
                state[i*2] = lane_vehicles_dict[self.all_lane_ids[i]]
                state[i*2 + 1] = lane_waiting_vehicles_dict[self.all_lane_ids[i]]

        if self.mode=="start_waiting":
            state = np.zeros(len(self.start_lane_ids), dtype=np.float32)
            for i in range(len(self.start_lane_ids)):
                state[i] = lane_waiting_vehicles_dict[self.start_lane_ids[i]]

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
    '''
    def _get_reward(self):
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()
        reward = 0.0


        if self.mode == "all_all":
            for (road_id, num_vehicles) in lane_waiting_vehicles_dict.items():
                if road_id in self.all_lane_ids:
                    reward -= self.sec_per_step * num_vehicles

        if self.mode == "start_waiting":
            for (road_id, num_vehicles) in lane_waiting_vehicles_dict.items():
                if road_id in self.start_lane_ids:
                    reward -= self.sec_per_step * num_vehicles

        return reward
    '''
    
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
                    total_weighted_waiting_time += waiting_time **2
                    num_waiting_vehicles += 1
                    max_waiting_time = max(max_waiting_time, waiting_time)
        if self.mode == "start_waiting":
            target_roads = set(lane_to_road[lane_id] for lane_id in self.start_lane_ids)
            for vehicle_id, waiting_time in self.vehicle_waiting_time.items():
                vehicle_info = self.cityflow.get_vehicle_info(vehicle_id)
                current_road = vehicle_info.get("road", None)
                if current_road in target_roads:
                    total_weighted_waiting_time += waiting_time **2
                    num_waiting_vehicles += 1
                    max_waiting_time = max(max_waiting_time, waiting_time)
        reward = -total_weighted_waiting_time / 10000.0
        avg_waiting_time = (sum(self.vehicle_waiting_time.values()) / num_waiting_vehicles) if num_waiting_vehicles > 0 else 0
        throughput = self.cityflow.get_vehicle_count() if hasattr(self.cityflow, "get_vehicle_count") else 0
        info = {
            "avg_waiting_time": avg_waiting_time,
            "max_waiting_time": max_waiting_time,
            "throughput": throughput,
            "reward": reward
        }
        return reward, info
    '''
    def _get_reward(self):
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()
        reward = 0.0

        if self.mode == "all_all":
            for (road_id, num_vehicles) in lane_waiting_vehicles_dict.items():
                if road_id in self.all_lane_ids:
                    reward -= self.sec_per_step * num_vehicles
            total_waiting = sum(num for road, num in lane_waiting_vehicles_dict.items() if road in self.all_lane_ids)
        elif self.mode == "start_waiting":
            for (road_id, num_vehicles) in lane_waiting_vehicles_dict.items():
                if road_id in self.start_lane_ids:
                    reward -= self.sec_per_step * num_vehicles

            total_waiting = sum(num for road, num in lane_waiting_vehicles_dict.items() if road in self.start_lane_ids)
        else:
            total_waiting = 0

        max_waiting_time = 0.0  
        throughput = self.cityflow.get_vehicle_count() if hasattr(self.cityflow, "get_vehicle_count") else 0
        avg_waiting_time = total_waiting

        info = {
            "avg_waiting_time": avg_waiting_time,
            "max_waiting_time": max_waiting_time,
            "throughput": throughput
        }

        return reward, info
    '''
    def set_replay_path(self, path):
        self.cityflow.set_replay_file(path)

    def seed(self, seed=None):
        self.cityflow.set_random_seed(seed)

    def get_path_to_config(self):
        return self.config_dir

    def set_save_replay(self, save_replay):
        self.cityflow.set_save_replay(save_replay)