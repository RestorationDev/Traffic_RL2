import cityflow
import numpy as np
import json
import os

class CityFlowEnv:
    def __init__(self, config_path, intersection_ids):
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_ids = intersection_ids  # list of strings
        self.action_space = [9] * len(intersection_ids)  # binary: 0 or 1
        self.step_time = 10

        # Load incoming lane info from JSON
        base_dir = os.path.dirname(os.path.abspath(__file__))  # path to envs/
        json_path = os.path.join(base_dir, "..", "mydata", "incoming_lanes.json")

        with open(json_path, "r") as f:
            self.incoming_lanes = json.load(f)

    def reset(self):
        self.eng.reset()
        return self.get_state()

    def get_state(self):
        state = []
        lane_vehicles = self.eng.get_lane_vehicle_count()  # query once per step
        for inter_id in self.intersection_ids:
            inter_state = self._extract_intersection_state(inter_id, lane_vehicles)
            state.append(inter_state)
        return np.array(state)

    def _extract_intersection_state(self, inter_id, lane_vehicles):
        # Use precomputed lane info from incoming_lanes.json
        lanes = self.incoming_lanes[inter_id]
        return [lane_vehicles.get(lane, 0) for lane in lanes]

    def step(self, actions):
        for inter_id, action in zip(self.intersection_ids, actions):
            self.eng.set_tl_phase(inter_id, action)
        
        for _ in range(self.step_time):
            self.eng.next_step()

        reward = self._compute_reward()
        state = self.get_state()
        done = self.eng.get_current_time() >= 3600  # 1 hour sim
        return state, reward, done, {}

    def _compute_reward(self):
        total_waiting = sum(self.eng.get_lane_waiting_vehicle_count().values())
        return -total_waiting
