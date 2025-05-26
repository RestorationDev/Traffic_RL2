import cityflow
import numpy as np

class CityFlowEnv:
    def __init__(self, config_path, intersection_ids):
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_ids = intersection_ids  # list of strings
        self.action_space = [2] * len(intersection_ids)  # binary: 0 or 1
        self.step_time = 10

    def reset(self):
        self.eng.reset()
        return self.get_state()

    def get_state(self):
        state = []
        for inter_id in self.intersection_ids:
            lane_vehicles = self.eng.get_lane_vehicle_count()
            # You can also use pressure, queue length, etc.
            inter_state = self._extract_intersection_state(inter_id, lane_vehicles)
            state.append(inter_state)
        return np.array(state)

    def _extract_intersection_state(self, inter_id, lane_vehicles):
        # Simple: sum vehicles on all incoming lanes
        lanes = self.eng.get_intersection_lane_ids(inter_id)["incoming"]
        return [lane_vehicles[lane] for lane in lanes]

    def step(self, actions):
        # Apply actions to all intersections
        for inter_id, action in zip(self.intersection_ids, actions):
            self.eng.set_tl_phase(inter_id, action)
        self.eng.next_step(self.step_time)

        reward = self._compute_reward()
        state = self.get_state()
        done = self.eng.get_current_time() >= 3600  # 1 hour sim
        return state, reward, done, {}

    def _compute_reward(self):
        total_waiting = sum(self.eng.get_lane_waiting_vehicle_count().values())
        return -total_waiting