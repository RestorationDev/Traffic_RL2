import cityflow
import numpy as np
import json
import os
from typing import Optional


def _resolve_cityflow_file(config_abs: str, ref: str) -> str:
    """Resolve a path from CityFlow config (relative to cwd or config directory)."""
    if os.path.isabs(ref):
        return ref
    ref = ref.replace("\\", "/").lstrip("./")
    for root in (os.getcwd(), os.path.dirname(config_abs)):
        cand = os.path.normpath(os.path.join(root, ref))
        if os.path.isfile(cand):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), ref))


class CityFlowEnvExp:
    """
    Experimental CityFlow env wrapper (keeps envs/intersection_env.py unchanged).

    Differences vs CityFlowEnv:
    - configurable episode horizon via max_sim_time
    - configurable engine thread_num
    - same state interface: reset() -> (n_inters, state_dim), step(actions) -> (state, reward, done, info)
    """

    def __init__(
        self,
        config_path: str,
        intersection_ids,
        reward_fn=None,
        *,
        step_time: int = 10,
        max_sim_time: int = 10000,
        thread_num: int = 1,
        incoming_lanes_path: Optional[str] = None,
        roadnet_path: Optional[str] = None,
    ):
        config_abs = os.path.abspath(config_path)
        cfg_dir = os.path.dirname(config_abs)
        with open(config_abs, "r") as cf:
            cityflow_cfg = json.load(cf)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if incoming_lanes_path is None:
            cand = os.path.join(cfg_dir, "incoming_lanes.json")
            if os.path.isfile(cand):
                incoming_lanes_path = cand
            else:
                incoming_lanes_path = os.path.join(base_dir, "..", "mydata", "incoming_lanes.json")
        if roadnet_path is None:
            rf = cityflow_cfg.get("roadnetFile", "roadnet.json")
            roadnet_path = _resolve_cityflow_file(config_abs, rf)
            if not os.path.isfile(roadnet_path):
                roadnet_path = os.path.join(base_dir, "..", "mydata", "roadnet.json")

        self.eng = cityflow.Engine(config_path, thread_num=thread_num)
        self.intersection_ids = list(intersection_ids)
        self.action_space = [9] * len(self.intersection_ids)
        self.step_time = int(step_time)
        self.max_sim_time = int(max_sim_time) if max_sim_time is not None else 10000

        self.prev_phase = None
        self.curr_phase = None
        self.prev_total_travel_time = 0
        self.vehicle_entry_time = {}
        self.total_travel_time = 0.0
        self.prev_vehicle_ids = set()
        self.total_exited = 0

        # Reward function override
        self.reward_fn = reward_fn if reward_fn else self._compute_reward

        # Load incoming lane info
        with open(incoming_lanes_path, "r") as f:
            self.incoming_lanes = json.load(f)

        with open(roadnet_path, "r") as f:
            self.roadnet_data = json.load(f)

    def reset(self):
        self.eng.reset()
        return self.get_state()

    def get_state(self):
        state = []
        lane_vehicles = self.eng.get_lane_vehicle_count()
        for inter_id in self.intersection_ids:
            lanes = self.incoming_lanes[inter_id]
            state.append([lane_vehicles.get(lane, 0) for lane in lanes])
        return np.array(state)

    def set_rwd_fn(self, fn):
        self.reward_fn = fn

    def step(self, actions):
        for inter_id, action in zip(self.intersection_ids, actions):
            self.eng.set_tl_phase(inter_id, int(action))

        for _ in range(self.step_time):
            self.eng.next_step()

        self.update_exited_vehicle_count()
        reward = self.reward_fn()
        state = self.get_state()
        done = self.eng.get_current_time() >= self.max_sim_time
        return state, reward, done, {}

    def update_exited_vehicle_count(self):
        current_vehicle_ids = set(self.eng.get_vehicles(include_waiting=True))
        exited = self.prev_vehicle_ids - current_vehicle_ids
        self.total_exited += len(exited)
        self.prev_vehicle_ids = current_vehicle_ids
        return self.total_exited

    # --- Reward helpers (mirrors intersection_env.py API) ---
    def _compute_reward(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicles = self.eng.get_lane_vehicles()
        passed_vehicles = sum(len(v) for k, v in lane_vehicles.items() if k.startswith("out"))
        total_delay = sum(lane_waiting.values())
        queue_std = np.std(list(lane_waiting.values())) if lane_waiting else 0.0
        num_stopped = sum(1 for v in lane_waiting.values() if v > 0)

        switch_penalty = 1 if self.prev_phase != self.curr_phase else 0

        w_passed = 0.5
        w_delay = 0.1
        w_fairness = 5.0
        w_switch = 1.0
        w_emission = 0.05

        reward = (
            w_passed * passed_vehicles
            - w_delay * total_delay
            - w_fairness * queue_std
            - w_switch * switch_penalty
            - w_emission * num_stopped
        )

        self.prev_phase = self.curr_phase
        return float(reward)

    def default_reward(self):
        return self._compute_reward()

    def pressure_only_reward(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        return -float(sum(lane_waiting.values()))

    def pressure_and_count_reward(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_count = self.eng.get_lane_vehicle_count()
        pressure = -float(sum(lane_waiting.values()))
        count = -float(sum(lane_count.values()))
        return pressure + 0.1 * count

