import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cityflow
import numpy as np


@dataclass
class Env1x3Config:
    config_path: str = "./mydata_1x3/config_rl.json"  # rlTrafficLight=true
    incoming_lanes_path: str = "./mydata_1x3/incoming_lanes.json"
    intersection_ids: Tuple[str, str, str] = ("intersection_1_1", "intersection_2_1", "intersection_3_1")
    action_interval_s: int = 10          # decide every 10 simulated seconds
    min_green_s: int = 20                # ignore switches until >= 20s in phase
    demand_end_s: int = 3600             # after this, no new vehicles should enter (dataset is ~1h)
    max_sim_time_s: int = 8400           # generous hard cap (episode horizon) in simulated seconds
    fairness_lambda: float = 0.2         # weight on std(queue_per_intersection)


class CityFlowEnv1x3:
    """
    1x3 corridor env (3 intersections) with keep/switch control.

    - Engine must be created with rlTrafficLight=true in config (use config_rl.json).
    - Action per intersection: 0=keep, 1=switch_to_next_phase (phase = (phase+1)%9).
    - Decision interval: action_interval_s CityFlow steps per env.step().
    - Min-green: min_green_s seconds before allowing a switch.
    - Observation per intersection:
        [waiting_count for each incoming lane] + onehot(phase_idx,9) + [time_in_phase_s]
      Then concatenated over intersections in fixed order intersection_ids.
    - Reward:
        r = - total_queue - fairness_lambda * std(queue_sum_per_intersection)
      computed at the *end* of each env.step() using lane waiting counts.
    """

    def __init__(self, cfg: Env1x3Config):
        self.cfg = cfg
        self.eng = cityflow.Engine(cfg.config_path, thread_num=1)

        self.intersection_ids = list(cfg.intersection_ids)
        self.n_inters = len(self.intersection_ids)
        self.n_phases = 9

        # Load incoming lanes mapping
        self.incoming_lanes: Dict[str, List[str]] = self._load_incoming_lanes(cfg.incoming_lanes_path)
        for iid in self.intersection_ids:
            if iid not in self.incoming_lanes:
                raise KeyError(f"Intersection {iid} missing from incoming_lanes.json")

        # Internal phase tracking (CityFlow Python binding lacks get_tl_phase)
        self.phase_idx: Dict[str, int] = {iid: 0 for iid in self.intersection_ids}
        self.time_in_phase_s: Dict[str, int] = {iid: 0 for iid in self.intersection_ids}

        # Observation sizing
        self.per_inter_lane_dim = len(self.incoming_lanes[self.intersection_ids[0]])
        if any(len(self.incoming_lanes[iid]) != self.per_inter_lane_dim for iid in self.intersection_ids):
            raise ValueError("All intersections must have same incoming lane count for a fixed-size obs.")
        self.per_inter_obs_dim = self.per_inter_lane_dim + self.n_phases + 1
        self.obs_dim = self.per_inter_obs_dim * self.n_inters

    def _load_incoming_lanes(self, path: str) -> Dict[str, List[str]]:
        # Resolve relative to repo root (cwd) if needed
        p = path
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(os.getcwd(), p))
        with open(p, "r") as f:
            return json.load(f)

    def reset(self) -> np.ndarray:
        self.eng.reset()
        for iid in self.intersection_ids:
            self.phase_idx[iid] = 0
            self.time_in_phase_s[iid] = 0
            # Ensure engine starts at phase 0 in RL mode
            self.eng.set_tl_phase(iid, 0)
        return self._get_obs()

    def set_replay(self, replay_log_file: Optional[str]) -> None:
        """
        Enable/disable replay logging.

        CityFlow prepends config 'dir' to the replay path, so this should be a relative path
        like './mydata_1x3/replays/ppo/replay_ep10.txt' (NOT '/app/...').
        """
        if not replay_log_file:
            self.eng.set_save_replay(False)
            return
        self.eng.set_save_replay(True)
        self.eng.set_replay_file(replay_log_file)

    def _get_obs(self) -> np.ndarray:
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        chunks = []
        for iid in self.intersection_ids:
            lanes = self.incoming_lanes[iid]
            q = np.array([lane_waiting.get(l, 0) for l in lanes], dtype=np.float32)

            onehot = np.zeros((self.n_phases,), dtype=np.float32)
            onehot[int(self.phase_idx[iid])] = 1.0

            tip = np.array([float(self.time_in_phase_s[iid])], dtype=np.float32)
            chunks.append(np.concatenate([q, onehot, tip], axis=0))
        return np.concatenate(chunks, axis=0)

    def _apply_action(self, actions: List[int]) -> None:
        # actions: list length 3 of {0,1}
        for iid, a in zip(self.intersection_ids, actions):
            a = int(a)
            if a not in (0, 1):
                raise ValueError("Action must be 0 (keep) or 1 (switch).")

            can_switch = self.time_in_phase_s[iid] >= int(self.cfg.min_green_s)
            if a == 1 and can_switch:
                self.phase_idx[iid] = (self.phase_idx[iid] + 1) % self.n_phases
                self.time_in_phase_s[iid] = 0
            else:
                self.time_in_phase_s[iid] += int(self.cfg.action_interval_s)

            self.eng.set_tl_phase(iid, int(self.phase_idx[iid]))

    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        self._apply_action(actions)

        # Advance simulation for action_interval_s seconds
        for _ in range(int(self.cfg.action_interval_s)):
            self.eng.next_step()

        # Reward at end of interval
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        per_inter_q = []
        total_q = 0.0
        for iid in self.intersection_ids:
            q_i = float(sum(lane_waiting.get(l, 0) for l in self.incoming_lanes[iid]))
            per_inter_q.append(q_i)
            total_q += q_i
        fairness = float(np.std(per_inter_q)) if len(per_inter_q) > 1 else 0.0
        reward = -total_q - float(self.cfg.fairness_lambda) * fairness

        obs = self._get_obs()
        now = float(self.eng.get_current_time())
        vehicle_count = int(self.eng.get_vehicle_count())
        cleared = (now >= float(self.cfg.demand_end_s)) and (vehicle_count == 0)
        capped = now >= float(self.cfg.max_sim_time_s)
        done = bool(cleared or capped)
        info = {
            "total_queue": total_q,
            "queue_std": fairness,
            "per_inter_queue": per_inter_q,
            "vehicle_count": vehicle_count,
            "cleared": cleared,
            "capped": capped,
            "time": now,
        }
        return obs, float(reward), bool(done), info

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cityflow
import numpy as np


@dataclass
class Env1x3Config:
    config_path: str = "./mydata_1x3/config_rl.json"  # rlTrafficLight=true
    incoming_lanes_path: str = "./mydata_1x3/incoming_lanes.json"
    intersection_ids: Tuple[str, str, str] = ("intersection_1_1", "intersection_2_1", "intersection_3_1")
    action_interval_s: int = 10          # decide every 10 simulated seconds
    min_green_s: int = 20                # ignore switches until >= 20s in phase
    demand_end_s: int = 3600             # after this, no new vehicles should enter (dataset is ~1h)
    max_sim_time_s: int = 8400           # generous hard cap (episode horizon) in simulated seconds
    fairness_lambda: float = 0.2         # weight on std(queue_per_intersection)


class CityFlowEnv1x3:
    """
    1x3 corridor env (3 intersections) with keep/switch control.

    - Engine must be created with rlTrafficLight=true in config (use config_rl.json).
    - Action per intersection: 0=keep, 1=switch_to_next_phase (phase = (phase+1)%9).
    - Decision interval: action_interval_s CityFlow steps per env.step().
    - Min-green: min_green_s seconds before allowing a switch.
    - Observation per intersection:
        [waiting_count for each incoming lane] + onehot(phase_idx,9) + [time_in_phase_s]
      Then concatenated over intersections in fixed order intersection_ids.
    - Reward:
        r = - total_queue - fairness_lambda * std(queue_sum_per_intersection)
      computed at the *end* of each env.step() using lane waiting counts.
    """

    def __init__(self, cfg: Env1x3Config):
        self.cfg = cfg
        self.eng = cityflow.Engine(cfg.config_path, thread_num=1)

        self.intersection_ids = list(cfg.intersection_ids)
        self.n_inters = len(self.intersection_ids)
        self.n_phases = 9

        # Load incoming lanes mapping
        self.incoming_lanes: Dict[str, List[str]] = self._load_incoming_lanes(cfg.incoming_lanes_path)
        for iid in self.intersection_ids:
            if iid not in self.incoming_lanes:
                raise KeyError(f"Intersection {iid} missing from incoming_lanes.json")

        # Internal phase tracking (CityFlow Python binding lacks get_tl_phase)
        self.phase_idx: Dict[str, int] = {iid: 0 for iid in self.intersection_ids}
        self.time_in_phase_s: Dict[str, int] = {iid: 0 for iid in self.intersection_ids}

        # Observation sizing
        self.per_inter_lane_dim = len(self.incoming_lanes[self.intersection_ids[0]])
        if any(len(self.incoming_lanes[iid]) != self.per_inter_lane_dim for iid in self.intersection_ids):
            raise ValueError("All intersections must have same incoming lane count for a fixed-size obs.")
        self.per_inter_obs_dim = self.per_inter_lane_dim + self.n_phases + 1
        self.obs_dim = self.per_inter_obs_dim * self.n_inters

    def _load_incoming_lanes(self, path: str) -> Dict[str, List[str]]:
        # Resolve relative to repo root (cwd) if needed
        p = path
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(os.getcwd(), p))
        with open(p, "r") as f:
            return json.load(f)

    def reset(self) -> np.ndarray:
        self.eng.reset()
        for iid in self.intersection_ids:
            self.phase_idx[iid] = 0
            self.time_in_phase_s[iid] = 0
            # Ensure engine starts at phase 0 in RL mode
            self.eng.set_tl_phase(iid, 0)
        return self._get_obs()

    def set_replay(self, replay_log_file: Optional[str]) -> None:
        """
        Enable/disable replay logging.

        CityFlow prepends config 'dir' to the replay path, so this should be a relative path
        like './mydata_1x3/replays/ppo/replay_ep10.txt' (NOT '/app/...').
        """
        if not replay_log_file:
            self.eng.set_save_replay(False)
            return
        self.eng.set_save_replay(True)
        self.eng.set_replay_file(replay_log_file)

    def _get_obs(self) -> np.ndarray:
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        chunks = []
        for iid in self.intersection_ids:
            lanes = self.incoming_lanes[iid]
            q = np.array([lane_waiting.get(l, 0) for l in lanes], dtype=np.float32)

            onehot = np.zeros((self.n_phases,), dtype=np.float32)
            onehot[int(self.phase_idx[iid])] = 1.0

            tip = np.array([float(self.time_in_phase_s[iid])], dtype=np.float32)
            chunks.append(np.concatenate([q, onehot, tip], axis=0))
        return np.concatenate(chunks, axis=0)

    def _apply_action(self, actions: List[int]) -> None:
        # actions: list length 3 of {0,1}
        for iid, a in zip(self.intersection_ids, actions):
            a = int(a)
            if a not in (0, 1):
                raise ValueError("Action must be 0 (keep) or 1 (switch).")

            can_switch = self.time_in_phase_s[iid] >= int(self.cfg.min_green_s)
            if a == 1 and can_switch:
                self.phase_idx[iid] = (self.phase_idx[iid] + 1) % self.n_phases
                self.time_in_phase_s[iid] = 0
            else:
                self.time_in_phase_s[iid] += int(self.cfg.action_interval_s)

            self.eng.set_tl_phase(iid, int(self.phase_idx[iid]))

    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        self._apply_action(actions)

        # Advance simulation for action_interval_s seconds
        for _ in range(int(self.cfg.action_interval_s)):
            self.eng.next_step()

        # Reward at end of interval
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        per_inter_q = []
        total_q = 0.0
        for iid in self.intersection_ids:
            q_i = float(sum(lane_waiting.get(l, 0) for l in self.incoming_lanes[iid]))
            per_inter_q.append(q_i)
            total_q += q_i
        fairness = float(np.std(per_inter_q)) if len(per_inter_q) > 1 else 0.0
        reward = -total_q - float(self.cfg.fairness_lambda) * fairness

        obs = self._get_obs()
        now = float(self.eng.get_current_time())
        vehicle_count = int(self.eng.get_vehicle_count())
        cleared = (now >= float(self.cfg.demand_end_s)) and (vehicle_count == 0)
        capped = now >= float(self.cfg.max_sim_time_s)
        done = bool(cleared or capped)
        info = {
            "total_queue": total_q,
            "queue_std": fairness,
            "per_inter_queue": per_inter_q,
            "vehicle_count": vehicle_count,
            "cleared": cleared,
            "capped": capped,
            "time": now,
        }
        return obs, float(reward), bool(done), info

