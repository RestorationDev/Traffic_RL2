import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.intersection_env import CityFlowEnv

if __name__ == "__main__":
    env = CityFlowEnv(
        config_path="config.json",  # make sure this path is correct *inside* Docker
        intersection_ids=["intersection_1_1"]
    )
    state = env.reset()
    print("Initial state:", state)
    