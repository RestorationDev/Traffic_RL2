from envs.intersection_env import CityFlowEnv

if __name__ == "__main__":
    env = CityFlowEnv(
        config_path="Kunal_Traffic_RL/mydata/cityflow.config",  # make sure this path is correct *inside* Docker
        intersection_ids=["intersection_1"]
    )
    state = env.reset()
    print("Initial state:", state)
    