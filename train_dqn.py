import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import time

from models.dqn import DQN
from envs.replay_buffer import ReplayBuffer

def train_dqn(env, episodes=100, gamma=0.99, epsilon=0.1, batch_size=64):
    intersection_id = env.intersection_ids[0]  # intersection 1
    state_dim = len(env.incoming_lanes[intersection_id])  # 8
    action_dim = 9

    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(capacity=10000)

    episode_rewards = []
    episode_times = []

    for ep in range(episodes):
        start_time = time.time()

        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = q_net(state_tensor).argmax().item()

            next_state, reward, done, _ = env.step([action])
            next_state = next_state[0]

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                s, a, r, s_, d = buffer.sample(batch_size)
                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a).unsqueeze(1)
                r = torch.tensor(r).unsqueeze(1)
                s_ = torch.tensor(s_, dtype=torch.float32)
                d = torch.tensor(d).unsqueeze(1)

                q_val = q_net(s).gather(1, a)

                with torch.no_grad():
                    next_q = target_net(s_).max(1)[0].unsqueeze(1)
                    target = r + gamma * next_q * (1 - d)

                loss = loss_fn(q_val, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        duration = time.time() - start_time
        episode_rewards.append(total_reward)
        episode_times.append(duration)

        if ep % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"[DQN] Episode {ep}: Total Reward = {total_reward:.2f}, Time = {duration:.2f}s")

    # Plot reward vs time vs episode
    episodes_range = list(range(episodes))

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color='tab:blue')
    ax1.plot(episodes_range, episode_rewards, label='Reward', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Time (s)', color='tab:red')
    ax2.plot(episodes_range, episode_times, label='Time', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Episode vs Total Reward and Time')
    plt.tight_layout()
    plt.savefig("reward_vs_time.png")
    plt.show()


if __name__ == "__main__":
    from envs.intersection_env import CityFlowEnv

    config_path = "mydata/config.json"
    intersection_ids = ["intersection_1_1"]

    env = CityFlowEnv(config_path=config_path, intersection_ids=intersection_ids)
    train_dqn(env)
