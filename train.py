import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from envs.intersection_env import CityFlowEnv
from models.dqn import DQN
from envs.replay_buffer import ReplayBuffer

# === SETUP ===
config_path = "./mydata/config.json"
intersection_id = "intersection_1_1"  # Change if needed

# Initialize environment
env = CityFlowEnv(config_path, [intersection_id])
# Setup replay saving every 50 episodes
def enable_replay_saving(env, episode):
    replay_path = f"./mydata/replays/replay_ep{episode}.txt"
    env.eng.set_save_replay(True)
    env.eng.set_replay_file(replay_path)

state_dim = len(env.incoming_lanes[intersection_id])  # e.g., number of incoming lanes
action_dim = 9  # Two phases: green or red

# Initialize networks
q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())  # Sync weights initially

# Optimizer and loss
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Experience replay
buffer = ReplayBuffer(capacity=10000)

# Hyperparameters
batch_size = 64
gamma = 0.99
epsilon = 0.1
episodes = 100

# === TRAINING LOOP ===
for ep in range(episodes):
    state = env.reset()[0]  # Get initial state for single intersection
    total_reward = 0
    done = False

    while not done:
        # Choose action (epsilon-greedy)
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = q_net(state_tensor).argmax().item()

        # Take action in environment
        next_state, reward, done, _ = env.step([action])
        next_state = next_state[0]

        # Store experience
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train the model
        if len(buffer) >= batch_size:
            s, a, r, s_, d = buffer.sample(batch_size)

            s = torch.tensor(s)
            a = torch.tensor(a).unsqueeze(1)
            r = torch.tensor(r).unsqueeze(1)
            s_ = torch.tensor(s_)
            d = torch.tensor(d).unsqueeze(1)

            # Current Q-value estimates
            q_val = q_net(s).gather(1, a)

            # Target Q-values
            with torch.no_grad():
                next_q = target_net(s_).max(1)[0].unsqueeze(1)
                target = r + gamma * next_q * (1 - d)

            # Compute loss and update
            loss = loss_fn(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Sync target network
    if ep % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}")
    
    if ep % 50 == 0:
        enable_replay_saving(env, ep)
    else:
        env.eng.set_save_replay(False)


print("Training complete.")
