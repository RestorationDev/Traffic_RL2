import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import argparse
import multiprocessing as mp
import queue as py_queue
from envs.intersection_env_exp import CityFlowEnvExp
from models.dqn_mlp import DQNMLP
from envs.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

# Setup replay saving every N episodes
def enable_replay_saving(env, episode, name):
    os.makedirs(f"./mydata/replays/{name}", exist_ok=True)
    replay_path = f"./mydata/replays/{name}/replay_ep{episode}.txt"
    env.eng.set_save_replay(True)
    env.eng.set_replay_file(replay_path)


def _cpu_state_dict(state_dict):
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def _worker_loop(
    wid: int,
    config_path: str,
    intersection_id: str,
    reward_name: str,
    step_time: int,
    max_sim_time: int,
    epsilon: float,
    action_dim: int,
    state_dim: int,
    save_replay: bool,
    exp_queue: mp.Queue,
    ctrl_queue: mp.Queue,
    incoming_lanes_path,
    roadnet_path,
):
    # Workers do action selection on CPU to avoid GPU sync overhead.
    policy = DQNMLP(state_dim, action_dim, hidden_dim=128, depth=3).cpu().eval()

    reward_fns_local = {
        "default": lambda env: env.default_reward(),
        "pressure and count": lambda env: env.pressure_and_count_reward(),
        "pressure only": lambda env: env.pressure_only_reward(),
    }

    env = CityFlowEnvExp(
        config_path,
        [intersection_id],
        step_time=step_time,
        max_sim_time=max_sim_time,
        thread_num=1,
        incoming_lanes_path=incoming_lanes_path,
        roadnet_path=roadnet_path,
    )
    env.set_rwd_fn(lambda: reward_fns_local[reward_name](env))

    ep = 0
    state = env.reset()[0]
    ep_reward = 0.0

    while True:
        # Apply any pending control messages (non-blocking).
        try:
            while True:
                msg = ctrl_queue.get_nowait()
                if msg is None:
                    return
                if isinstance(msg, dict) and msg.get("type") == "weights":
                    policy.load_state_dict(msg["state_dict"], strict=True)
                elif isinstance(msg, dict) and msg.get("type") == "epsilon":
                    epsilon = float(msg["value"])
        except py_queue.Empty:
            pass

        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = int(torch.argmax(policy(st), dim=1).item())

        next_state, reward, done, _ = env.step([action])
        next_state = next_state[0]

        exp_queue.put((state, action, float(reward), next_state, bool(done)))
        state = next_state
        ep_reward += float(reward)

        if done:
            # Optional replay saving every 10 episodes.
            if save_replay and (ep % 10 == 0):
                # Include worker id to avoid collisions if multiple runs mount same directory.
                os.makedirs(f"./mydata/replays/{reward_name}", exist_ok=True)
                replay_path = f"./mydata/replays/{reward_name}/replay_w{wid}_ep{ep}.txt"
                env.eng.set_save_replay(True)
                env.eng.set_replay_file(replay_path)
            else:
                env.eng.set_save_replay(False)

            exp_queue.put(("episode_end", wid, ep, ep_reward))
            ep += 1
            state = env.reset()[0]
            ep_reward = 0.0

def main():
    # === SETUP ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="./mydata/config.json")
    ap.add_argument(
        "--incoming_lanes",
        type=str,
        default=None,
        help="Optional path to incoming_lanes.json (default: next to --config, else mydata/).",
    )
    ap.add_argument(
        "--roadnet",
        type=str,
        default=None,
        help="Optional roadnet JSON path (default: roadnetFile from CityFlow config).",
    )
    ap.add_argument("--intersection", type=str, default="intersection_1_1")
    ap.add_argument(
        "--reward",
        type=str,
        default="default",
        choices=["default", "pressure and count", "pressure only"],
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_sim_time", type=int, default=2000, help="Early stop per episode (CityFlow seconds).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--update_every", type=int, default=4, help="Train every N env steps once buffer is warm.")
    ap.add_argument("--updates_per_step", type=int, default=1, help="Gradient updates per training event.")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target_update_every", type=int, default=10, help="Sync target net every N episodes.")
    ap.add_argument("--save_replay", action="store_true", help="Enable replay saving (slower).")
    ap.add_argument("--plot", action="store_true", help="Save reward plot (slower).")
    ap.add_argument("--step_time", type=int, default=10, help="CityFlow steps per RL action in env.step().")
    ap.add_argument("--num_workers", type=int, default=0, help="Parallel env workers (0 = single process).")
    ap.add_argument("--sync_every", type=int, default=200, help="Learner steps between policy sync to workers.")
    ap.add_argument("--hidden_dim", type=int, default=128, help="DQN hidden size (bigger = more GPU work).")
    ap.add_argument("--depth", type=int, default=3, help="DQN MLP depth.")
    args = ap.parse_args()

    config_path = args.config
    intersection_id = args.intersection

    reward_fns = {
        "default": lambda env: env.default_reward(),
        "pressure and count": lambda env: env.pressure_and_count_reward(),
        "pressure only": lambda env: env.pressure_only_reward(),
    }

    name = args.reward
    fn = reward_fns[name]
    print(f"\nTraining with reward: {name}")

    env = CityFlowEnvExp(
        config_path,
        [intersection_id],
        step_time=args.step_time,
        max_sim_time=args.max_sim_time,
        incoming_lanes_path=args.incoming_lanes,
        roadnet_path=args.roadnet,
    )
    env.set_rwd_fn(lambda: fn(env))

    action_dim = 9
    state_dim = len(env.incoming_lanes[intersection_id])

    q_net = DQNMLP(state_dim, action_dim, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    target_net = DQNMLP(state_dim, action_dim, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    buffer = ReplayBuffer(capacity=10000)

    batch_size = args.batch_size
    gamma = args.gamma
    epsilon = args.epsilon
    episodes = args.episodes
    update_every = max(1, args.update_every)
    updates_per_step = max(1, args.updates_per_step)

    reward_log = []
    global_step = 0

    def _train_step():
        s, a, r, s_, d = buffer.sample(batch_size)

        s = torch.tensor(s, dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
        s_ = torch.tensor(s_, dtype=torch.float32, device=device)
        d = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

        q_val = q_net(s).gather(1, a)
        with torch.no_grad():
            next_q = target_net(s_).max(1)[0].unsqueeze(1)
            target = r + gamma * next_q * (1 - d)

        loss = loss_fn(q_val, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    if args.num_workers <= 0:
        for ep in range(episodes):
            state = env.reset()[0]
            total_reward = 0.0
            done = False

            while not done:
                if random.random() < epsilon:
                    action = random.randint(0, action_dim - 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        action = int(torch.argmax(q_net(state_tensor), dim=1).item())

                next_state, reward, done, _ = env.step([action])
                next_state = next_state[0]

                buffer.push(state, action, float(reward), next_state, bool(done))
                state = next_state
                total_reward += float(reward)

                if len(buffer) >= batch_size and (global_step % update_every == 0):
                    for _ in range(updates_per_step):
                        _train_step()
                global_step += 1

            reward_log.append(total_reward)

            if ep % max(1, args.target_update_every) == 0:
                target_net.load_state_dict(q_net.state_dict())
                print(f"Episode {ep}: Total Reward = {total_reward:.2f}")

            if args.save_replay and (ep % 10 == 0):
                enable_replay_saving(env, ep, name)
            else:
                env.eng.set_save_replay(False)
    else:
        ctx = mp.get_context("spawn")
        exp_queue: mp.Queue = ctx.Queue(maxsize=5000)
        ctrl_queues = [ctx.Queue() for _ in range(args.num_workers)]

        workers = []
        for wid in range(args.num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(
                    wid,
                    config_path,
                    intersection_id,
                    name,
                    args.step_time,
                    args.max_sim_time,
                    epsilon,
                    action_dim,
                    state_dim,
                    args.save_replay,
                    exp_queue,
                    ctrl_queues[wid],
                    args.incoming_lanes,
                    args.roadnet,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        cpu_sd = _cpu_state_dict(q_net.state_dict())
        for cq in ctrl_queues:
            cq.put({"type": "weights", "state_dict": cpu_sd})

        episodes_done = 0
        last_sync_at = 0

        while episodes_done < episodes:
            item = exp_queue.get()
            if isinstance(item, tuple) and len(item) == 4 and item[0] == "episode_end":
                _, wid, ep_idx, ep_reward = item
                reward_log.append(float(ep_reward))
                episodes_done += 1
                print(f"Episode {episodes_done-1}: Total Reward = {float(ep_reward):.2f} (worker {wid})")
                if (episodes_done - 1) % max(1, args.target_update_every) == 0:
                    target_net.load_state_dict(q_net.state_dict())
                continue

            s, a, r, s2, d = item
            buffer.push(s, a, r, s2, d)
            global_step += 1

            if len(buffer) >= batch_size and (global_step % update_every == 0):
                for _ in range(updates_per_step):
                    _train_step()

            if (global_step - last_sync_at) >= max(1, args.sync_every):
                cpu_sd = _cpu_state_dict(q_net.state_dict())
                for cq in ctrl_queues:
                    cq.put({"type": "weights", "state_dict": cpu_sd})
                last_sync_at = global_step

        for cq in ctrl_queues:
            cq.put(None)
        for p in workers:
            p.join(timeout=5)

    if args.plot:
        plt.figure()
        plt.plot(range(episodes), reward_log)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Reward per Episode ({name})")
        plt.grid(True)
        plt.savefig(f"mydata/rewards_plot_{name.replace(' ', '_')}.png")
        plt.close()

    print("Training complete.")


if __name__ == "__main__":
    main()
