import argparse
from dataclasses import asdict
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from envs.intersection_env_1x3 import CityFlowEnv1x3, Env1x3Config
from models.ppo_mlp import PPOMLP


def _discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    acc = 0.0
    for i in reversed(range(len(x))):
        acc = float(x[i]) + gamma * acc
        y[i] = acc
    return y


def _gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    # values: length T+1 (bootstrap last)
    adv = np.zeros((len(rewards),), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv


@torch.no_grad()
def _rollout(
    env: CityFlowEnv1x3,
    model: PPOMLP,
    device: torch.device,
    horizon: int,
    *,
    start_episode_idx: int,
    save_replay_every: int,
    save_replay_dir: str,
    stop_on_success_under_s: float,
) -> Tuple[Dict[str, np.ndarray], int, bool, float]:
    """
    Returns: (rollout_batch, next_episode_idx, should_stop, success_time)
    """
    obs = env.reset()
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    episode_idx = int(start_episode_idx)
    should_stop = False
    success_time = float("inf")

    def _maybe_enable_replay(ep: int) -> None:
        if save_replay_every <= 0:
            env.set_replay(None)
            return
        if (ep % save_replay_every) == 0:
            os.makedirs(save_replay_dir, exist_ok=True)
            env.set_replay(os.path.join(save_replay_dir, f"replay_ep{ep}.txt").replace("\\", "/"))
        else:
            env.set_replay(None)

    _maybe_enable_replay(episode_idx)

    for _ in range(horizon):
        ot = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, v = model(ot)
        dist = torch.distributions.Bernoulli(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a).sum(dim=-1)  # sum over 3 bits

        actions = a.squeeze(0).to(torch.int64).cpu().numpy().tolist()
        next_obs, rew, done, info = env.step(actions)

        obs_buf.append(obs)
        act_buf.append(actions)
        logp_buf.append(float(logp.item()))
        rew_buf.append(float(rew))
        done_buf.append(float(done))
        val_buf.append(float(v.item()))

        obs = next_obs
        if done:
            # Success condition: cleared and time < threshold
            if bool(info.get("cleared", False)) and float(info.get("time", 1e9)) < float(stop_on_success_under_s):
                should_stop = True
                success_time = float(info.get("time", float("inf")))
                break
            episode_idx += 1
            obs = env.reset()
            _maybe_enable_replay(episode_idx)

    # Bootstrap value
    ot = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    _, v_last = model(ot)
    val_buf.append(float(v_last.item()))

    batch = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "act": np.asarray(act_buf, dtype=np.int64),
        "logp": np.asarray(logp_buf, dtype=np.float32),
        "rew": np.asarray(rew_buf, dtype=np.float32),
        "done": np.asarray(done_buf, dtype=np.float32),
        "val": np.asarray(val_buf, dtype=np.float32),  # T+1
    }
    return batch, episode_idx, should_stop, success_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="./mydata_1x3/config_rl.json")
    ap.add_argument("--incoming_lanes", type=str, default="./mydata_1x3/incoming_lanes.json")
    ap.add_argument("--steps_per_iter", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)

    ap.add_argument("--action_interval", type=int, default=10)
    ap.add_argument("--min_green", type=int, default=20)
    ap.add_argument("--max_sim_time", type=int, default=8400)
    ap.add_argument("--demand_end", type=int, default=3600)
    ap.add_argument("--fairness_lambda", type=float, default=0.2)
    ap.add_argument("--stop_on_success_under", type=float, default=4000.0)
    ap.add_argument("--save_replay_every", type=int, default=0, help="0 disables. Otherwise save replay every N episodes.")
    ap.add_argument("--save_replay_dir", type=str, default="./mydata_1x3/replays/ppo")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env_cfg = Env1x3Config(
        config_path=args.config,
        incoming_lanes_path=args.incoming_lanes,
        action_interval_s=int(args.action_interval),
        min_green_s=int(args.min_green),
        max_sim_time_s=int(args.max_sim_time),
        demand_end_s=int(args.demand_end),
        fairness_lambda=float(args.fairness_lambda),
    )
    env = CityFlowEnv1x3(env_cfg)
    print(f"Env obs_dim={env.obs_dim} per_inter_lane_dim={env.per_inter_lane_dim}")
    print(f"Env config: {asdict(env_cfg)}")

    model = PPOMLP(env.obs_dim, action_dim=3, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    episode_idx = 0
    for it in range(int(args.iters)):
        roll, episode_idx, should_stop, success_time = _rollout(
            env,
            model,
            device,
            horizon=int(args.steps_per_iter),
            start_episode_idx=episode_idx,
            save_replay_every=int(args.save_replay_every),
            save_replay_dir=str(args.save_replay_dir),
            stop_on_success_under_s=float(args.stop_on_success_under),
        )
        obs = roll["obs"]
        act = roll["act"]
        logp_old = roll["logp"]
        rew = roll["rew"]
        done = roll["done"]
        val = roll["val"]  # T+1
        if obs.shape[0] == 0:
            print("No samples collected (early stop before horizon).")
            break

        adv = _gae(rew, val, done, gamma=float(args.gamma), lam=float(args.gae_lambda))
        ret = adv + val[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        act_t = torch.tensor(act, dtype=torch.float32, device=device)  # Bernoulli expects float {0,1}
        logp_old_t = torch.tensor(logp_old, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        n = obs_t.shape[0]
        idx = np.arange(n)

        last_pi_loss = last_v_loss = last_ent = 0.0
        for _ in range(int(args.epochs)):
            np.random.shuffle(idx)
            for start in range(0, n, int(args.batch_size)):
                b = idx[start : start + int(args.batch_size)]
                logits, v = model(obs_t[b])
                dist = torch.distributions.Bernoulli(logits=logits)
                logp = dist.log_prob(act_t[b]).sum(dim=-1)
                ent = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(logp - logp_old_t[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1.0 - float(args.clip), 1.0 + float(args.clip)) * adv_t[b]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(v, ret_t[b])
                loss = pi_loss + float(args.vf_coef) * v_loss - float(args.ent_coef) * ent

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                opt.step()

                last_pi_loss = float(pi_loss.item())
                last_v_loss = float(v_loss.item())
                last_ent = float(ent.item())

        print(
            f"Iter {it:03d} | "
            f"rew_mean={rew.mean():.2f} adv_mean={adv.mean():.2f} | "
            f"pi_loss={last_pi_loss:.3f} v_loss={last_v_loss:.3f} ent={last_ent:.3f}"
        )

        if should_stop:
            print(f"SUCCESS: cleared under {float(args.stop_on_success_under):.0f}s at t={success_time:.0f}s. Stopping.")
            break


if __name__ == "__main__":
    main()

