# ablation.py — sweep K (history window size) for AA-DQN on Acrobot-v1

import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from copy import deepcopy

import config
from models import AttentionQNetwork
from replay_buffer import ReplayBuffer

K_VALUES = [1, 2, 4, 8]   # history window sizes to sweep
SEEDS    = [42, 123, 456]
OUT_PATH = os.path.join(config.RESULTS_DIR, "ablation_k.json")


# ------------------------------------------------------------------
class AblationAgent:
    """AttentionDQNAgent with configurable K, for ablation."""

    def __init__(self, history_len: int, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device      = torch.device("cpu")
        self.epsilon     = config.EPSILON_START
        self.history_len = history_len
        self._history    = deque(maxlen=history_len)

        self.online_net = AttentionQNetwork(
            state_dim   = config.STATE_DIM,
            action_dim  = config.ACTION_DIM,
            history_len = history_len,
            embed_dim   = config.EMBED_DIM,
            num_heads   = min(config.NUM_HEADS, config.EMBED_DIM),
            hidden_dim  = config.HIDDEN_DIM,
        ).to(self.device)
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=config.LR_ATTN)
        self.buffer = ReplayBuffer(config.BUFFER_SIZE)

    def reset_history(self, state):
        self._history.clear()
        for _ in range(self.history_len):
            self._history.append(state.copy())

    def push_state(self, state):
        self._history.append(state.copy())

    def get_history(self):
        return np.concatenate(list(self._history), axis=0)

    def select_action(self, history, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(config.ACTION_DIM)
        h_t = torch.FloatTensor(history).unsqueeze(0)
        with torch.no_grad():
            q = self.online_net(h_t)
        return int(q.argmax(dim=1).item())

    def store(self, h, a, r, nh, d):
        self.buffer.push(h, a, r, nh, d)

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END,
                          self.epsilon * config.EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self):
        if len(self.buffer) < config.BATCH_SIZE:
            return None
        histories, actions, rewards, next_histories, dones = \
            self.buffer.sample(config.BATCH_SIZE)

        h  = torch.FloatTensor(histories)
        a  = torch.LongTensor(actions)
        r  = torch.FloatTensor(rewards)
        nh = torch.FloatTensor(next_histories)
        d  = torch.FloatTensor(dones)

        q_values = self.online_net(h).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a   = self.online_net(nh).argmax(dim=1, keepdim=True)
            q_next   = self.target_net(nh).gather(1, best_a).squeeze(1)
            q_target = r + config.GAMMA * q_next * (1 - d)

        loss = F.smooth_l1_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()


# ------------------------------------------------------------------
def train_one(k: int, seed: int) -> dict:
    print(f"  K={k}  seed={seed}")
    env = gym.make(config.ENV_NAME)
    agent = AblationAgent(history_len=k, seed=seed)
    episode_returns = []

    for ep in range(1, config.NUM_EPISODES + 1):
        obs, _ = env.reset(seed=seed + ep)
        agent.reset_history(obs)
        agent.push_state(obs)
        ep_return = 0.0
        done = False

        while not done:
            hist   = agent.get_history()
            action = agent.select_action(hist)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.push_state(next_obs)
            next_hist = agent.get_history()
            agent.store(hist, action, reward, next_hist, float(done))
            agent.train_step()
            ep_return += reward

        agent.decay_epsilon()
        if ep % config.TARGET_UPDATE == 0:
            agent.update_target()
        episode_returns.append(ep_return)

        if ep % 100 == 0:
            avg = np.mean(episode_returns[-100:])
            print(f"    ep {ep:4d} | avg-100: {avg:7.1f} | eps: {agent.epsilon:.3f}")

    env.close()
    return {"episode_returns": episode_returns,
            "last50_mean": float(np.mean(episode_returns[-50:])),
            "last50_std":  float(np.std(episode_returns[-50:]))}


# ------------------------------------------------------------------
def run_ablation():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"Resuming from {OUT_PATH}")
    else:
        results = {}

    for k in K_VALUES:
        key = str(k)
        if key not in results:
            results[key] = {}
        print(f"\n=== K={k} ===")
        for seed in SEEDS:
            skey = str(seed)
            if skey in results[key]:
                print(f"  Skipping K={k} seed={seed} (done)")
                continue
            results[key][skey] = train_one(k, seed)
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved K={k} seed={seed}")

    # Print summary table
    print("\n" + "="*50)
    print(f"  {'K':>4}  {'Mean (last 50)':>16}  {'Std':>8}")
    print("="*50)
    for k in K_VALUES:
        means = [results[str(k)][str(s)]["last50_mean"] for s in SEEDS]
        print(f"  K={k:>2}  {np.mean(means):>16.1f}  {np.std(means):>8.1f}")
    print("="*50)


if __name__ == "__main__":
    run_ablation()