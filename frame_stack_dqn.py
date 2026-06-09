# frame_stack_dqn.py
# Frame-stacking DQN baseline for comparison against AA-DQN
# Drop into your project folder and run: python frame_stack_dqn.py

import json
import os
import random
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─── Config ───────────────────────────────────────────────────────────
ENV_NAME      = "Acrobot-v1"
STATE_DIM     = 6
ACTION_DIM    = 3
K_VALUES      = [1, 2, 4]       # history window sizes to test
SEEDS         = [42, 123, 456]
LR            = 3e-4            # same as AA-DQN for fair comparison
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.997
BATCH_SIZE    = 64
BUFFER_SIZE   = 50_000
TARGET_UPDATE = 10
HIDDEN_DIM    = 128
NUM_EPISODES  = 800
RESULTS_DIR   = "results"
OUT_PATH      = os.path.join(RESULTS_DIR, "framestack_results.json")

# AA-DQN results for reference (from your ablation)
AA_DQN_RESULTS = {
    1: (-119.7, 19.9),
    2: (-117.9,  9.0),
    4: (-178.1, 52.8),
}
DQN_BASELINE = (-171.0, 16.0)

# ─── Replay Buffer ────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)

# ─── Network ──────────────────────────────────────────────────────────
class FrameStackNet(nn.Module):
    """
    Simple MLP that takes K concatenated states as input.
    No attention — pure concatenation baseline.
    """
    def __init__(self, state_dim, action_dim, k, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ─── Agent ────────────────────────────────────────────────────────────
class FrameStackAgent:
    def __init__(self, k, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.k       = k
        self.eps     = EPSILON_START
        self.history = deque(maxlen=k)
        self.device  = torch.device("cpu")

        self.online = FrameStackNet(STATE_DIM, ACTION_DIM, k).to(self.device)
        self.target = deepcopy(self.online)
        self.target.eval()

        self.opt    = optim.Adam(self.online.parameters(), lr=LR)
        self.buf    = ReplayBuffer(BUFFER_SIZE)

    # history helpers
    def reset(self, obs):
        self.history.clear()
        for _ in range(self.k):
            self.history.append(obs.copy())

    def push(self, obs):
        self.history.append(obs.copy())

    def get_hist(self):
        return np.concatenate(list(self.history))

    # action
    def act(self, hist, greedy=False):
        if not greedy and np.random.rand() < self.eps:
            return np.random.randint(ACTION_DIM)
        with torch.no_grad():
            q = self.online(torch.FloatTensor(hist).unsqueeze(0))
        return int(q.argmax(1).item())

    # training
    def store(self, h, a, r, nh, d):
        self.buf.push(h, a, r, nh, d)

    def decay(self):
        self.eps = max(EPSILON_END, self.eps * EPSILON_DECAY)

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def learn(self):
        if len(self.buf) < BATCH_SIZE:
            return
        h, a, r, nh, d = self.buf.sample(BATCH_SIZE)
        h  = torch.FloatTensor(h)
        a  = torch.LongTensor(a)
        r  = torch.FloatTensor(r)
        nh = torch.FloatTensor(nh)
        d  = torch.FloatTensor(d)

        q  = self.online(h).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.online(nh).argmax(1, keepdim=True)
            qt   = self.target(nh).gather(1, best).squeeze(1)
            y    = r + GAMMA * qt * (1 - d)

        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

# ─── Training loop ────────────────────────────────────────────────────
def train(k, seed):
    env    = gym.make(ENV_NAME)
    agent  = FrameStackAgent(k, seed)
    ep_ret = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset(seed=seed + ep)
        agent.reset(obs)
        agent.push(obs)
        total = 0.0
        done  = False

        while not done:
            h      = agent.get_hist()
            action = agent.act(h)
            nobs, rew, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.push(nobs)
            nh = agent.get_hist()
            agent.store(h, action, rew, nh, float(done))
            agent.learn()
            total += rew

        agent.decay()
        if ep % TARGET_UPDATE == 0:
            agent.sync_target()
        ep_ret.append(total)

        if ep % 100 == 0:
            avg = np.mean(ep_ret[-100:])
            print(f"    ep {ep:4d} | avg-100: {avg:7.1f} | eps: {agent.eps:.3f}")

    env.close()
    return {
        "episode_returns": ep_ret,
        "last50_mean":     float(np.mean(ep_ret[-50:])),
        "last50_std":      float(np.std(ep_ret[-50:])),
        "params":          agent.online.count_params(),
    }

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {}

    for k in K_VALUES:
        key = str(k)
        if key not in results:
            results[key] = {}
        print(f"\n{'='*50}\nFrameStack  K={k}\n{'='*50}")

        for seed in SEEDS:
            if str(seed) in results[key]:
                print(f"  K={k} seed={seed} already done, skipping")
                continue
            print(f"\n  Running K={k} seed={seed} ...")
            results[key][str(seed)] = train(k, seed)
            json.dump(results, open(OUT_PATH, "w"), indent=2)
            print(f"  Saved → {OUT_PATH}")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'Method':<26} {'Mean':>10}  {'Std':>8}  {'Params':>8}")
    print(f"{'='*65}")
    print(f"  {'DQN baseline':<26} {DQN_BASELINE[0]:>10.1f}  "
          f"{DQN_BASELINE[1]:>8.1f}  {'~17K':>8}")
    print(f"  {'-'*63}")
    for k in K_VALUES:
        ms = [results[str(k)][str(s)]["last50_mean"] for s in SEEDS]
        ss = [results[str(k)][str(s)]["last50_std"]  for s in SEEDS]
        pc = results[str(k)][str(SEEDS[0])]["params"]
        print(f"  {'FrameStack K='+str(k):<26} {np.mean(ms):>10.1f}  "
              f"{np.mean(ss):>8.1f}  {pc:>8,}")
        if k in AA_DQN_RESULTS:
            m, s = AA_DQN_RESULTS[k]
            print(f"  {'AA-DQN     K='+str(k):<26} {m:>10.1f}  "
                  f"{s:>8.1f}  {'~120K':>8}")
        print()
    print(f"{'='*65}")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()