# extra_experiments.py
# Runs all additional experiments for reviewer responses:
#   1. Frame-stacking DQN  (K=1,2,4 x 3 seeds) — isolates attention vs concatenation
#   2. DQN at lr=3e-4      (3 seeds)            — isolates architecture vs LR
#   3. Extra seeds K=1,2   (seeds 7,11)          — strengthens ablation statistics
#
# Run: python extra_experiments.py
# Saves: results/extra_results.json (crash-safe, resumes on restart)

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
OUT_PATH      = os.path.join(RESULTS_DIR, "extra_results.json")

SEEDS_MAIN  = [42, 123, 456]
SEEDS_EXTRA = [7, 11]          # extra seeds for ablation strengthening


# ─── Replay Buffer ────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ─── Networks ─────────────────────────────────────────────────────────
class MLPDQN(nn.Module):
    """Vanilla MLP — used for DQN baseline and frame-stacking."""
    def __init__(self, input_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ─── Generic Agent ────────────────────────────────────────────────────
class Agent:
    """
    Works for both vanilla DQN and frame-stacking DQN.
    history_len=1 → vanilla DQN (single state input)
    history_len=K → frame-stacking (K states concatenated)
    """
    def __init__(self, history_len, lr, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.k       = history_len
        self.eps     = EPSILON_START
        self.history = deque(maxlen=history_len)

        self.online = MLPDQN(STATE_DIM * history_len, ACTION_DIM)
        self.target = deepcopy(self.online)
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = ReplayBuffer(BUFFER_SIZE)

    def reset(self, obs):
        self.history.clear()
        for _ in range(self.k):
            self.history.append(obs.copy())

    def push(self, obs):
        self.history.append(obs.copy())

    def get_hist(self):
        return np.concatenate(list(self.history))

    def act(self, hist, greedy=False):
        if not greedy and np.random.rand() < self.eps:
            return np.random.randint(ACTION_DIM)
        with torch.no_grad():
            q = self.online(torch.FloatTensor(hist).unsqueeze(0))
        return int(q.argmax(1).item())

    def store(self, h, a, r, nh, d):
        self.buf.push(h, a, r, nh, d)

    def decay(self):
        self.eps = max(EPSILON_END, self.eps * EPSILON_DECAY)

    def sync(self):
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

        q = self.online(h).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.online(nh).argmax(1, keepdim=True)
            qt   = self.target(nh).gather(1, best).squeeze(1)
            y    = r + GAMMA * qt * (1 - d)

        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()


# ─── Attention Agent (for extra ablation seeds) ───────────────────────
class AttentionNet(nn.Module):
    def __init__(self, state_dim, action_dim, k, embed_dim=64, heads=4, hidden=128):
        super().__init__()
        self.state_dim   = state_dim
        self.k           = k
        self.embed_dim   = embed_dim
        self.encoder     = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU())
        self.pos_emb     = nn.Embedding(k, embed_dim)
        self.attn        = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm        = nn.LayerNorm(embed_dim)
        self.q_head      = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim))

    def forward(self, x):
        b = x.shape[0]
        tokens = x.view(b, self.k, self.state_dim)
        tokens = self.encoder(tokens)
        pos    = torch.arange(self.k, device=x.device)
        tokens = tokens + self.pos_emb(pos)
        out, _ = self.attn(tokens, tokens, tokens)
        out    = self.norm(out)
        return self.q_head(out[:, -1, :])

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class AttentionAgent:
    def __init__(self, k, lr, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.k       = k
        self.eps     = EPSILON_START
        self.history = deque(maxlen=k)

        self.online = AttentionNet(STATE_DIM, ACTION_DIM, k)
        self.target = deepcopy(self.online)
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = ReplayBuffer(BUFFER_SIZE)

    def reset(self, obs):
        self.history.clear()
        for _ in range(self.k):
            self.history.append(obs.copy())

    def push(self, obs):
        self.history.append(obs.copy())

    def get_hist(self):
        return np.concatenate(list(self.history))

    def act(self, hist, greedy=False):
        if not greedy and np.random.rand() < self.eps:
            return np.random.randint(ACTION_DIM)
        with torch.no_grad():
            q = self.online(torch.FloatTensor(hist).unsqueeze(0))
        return int(q.argmax(1).item())

    def store(self, h, a, r, nh, d):
        self.buf.push(h, a, r, nh, d)

    def decay(self):
        self.eps = max(EPSILON_END, self.eps * EPSILON_DECAY)

    def sync(self):
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

        q = self.online(h).gather(1, a.unsqueeze(1)).squeeze(1)
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
def train(agent, seed, label):
    env    = gym.make(ENV_NAME)
    ep_ret = []
    print(f"\n  [{label}] seed={seed}")

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
            agent.sync()
        ep_ret.append(total)

        if ep % 200 == 0:
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

    def save():
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    def skip(key, seed):
        return str(seed) in results.get(key, {})

    def record(key, seed, data):
        if key not in results:
            results[key] = {}
        results[key][str(seed)] = data
        save()
        print(f"  Saved → {key} seed={seed}")

    # ── 1. Frame-stacking DQN ─────────────────────────────────────────
    print("\n" + "="*55)
    print("EXPERIMENT 1: Frame-Stacking DQN (lr=3e-4, same as AA-DQN)")
    print("="*55)
    for k in [1, 2, 4]:
        key = f"framestack_k{k}"
        for seed in SEEDS_MAIN:
            if skip(key, seed):
                print(f"  Skipping {key} seed={seed}")
                continue
            agent = Agent(history_len=k, lr=3e-4, seed=seed)
            data  = train(agent, seed, f"FrameStack K={k}")
            record(key, seed, data)

    # ── 2. LR Control: DQN at lr=3e-4 ────────────────────────────────
    print("\n" + "="*55)
    print("EXPERIMENT 2: Vanilla DQN at lr=3e-4 (LR control)")
    print("="*55)
    for seed in SEEDS_MAIN:
        key = "dqn_lr3e4"
        if skip(key, seed):
            print(f"  Skipping {key} seed={seed}")
            continue
        agent = Agent(history_len=1, lr=3e-4, seed=seed)
        data  = train(agent, seed, "DQN lr=3e-4")
        record(key, seed, data)

    # ── 3. Extra seeds for K=1,2 ablation ────────────────────────────
    print("\n" + "="*55)
    print("EXPERIMENT 3: AA-DQN extra seeds (K=1, K=2 only)")
    print("="*55)
    for k in [1, 2]:
        key = f"aadqn_extra_k{k}"
        for seed in SEEDS_EXTRA:
            if skip(key, seed):
                print(f"  Skipping {key} seed={seed}")
                continue
            agent = AttentionAgent(k=k, lr=3e-4, seed=seed)
            data  = train(agent, seed, f"AA-DQN K={k} extra seed")
            record(key, seed, data)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  {'Method':<30} {'Mean':>10}  {'Std':>8}  {'Params':>8}")
    print("="*65)

    # DQN baselines
    print(f"  {'DQN (lr=1e-3, original)':<30} {-171.0:>10.1f}  {16.0:>8.1f}  {'~17K':>8}")

    if "dqn_lr3e4" in results:
        ms = [results["dqn_lr3e4"][str(s)]["last50_mean"] for s in SEEDS_MAIN]
        pc = results["dqn_lr3e4"][str(SEEDS_MAIN[0])]["params"]
        print(f"  {'DQN (lr=3e-4, control)':<30} {np.mean(ms):>10.1f}  "
              f"{np.std(ms):>8.1f}  {pc:>8,}")

    print()
    for k in [1, 2, 4]:
        key = f"framestack_k{k}"
        if key in results and len(results[key]) == 3:
            ms = [results[key][str(s)]["last50_mean"] for s in SEEDS_MAIN]
            pc = results[key][str(SEEDS_MAIN[0])]["params"]
            print(f"  {'FrameStack K='+str(k):<30} {np.mean(ms):>10.1f}  "
                  f"{np.std(ms):>8.1f}  {pc:>8,}")

        # AA-DQN original + extra seeds combined
        orig = {1: (-119.7, 19.9), 2: (-117.9, 9.0), 4: (-178.1, 52.8)}
        if k in orig:
            print(f"  {'AA-DQN K='+str(k)+' (original 3 seeds)':<30} "
                  f"{orig[k][0]:>10.1f}  {orig[k][1]:>8.1f}  {'~120K':>8}")
        print()

    print("="*65)
    print(f"\nAll results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()