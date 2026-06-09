# agent.py — DQNAgent and AttentionDQNAgent

from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import config
from models import VanillaDQN, AttentionQNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Standard DQN agent with:
      - epsilon-greedy exploration (decayed per episode)
      - target network (hard update every TARGET_UPDATE episodes)
      - uniform experience replay
    """

    def __init__(self, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Force CPU — avoids Windows CUDA driver errors on long runs
        self.device  = torch.device("cpu")
        self.epsilon = config.EPSILON_START
        self.steps   = 0

        self.online_net = VanillaDQN(
            config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM
        ).to(self.device)
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.LR_DQN)
        self.buffer    = ReplayBuffer(config.BUFFER_SIZE)

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(config.ACTION_DIM)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(state_t)
        return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END,
                          self.epsilon * config.EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    def train_step(self) -> float | None:
        if len(self.buffer) < config.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(config.BATCH_SIZE)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for taken actions
        q_values = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: online selects action, target evaluates it
        with torch.no_grad():
            best_actions = self.online_net(ns).argmax(dim=1, keepdim=True)
            q_next       = self.target_net(ns).gather(1, best_actions).squeeze(1)
            q_target     = r + config.GAMMA * q_next * (1 - d)

        loss = F.smooth_l1_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()


# ======================================================================

class AttentionDQNAgent:
    """
    Attention-augmented DQN agent.
    The key difference: the "state" fed to the network is a window of the
    last K raw states, concatenated. Self-attention across these K tokens
    lets the agent reason about temporal patterns.
    """

    def __init__(self, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Force CPU — avoids Windows CUDA driver errors on long runs
        self.device      = torch.device("cpu")
        self.epsilon     = config.EPSILON_START
        self.history_len = config.HISTORY_LEN
        self._history: deque = deque(maxlen=config.HISTORY_LEN)

        self.online_net = AttentionQNetwork(
            state_dim   = config.STATE_DIM,
            action_dim  = config.ACTION_DIM,
            history_len = config.HISTORY_LEN,
            embed_dim   = config.EMBED_DIM,
            num_heads   = config.NUM_HEADS,
            hidden_dim  = config.HIDDEN_DIM,
        ).to(self.device)
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.LR_ATTN)
        self.buffer    = ReplayBuffer(config.BUFFER_SIZE)

    # ------------------------------------------------------------------
    def reset_history(self, initial_state: np.ndarray):
        """Call at the start of each episode to pad history."""
        self._history.clear()
        for _ in range(self.history_len):
            self._history.append(initial_state.copy())

    def push_state(self, state: np.ndarray):
        self._history.append(state.copy())

    def get_history(self) -> np.ndarray:
        """Returns the flattened history: shape (K * state_dim,)"""
        return np.concatenate(list(self._history), axis=0)

    # ------------------------------------------------------------------
    def select_action(self, history: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(config.ACTION_DIM)
        h_t = torch.FloatTensor(history).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(h_t)
        return int(q.argmax(dim=1).item())

    def store(self, history, action, reward, next_history, done):
        self.buffer.push(history, action, reward, next_history, done)

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END,
                          self.epsilon * config.EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    def train_step(self) -> float | None:
        if len(self.buffer) < config.BATCH_SIZE:
            return None

        histories, actions, rewards, next_histories, dones = \
            self.buffer.sample(config.BATCH_SIZE)

        h  = torch.FloatTensor(histories).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        nh = torch.FloatTensor(next_histories).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        q_values = self.online_net(h).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self.online_net(nh).argmax(dim=1, keepdim=True)
            q_next       = self.target_net(nh).gather(1, best_actions).squeeze(1)
            q_target     = r + config.GAMMA * q_next * (1 - d)

        loss = F.smooth_l1_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()