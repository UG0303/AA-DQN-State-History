# models.py — Q-network architectures

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaDQN(nn.Module):
    """Standard MLP Q-network used as the baseline."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, state_dim)
        return self.net(x)


class AttentionQNetwork(nn.Module):
    """
    Attention-augmented Q-network.

    Input: concatenation of K recent states → shape (batch, K * state_dim)
    Architecture:
        1. Linear projection of each state → token embeddings
        2. Multi-head self-attention across the K tokens
        3. Take the last token's attended representation
        4. MLP Q-head → action values
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_len: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.state_dim   = state_dim
        self.history_len = history_len
        self.embed_dim   = embed_dim

        # Project each raw state to a token embedding
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
        )

        # Learnable positional embeddings (one per history position)
        self.pos_embedding = nn.Embedding(history_len, embed_dim)

        # Self-attention over the K tokens
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm + Q-head
        self.norm  = nn.LayerNorm(embed_dim)
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, history_len * state_dim)
        batch = x.shape[0]

        # Reshape → (batch, K, state_dim)
        tokens = x.view(batch, self.history_len, self.state_dim)

        # Encode each state token → (batch, K, embed_dim)
        tokens = self.state_encoder(tokens)

        # Add positional embeddings
        positions = torch.arange(self.history_len, device=x.device)
        tokens = tokens + self.pos_embedding(positions)  # broadcast over batch

        # Self-attention
        attn_out, _ = self.attention(tokens, tokens, tokens)  # (batch, K, embed_dim)
        attn_out = self.norm(attn_out)

        # Use the most-recent token's representation (last position)
        last_token = attn_out[:, -1, :]  # (batch, embed_dim)

        return self.q_head(last_token)
