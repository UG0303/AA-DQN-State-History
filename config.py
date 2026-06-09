# config.py — all hyperparameters in one place

ENV_NAME       = "Acrobot-v1"
STATE_DIM      = 6       # Acrobot-v1 observation space
ACTION_DIM     = 3       # Acrobot-v1 action space

# --- Shared DQN hyperparameters ---
LR_DQN         = 1e-3    # vanilla DQN learning rate (proven to work)
LR_ATTN        = 3e-4    # attention-DQN needs smaller LR (larger model)
GAMMA          = 0.99
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.997   # slower decay → more exploration for harder attn model
BATCH_SIZE     = 64
BUFFER_SIZE    = 50_000
TARGET_UPDATE  = 10      # update target net every N episodes
HIDDEN_DIM     = 128
NUM_EPISODES   = 800     # more episodes so attention has time to converge
SEEDS          = [42, 123, 456]

# --- Attention-DQN specific ---
HISTORY_LEN    = 4       # K: number of past states fed to attention
EMBED_DIM      = 64      # projection dimension per state token
NUM_HEADS      = 4       # must divide EMBED_DIM evenly

# --- Logging ---
EVAL_EVERY     = 10      # evaluate (no exploration) every N episodes
EVAL_EPISODES  = 5
RESULTS_DIR    = "results"