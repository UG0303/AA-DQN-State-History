# train.py — train vanilla DQN and Attention-DQN on Acrobot-v1

import json
import os
import time
from typing import Literal

import gymnasium as gym
import numpy as np

import config
from agent import DQNAgent, AttentionDQNAgent


AgentType = Literal["dqn", "attention_dqn"]


# ----------------------------------------------------------------------
def evaluate(agent, env, n_episodes: int = 5) -> float:
    returns = []
    for _ in range(n_episodes):
        if isinstance(agent, AttentionDQNAgent):
            obs, _ = env.reset()
            agent.reset_history(obs)
            agent.push_state(obs)
            done = False
            total = 0.0
            while not done:
                hist = agent.get_history()
                action = agent.select_action(hist, greedy=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                agent.push_state(obs)
                done = terminated or truncated
                total += reward
        else:
            obs, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                action = agent.select_action(obs, greedy=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
        returns.append(total)
    return float(np.mean(returns))


# ----------------------------------------------------------------------
def train_one_seed(agent_type: AgentType, seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  {agent_type.upper()}  |  seed={seed}")
    print(f"{'='*60}")

    env      = gym.make(config.ENV_NAME)
    eval_env = gym.make(config.ENV_NAME)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)

    agent = DQNAgent(seed=seed) if agent_type == "dqn" else AttentionDQNAgent(seed=seed)

    episode_returns: list[float] = []
    eval_returns:    list[tuple] = []
    t0 = time.time()

    for ep in range(1, config.NUM_EPISODES + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0

        if isinstance(agent, AttentionDQNAgent):
            agent.reset_history(obs)
            agent.push_state(obs)

        done = False
        while not done:
            if isinstance(agent, AttentionDQNAgent):
                hist   = agent.get_history()
                action = agent.select_action(hist)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.push_state(next_obs)
                next_hist = agent.get_history()
                agent.store(hist, action, reward, next_hist, float(done))
            else:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store(obs, action, reward, next_obs, float(done))
                obs = next_obs

            agent.train_step()
            ep_return += reward

        agent.decay_epsilon()
        if ep % config.TARGET_UPDATE == 0:
            agent.update_target()

        episode_returns.append(ep_return)

        if ep % config.EVAL_EVERY == 0:
            mean_eval = evaluate(agent, eval_env, config.EVAL_EPISODES)
            eval_returns.append((ep, mean_eval))
            avg_100 = np.mean(episode_returns[-100:])
            print(
                f"  ep {ep:4d} | "
                f"avg-100: {avg_100:7.1f} | "
                f"eval: {mean_eval:7.1f} | "
                f"eps: {agent.epsilon:.3f}"
            )

    total_time = time.time() - t0
    env.close()
    eval_env.close()

    return {
        "episode_returns": episode_returns,
        "eval_returns":    eval_returns,
        "total_time_s":    round(total_time, 1),
    }


# ----------------------------------------------------------------------
def run_all():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "results.json")

    # Load existing results (we already have good DQN results from before)
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {out_path}")
    else:
        all_results = {"dqn": {}, "attention_dqn": {}}

    for agent_type in ("dqn", "attention_dqn"):
        for seed in config.SEEDS:
            if str(seed) in all_results.get(agent_type, {}):
                print(f"\nSkipping {agent_type} seed={seed} (already done)")
                continue
            result = train_one_seed(agent_type, seed)
            all_results[agent_type][str(seed)] = result
            # Save after every seed — crash-safe
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  --> Saved progress to {out_path}")

    print(f"\nAll done! Results at {out_path}")
    return all_results


# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_all()