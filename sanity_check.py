#!/usr/bin/env python3
# sanity_check.py — run 5 episodes of each agent to verify no errors

import gymnasium as gym
import config
from agent import DQNAgent, AttentionDQNAgent


def run_quick(agent_type: str, n_episodes: int = 5):
    env   = gym.make(config.ENV_NAME)
    seed  = 42
    agent = DQNAgent(seed=seed) if agent_type == "dqn" else AttentionDQNAgent(seed=seed)
    print(f"\n--- {agent_type.upper()} sanity check ({n_episodes} episodes) ---")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_return = 0.0
        done = False

        if isinstance(agent, AttentionDQNAgent):
            agent.reset_history(obs)
            agent.push_state(obs)

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

            loss = agent.train_step()
            ep_return += reward

        agent.decay_epsilon()
        print(f"  ep {ep+1}: return={ep_return:.1f}  "
              f"buffer={len(agent.buffer)}  "
              f"epsilon={agent.epsilon:.3f}  "
              f"loss={'n/a' if loss is None else f'{loss:.4f}'}")

    env.close()
    print(f"  ✓ {agent_type.upper()} OK")


if __name__ == "__main__":
    run_quick("dqn")
    run_quick("attention_dqn")
    print("\n✓ All checks passed. Run `python train.py` to start full training.")
