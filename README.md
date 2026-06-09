# Beyond Memoryless Control: Attention-Augmented DQN with State History Windows

**NTU Deep Reinforcement Learning Mini-Conference 2026**  
Udit Goyal · National Taiwan University (Exchange Student)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green.svg)](https://gymnasium.farama.org/)

---

## Overview

Standard DQN estimates Q-values from a single observation, ignoring temporal context. This project proposes **AA-DQN (Attention-Augmented DQN)**, which maintains a fixed-length window of K recent states and processes them through a lightweight multi-head self-attention module.

Through systematic experimentation and peer review, we discovered that:

1. **Learning rate is the primary driver** — DQN at lr=3e-4 (matching AA-DQN) achieves −98.6, explaining most of the apparent gain
2. **FrameStack outperforms AA-DQN** at all K values with 6× fewer parameters
3. **K=2 is the sweet spot** for FrameStack (−101.4 ± 2.2), with the lowest variance of any configuration tested
4. **Attention variance explodes at K=4** (±52.8 vs ±2.2 for FrameStack K=2)

These are honest findings reported transparently, including results that do not favour the proposed method.

---

## Results Summary

| Method | Mean Return | Std | Params |
|--------|------------|-----|--------|
| DQN baseline (lr=1e-3) | −171.0 | ±16.0 | ~17K |
| DQN control (lr=3e-4) | −98.6 | ±6.8 | ~17K |
| FrameStack K=1 | −98.6 | ±6.8 | ~18K |
| **FrameStack K=2** | **−101.4** | **±2.2** | **~18K** |
| FrameStack K=4 | −140.3 | ±15.0 | ~20K |
| AA-DQN K=1 | −134.5 | ±28.4 | ~120K |
| AA-DQN K=2 | −129.7 | ±18.1 | ~120K |
| AA-DQN K=4 | −178.1 | ±52.8 | ~120K |

All results on **Acrobot-v1**, mean ± std over 3–5 random seeds, last 50 episodes.

---

## Project Structure

```
AA-DQN-State-History/
├── config.py               # All hyperparameters
├── models.py               # VanillaDQN and AttentionQNetwork
├── agent.py                # DQNAgent and AttentionDQNAgent
├── replay_buffer.py        # Uniform experience replay
├── train.py                # Main training script (DQN vs AA-DQN)
├── ablation.py             # K sweep ablation (K=1,2,4,8)
├── frame_stack_dqn.py      # FrameStack MLP baseline
├── extra_experiments.py    # LR control + extra seeds
├── plot_results.py         # Generate all figures from JSON results
├── sanity_check.py         # Verify agents run correctly
├── results/
│   ├── results.json            # DQN vs AA-DQN (3 seeds)
│   ├── ablation_k.json         # K ablation (K=1,2,4,8 x 3 seeds)
│   ├── framestack_results.json # FrameStack baseline
│   └── extra_results.json      # LR control + extra seeds
├── figures/
│   ├── fig1_training_curves.png
│   ├── fig2_eval_curves.png
│   ├── fig3_ablation_k.png
│   └── fig4_full_comparison.png
└── paper/
    ├── neurips_2026.tex
    └── AA_DQN_NTU_MiniConf2026.pdf
```

---

## Installation

```bash
pip install torch gymnasium numpy matplotlib
```

---

## Usage

### Train DQN vs AA-DQN (main comparison)
```bash
python train.py
```
Trains both agents across 3 seeds. Saves to `results/results.json`.

### Run K ablation
```bash
python ablation.py
```
Sweeps K ∈ {1, 2, 4, 8}. Saves to `results/ablation_k.json`.

### Run FrameStack baseline
```bash
python frame_stack_dqn.py
```
Trains FrameStack MLP at K=1,2,4. Saves to `results/framestack_results.json`.

### Run all additional experiments
```bash
python extra_experiments.py
```
Runs LR control (DQN at lr=3e-4) and extra seeds for K=1,2.

### Generate all figures
```bash
python plot_results.py
```
Requires all 4 JSON files in `results/`. Saves PNG and PDF figures to `figures/`.

---

## Key Findings

### Learning Rate Confound
The original comparison used DQN at lr=1e-3 vs AA-DQN at lr=3e-4. Running DQN at lr=3e-4 achieves −98.6, nearly matching AA-DQN K=2 (−117.9). **The LR difference explains most of the apparent gain.**

### FrameStack vs Attention
Simple state concatenation (FrameStack MLP) outperforms multi-head self-attention at K=1, 2, and 4 — with 6× fewer parameters and much lower variance. This suggests attention does not provide a meaningful advantage over simple history concatenation at small K on low-dimensional fully-observable tasks.

### K Sweep Trade-off
The non-monotonic K sweep (K=2 best for both methods, K=4 worst for AA-DQN) reveals a fundamental trade-off: larger history windows increase optimisation difficulty faster than they add useful temporal context.

---

## Hardware & Runtime

- **Hardware:** NVIDIA RTX 3050 GPU (Windows 11)
- **Training time:** ~1–2 hours per seed per agent
- **Full reproduction (all scripts):** ~12–15 hours total
- Note: CPU fallback used for long runs due to CUDA stability on Windows

---