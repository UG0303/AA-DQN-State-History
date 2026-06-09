# plot_results.py
# Generates all paper figures from results JSON files.
#
# Required files in results/ folder:
#   results.json          — main DQN vs AA-DQN comparison (3 seeds)
#   ablation_k.json       — K sweep ablation (K=1,2,4,8 x 3 seeds)
#   framestack_results.json — FrameStack baseline (K=1,2,4 x 3 seeds)
#   extra_results.json    — LR control + extra seeds
#
# Usage:
#   python plot_results.py
#
# Output: figures/ folder with PNG and PDF versions of all plots

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─── Paths ────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

RESULTS_PATH    = os.path.join(RESULTS_DIR, "results.json")
ABLATION_PATH   = os.path.join(RESULTS_DIR, "ablation_k.json")
FRAMESTACK_PATH = os.path.join(RESULTS_DIR, "framestack_results.json")
EXTRA_PATH      = os.path.join(RESULTS_DIR, "extra_results.json")

# ─── Constants ────────────────────────────────────────────────────────
SEEDS       = [42, 123, 456]
SEEDS_EXTRA = [7, 11]
K_VALUES    = [1, 2, 4, 8]

C = {
    "dqn":   "#4878CF",
    "aadqn": "#E55C47",
    "fs":    "#16A34A",
    "lr":    "#0891B2",
}

LABELS = {
    "dqn":   "Vanilla DQN (lr=1e-3)",
    "aadqn": "AA-DQN (ours)",
    "fs":    "FrameStack MLP",
    "lr":    "DQN control (lr=3e-4)",
}


# ─── Helpers ──────────────────────────────────────────────────────────
def smooth(data: list, window: int = 30) -> np.ndarray:
    arr = np.array(data, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode="same")


def save(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {name}.png / .pdf")
    plt.close(fig)


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}\nRun the training scripts first.")
    with open(path) as f:
        return json.load(f)


# ─── Figure 1: Training curves (DQN vs AA-DQN) ────────────────────────
def plot_training_curves(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4))

    for agent, color, label in [
        ("dqn",           C["dqn"],   LABELS["dqn"]),
        ("attention_dqn", C["aadqn"], "AA-DQN K=4 (lr=3e-4)"),
    ]:
        seed_returns = [results[agent][str(s)]["episode_returns"] for s in SEEDS]
        min_len = min(len(x) for x in seed_returns)
        arr  = np.array([x[:min_len] for x in seed_returns])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        eps  = np.arange(1, min_len + 1)
        ax.plot(eps, smooth(mean), label=label, color=color, linewidth=2)
        ax.fill_between(eps, smooth(mean - std), smooth(mean + std),
                        alpha=0.2, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("Training Curves on Acrobot-v1 (mean ± std, 3 seeds)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig1_training_curves")


# ─── Figure 2: Greedy eval curves ─────────────────────────────────────
def plot_eval_curves(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4))

    for agent, color, label in [
        ("dqn",           C["dqn"],   LABELS["dqn"]),
        ("attention_dqn", C["aadqn"], "AA-DQN K=4"),
    ]:
        all_eps, all_ret = [], []
        for s in SEEDS:
            ev = results[agent][str(s)]["eval_returns"]
            all_eps.append([x[0] for x in ev])
            all_ret.append([x[1] for x in ev])
        eps  = np.array(all_eps[0])
        mean = np.array(all_ret).mean(axis=0)
        std  = np.array(all_ret).std(axis=0)
        ax.plot(eps, mean, label=label, color=color,
                linewidth=2, marker="o", markersize=3)
        ax.fill_between(eps, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Greedy Eval Return", fontsize=12)
    ax.set_title("Evaluation Returns on Acrobot-v1 (mean ± std, 3 seeds)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig2_eval_curves")


# ─── Figure 3: K Ablation — AA-DQN vs FrameStack ──────────────────────
def plot_ablation(ablation: dict, framestack: dict, extra: dict):
    fig, ax = plt.subplots(figsize=(8, 4))

    # AA-DQN — K=1,2 get extra seeds; K=4,8 use original 3
    aa_means, aa_stds = [], []
    for k in K_VALUES:
        ms = [ablation[str(k)][str(s)]["last50_mean"] for s in SEEDS]
        if k in [1, 2] and extra:
            ms += [extra[f"aadqn_extra_k{k}"][str(s)]["last50_mean"]
                   for s in SEEDS_EXTRA]
        aa_means.append(np.mean(ms))
        aa_stds.append(np.std(ms))

    # FrameStack — K=1,2,4
    fs_k     = [1, 2, 4]
    fs_means = [np.mean([framestack[str(k)][str(s)]["last50_mean"]
                         for s in SEEDS]) for k in fs_k]
    fs_stds  = [np.std([framestack[str(k)][str(s)]["last50_mean"]
                        for s in SEEDS]) for k in fs_k]

    ax.errorbar(K_VALUES, aa_means, yerr=aa_stds,
                fmt="o-",  color=C["aadqn"], linewidth=2, markersize=8,
                capsize=5, label=LABELS["aadqn"])
    ax.errorbar(fs_k, fs_means, yerr=fs_stds,
                fmt="s--", color=C["fs"],    linewidth=2, markersize=8,
                capsize=5, label=LABELS["fs"])

    # Baseline reference lines
    ax.axhline(-171.0, color=C["dqn"], linewidth=1.8, linestyle=":",
               label=LABELS["dqn"])
    ax.axhline(-98.6,  color=C["lr"],  linewidth=1.8, linestyle=":",
               label=LABELS["lr"])
    ax.fill_between([0.5, 8.5], [-171.0-16, -171.0-16],
                    [-171.0+16, -171.0+16], alpha=0.08, color=C["dqn"])
    ax.fill_between([0.5, 8.5], [-98.6-6.8, -98.6-6.8],
                    [-98.6+6.8, -98.6+6.8], alpha=0.08, color=C["lr"])

    ax.set_xlabel("History Window Size K", fontsize=12)
    ax.set_ylabel("Mean Return (last 50 eps)", fontsize=12)
    ax.set_title(
        "K Ablation: AA-DQN vs FrameStack vs DQN Baselines on Acrobot-v1",
        fontsize=12)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig3_ablation_k")


# ─── Figure 4: Full comparison bar chart ──────────────────────────────
def plot_full_comparison(ablation: dict, framestack: dict, extra: dict):
    fig, ax = plt.subplots(figsize=(9, 4))

    dqn_lr_means = [extra["dqn_lr3e4"][str(s)]["last50_mean"] for s in SEEDS]

    methods = [
        "DQN\n(lr=1e-3)",
        "DQN\n(lr=3e-4)",
        "FrameStack\nK=1",
        "FrameStack\nK=2",
        "AA-DQN\nK=1",
        "AA-DQN\nK=2",
        "AA-DQN\nK=4",
    ]
    aa_k1 = ([ablation["1"][str(s)]["last50_mean"] for s in SEEDS] +
             [extra["aadqn_extra_k1"][str(s)]["last50_mean"] for s in SEEDS_EXTRA])
    aa_k2 = ([ablation["2"][str(s)]["last50_mean"] for s in SEEDS] +
             [extra["aadqn_extra_k2"][str(s)]["last50_mean"] for s in SEEDS_EXTRA])

    means = [
        -171.0,
        np.mean(dqn_lr_means),
        np.mean([framestack["1"][str(s)]["last50_mean"] for s in SEEDS]),
        np.mean([framestack["2"][str(s)]["last50_mean"] for s in SEEDS]),
        np.mean(aa_k1),
        np.mean(aa_k2),
        np.mean([ablation["4"][str(s)]["last50_mean"] for s in SEEDS]),
    ]
    stds = [
        16.0,
        np.std(dqn_lr_means),
        np.std([framestack["1"][str(s)]["last50_mean"] for s in SEEDS]),
        np.std([framestack["2"][str(s)]["last50_mean"] for s in SEEDS]),
        np.std(aa_k1),
        np.std(aa_k2),
        np.std([ablation["4"][str(s)]["last50_mean"] for s in SEEDS]),
    ]
    colors = [C["dqn"], C["lr"], C["fs"], C["fs"],
              C["aadqn"], C["aadqn"], C["aadqn"]]

    bars = ax.bar(methods, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, width=0.6)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m - 8,
                f"{m:.1f}", ha="center", fontsize=9,
                fontweight="bold", color="white")

    ax.set_ylabel("Mean Return (last 50 eps)", fontsize=12)
    ax.set_title("Full Comparison: All Methods on Acrobot-v1", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [
        Patch(color=C["dqn"],   label="DQN baseline"),
        Patch(color=C["lr"],    label="DQN lr control"),
        Patch(color=C["fs"],    label="FrameStack MLP"),
        Patch(color=C["aadqn"], label="AA-DQN (ours)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_full_comparison")


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("Loading results...")
    results    = load_json(RESULTS_PATH)
    ablation   = load_json(ABLATION_PATH)
    framestack = load_json(FRAMESTACK_PATH)
    extra      = load_json(EXTRA_PATH)

    print("\nGenerating figures...")
    plot_training_curves(results)
    plot_eval_curves(results)
    plot_ablation(ablation, framestack, extra)
    plot_full_comparison(ablation, framestack, extra)

    print(f"\nAll figures saved to ./{FIGURES_DIR}/")
    print("Files: fig1_training_curves, fig2_eval_curves,")
    print("       fig3_ablation_k, fig4_full_comparison")
    print("       (each as .png and .pdf)")


if __name__ == "__main__":
    main()