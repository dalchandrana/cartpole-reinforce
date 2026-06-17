"""
Utility functions for the CartPole REINFORCE project.

Contains:
    - Reward-to-go (discounted return) computation  (FR-3)
    - Baseline subtraction                           (FR-4)
    - Training reward curve plotting                 (FR-9)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless / SSH sessions
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_rewards_to_go(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted reward-to-go for every timestep, working *backward*.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Implementation works from the last timestep backward so each G_t is
    computed in O(1) from G_{t+1}.

    Args:
        rewards: List of per-timestep rewards from a single episode.
        gamma: Discount factor (default 0.99 per PRD Section 3.5).

    Returns:
        List of discounted returns, same length as *rewards*.
    """
    rewards_to_go: List[float] = []
    running_return: float = 0.0
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        rewards_to_go.insert(0, running_return)
    return rewards_to_go


def apply_baseline(returns: torch.Tensor) -> torch.Tensor:
    """Subtract the batch-mean return as a simple baseline (PRD Section 3.6).

    This does NOT change the expected gradient direction but dramatically
    reduces variance, leading to faster and more stable learning.

    Args:
        returns: 1-D tensor of discounted returns for one episode.

    Returns:
        Normalised returns (zero-mean).  If std ≈ 0 we skip division
        to avoid numerical issues on very short / constant-reward episodes.
    """
    mean = returns.mean()
    std = returns.std()
    if std < 1e-8:
        return returns - mean
    return (returns - mean) / (std + 1e-8)


def plot_training_curve(
    rewards: List[float],
    save_path: str | Path = "experiments/reward_curve.png",
    window: int = 50,
) -> None:
    """Plot episode reward curve with a rolling-average overlay (FR-9).

    Args:
        rewards: List of total rewards per episode (one entry per episode).
        save_path: Where to save the PNG.
        window: Window size for the rolling average line.
    """
    episodes = np.arange(1, len(rewards) + 1)
    rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="#5e81ac", label="Episode reward")
    ax.plot(
        episodes[window - 1 :],
        rolling_avg,
        color="#bf616a",
        linewidth=2,
        label=f"Rolling avg (window={window})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("REINFORCE on CartPole-v1 — Training Reward Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Reward curve saved → {save_path}")


def save_reward_log(
    rewards: List[float],
    save_path: str | Path = "experiments/reward_log.csv",
) -> None:
    """Persist per-episode rewards to a CSV file for later analysis (FR-7).

    Args:
        rewards: Total reward per episode.
        save_path: Destination CSV path.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("episode,reward\n")
        for i, r in enumerate(rewards, start=1):
            f.write(f"{i},{r}\n")
    print(f"[utils] Reward log saved → {save_path}")
