#!/usr/bin/env python3
"""
train.py — Main REINFORCE training loop for CartPole-v1.

Covers functional requirements:
    FR-2  Episode collection
    FR-5  REINFORCE loss + gradient sanity check
    FR-6  Training until solved threshold
    FR-7  Per-episode reward logging
    FR-9  Reward curve plot on completion
    FR-10 Model checkpoint saving

Usage:
    python train.py                        # defaults
    python train.py --episodes 2000 --lr 0.01 --gamma 0.99 --seed 42
    python train.py --no-baseline          # ablation: disable baseline subtraction
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from policy import PolicyNetwork
from utils import (
    apply_baseline,
    compute_rewards_to_go,
    plot_training_curve,
    save_reward_log,
)


# ---------------------------------------------------------------------------
# Episode collection (FR-2)
# ---------------------------------------------------------------------------

def collect_episode(
    env: gym.Env,
    policy: PolicyNetwork,
) -> Tuple[List[torch.Tensor], List[float]]:
    """Run one full episode using *policy*, recording log-probs and rewards.

    The agent samples actions (not argmax) to ensure exploration during
    training — see PRD Section 4.3 step 3.

    Args:
        env: A Gymnasium CartPole-v1 environment instance.
        policy: The current policy network.

    Returns:
        log_probs: List of log-probability tensors, one per timestep.
        rewards: List of float rewards, one per timestep.
    """
    state, _ = env.reset()
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state)
        action, log_prob = policy.select_action(state_tensor)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(float(reward))
        state = next_state

    return log_probs, rewards


# ---------------------------------------------------------------------------
# REINFORCE update (FR-5)
# ---------------------------------------------------------------------------

def reinforce_update(
    optimizer: optim.Optimizer,
    log_probs: List[torch.Tensor],
    rewards: List[float],
    gamma: float,
    use_baseline: bool = True,
) -> float:
    """Compute REINFORCE loss, backpropagate, and step the optimiser.

    Loss = -mean( log_prob(a_t) * G_t )     (PRD Section 3.4)

    Args:
        optimizer: The Adam (or other) optimiser wrapping the policy params.
        log_probs: Log-probabilities of actions taken during the episode.
        rewards: Raw per-timestep rewards from the episode.
        gamma: Discount factor for reward-to-go.
        use_baseline: Whether to subtract baseline (FR-4).

    Returns:
        The scalar loss value (for logging).
    """
    # FR-3: discounted reward-to-go, computed backward
    returns = compute_rewards_to_go(rewards, gamma)
    returns_tensor = torch.FloatTensor(returns)

    # FR-4: baseline subtraction
    if use_baseline:
        returns_tensor = apply_baseline(returns_tensor)

    # FR-5: REINFORCE policy gradient loss
    policy_loss = []
    for log_prob, G in zip(log_probs, returns_tensor):
        policy_loss.append(-log_prob * G)

    loss = torch.stack(policy_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Gradient sanity check (FR-5 — "verify gradients flow correctly")
# ---------------------------------------------------------------------------

def gradient_sanity_check(policy: PolicyNetwork, env: gym.Env) -> None:
    """Run one episode, compute loss, and verify that all gradients are non-zero.

    This catches silent bugs like detached tensors or frozen layers.
    """
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    log_probs, rewards = collect_episode(env, policy)
    reinforce_update(optimizer, log_probs, rewards, gamma=0.99, use_baseline=True)

    all_ok = True
    for name, param in policy.named_parameters():
        if param.grad is None or torch.all(param.grad == 0):
            print(f"  ⚠️  {name}: gradient is zero or None!")
            all_ok = False
        else:
            print(f"  ✓  {name}: grad norm = {param.grad.norm().item():.6f}")

    if all_ok:
        print("  ✅ Gradient sanity check PASSED — all gradients are non-zero.\n")
    else:
        print("  ❌ Gradient sanity check FAILED — some gradients are missing!\n")


# ---------------------------------------------------------------------------
# Main training loop (FR-6)
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Full REINFORCE training run.

    Args:
        args: Parsed CLI arguments (episodes, lr, gamma, seed, etc.).
    """
    # --- Reproducibility (NFR: random seed must be settable) ---------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Environment -------------------------------------------------------
    env = gym.make("CartPole-v1")

    # --- Policy & Optimiser ------------------------------------------------
    policy = PolicyNetwork(hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # --- Gradient sanity check (FR-5) --------------------------------------
    print("🔍 Running gradient sanity check …")
    gradient_sanity_check(policy, env)

    # Re-initialise after sanity check so the actual training starts clean
    policy = PolicyNetwork(hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Training loop (FR-6) ----------------------------------------------
    episode_rewards: List[float] = []
    start_time = time.time()

    print(f"🚀 Starting REINFORCE training — {args.episodes} episodes")
    print(f"   lr={args.lr}  gamma={args.gamma}  hidden={args.hidden_dim}  "
          f"baseline={'ON' if args.baseline else 'OFF'}  seed={args.seed}\n")

    for ep in range(1, args.episodes + 1):
        log_probs, rewards = collect_episode(env, policy)
        loss = reinforce_update(
            optimizer, log_probs, rewards, args.gamma, use_baseline=args.baseline
        )
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Logging (FR-7)
        if ep % 50 == 0 or ep == 1:
            recent_avg = np.mean(episode_rewards[-50:])
            elapsed = time.time() - start_time
            print(
                f"  Episode {ep:>5d} | "
                f"Reward {total_reward:>6.1f} | "
                f"Avg(50) {recent_avg:>6.1f} | "
                f"Loss {loss:>8.4f} | "
                f"Time {elapsed:>6.1f}s"
            )

    elapsed = time.time() - start_time
    print(f"\n✅ Training complete — {args.episodes} episodes in {elapsed:.1f}s")

    # --- Save model checkpoint (FR-10) -------------------------------------
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "policy_reinforce.pt"
    torch.save(policy.state_dict(), ckpt_path)
    print(f"💾 Model saved → {ckpt_path}")

    # --- Save experiment config alongside the model ------------------------
    config = {
        "episodes": args.episodes,
        "lr": args.lr,
        "gamma": args.gamma,
        "hidden_dim": args.hidden_dim,
        "baseline": args.baseline,
        "seed": args.seed,
        "final_avg_reward_50": float(np.mean(episode_rewards[-50:])),
        "training_time_seconds": round(elapsed, 2),
    }
    config_path = ckpt_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Config saved → {config_path}")

    # --- Reward log + plot (FR-7, FR-9) ------------------------------------
    save_reward_log(episode_rewards)
    plot_training_curve(episode_rewards)

    env.close()
    print("\n🎯 Next step: run  python evaluate.py  to test the trained policy.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults from the PRD."""
    parser = argparse.ArgumentParser(
        description="Train a REINFORCE agent on CartPole-v1"
    )
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training episodes (default: 2000)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for Adam (default: 0.01)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden layer size (default: 128)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false",
                        help="Disable baseline subtraction (ablation)")
    parser.set_defaults(baseline=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
