#!/usr/bin/env python3
"""
evaluate.py — Evaluate the trained REINFORCE policy on CartPole-v1.

Runs the policy *greedily* (argmax, not sampling) for 100 episodes and
reports the average reward (FR-8).  The CartPole-v1 solved threshold is
an average reward ≥ 475 over 100 consecutive episodes.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/policy_reinforce.pt --episodes 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from policy import PolicyNetwork


def evaluate(
    policy: PolicyNetwork,
    env: gym.Env,
    num_episodes: int = 100,
) -> list[float]:
    """Run the trained policy greedily for *num_episodes* and collect rewards.

    Greedy means argmax over action probabilities — no sampling, no
    exploration.  This is the correct evaluation protocol (PRD Section 5.1,
    FR-8).

    Args:
        policy: A trained PolicyNetwork in eval mode.
        env: The CartPole-v1 environment.
        num_episodes: How many episodes to run.

    Returns:
        List of total rewards, one per episode.
    """
    policy.eval()
    episode_rewards: list[float] = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                probs = policy(state_tensor)
                action = torch.argmax(probs).item()  # greedy — argmax

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)

    return episode_rewards


def main(args: argparse.Namespace) -> None:
    """Load checkpoint, evaluate, report results."""
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print("   Run  python train.py  first to train the policy.")
        return

    # Load the trained policy
    policy = PolicyNetwork(hidden_dim=args.hidden_dim)
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    print(f"✅ Loaded policy from {ckpt_path}")

    env = gym.make("CartPole-v1")

    # Evaluate
    print(f"🔍 Evaluating over {args.episodes} episodes (greedy / argmax) …\n")
    rewards = evaluate(policy, env, args.episodes)

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print(f"  Average reward : {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"  Min / Max      : {min_reward:.0f} / {max_reward:.0f}")
    print()

    solved = avg_reward >= 475
    if solved:
        print(f"  🏆 SOLVED!  Average {avg_reward:.1f} ≥ 475 threshold.")
    else:
        print(f"  ⚠️  Not yet solved — average {avg_reward:.1f} < 475.")
        print("   Try training for more episodes or tuning hyperparameters.")

    # Save evaluation log
    log_path = Path("experiments/evaluation_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Average reward: {avg_reward:.2f}\n")
        f.write(f"Std: {std_reward:.2f}\n")
        f.write(f"Min: {min_reward:.0f}\n")
        f.write(f"Max: {max_reward:.0f}\n")
        f.write(f"Solved: {solved}\n\n")
        f.write("Per-episode rewards:\n")
        for i, r in enumerate(rewards, 1):
            f.write(f"  Episode {i:>3d}: {r:.0f}\n")
    print(f"\n  📝 Evaluation log saved → {log_path}")

    env.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained REINFORCE policy on CartPole-v1"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/policy_reinforce.pt",
                        help="Path to the saved model weights")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes (default: 100)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden layer size — must match the training config")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
