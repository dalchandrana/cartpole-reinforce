"""
PolicyNetwork — a small feedforward neural network for CartPole-v1.

Architecture (from PRD Section 4.2):
    Input:  4 units  (cart position, cart velocity, pole angle, pole angular velocity)
    Hidden: 128 units, ReLU activation
    Output: 2 units  (logits for push-left / push-right)
    Final:  Softmax → valid probability distribution over actions
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Two-layer feedforward policy network mapping state → action probabilities."""

    def __init__(self, state_dim: int = 4, hidden_dim: int = 128, action_dim: int = 2) -> None:
        """Initialise the policy network.

        Args:
            state_dim: Dimension of the observation / state vector.
            hidden_dim: Number of units in the hidden layer.
            action_dim: Number of discrete actions available.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state tensor → action probability distribution.

        Args:
            state: A tensor of shape (state_dim,) or (batch, state_dim).

        Returns:
            Action probabilities of shape (action_dim,) or (batch, action_dim).
        """
        return self.network(state)

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Sample an action from the policy and return it with its log-probability.

        This uses *sampling* (not argmax) — sampling is essential for exploration
        during training (see PRD Section 4.3, step 3).

        Args:
            state: A 1-D tensor of shape (state_dim,).

        Returns:
            action: The integer action sampled from the distribution.
            log_prob: The log-probability of the sampled action (needed for the loss).
        """
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
