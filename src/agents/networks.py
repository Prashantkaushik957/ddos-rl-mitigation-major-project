"""
Neural network architectures for DRL agents.

Contains:
    - QNetwork: For DQN and Double DQN agents
    - ActorCritic: For PPO agent (shared backbone, separate heads)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from src.config import DQN_HIDDEN_DIMS, PPO_HIDDEN_DIMS, NUM_ACTIONS


class QNetwork(nn.Module):
    """
    Q-Network for DQN / Double DQN.

    Architecture: MLP with configurable hidden layers.
    Input → Hidden1 (ReLU, BN) → Hidden2 (ReLU, BN) → Q-values

    Output: Q-value for each action (no activation).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = DQN_HIDDEN_DIMS

        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions given a state.

        Args:
            state: (batch_size, state_dim)

        Returns:
            Q-values: (batch_size, action_dim)
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.

    Separates the value and advantage streams for better learning.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = DQN_HIDDEN_DIMS

        # Shared feature extraction backbone
        backbone_layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            backbone_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        self.backbone = nn.Sequential(*backbone_layers)

        last_hidden = hidden_dims[-1] if hidden_dims else state_dim

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values using dueling architecture.
        Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        """
        features = self.backbone(state)
        value = self.value_stream(features)            # (batch, 1)
        advantage = self.advantage_stream(features)    # (batch, action_dim)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Architecture:
        Shared backbone → Actor head (policy logits)
                        → Critic head (state value)

    The shared backbone learns common representations,
    while the heads specialize for policy and value estimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = PPO_HIDDEN_DIMS

        # Shared backbone
        backbone_layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh(),
            ])
            prev_dim = h_dim
        self.backbone = nn.Sequential(*backbone_layers)

        # Actor head: outputs policy logits
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # Critic head: outputs state value
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Smaller init for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network.

        Args:
            state: (batch_size, state_dim)

        Returns:
            Tuple of (action_logits, state_value)
            - action_logits: (batch_size, action_dim) — unnormalized log probs
            - state_value: (batch_size, 1) — estimated V(s)
        """
        features = self.backbone(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Sample an action from the policy.

        Args:
            state: Single state observation.
            deterministic: If True, take the action with highest probability.

        Returns:
            (action, log_prob, value)
        """
        logits, value = self.forward(state.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: Batch of states.
            actions: Batch of actions taken.

        Returns:
            (log_probs, values, entropy)
        """
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


