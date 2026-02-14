"""
Proximal Policy Optimization (PPO) Agent.

This is the PRIMARY agent in our framework. PPO is an on-policy
actor-critic method that uses clipped surrogate objectives for
stable policy updates.

Key components:
    - ActorCritic network with shared backbone
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate loss with entropy bonus
    - Mini-batch updates over multiple epochs per rollout
"""

import logging
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.agents.networks import ActorCritic
from src.config import (
    PPO_LEARNING_RATE,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RATIO,
    PPO_ENTROPY_COEF,
    PPO_VALUE_LOSS_COEF,
    PPO_MAX_GRAD_NORM,
    PPO_UPDATE_EPOCHS,
    PPO_BATCH_SIZE,
    PPO_ROLLOUT_LENGTH,
    PPO_HIDDEN_DIMS,
    AUTOENCODER_LATENT_DIM,
    NUM_ACTIONS,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    Buffer for storing PPO rollout data.

    Stores a fixed-length trajectory of experiences and computes
    advantages using Generalized Advantage Estimation (GAE).
    """

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ):
        """Store one step of experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages.

        Args:
            last_value: V(s_T) — value of the final state.
            gamma: Discount factor.
            gae_lambda: GAE lambda for bias-variance tradeoff.

        Returns:
            (returns, advantages) — both as numpy arrays.
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = float(self.dones[t + 1])

            delta = (
                self.rewards[t]
                + gamma * next_value * (1 - next_done)
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages

    def get_batches(
        self,
        batch_size: int,
        returns: np.ndarray,
        advantages: np.ndarray,
    ):
        """
        Generate random mini-batches for PPO updates.

        Yields mini-batches of (states, actions, old_log_probs, returns, advantages).
        """
        n = len(self.states)
        indices = np.random.permutation(n)

        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield (
                states[batch_idx],
                actions[batch_idx],
                old_log_probs[batch_idx],
                returns[batch_idx],
                advantages[batch_idx],
            )

    def clear(self):
        """Clear the buffer for the next rollout."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization agent for DDoS mitigation.

    Features:
        - Clipped surrogate objective for stable policy updates
        - GAE for advantage estimation
        - Entropy bonus for exploration
        - Mini-batch updates over multiple epochs
    """

    def __init__(
        self,
        state_dim: int = AUTOENCODER_LATENT_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: list = None,
        lr: float = PPO_LEARNING_RATE,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
        clip_ratio: float = PPO_CLIP_RATIO,
        entropy_coef: float = PPO_ENTROPY_COEF,
        value_loss_coef: float = PPO_VALUE_LOSS_COEF,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        update_epochs: int = PPO_UPDATE_EPOCHS,
        batch_size: int = PPO_BATCH_SIZE,
        rollout_length: int = PPO_ROLLOUT_LENGTH,
        device: Optional[str] = None,
    ):
        if hidden_dims is None:
            hidden_dims = PPO_HIDDEN_DIMS

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length

        # Network
        self.network = ActorCritic(
            state_dim, action_dim, hidden_dims
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Tracking
        self.total_steps = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using the current policy.

        Args:
            state: Current observation.
            deterministic: If True, take argmax action.

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            action, log_prob, value = self.network.get_action(
                state_t, deterministic=deterministic
            )
        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ):
        """Store a transition in the rollout buffer."""
        self.buffer.push(state, action, reward, log_prob, value, done)
        self.total_steps += 1

    def update(self, last_value: float) -> dict:
        """
        Perform PPO update using collected rollout.

        Args:
            last_value: V(s_T) from the critic for bootstrapping.

        Returns:
            Dictionary with loss statistics.
        """
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.update_epochs):
            for batch in self.buffer.get_batches(
                self.batch_size, returns, advantages
            ):
                states_b, actions_b, old_log_probs_b, returns_b, advantages_b = batch

                # Convert to tensors
                states_t = torch.FloatTensor(states_b).to(self.device)
                actions_t = torch.LongTensor(actions_b).to(self.device)
                old_log_probs_t = torch.FloatTensor(old_log_probs_b).to(self.device)
                returns_t = torch.FloatTensor(returns_b).to(self.device)
                advantages_t = torch.FloatTensor(advantages_b).to(self.device)

                # Evaluate actions under current policy
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    states_t, actions_t
                )

                # ──────────────────────────────────────
                # Policy loss (clipped surrogate)
                # ──────────────────────────────────────
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # ──────────────────────────────────────
                # Value loss
                # ──────────────────────────────────────
                value_loss = F.mse_loss(values, returns_t)

                # ──────────────────────────────────────
                # Entropy bonus
                # ──────────────────────────────────────
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        # Clear buffer
        self.buffer.clear()

        # Store losses
        avg_policy_loss = total_policy_loss / max(n_updates, 1)
        avg_value_loss = total_value_loss / max(n_updates, 1)
        avg_entropy = total_entropy / max(n_updates, 1)

        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy)

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "n_updates": n_updates,
        }

    def save(self, path: Optional[str] = None):
        """Save PPO agent state."""
        if path is None:
            path = str(MODELS_DIR / "ppo_agent.pt")
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropy_losses": self.entropy_losses,
        }, path)
        logger.info(f"PPO agent saved to {path}")

    def load(self, path: str):
        """Load PPO agent state."""
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.policy_losses = checkpoint["policy_losses"]
        self.value_losses = checkpoint["value_losses"]
        self.entropy_losses = checkpoint["entropy_losses"]
        logger.info(f"PPO agent loaded from {path}")
