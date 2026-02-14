"""
Double Deep Q-Network (DDQN) Agent.

Extends DQN by decoupling action selection from action evaluation
to reduce overestimation bias in Q-value estimates.

Key difference from DQN:
    - DQN:  target = r + γ * max_a Q_target(s', a)
    - DDQN: target = r + γ * Q_target(s', argmax_a Q_online(s', a))
"""

import logging
from typing import Optional

import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.config import (
    DQN_LEARNING_RATE,
    DQN_GAMMA,
    DQN_EPSILON_START,
    DQN_EPSILON_END,
    DQN_EPSILON_DECAY_STEPS,
    DQN_REPLAY_BUFFER_SIZE,
    DQN_BATCH_SIZE,
    DQN_TARGET_UPDATE_TAU,
    DQN_HIDDEN_DIMS,
    AUTOENCODER_LATENT_DIM,
    NUM_ACTIONS,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class DDQNAgent(DQNAgent):
    """
    Double DQN agent — inherits from DQN, overrides the training step
    to use the double Q-learning target.

    This reduces the well-known overestimation bias of standard DQN
    by using the online network to SELECT the best action, and the
    target network to EVALUATE it.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self) -> Optional[float]:
        """
        Training step using Double Q-Learning.

        Target: r + γ * Q_target(s', argmax_a Q_online(s', a))
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a)
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ──────────────────────────────────────
        # DOUBLE DQN TARGET
        # ──────────────────────────────────────
        with torch.no_grad():
            # Step 1: Use ONLINE network to select best actions
            next_q_online = self.q_network(next_states_t)
            best_actions = next_q_online.argmax(dim=1)

            # Step 2: Use TARGET network to evaluate those actions
            next_q_target = self.target_network(next_states_t)
            next_q_values = next_q_target.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            # Compute target
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        # Loss and update
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def save(self, path: Optional[str] = None):
        """Save DDQN agent state."""
        if path is None:
            path = str(MODELS_DIR / "ddqn_agent.pt")
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_losses": self.training_losses,
        }, path)
        logger.info(f"DDQN agent saved to {path}")
