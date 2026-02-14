"""
Deep Q-Network (DQN) Agent.

Standard DQN with experience replay buffer, epsilon-greedy exploration,
and soft target network updates.
"""

import logging
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.networks import QNetwork
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


class ReplayBuffer:
    """Fixed-size circular buffer for storing experience tuples."""

    def __init__(self, capacity: int = DQN_REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for DDoS mitigation in SDN.

    Features:
        - Experience replay for stable training
        - Epsilon-greedy exploration with linear decay
        - Soft target network updates (Polyak averaging)
    """

    def __init__(
        self,
        state_dim: int = AUTOENCODER_LATENT_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: list = None,
        lr: float = DQN_LEARNING_RATE,
        gamma: float = DQN_GAMMA,
        epsilon_start: float = DQN_EPSILON_START,
        epsilon_end: float = DQN_EPSILON_END,
        epsilon_decay_steps: int = DQN_EPSILON_DECAY_STEPS,
        buffer_size: int = DQN_REPLAY_BUFFER_SIZE,
        batch_size: int = DQN_BATCH_SIZE,
        tau: float = DQN_TARGET_UPDATE_TAU,
        device: Optional[str] = None,
    ):
        if hidden_dims is None:
            hidden_dims = DQN_HIDDEN_DIMS

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_step = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Tracking
        self.total_steps = 0
        self.training_losses = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current observation.
            deterministic: If True, always exploit (no exploration).

        Returns:
            Selected action index.
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store a transition and update epsilon."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_step
            self.epsilon = max(self.epsilon, self.epsilon_end)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step on a minibatch from the replay buffer.

        Returns:
            Training loss, or None if buffer too small.
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

        # Current Q-values
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1)[0]
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

    def _soft_update(self):
        """Polyak averaging to update target network."""
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: Optional[str] = None):
        """Save agent state."""
        if path is None:
            path = str(MODELS_DIR / "dqn_agent.pt")
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_losses": self.training_losses,
        }, path)
        logger.info(f"DQN agent saved to {path}")

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        self.training_losses = checkpoint["training_losses"]
        logger.info(f"DQN agent loaded from {path}")
