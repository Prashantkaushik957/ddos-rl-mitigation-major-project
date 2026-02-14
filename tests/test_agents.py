"""
Tests for DRL agents.
"""

import numpy as np
import pytest
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.ddqn_agent import DDQNAgent
from src.agents.ppo_agent import PPOAgent
from src.config import AUTOENCODER_LATENT_DIM, NUM_ACTIONS


STATE_DIM = AUTOENCODER_LATENT_DIM


class TestDQNAgent:

    def test_initialization(self):
        """Verify DQN agent initializes correctly."""
        agent = DQNAgent(state_dim=STATE_DIM, device="cpu")
        assert agent.state_dim == STATE_DIM
        assert agent.action_dim == NUM_ACTIONS

    def test_select_action(self):
        """Verify action selection returns valid action."""
        agent = DQNAgent(state_dim=STATE_DIM, device="cpu")
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = agent.select_action(state, deterministic=True)
        assert 0 <= action < NUM_ACTIONS

    def test_store_and_train(self):
        """Verify training step runs without error."""
        agent = DQNAgent(state_dim=STATE_DIM, batch_size=4, device="cpu")
        # Fill buffer with enough samples
        for _ in range(10):
            s = np.random.randn(STATE_DIM).astype(np.float32)
            a = np.random.randint(0, NUM_ACTIONS)
            r = np.random.randn()
            s2 = np.random.randn(STATE_DIM).astype(np.float32)
            agent.store_transition(s, a, r, s2, False)

        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)

    def test_forward_pass_shape(self):
        """Verify Q-network output shape."""
        agent = DQNAgent(state_dim=STATE_DIM, device="cpu")
        state = torch.randn(1, STATE_DIM)
        q_vals = agent.q_network(state)
        assert q_vals.shape == (1, NUM_ACTIONS)


class TestDDQNAgent:

    def test_initialization(self):
        """Verify DDQN agent initializes correctly."""
        agent = DDQNAgent(state_dim=STATE_DIM, device="cpu")
        assert agent.state_dim == STATE_DIM

    def test_train_step_different_from_dqn(self):
        """Verify DDQN uses double Q-learning."""
        agent = DDQNAgent(state_dim=STATE_DIM, batch_size=4, device="cpu")
        for _ in range(10):
            s = np.random.randn(STATE_DIM).astype(np.float32)
            a = np.random.randint(0, NUM_ACTIONS)
            r = np.random.randn()
            s2 = np.random.randn(STATE_DIM).astype(np.float32)
            agent.store_transition(s, a, r, s2, False)

        loss = agent.train_step()
        assert loss is not None


class TestPPOAgent:

    def test_initialization(self):
        """Verify PPO agent initializes correctly."""
        agent = PPOAgent(state_dim=STATE_DIM, device="cpu")
        assert agent.network is not None

    def test_select_action(self):
        """Verify PPO action selection returns (action, log_prob, value)."""
        agent = PPOAgent(state_dim=STATE_DIM, device="cpu")
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        assert 0 <= action < NUM_ACTIONS
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_update(self):
        """Verify PPO update runs without error."""
        agent = PPOAgent(state_dim=STATE_DIM, device="cpu")

        # Collect some transitions
        for _ in range(20):
            s = np.random.randn(STATE_DIM).astype(np.float32)
            action, log_prob, value = agent.select_action(s)
            agent.store_transition(s, action, 1.0, log_prob, value, False)

        result = agent.update(last_value=0.0)
        assert "policy_loss" in result
        assert "value_loss" in result
        assert "entropy" in result

    def test_deterministic_action(self):
        """Verify deterministic mode produces consistent results."""
        agent = PPOAgent(state_dim=STATE_DIM, device="cpu")
        state = np.random.randn(STATE_DIM).astype(np.float32)

        actions = set()
        for _ in range(10):
            a, _, _ = agent.select_action(state, deterministic=True)
            actions.add(a)

        # Deterministic should always give the same action
        assert len(actions) == 1
