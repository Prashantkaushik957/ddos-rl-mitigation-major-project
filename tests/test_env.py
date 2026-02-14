"""
Tests for the SDN Gymnasium environment.
"""

import numpy as np
import pytest

from src.env.sdn_env import SDNEnvironment
from src.config import NUM_ACTIONS, AUTOENCODER_LATENT_DIM


@pytest.fixture
def sample_env():
    """Create a small test environment."""
    np.random.seed(42)
    n_samples = 100
    features = np.random.randn(n_samples, AUTOENCODER_LATENT_DIM).astype(np.float32)
    labels = np.random.randint(0, 2, size=n_samples).astype(np.int64)
    return SDNEnvironment(features, labels, batch_size=50, seed=42)


class TestSDNEnvironment:

    def test_observation_space_shape(self, sample_env):
        """Verify obs space matches expected latent dimensions."""
        assert sample_env.observation_space.shape == (AUTOENCODER_LATENT_DIM,)

    def test_action_space_size(self, sample_env):
        """Verify we have exactly 3 actions."""
        assert sample_env.action_space.n == NUM_ACTIONS
        assert sample_env.action_space.n == 3

    def test_reset_returns_valid_obs(self, sample_env):
        """Verify reset returns observation of correct shape."""
        obs, info = sample_env.reset()
        assert obs.shape == (AUTOENCODER_LATENT_DIM,)
        assert isinstance(info, dict)

    def test_step_returns_correct_format(self, sample_env):
        """Verify step returns (obs, reward, terminated, truncated, info)."""
        sample_env.reset()
        obs, reward, terminated, truncated, info = sample_env.step(0)
        assert obs.shape == (AUTOENCODER_LATENT_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_completes(self, sample_env):
        """Verify episode terminates after processing all flows."""
        obs, _ = sample_env.reset()
        total_reward = 0
        done = False
        truncated = False
        steps = 0

        while not (done or truncated):
            action = sample_env.action_space.sample()
            obs, reward, done, truncated, info = sample_env.step(action)
            total_reward += reward
            steps += 1

        assert steps <= 50  # batch_size
        assert "accuracy" in info
        assert "f1_score" in info

    def test_reward_values(self, sample_env):
        """Verify reward values are within expected ranges."""
        sample_env.reset()
        rewards = []
        for _ in range(10):
            action = sample_env.action_space.sample()
            _, reward, _, _, _ = sample_env.step(action)
            rewards.append(reward)

        # Rewards should be within the defined range
        for r in rewards:
            assert -3.0 <= r <= 2.0

    def test_info_contains_metrics(self, sample_env):
        """Verify info dict contains all required metrics."""
        sample_env.reset()
        _, _, _, _, info = sample_env.step(0)

        required_keys = [
            "step", "total_flows", "episode_reward",
            "accuracy", "precision", "recall", "f1_score",
            "true_positives", "false_positives",
            "true_negatives", "false_negatives", "fp_rate",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_metrics_consistency(self, sample_env):
        """Verify TP + FP + TN + FN = total steps."""
        obs, _ = sample_env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action = sample_env.action_space.sample()
            obs, _, done, truncated, info = sample_env.step(action)

        total = info["true_positives"] + info["false_positives"] + \
                info["true_negatives"] + info["false_negatives"]
        assert total == info["step"]
