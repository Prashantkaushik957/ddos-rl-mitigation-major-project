"""
Custom Gymnasium environment simulating an SDN controller
making per-flow decisions for DDoS attack mitigation.

The agent observes latent features of each network flow and chooses
one of three actions: ALLOW, RATE_LIMIT, or DROP.
"""

import logging
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import (
    NUM_ACTIONS,
    ACTION_NAMES,
    BATCH_SIZE_ENV,
    FALSE_POSITIVE_THRESHOLD,
    REWARD_CORRECT_BENIGN,
    REWARD_CORRECT_ATTACK,
    REWARD_FALSE_POSITIVE,
    REWARD_FALSE_NEGATIVE,
    REWARD_RATE_LIMIT_ATTACK,
    AUTOENCODER_LATENT_DIM,
)

logger = logging.getLogger(__name__)


class SDNEnvironment(gym.Env):
    """
    SDN Controller Environment for DDoS Mitigation.

    This environment simulates an SDN controller that processes network
    flows one at a time. Each flow is characterized by a latent feature
    vector from the autoencoder. The agent must decide whether to ALLOW,
    RATE_LIMIT, or DROP each flow.

    Observation Space:
        Box of shape (latent_dim,) — latent features of the current flow.

    Action Space:
        Discrete(3):
            0 = ALLOW   — let the flow through
            1 = RATE_LIMIT — throttle the flow
            2 = DROP    — block the flow entirely

    Reward Structure:
        - Correctly allowing benign traffic: +1.0
        - Correctly dropping attack traffic: +2.0
        - Rate-limiting attack traffic: +1.5 (partial credit)
        - False positive (blocking benign): -3.0
        - False negative (allowing attack): -1.0

    Episode terminates when:
        - All flows in the batch have been processed, OR
        - False positive rate exceeds the threshold
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = BATCH_SIZE_ENV,
        latent_dim: int = AUTOENCODER_LATENT_DIM,
        fp_threshold: float = FALSE_POSITIVE_THRESHOLD,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            features: Feature matrix (n_samples, latent_dim).
            labels: Binary labels (0=benign, 1=attack).
            batch_size: Number of flows per episode.
            latent_dim: Dimensionality of the latent feature space.
            fp_threshold: Episode terminates if FP rate exceeds this.
            shuffle: Whether to shuffle flows at episode start.
            seed: Random seed.
        """
        super().__init__()

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.batch_size = min(batch_size, len(features))
        self.latent_dim = latent_dim
        self.fp_threshold = fp_threshold
        self.shuffle = shuffle

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(latent_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Episode state
        self._rng = np.random.RandomState(seed)
        self._episode_flows: np.ndarray = None
        self._episode_labels: np.ndarray = None
        self._current_step: int = 0
        self._total_benign: int = 0
        self._false_positives: int = 0
        self._true_positives: int = 0
        self._false_negatives: int = 0
        self._true_negatives: int = 0
        self._episode_reward: float = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Samples a random batch of flows from the dataset.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Sample a batch of flows
        indices = self._rng.choice(
            len(self.features), size=self.batch_size, replace=False
        )
        if self.shuffle:
            self._rng.shuffle(indices)

        self._episode_flows = self.features[indices]
        self._episode_labels = self.labels[indices]
        self._current_step = 0
        self._false_positives = 0
        self._true_positives = 0
        self._false_negatives = 0
        self._true_negatives = 0
        self._total_benign = int((self._episode_labels == 0).sum())
        self._episode_reward = 0.0

        obs = self._episode_flows[0]
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Process one flow with the chosen action.

        Args:
            action: 0=ALLOW, 1=RATE_LIMIT, 2=DROP

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        true_label = self._episode_labels[self._current_step]
        reward = self._compute_reward(action, true_label)
        self._update_metrics(action, true_label)
        self._episode_reward += reward

        self._current_step += 1

        # Check termination
        terminated = self._current_step >= self.batch_size
        truncated = self._check_fp_threshold()

        # Next observation
        if terminated or truncated:
            obs = np.zeros(self.latent_dim, dtype=np.float32)
        else:
            obs = self._episode_flows[self._current_step]

        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: int, true_label: int) -> float:
        """
        Compute reward based on action correctness.

        Args:
            action: Agent's chosen action.
            true_label: 0=benign, 1=attack.

        Returns:
            Scalar reward value.
        """
        if true_label == 0:  # Benign traffic
            if action == 0:  # ALLOW — correct
                return REWARD_CORRECT_BENIGN
            else:  # RATE_LIMIT or DROP — false positive
                return REWARD_FALSE_POSITIVE
        else:  # Attack traffic
            if action == 2:  # DROP — best response
                return REWARD_CORRECT_ATTACK
            elif action == 1:  # RATE_LIMIT — partial credit
                return REWARD_RATE_LIMIT_ATTACK
            else:  # ALLOW — false negative
                return REWARD_FALSE_NEGATIVE

    def _update_metrics(self, action: int, true_label: int):
        """Update confusion matrix counters."""
        is_block = action > 0  # RATE_LIMIT or DROP
        is_attack = true_label == 1

        if is_attack and is_block:
            self._true_positives += 1
        elif is_attack and not is_block:
            self._false_negatives += 1
        elif not is_attack and is_block:
            self._false_positives += 1
        else:
            self._true_negatives += 1

    def _check_fp_threshold(self) -> bool:
        """Check if false positive rate exceeds threshold."""
        if self._total_benign == 0:
            return False
        fp_rate = self._false_positives / max(self._total_benign, 1)
        return fp_rate > self.fp_threshold

    def _get_info(self) -> dict:
        """Return episode metrics."""
        total = max(self._current_step, 1)
        tp = self._true_positives
        fp = self._false_positives
        fn = self._false_negatives
        tn = self._true_negatives

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / total

        return {
            "step": self._current_step,
            "total_flows": self.batch_size,
            "episode_reward": self._episode_reward,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "fp_rate": fp / max(self._total_benign, 1),
        }

    def get_episode_summary(self) -> dict:
        """Get a summary of the completed episode."""
        return self._get_info()
