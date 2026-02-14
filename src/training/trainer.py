"""
Unified training framework for all DRL agents.

Supports training DQN, DDQN, and PPO agents on the SDN environment
with TensorBoard logging, model checkpointing, and early stopping.
"""

import logging
import time
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.env.sdn_env import SDNEnvironment
from src.agents.dqn_agent import DQNAgent
from src.agents.ddqn_agent import DDQNAgent
from src.agents.ppo_agent import PPOAgent
from src.config import (
    NUM_EPISODES,
    CHECKPOINT_INTERVAL,
    LOG_INTERVAL,
    EARLY_STOPPING_PATIENCE,
    LOGS_DIR,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for DRL agents.

    Handles the training loop, logging, checkpointing, and
    early stopping for all agent types (DQN, DDQN, PPO).
    """

    def __init__(
        self,
        agent,
        env: SDNEnvironment,
        val_env: Optional[SDNEnvironment] = None,
        num_episodes: int = NUM_EPISODES,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        log_interval: int = LOG_INTERVAL,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        experiment_name: Optional[str] = None,
    ):
        self.agent = agent
        self.env = env
        self.val_env = val_env
        self.num_episodes = num_episodes
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.early_stopping_patience = early_stopping_patience

        # Determine agent type
        self.agent_type = type(agent).__name__

        # Experiment name for logging
        if experiment_name is None:
            experiment_name = f"{self.agent_type}_{int(time.time())}"
        self.experiment_name = experiment_name

        # TensorBoard writer
        log_dir = LOGS_DIR / experiment_name
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Training history
        self.episode_rewards = []
        self.episode_metrics = []
        self.best_reward = float("-inf")
        self.patience_counter = 0

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            Training history dictionary.
        """
        logger.info("=" * 60)
        logger.info(f"Training {self.agent_type}")
        logger.info(f"Episodes: {self.num_episodes}")
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info("=" * 60)

        start_time = time.time()

        for episode in tqdm(range(1, self.num_episodes + 1), desc=self.agent_type):
            if isinstance(self.agent, PPOAgent):
                metrics = self._train_episode_ppo(episode)
            else:
                metrics = self._train_episode_dqn(episode)

            self.episode_rewards.append(metrics["episode_reward"])
            self.episode_metrics.append(metrics)

            # Logging
            self._log_episode(episode, metrics)

            # Checkpointing
            if episode % self.checkpoint_interval == 0:
                self.agent.save(
                    str(MODELS_DIR / f"{self.agent_type}_ep{episode}.pt")
                )

            # Early stopping check
            if self._check_early_stopping(metrics["episode_reward"]):
                logger.info(
                    f"Early stopping at episode {episode} "
                    f"(best reward: {self.best_reward:.2f})"
                )
                break

            # Print summary
            if episode % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                logger.info(
                    f"[{self.agent_type}] Episode {episode}/{self.num_episodes} "
                    f"| Avg Reward: {avg_reward:.2f} "
                    f"| Acc: {metrics['accuracy']:.4f} "
                    f"| F1: {metrics['f1_score']:.4f}"
                )

        elapsed = time.time() - start_time
        self.writer.close()

        # Save final model
        self.agent.save()

        logger.info(f"\nTraining complete in {elapsed:.1f}s")
        logger.info(f"Best episode reward: {self.best_reward:.2f}")

        return {
            "agent_type": self.agent_type,
            "episode_rewards": self.episode_rewards,
            "episode_metrics": self.episode_metrics,
            "best_reward": self.best_reward,
            "training_time": elapsed,
            "n_episodes": len(self.episode_rewards),
        }

    def _train_episode_dqn(self, episode: int) -> dict:
        """Training episode for DQN/DDQN agents."""
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        n_steps = 0

        done = False
        truncated = False

        while not (done or truncated):
            # Select action
            action = self.agent.select_action(obs)

            # Take step
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Store and train
            self.agent.store_transition(obs, action, reward, next_obs, done or truncated)
            loss = self.agent.train_step()

            if loss is not None:
                episode_loss += loss

            episode_reward += reward
            obs = next_obs
            n_steps += 1

        info["episode_reward"] = episode_reward
        info["avg_loss"] = episode_loss / max(n_steps, 1)
        info["epsilon"] = getattr(self.agent, "epsilon", 0)
        return info

    def _train_episode_ppo(self, episode: int) -> dict:
        """Training episode for PPO agent."""
        obs, info = self.env.reset()
        episode_reward = 0.0

        done = False
        truncated = False

        while not (done or truncated):
            # Select action
            action, log_prob, value = self.agent.select_action(obs)

            # Take step
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Store transition
            self.agent.store_transition(
                obs, action, reward, log_prob, value, done or truncated
            )

            episode_reward += reward
            obs = next_obs

        # PPO update at end of episode
        # Get value of last state for bootstrapping
        with torch.no_grad():
            last_state = torch.FloatTensor(obs).to(self.agent.device)
            _, last_value = self.agent.network(last_state.unsqueeze(0))
            last_value = last_value.item()

        update_info = self.agent.update(last_value if not (done or truncated) else 0.0)

        info["episode_reward"] = episode_reward
        info["policy_loss"] = update_info["policy_loss"]
        info["value_loss"] = update_info["value_loss"]
        info["entropy"] = update_info["entropy"]
        return info

    def _log_episode(self, episode: int, metrics: dict):
        """Log episode metrics to TensorBoard."""
        self.writer.add_scalar("reward/episode", metrics["episode_reward"], episode)
        self.writer.add_scalar("metrics/accuracy", metrics["accuracy"], episode)
        self.writer.add_scalar("metrics/precision", metrics["precision"], episode)
        self.writer.add_scalar("metrics/recall", metrics["recall"], episode)
        self.writer.add_scalar("metrics/f1_score", metrics["f1_score"], episode)
        self.writer.add_scalar("metrics/fp_rate", metrics["fp_rate"], episode)

        if "epsilon" in metrics:
            self.writer.add_scalar("agent/epsilon", metrics["epsilon"], episode)
        if "avg_loss" in metrics:
            self.writer.add_scalar("loss/avg", metrics["avg_loss"], episode)
        if "policy_loss" in metrics:
            self.writer.add_scalar("loss/policy", metrics["policy_loss"], episode)
        if "value_loss" in metrics:
            self.writer.add_scalar("loss/value", metrics["value_loss"], episode)
        if "entropy" in metrics:
            self.writer.add_scalar("agent/entropy", metrics["entropy"], episode)

    def _check_early_stopping(self, reward: float) -> bool:
        """Check if training should stop early."""
        if reward > self.best_reward:
            self.best_reward = reward
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience

    def evaluate(
        self,
        env: Optional[SDNEnvironment] = None,
        n_episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate the trained agent (deterministic).

        Args:
            env: Environment to evaluate on (defaults to val_env).
            n_episodes: Number of evaluation episodes.

        Returns:
            Average metrics over evaluation episodes.
        """
        if env is None:
            env = self.val_env if self.val_env is not None else self.env

        all_metrics = []
        total_reward = 0.0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            ep_reward = 0.0

            while not (done or truncated):
                if isinstance(self.agent, PPOAgent):
                    action, _, _ = self.agent.select_action(obs, deterministic=True)
                else:
                    action = self.agent.select_action(obs, deterministic=True)

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward

            info["episode_reward"] = ep_reward
            all_metrics.append(info)
            total_reward += ep_reward

        # Average metrics
        avg = {
            "avg_reward": total_reward / n_episodes,
            "avg_accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "avg_precision": np.mean([m["precision"] for m in all_metrics]),
            "avg_recall": np.mean([m["recall"] for m in all_metrics]),
            "avg_f1": np.mean([m["f1_score"] for m in all_metrics]),
            "avg_fp_rate": np.mean([m["fp_rate"] for m in all_metrics]),
        }

        logger.info(f"\n{self.agent_type} Evaluation ({n_episodes} episodes):")
        for k, v in avg.items():
            logger.info(f"  {k}: {v:.4f}")

        return avg
