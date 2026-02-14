"""
Main entry point for the DDoS RL Mitigation project.

Usage:
    python main.py                    # Full pipeline (train + evaluate)
    python main.py --mode train       # Train only
    python main.py --mode evaluate    # Evaluate only
    python main.py --test-run         # Quick test with synthetic data
"""

import argparse
import logging
import sys
import warnings

import numpy as np
import torch

from src.config import (
    RANDOM_SEED,
    AUTOENCODER_LATENT_DIM,
    NUM_EPISODES,
    MODELS_DIR,
)
from src.data.loader import generate_synthetic_dataset, load_and_prepare
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureExtractor
from src.env.sdn_env import SDNEnvironment
from src.agents.dqn_agent import DQNAgent
from src.agents.ddqn_agent import DDQNAgent
from src.agents.ppo_agent import PPOAgent
from src.baselines.ml_baselines import BaselineModels
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def set_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(use_synthetic: bool = False) -> dict:
    """
    Prepare data for training and evaluation.

    Args:
        use_synthetic: If True, use synthetic data (for testing).

    Returns:
        Dictionary with train/val/test splits.
    """
    if use_synthetic:
        logger.info("Using synthetic dataset for development/testing")
        X_train_raw, y_train = generate_synthetic_dataset(
            n_samples=8000, n_features=20, seed=RANDOM_SEED
        )
        X_val_raw, y_val = generate_synthetic_dataset(
            n_samples=1000, n_features=20, seed=RANDOM_SEED + 1
        )
        X_test_raw, y_test = generate_synthetic_dataset(
            n_samples=1000, n_features=20, seed=RANDOM_SEED + 2
        )
    else:
        logger.info("Loading and preprocessing CICDDoS2019 dataset...")
        df = load_and_prepare(binary=True, balance=True)
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(df)
        X_train_raw = data["X_train"]
        X_val_raw = data["X_val"]
        X_test_raw = data["X_test"]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]

    # Feature extraction with autoencoder
    logger.info("Training autoencoder for feature extraction...")
    input_dim = X_train_raw.shape[1]
    extractor = FeatureExtractor(input_dim=input_dim)
    extractor.fit(X_train_raw, X_val_raw)

    X_train = extractor.transform(X_train_raw)
    X_val = extractor.transform(X_val_raw)
    X_test = extractor.transform(X_test_raw)

    extractor.save()

    logger.info(
        f"Feature extraction complete: {input_dim} → {X_train.shape[1]} dims"
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "extractor": extractor,
    }


def train_rl_agents(data: dict, num_episodes: int = NUM_EPISODES) -> dict:
    """
    Train all DRL agents and return training histories.
    """
    state_dim = data["X_train"].shape[1]

    # Create environments
    train_env = SDNEnvironment(data["X_train"], data["y_train"])
    val_env = SDNEnvironment(data["X_val"], data["y_val"])

    results = {}
    agents = {}

    # ── DQN ──
    logger.info("\n" + "=" * 60)
    logger.info("Training DQN Agent")
    logger.info("=" * 60)
    dqn = DQNAgent(state_dim=state_dim)
    trainer = Trainer(
        dqn, train_env, val_env,
        num_episodes=num_episodes,
        experiment_name="DQN",
    )
    results["DQN"] = trainer.train()
    agents["DQN"] = dqn

    # ── DDQN ──
    logger.info("\n" + "=" * 60)
    logger.info("Training Double DQN Agent")
    logger.info("=" * 60)
    ddqn = DDQNAgent(state_dim=state_dim)
    trainer = Trainer(
        ddqn, train_env, val_env,
        num_episodes=num_episodes,
        experiment_name="DDQN",
    )
    results["DDQN"] = trainer.train()
    agents["DDQN"] = ddqn

    # ── PPO ──
    logger.info("\n" + "=" * 60)
    logger.info("Training PPO Agent")
    logger.info("=" * 60)
    ppo = PPOAgent(state_dim=state_dim)
    trainer = Trainer(
        ppo, train_env, val_env,
        num_episodes=num_episodes,
        experiment_name="PPO",
    )
    results["PPO"] = trainer.train()
    agents["PPO"] = ppo

    return {"results": results, "agents": agents}


def train_baselines(data: dict) -> dict:
    """Train traditional ML baselines."""
    logger.info("\n" + "=" * 60)
    logger.info("Training ML Baselines")
    logger.info("=" * 60)

    baselines = BaselineModels()
    baselines.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    baseline_results = baselines.evaluate(data["X_test"], data["y_test"])
    baselines.save()

    return {"baselines": baselines, "results": baseline_results}


def evaluate_and_visualize(
    rl_results: dict, baseline_results: dict, data: dict
):
    """Generate all evaluation plots and tables."""
    import time as _time
    from sklearn.metrics import confusion_matrix as _cm

    evaluator = Evaluator()

    # 1. Training reward curves
    reward_dict = {}
    for name, res in rl_results["results"].items():
        reward_dict[name] = res["episode_rewards"]
    evaluator.plot_training_rewards(reward_dict)

    # 2. Evaluate RL agents on test set + collect action distributions
    test_env = SDNEnvironment(data["X_test"], data["y_test"])
    rl_test_metrics = {}
    action_counts = {}
    rl_predictions = {}

    for name, agent in rl_results["agents"].items():
        trainer = Trainer(agent, test_env, experiment_name=f"{name}_eval")

        # Measure inference time
        start_t = _time.time()
        metrics = trainer.evaluate(test_env, n_episodes=20)
        elapsed_t = _time.time() - start_t

        rl_test_metrics[name] = {
            "accuracy": metrics["avg_accuracy"],
            "precision": metrics["avg_precision"],
            "recall": metrics["avg_recall"],
            "f1_score": metrics["avg_f1"],
            "inference_time": elapsed_t,
        }

        # Collect actions on one full pass for confusion matrix + action dist
        obs, _ = test_env.reset()
        preds = []
        trues = []
        acts = {0: 0, 1: 0, 2: 0}
        done = truncated = False
        step_idx = 0
        while not (done or truncated):
            if isinstance(agent, PPOAgent):
                action, _, _ = agent.select_action(obs, deterministic=True)
            else:
                action = agent.select_action(obs, deterministic=True)
            acts[action] = acts.get(action, 0) + 1
            true_label = test_env._episode_labels[step_idx]
            pred_label = 0 if action == 0 else 1  # ALLOW=benign, else=attack
            preds.append(pred_label)
            trues.append(true_label)
            obs, _, done, truncated, _ = test_env.step(action)
            step_idx += 1

        action_counts[name] = acts

        # Confusion matrix
        cm = _cm(trues, preds)
        evaluator.plot_confusion_matrix(
            cm, labels=["Benign", "Attack"], model_name=name
        )

    # 3. Action distribution chart
    evaluator.plot_action_distribution(action_counts)

    # 4. Confusion matrices for baselines
    for name, metrics in baseline_results["results"].items():
        if "confusion_matrix" in metrics:
            evaluator.plot_confusion_matrix(
                metrics["confusion_matrix"],
                labels=["Benign", "Attack"],
                model_name=name,
            )

    # 5. Combine RL + baseline results
    all_results = {}
    all_results.update(rl_test_metrics)
    for name, metrics in baseline_results["results"].items():
        all_results[name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "auc_roc": metrics.get("auc_roc"),
        }

    # 6. Comparison chart
    evaluator.plot_comparison_bar(all_results)

    # 7. LaTeX table
    latex = evaluator.generate_latex_table(all_results)
    logger.info(f"\nLaTeX Table:\n{latex}")

    logger.info("\n" + "=" * 60)
    logger.info("All visualizations generated in paper/figures/")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="DDoS Attack Mitigation Using Deep Reinforcement Learning"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "full"],
        default="full",
        help="Run mode",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Quick test with synthetic data (5 episodes)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes",
    )
    args = parser.parse_args()

    set_seeds()

    # Determine settings
    use_synthetic = args.test_run
    num_episodes = args.episodes or (5 if args.test_run else NUM_EPISODES)

    logger.info("=" * 60)
    logger.info("DDoS Attack Mitigation Using Deep Reinforcement Learning")
    logger.info("=" * 60)
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Synthetic data: {use_synthetic}")

    # Step 1: Prepare data
    data = prepare_data(use_synthetic=use_synthetic)

    if args.mode in ("train", "full"):
        # Step 2: Train RL agents
        rl_results = train_rl_agents(data, num_episodes=num_episodes)

        # Step 3: Train baselines
        baseline_results = train_baselines(data)

        if args.mode == "full":
            # Step 4: Evaluate and visualize
            evaluate_and_visualize(rl_results, baseline_results, data)

    logger.info("\nDone! ✓")


if __name__ == "__main__":
    main()
