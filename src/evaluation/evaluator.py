"""
Evaluation and visualization module.

Generates metrics, confusion matrices, ROC curves, comparison tables,
and training reward curves for the research paper.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PAPER_FIGURES_DIR,
    CONFUSION_MATRIX_FIGSIZE,
    COMPARISON_FIGSIZE,
    ACTION_NAMES,
)

logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


class Evaluator:
    """
    Evaluation and visualization for DDoS mitigation results.

    Generates publication-quality figures and tables for the
    research paper.
    """

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self.output_dir = PAPER_FIGURES_DIR
        else:
            from pathlib import Path
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_rewards(
        self,
        rewards_dict: Dict[str, List[float]],
        window: int = 20,
        title: str = "Training Reward Curves",
        filename: str = "training_rewards.png",
    ):
        """
        Plot training reward curves for all agents.

        Args:
            rewards_dict: {agent_name: [episode_rewards]}
            window: Smoothing window for moving average.
            title: Plot title.
            filename: Output filename.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            "DQN": "#e74c3c",
            "DDQN": "#3498db",
            "PPO": "#2ecc71",
        }

        for name, rewards in rewards_dict.items():
            episodes = np.arange(1, len(rewards) + 1)
            color = colors.get(name, "#95a5a6")

            # Raw rewards (light)
            ax.plot(episodes, rewards, alpha=0.2, color=color, linewidth=0.8)

            # Smoothed rewards (bold)
            if len(rewards) >= window:
                smoothed = np.convolve(
                    rewards, np.ones(window) / window, mode="valid"
                )
                ax.plot(
                    episodes[window - 1:], smoothed,
                    label=f"{name} (smoothed)",
                    color=color, linewidth=2.0,
                )
            else:
                ax.plot(episodes, rewards, label=name, color=color, linewidth=2.0)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        model_name: str,
        filename: Optional[str] = None,
    ):
        """
        Plot a heatmap confusion matrix.

        Args:
            cm: Confusion matrix array.
            labels: Class labels.
            model_name: Name for the title.
            filename: Output filename.
        """
        if filename is None:
            filename = f"confusion_matrix_{model_name.lower()}.png"

        fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix — {model_name}")

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_comparison_bar(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Model Comparison",
        filename: str = "model_comparison.png",
    ):
        """
        Plot grouped bar chart comparing metrics across models.

        Args:
            results: {model_name: {metric: value}}
            metrics: Which metrics to plot.
            title: Chart title.
            filename: Output filename.
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score"]

        models = list(results.keys())
        n_metrics = len(metrics)
        n_models = len(models)

        fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)

        x = np.arange(n_metrics)
        width = 0.8 / n_models

        colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
            "#9b59b6", "#1abc9c",
        ]

        for i, model in enumerate(models):
            values = [results[model].get(m, 0) for m in metrics]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, values,
                width=width,
                label=model,
                color=colors[i % len(colors)],
                edgecolor="white",
                linewidth=0.5,
            )
            # Value labels on top
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )

        metric_labels = [m.replace("_", " ").title() for m in metrics]
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict],
        filename: str = "roc_curves.png",
    ):
        """
        Plot ROC curves for multiple models.

        Args:
            roc_data: {model_name: {"fpr": array, "tpr": array, "auc": float}}
            filename: Output filename.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
            "#9b59b6", "#1abc9c",
        ]

        for i, (name, data) in enumerate(roc_data.items()):
            ax.plot(
                data["fpr"], data["tpr"],
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} (AUC = {data['auc']:.4f})",
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_action_distribution(
        self,
        action_counts: Dict[str, Dict[int, int]],
        filename: str = "action_distribution.png",
    ):
        """
        Plot action distribution per agent.

        Args:
            action_counts: {agent_name: {action_id: count}}
        """
        fig, axes = plt.subplots(
            1, len(action_counts),
            figsize=(5 * len(action_counts), 5),
            sharey=True,
        )
        if len(action_counts) == 1:
            axes = [axes]

        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        action_labels = [ACTION_NAMES[i] for i in range(len(ACTION_NAMES))]

        for ax, (agent_name, counts) in zip(axes, action_counts.items()):
            values = [counts.get(i, 0) for i in range(len(ACTION_NAMES))]
            total = sum(values) or 1
            percentages = [v / total * 100 for v in values]

            bars = ax.bar(action_labels, percentages, color=colors, edgecolor="white")
            ax.set_title(agent_name)
            ax.set_ylabel("Percentage (%)" if ax == axes[0] else "")

            for bar, pct in zip(bars, percentages):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.1f}%",
                    ha="center", fontsize=10,
                )

        fig.suptitle("Action Distribution by Agent", fontsize=14, y=1.02)
        fig.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def generate_latex_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        caption: str = "Performance comparison of DRL agents and ML baselines.",
        label: str = "tab:comparison",
    ) -> str:
        """
        Generate a LaTeX table for the research paper.

        Returns:
            LaTeX table string.
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

        # Header
        header_labels = [m.replace("_", " ").title() for m in metrics]
        header = " & ".join(["Model"] + header_labels) + " \\\\"

        rows = []
        for model, vals in results.items():
            row_vals = []
            for m in metrics:
                v = vals.get(m, None)
                if v is not None:
                    row_vals.append(f"{v:.4f}")
                else:
                    row_vals.append("—")
            rows.append(f"    {model} & " + " & ".join(row_vals) + " \\\\")

        col_spec = "l" + "c" * len(metrics)

        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
    {header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

        # Also save to file
        path = self.output_dir / "comparison_table.tex"
        with open(path, "w") as f:
            f.write(latex)
        logger.info(f"LaTeX table saved to {path}")

        return latex
