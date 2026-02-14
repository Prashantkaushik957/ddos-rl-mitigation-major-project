"""
Traditional machine learning baselines for comparison.

Implements Random Forest, SVM, and XGBoost classifiers
trained on the same preprocessed features for fair comparison
with the DRL agents.
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib

from src.config import (
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    SVM_KERNEL,
    SVM_C,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    RANDOM_SEED,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class BaselineModels:
    """
    Collection of traditional ML classifiers for DDoS detection.

    Models:
        - Random Forest
        - Support Vector Machine (SVM)
        - XGBoost

    All models are trained on the same feature set for fair
    comparison with the DRL agents.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.models = {}
        self.results = {}
        self.training_times = {}

        self._init_models()

    def _init_models(self):
        """Initialize all baseline models."""
        self.models["RandomForest"] = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=self.seed,
            n_jobs=-1,
            class_weight="balanced",
        )

        self.models["SVM"] = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            random_state=self.seed,
            probability=True,
            class_weight="balanced",
        )

        try:
            from xgboost import XGBClassifier
            self.models["XGBoost"] = XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                random_state=self.seed,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
                scale_pos_weight=1,
            )
        except ImportError:
            logger.warning("XGBoost not installed, skipping XGBoost baseline.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train all baseline models.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (for reporting val accuracy).
            y_val: Validation labels.

        Returns:
            Dictionary of model_name → training time (seconds).
        """
        logger.info("=" * 60)
        logger.info("Training baseline models")
        logger.info("=" * 60)

        for name, model in self.models.items():
            logger.info(f"\nTraining {name}...")
            start_time = time.time()

            model.fit(X_train, y_train)

            elapsed = time.time() - start_time
            self.training_times[name] = elapsed
            logger.info(f"  {name} trained in {elapsed:.2f}s")

            # Validation accuracy
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                logger.info(f"  Validation accuracy: {val_acc:.4f}")

        return self.training_times

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Evaluate all trained baseline models on test data.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary of model_name → metrics dictionary.
        """
        logger.info("=" * 60)
        logger.info("Evaluating baseline models")
        logger.info("=" * 60)

        for name, model in self.models.items():
            logger.info(f"\nEvaluating {name}:")

            # Predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time

            # Probabilities for AUC-ROC
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                y_prob = None
                auc = None

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            cm = confusion_matrix(y_test, y_pred)

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auc_roc": auc,
                "confusion_matrix": cm,
                "inference_time": inference_time,
                "inference_time_per_sample": inference_time / len(X_test),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }

            self.results[name] = metrics

            logger.info(f"  Accuracy:  {acc:.4f}")
            logger.info(f"  Precision: {prec:.4f}")
            logger.info(f"  Recall:    {rec:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            if auc is not None:
                logger.info(f"  AUC-ROC:   {auc:.4f}")
            logger.info(f"  Inference time: {inference_time:.4f}s ({inference_time/len(X_test)*1000:.4f}ms/sample)")

        return self.results

    def save(self, directory: Optional[str] = None):
        """Save all trained models."""
        if directory is None:
            directory = str(MODELS_DIR)

        for name, model in self.models.items():
            path = f"{directory}/{name.lower()}_model.joblib"
            joblib.dump(model, path)
            logger.info(f"Saved {name} to {path}")

    def get_comparison_table(self) -> dict:
        """
        Get a formatted comparison table of all baseline results.

        Returns:
            Dictionary suitable for creating a pandas DataFrame.
        """
        table = {
            "Model": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1-Score": [],
            "AUC-ROC": [],
            "Training Time (s)": [],
            "Inference Time (ms/sample)": [],
        }

        for name, metrics in self.results.items():
            table["Model"].append(name)
            table["Accuracy"].append(f"{metrics['accuracy']:.4f}")
            table["Precision"].append(f"{metrics['precision']:.4f}")
            table["Recall"].append(f"{metrics['recall']:.4f}")
            table["F1-Score"].append(f"{metrics['f1_score']:.4f}")
            table["AUC-ROC"].append(
                f"{metrics['auc_roc']:.4f}" if metrics["auc_roc"] else "N/A"
            )
            table["Training Time (s)"].append(
                f"{self.training_times.get(name, 0):.2f}"
            )
            table["Inference Time (ms/sample)"].append(
                f"{metrics['inference_time_per_sample']*1000:.4f}"
            )

        return table
