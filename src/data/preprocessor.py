"""
Data preprocessing pipeline for network traffic features.

Handles cleaning, feature selection, normalization, and train/test splitting.
"""

import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif

from src.config import (
    NUM_SELECTED_FEATURES,
    NORMALIZATION_METHOD,
    TEST_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    RANDOM_SEED,
    LABEL_COLUMN,
)

logger = logging.getLogger(__name__)


# Columns to always drop (identifiers, timestamps, etc.)
DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Port",
    "Src Port",
    "Destination IP",
    "Dst IP",
    "Destination Port",
    "Dst Port",
    "Timestamp",
    "Protocol",
    "SimillarHTTP",
    "Unnamed: 0",
]


class DataPreprocessor:
    """
    End-to-end preprocessing pipeline for network traffic data.

    Steps:
        1. Drop non-numeric / identifier columns
        2. Handle missing values and infinities
        3. Remove low-variance and highly-correlated features
        4. Select top-k features using mutual information
        5. Normalize features
        6. Split into train / validation / test sets
    """

    def __init__(
        self,
        n_features: int = NUM_SELECTED_FEATURES,
        normalization: str = NORMALIZATION_METHOD,
        test_ratio: float = TEST_SPLIT_RATIO,
        val_ratio: float = VAL_SPLIT_RATIO,
        seed: int = RANDOM_SEED,
    ):
        self.n_features = n_features
        self.normalization = normalization
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        self.scaler = None
        self.selected_features: List[str] = []
        self.feature_importances: Optional[np.ndarray] = None

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-numeric columns, handle NaN and infinity values."""
        df = df.copy()

        # Drop identifier columns
        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        if cols_to_drop:
            logger.info(f"Dropping identifier columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # Keep only numeric columns (plus label columns)
        label_cols = [c for c in [LABEL_COLUMN, "binary_label", "multi_label"] if c in df.columns]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = list(set(numeric_cols + label_cols))
        df = df[keep_cols]

        # Replace inf with NaN, then fill NaN with column median
        df = df.replace([np.inf, -np.inf], np.nan)
        n_nan = df.isna().sum().sum()
        if n_nan > 0:
            logger.info(f"Filling {n_nan} NaN values with column medians")
            numeric_only = df.select_dtypes(include=[np.number])
            df[numeric_only.columns] = numeric_only.fillna(numeric_only.median())

        # Drop any remaining NaN rows
        df = df.dropna()

        # Drop duplicate rows
        n_before = len(df)
        df = df.drop_duplicates()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(f"Dropped {n_dropped} duplicate rows")

        logger.info(f"Cleaned dataset shape: {df.shape}")
        return df

    def remove_low_variance(
        self, df: pd.DataFrame, threshold: float = 0.01
    ) -> pd.DataFrame:
        """Remove features with variance below threshold."""
        label_cols = [c for c in [LABEL_COLUMN, "binary_label", "multi_label"] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in label_cols]

        variances = df[feature_cols].var()
        low_var = variances[variances < threshold].index.tolist()

        if low_var:
            logger.info(f"Removing {len(low_var)} low-variance features: {low_var[:5]}...")
            df = df.drop(columns=low_var)

        return df

    def remove_correlated(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove features that are highly correlated with each other."""
        label_cols = [c for c in [LABEL_COLUMN, "binary_label", "multi_label"] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in label_cols]

        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        if to_drop:
            logger.info(
                f"Removing {len(to_drop)} highly-correlated features "
                f"(threshold={threshold})"
            )
            df = df.drop(columns=to_drop)

        return df

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select top-k features using mutual information.

        Args:
            X: Feature matrix.
            y: Label vector.
            feature_names: Names of features.

        Returns:
            Tuple of (selected features matrix, selected feature names).
        """
        n_select = min(self.n_features, X.shape[1])
        logger.info(f"Selecting top {n_select} features using mutual information...")

        mi_scores = mutual_info_classif(
            X, y, discrete_features=False, random_state=self.seed
        )
        self.feature_importances = mi_scores

        # Get indices of top features
        top_indices = np.argsort(mi_scores)[-n_select:][::-1]
        self.selected_features = [feature_names[i] for i in top_indices]

        logger.info(f"Selected features: {self.selected_features}")
        logger.info(
            f"MI scores: {[f'{mi_scores[i]:.4f}' for i in top_indices]}"
        )

        return X[:, top_indices], self.selected_features

    def normalize(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
        """
        Fit scaler on training data and transform all splits.

        Returns:
            Tuple of (X_train, X_val, X_test) normalized.
        """
        if self.normalization == "minmax":
            self.scaler = MinMaxScaler()
        elif self.normalization == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        logger.info(
            f"Normalized with {self.normalization}. "
            f"Train range: [{X_train.min():.4f}, {X_train.max():.4f}]"
        )
        return X_train, X_val, X_test

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split into train / validation / test sets with stratification.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.seed,
            stratify=y,
        )

        # Second split: train vs val
        relative_val_ratio = self.val_ratio / (1 - self.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=relative_val_ratio,
            random_state=self.seed,
            stratify=y_trainval,
        )

        logger.info(
            f"Split sizes — Train: {X_train.shape[0]}, "
            f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform(
        self, df: pd.DataFrame, label_col: str = "binary_label"
    ) -> dict:
        """
        Run the full preprocessing pipeline.

        Args:
            df: Raw DataFrame with features and labels.
            label_col: Column name for the target label.

        Returns:
            Dictionary with keys:
                'X_train', 'X_val', 'X_test',
                'y_train', 'y_val', 'y_test',
                'feature_names', 'scaler'
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 60)

        # Step 1: Clean
        df = self.clean(df)

        # Step 2: Remove low-variance features
        df = self.remove_low_variance(df)

        # Step 3: Remove highly-correlated features
        df = self.remove_correlated(df)

        # Step 4: Separate features and labels
        label_cols = [c for c in [LABEL_COLUMN, "binary_label", "multi_label"] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in label_cols]

        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64)

        # Step 5: Feature selection
        X, selected_names = self.select_features(X, y, feature_cols)

        # Step 6: Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Step 7: Normalize
        X_train, X_val, X_test = self.normalize(X_train, X_val, X_test)

        logger.info("Preprocessing pipeline complete!")
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": selected_names,
            "scaler": self.scaler,
        }
