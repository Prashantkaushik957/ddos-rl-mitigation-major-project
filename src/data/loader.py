"""
Dataset loader for CICDDoS2019.

Handles loading raw CSV files, merging multiple attack-day files,
and producing a unified DataFrame with consistent labels.
"""

import os
import glob
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    LABEL_COLUMN,
    BINARY_LABELS,
    MAX_SAMPLES_PER_CLASS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def load_raw_csvs(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load all CSV files from the raw data directory and concatenate them.

    The CICDDoS2019 dataset is split across multiple CSV files
    corresponding to different days and attack types.

    Args:
        data_dir: Path to directory containing CSV files.
                  Defaults to DATA_RAW_DIR from config.

    Returns:
        Combined DataFrame with all traffic flows.
    """
    if data_dir is None:
        data_dir = str(DATA_RAW_DIR)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            f"Please download the CICDDoS2019 dataset from "
            f"https://www.unb.ca/cic/datasets/ddos-2019.html "
            f"and place the CSV files in {data_dir}"
        )

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
    dfs = []
    for f in csv_files:
        logger.info(f"  Loading: {os.path.basename(f)}")
        try:
            df = pd.read_csv(f, low_memory=False, encoding="utf-8")
            # Strip whitespace from column names (common issue in CIC datasets)
            df.columns = df.columns.str.strip()
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Failed to load {f}: {e}")
            continue

    if not dfs:
        raise ValueError("No CSV files could be loaded successfully.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined.shape}")
    return combined


def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert multi-class attack labels to binary: BENIGN (0) vs ATTACK (1).

    Args:
        df: DataFrame with a 'Label' column.

    Returns:
        DataFrame with an additional 'binary_label' column.
    """
    df = df.copy()

    # Normalize label strings
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    # Create binary label
    df["binary_label"] = df[LABEL_COLUMN].apply(
        lambda x: BINARY_LABELS.get(x, 1)
    )

    label_counts = df["binary_label"].value_counts()
    logger.info(f"Binary label distribution:\n{label_counts}")
    return df


def create_multiclass_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode multi-class labels as integers for multi-class classification.

    Args:
        df: DataFrame with a 'Label' column.

    Returns:
        DataFrame with an additional 'multi_label' column and a label mapping.
    """
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    unique_labels = sorted(df[LABEL_COLUMN].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    df["multi_label"] = df[LABEL_COLUMN].map(label_map)
    logger.info(f"Multi-class labels ({len(unique_labels)} classes): {label_map}")
    return df


def balance_dataset(
    df: pd.DataFrame,
    label_col: str = "binary_label",
    max_per_class: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Balance the dataset by undersampling the majority class.

    Args:
        df: DataFrame with labels.
        label_col: Column to balance on.
        max_per_class: Maximum samples per class. Defaults to config value.
        seed: Random seed for reproducibility.

    Returns:
        Balanced DataFrame.
    """
    if max_per_class is None:
        max_per_class = MAX_SAMPLES_PER_CLASS

    balanced_parts = []
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        if len(subset) > max_per_class:
            subset = subset.sample(n=max_per_class, random_state=seed)
        balanced_parts.append(subset)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(f"Balanced dataset shape: {balanced.shape}")
    logger.info(
        f"Balanced distribution:\n{balanced[label_col].value_counts()}"
    )
    return balanced


def load_and_prepare(
    data_dir: Optional[str] = None,
    binary: bool = True,
    balance: bool = True,
    max_per_class: Optional[int] = None,
) -> pd.DataFrame:
    """
    Full pipeline: load CSVs → label → balance.

    Args:
        data_dir: Path to raw CSV directory.
        binary: If True, create binary labels. Otherwise, multi-class.
        balance: If True, balance the dataset.
        max_per_class: Maximum samples per class.

    Returns:
        Prepared DataFrame ready for preprocessing.
    """
    df = load_raw_csvs(data_dir)

    if binary:
        df = create_binary_labels(df)
        label_col = "binary_label"
    else:
        df = create_multiclass_labels(df)
        label_col = "multi_label"

    if balance:
        df = balance_dataset(df, label_col=label_col, max_per_class=max_per_class)

    return df


def generate_synthetic_dataset(
    n_samples: int = 10000,
    n_features: int = 20,
    attack_ratio: float = 0.4,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset for testing and development when
    the real CICDDoS2019 dataset is not available.

    Creates realistic-ish network flow features with distinct
    distributions for benign vs attack traffic.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features.
        attack_ratio: Proportion of attack samples.
        seed: Random seed.

    Returns:
        Tuple of (features, labels) as numpy arrays.
    """
    rng = np.random.RandomState(seed)
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack

    # Benign traffic: lower packet rates, normal distributions
    benign_features = rng.normal(loc=0.3, scale=0.15, size=(n_benign, n_features))
    benign_features = np.clip(benign_features, 0, 1)

    # Attack traffic: higher packet rates, different distribution
    attack_features = rng.normal(loc=0.7, scale=0.2, size=(n_attack, n_features))
    attack_features = np.clip(attack_features, 0, 1)

    # Add some noise to make it non-trivial
    # Features 0-4: flow-level stats (more discriminative)
    benign_features[:, :5] *= 0.5
    attack_features[:, :5] *= 1.5
    attack_features[:, :5] = np.clip(attack_features[:, :5], 0, 1)

    # Features 5-9: packet-level stats (moderately discriminative)
    benign_features[:, 5:10] += rng.uniform(-0.1, 0.1, (n_benign, 5))
    attack_features[:, 5:10] += rng.uniform(0.1, 0.3, (n_attack, 5))

    # Features 10+: less discriminative
    noise = rng.normal(0, 0.1, (n_samples, n_features - 10))

    features = np.vstack([benign_features, attack_features])
    features[:, 10:] += noise
    features = np.clip(features, 0, 1)

    labels = np.concatenate([
        np.zeros(n_benign, dtype=np.int64),
        np.ones(n_attack, dtype=np.int64),
    ])

    # Shuffle
    perm = rng.permutation(n_samples)
    return features[perm], labels[perm]
