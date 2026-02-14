"""
Tests for data preprocessing pipeline.
"""

import numpy as np
import pytest

from src.data.loader import generate_synthetic_dataset
from src.data.feature_engineer import Autoencoder, FeatureExtractor
from src.config import AUTOENCODER_LATENT_DIM


class TestSyntheticData:

    def test_shape(self):
        """Verify synthetic data has expected shape."""
        X, y = generate_synthetic_dataset(n_samples=100, n_features=20)
        assert X.shape == (100, 20)
        assert y.shape == (100,)

    def test_labels_binary(self):
        """Verify labels are binary (0 or 1)."""
        X, y = generate_synthetic_dataset(n_samples=100)
        assert set(np.unique(y)) <= {0, 1}

    def test_features_in_range(self):
        """Verify features are in [0, 1] range."""
        X, y = generate_synthetic_dataset(n_samples=100)
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_attack_ratio(self):
        """Verify attack ratio is approximately correct."""
        X, y = generate_synthetic_dataset(n_samples=10000, attack_ratio=0.4)
        actual_ratio = y.sum() / len(y)
        assert abs(actual_ratio - 0.4) < 0.05


class TestAutoencoder:

    def test_output_shape(self):
        """Verify autoencoder produces correct latent dimension."""
        import torch
        model = Autoencoder(input_dim=20, latent_dim=10)
        x = torch.randn(32, 20)
        reconstructed, latent = model(x)
        assert reconstructed.shape == (32, 20)
        assert latent.shape == (32, 10)

    def test_encode_decode(self):
        """Verify encode and decode work independently."""
        import torch
        model = Autoencoder(input_dim=20, latent_dim=10)
        x = torch.randn(16, 20)
        z = model.encode(x)
        assert z.shape == (16, 10)
        x_hat = model.decode(z)
        assert x_hat.shape == (16, 20)


class TestFeatureExtractor:

    def test_fit_transform(self):
        """Verify feature extractor reduces dimensionality."""
        X, _ = generate_synthetic_dataset(n_samples=200, n_features=20)
        X_val, _ = generate_synthetic_dataset(n_samples=50, n_features=20, seed=99)

        extractor = FeatureExtractor(
            input_dim=20,
            latent_dim=AUTOENCODER_LATENT_DIM,
            epochs=3,
            device="cpu",
        )
        X_latent, X_val_latent = extractor.fit_transform(X, X_val)

        assert X_latent.shape == (200, AUTOENCODER_LATENT_DIM)
        assert X_val_latent.shape == (50, AUTOENCODER_LATENT_DIM)

    def test_transform_only(self):
        """Verify transform works after fit."""
        X, _ = generate_synthetic_dataset(n_samples=100, n_features=20)

        extractor = FeatureExtractor(
            input_dim=20, latent_dim=10, epochs=2, device="cpu"
        )
        extractor.fit(X)

        X_new, _ = generate_synthetic_dataset(n_samples=30, n_features=20, seed=99)
        X_latent = extractor.transform(X_new)
        assert X_latent.shape == (30, 10)
