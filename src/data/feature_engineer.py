"""
Autoencoder-based feature extraction for dimensionality reduction.

This is a key component of the novel contribution — using a convolutional
autoencoder to compress selected features into a compact latent
representation that captures essential traffic patterns.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    AE_HIDDEN_DIMS,
    AE_LEARNING_RATE,
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_PATIENCE,
    AUTOENCODER_LATENT_DIM,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder for feature extraction.

    Architecture:
        Encoder: input_dim → hidden[0] → hidden[1] → latent_dim
        Decoder: latent_dim → hidden[1] → hidden[0] → input_dim

    Uses batch normalization and LeakyReLU activations for stable training.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = AUTOENCODER_LATENT_DIM,
        hidden_dims: list = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = AE_HIDDEN_DIMS

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Output in [0, 1]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode → decode.

        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class FeatureExtractor:
    """
    Wrapper for training the autoencoder and extracting latent features.

    Usage:
        extractor = FeatureExtractor(input_dim=20)
        extractor.fit(X_train)
        X_train_latent = extractor.transform(X_train)
        X_test_latent = extractor.transform(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = AUTOENCODER_LATENT_DIM,
        hidden_dims: list = None,
        lr: float = AE_LEARNING_RATE,
        batch_size: int = AE_BATCH_SIZE,
        epochs: int = AE_EPOCHS,
        patience: int = AE_PATIENCE,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Train the autoencoder on training data.

        Args:
            X_train: Training features (n_samples, n_features).
            X_val: Validation features for early stopping.

        Returns:
            Dictionary with training history.
        """
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val))
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training autoencoder on {self.device}")
        logger.info(
            f"  input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
            f"epochs={self.epochs}"
        )

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                reconstructed, _ = self.model(batch)
                loss = self.criterion(reconstructed, batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(train_dataset)
            self.train_losses.append(epoch_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_total = 0.0
                with torch.no_grad():
                    for (batch,) in val_loader:
                        batch = batch.to(self.device)
                        reconstructed, _ = self.model(batch)
                        val_total += self.criterion(reconstructed, batch).item() * batch.size(0)
                val_loss = val_total / len(X_val)
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_best_model()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(best val_loss={best_val_loss:.6f})"
                        )
                        self._load_best_model()
                        break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  Epoch {epoch + 1}/{self.epochs} — train_loss: {epoch_loss:.6f}"
                if val_loss is not None:
                    msg += f", val_loss: {val_loss:.6f}"
                logger.info(msg)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract latent features from input data.

        Args:
            X: Input features (n_samples, input_dim).

        Returns:
            Latent features (n_samples, latent_dim).
        """
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            _, latent = self.model(tensor)
            return latent.cpu().numpy()

    def fit_transform(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Train autoencoder and return latent representations.

        Returns:
            Tuple of (X_train_latent, X_val_latent or None)
        """
        self.fit(X_train, X_val)
        X_train_latent = self.transform(X_train)

        X_val_latent = None
        if X_val is not None:
            X_val_latent = self.transform(X_val)

        return X_train_latent, X_val_latent

    def _save_best_model(self):
        """Save the best model checkpoint."""
        path = MODELS_DIR / "autoencoder_best.pt"
        torch.save(self.model.state_dict(), path)

    def _load_best_model(self):
        """Load the best model checkpoint."""
        path = MODELS_DIR / "autoencoder_best.pt"
        if path.exists():
            self.model.load_state_dict(torch.load(path, weights_only=True))

    def save(self, path: Optional[str] = None):
        """Save the complete feature extractor state."""
        if path is None:
            path = str(MODELS_DIR / "feature_extractor.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, path)
        logger.info(f"Feature extractor saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "FeatureExtractor":
        """Load a saved feature extractor."""
        checkpoint = torch.load(path, weights_only=False)
        extractor = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            device=device,
        )
        extractor.model.load_state_dict(checkpoint["model_state_dict"])
        extractor.train_losses = checkpoint["train_losses"]
        extractor.val_losses = checkpoint["val_losses"]
        logger.info(f"Feature extractor loaded from {path}")
        return extractor
