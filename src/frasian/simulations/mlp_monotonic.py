"""Monotonic neural network for optimal eta prediction.

This module provides a neural network that predicts eta'*(w, alpha, Delta')
with guaranteed monotonicity in Delta'.

Architecture uses partial monotonicity: the network is unconstrained in (w, alpha)
but monotonically increasing in Delta' through positive-weight pathways.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path


# Standardization for uniform [0, 1] inputs
UNIFORM_MEAN = 0.5
UNIFORM_STD = 1.0 / np.sqrt(12)


def standardize_uniform(x: np.ndarray) -> np.ndarray:
    """Standardize assuming uniform [0, 1] distribution."""
    return (x - UNIFORM_MEAN) / UNIFORM_STD


class MonotonicEtaMLP:
    """Neural network predicting eta'*(w, alpha, Delta') with monotonicity in Delta'.

    Uses a partial monotonicity architecture:
    - Unconstrained pathway for (w, alpha) interactions
    - Positive-weight pathway for Delta' to ensure monotonicity
    - Cross-terms allow (w, alpha) to modulate the Delta' response

    The network guarantees d(eta'*)/d(Delta') >= 0 by construction.
    """

    def __init__(
        self,
        shared_sizes: Tuple[int, ...] = (64, 64),
        mono_sizes: Tuple[int, ...] = (64, 64),
        cross_size: int = 32,
    ):
        """Initialize monotonic network.

        Parameters
        ----------
        shared_sizes : tuple of int
            Hidden layer sizes for (w, alpha) pathway
        mono_sizes : tuple of int
            Hidden layer sizes for monotonic Delta' pathway
        cross_size : int
            Size of cross-interaction layer
        """
        self.shared_sizes = shared_sizes
        self.mono_sizes = mono_sizes
        self.cross_size = cross_size
        self.model = None
        self.device = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _build_model(self):
        """Build the monotonic neural network."""
        import torch
        import torch.nn as nn

        class MonotonicNet(nn.Module):
            """Monotonic network using strictly monotonic activations."""

            def __init__(self, shared_sizes, mono_sizes, cross_size):
                super().__init__()

                # ============================================================
                # Shared pathway for (w, alpha) - unconstrained
                # ============================================================
                self.shared_layers = nn.ModuleList()
                in_features = 2  # w, alpha (standardized)
                for hidden_size in shared_sizes:
                    self.shared_layers.append(nn.Linear(in_features, hidden_size))
                    in_features = hidden_size
                self.shared_out_dim = in_features

                # ============================================================
                # Monotonic pathway for Delta' - positive weights + ReLU
                # ReLU is strictly monotonic (non-decreasing)
                # ============================================================
                self.mono_layers = nn.ModuleList()
                in_features = 1  # delta_prime (standardized)
                for hidden_size in mono_sizes:
                    layer = nn.Linear(in_features, hidden_size)
                    self.mono_layers.append(layer)
                    in_features = hidden_size
                self.mono_out_dim = in_features

                # ============================================================
                # Output layers
                # ============================================================
                # Base output from shared pathway (unconstrained)
                self.output_base = nn.Linear(self.shared_out_dim, 1)
                # Scale factor from shared pathway (will be made positive)
                self.output_scale = nn.Linear(self.shared_out_dim, 1)
                # Monotonic output (positive weights)
                self.output_mono = nn.Linear(self.mono_out_dim, 1)

                self._init_weights()

            def _init_weights(self):
                """Initialize weights for stable training."""
                for layer in self.shared_layers:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
                for layer in self.mono_layers:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    layer.weight.data.abs_()
                    nn.init.zeros_(layer.bias)
                nn.init.xavier_normal_(self.output_base.weight)
                nn.init.xavier_normal_(self.output_scale.weight)
                nn.init.xavier_normal_(self.output_mono.weight)
                self.output_mono.weight.data.abs_()

            def forward(self, x):
                """Forward pass with strict monotonicity guarantee.

                Architecture:
                    output = sigmoid(base(w,α) + softplus(scale(w,α)) * mono(Δ'))

                Where:
                - base(w,α): unconstrained function of (w, α)
                - scale(w,α): positive scaling factor
                - mono(Δ'): strictly increasing function of Δ' (positive weights + ReLU)

                This guarantees ∂output/∂Δ' ≥ 0 because:
                - softplus(scale) > 0
                - ∂mono/∂Δ' ≥ 0 (positive weights + ReLU)
                - sigmoid is strictly increasing
                """
                import torch
                import torch.nn.functional as F

                w_alpha = x[:, :2]
                delta_prime = x[:, 2:3]

                # Shared pathway (unconstrained)
                h_shared = w_alpha
                for layer in self.shared_layers:
                    h_shared = F.gelu(layer(h_shared))

                # Monotonic pathway (positive weights + ReLU = strictly increasing)
                h_mono = delta_prime
                for layer in self.mono_layers:
                    # |W| ensures positive weights, ReLU is monotonic
                    h_mono = F.relu(F.linear(h_mono, layer.weight.abs(), layer.bias))

                # Outputs
                base = self.output_base(h_shared)
                scale = F.softplus(self.output_scale(h_shared))  # Always > 0
                mono = F.linear(h_mono, self.output_mono.weight.abs(),
                               self.output_mono.bias)

                # Combine: base + positive_scale * increasing_mono
                output = base + scale * mono

                # Bounded output via sigmoid
                output = 0.01 + 0.98 * torch.sigmoid(output)

                return output

        return MonotonicNet(self.shared_sizes, self.mono_sizes, self.cross_size)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.1,
        verbose: bool = True,
    ) -> Dict:
        """Train the monotonic model.

        Uses the same training setup as WidthRatioMLP:
        - AdamW optimizer
        - OneCycleLR scheduler
        - GELU activations (softplus for monotonic paths)

        Parameters
        ----------
        X : ndarray of shape (n_samples, 3)
            Input features [w, alpha, delta_prime] in [0, 1]
        y : ndarray of shape (n_samples,)
            Target values (eta_prime_star) in [0, 1]
        max_epochs : int
            Number of training epochs
        batch_size : int
            Mini-batch size
        lr : float
            Peak learning rate for OneCycleLR
        weight_decay : float
            AdamW weight decay
        val_fraction : float
            Fraction of data for validation
        verbose : bool
            Print training progress

        Returns
        -------
        history : dict with train_losses, val_losses
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"Training on: {self.device}")

        # Standardize inputs
        X_std = standardize_uniform(X)

        # Train/val split with different seed than width ratio MLP
        n_samples = len(X)
        n_val = int(n_samples * val_fraction)
        indices = np.random.RandomState(123).permutation(n_samples)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train = torch.FloatTensor(X_std[train_idx]).to(self.device)
        y_train = torch.FloatTensor(y[train_idx]).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_std[val_idx]).to(self.device)
        y_val = torch.FloatTensor(y[val_idx]).unsqueeze(1).to(self.device)

        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Build model
        self.model = self._build_model().to(self.device)

        # Optimizer and scheduler (same as WidthRatioMLP)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        steps_per_epoch = len(train_loader)
        total_steps = max_epochs * steps_per_epoch

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Loss function
        criterion = nn.MSELoss()

        # Training loop
        self.train_losses = []
        self.val_losses = []

        if verbose:
            print(f"Training for {max_epochs} epochs ({total_steps} steps)")
            print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            print(f"  Batch size: {batch_size}, Steps/epoch: {steps_per_epoch}")
            print(f"  Architecture: shared={self.shared_sizes}, mono={self.mono_sizes}")

        for epoch in range(max_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            self.val_losses.append(val_loss)

            # Print progress
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1:4d}/{max_epochs}: "
                      f"train={avg_train_loss:.6f}, "
                      f"val={val_loss:.6f}, "
                      f"lr={current_lr:.2e}", flush=True)

        if verbose:
            print(f"Training complete. Final val_loss: {self.val_losses[-1]:.6f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict optimal eta' for given inputs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 3)
            Input features [w, alpha, delta_prime] in raw [0, 1] scale

        Returns
        -------
        eta_prime : ndarray of shape (n_samples,)
            Predicted optimal eta' values
        """
        import torch

        if self.model is None:
            raise ValueError("Model not trained yet")

        X = np.atleast_2d(X)
        X_std = standardize_uniform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_std).to(self.device)
            pred = self.model(X_tensor).cpu().numpy().flatten()

        return pred

    def save(self, model_path: str) -> None:
        """Save model to disk."""
        import torch

        state = {
            'model_state_dict': self.model.state_dict(),
            'shared_sizes': self.shared_sizes,
            'mono_sizes': self.mono_sizes,
            'cross_size': self.cross_size,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(state, model_path)

    @classmethod
    def load(cls, model_path: str) -> 'MonotonicEtaMLP':
        """Load model from disk."""
        import torch

        state = torch.load(model_path, map_location='cpu', weights_only=False)

        mlp = cls(
            shared_sizes=state['shared_sizes'],
            mono_sizes=state['mono_sizes'],
            cross_size=state['cross_size'],
        )
        mlp.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlp.model = mlp._build_model().to(mlp.device)
        mlp.model.load_state_dict(state['model_state_dict'])
        mlp.train_losses = state.get('train_losses', [])
        mlp.val_losses = state.get('val_losses', [])

        return mlp


class OptimalEtaPredictor:
    """Fast predictor for optimal eta using the monotonic MLP.

    Replaces the lookup table with direct neural network inference.
    Provides the same interface as OptimalEtaLookup.
    """

    def __init__(self, mono_mlp: MonotonicEtaMLP):
        """Initialize predictor.

        Parameters
        ----------
        mono_mlp : MonotonicEtaMLP
            Trained monotonic neural network
        """
        self.mono_mlp = mono_mlp

    def get_optimal_eta(
        self,
        w: float,
        alpha: float,
        delta: float,
    ) -> float:
        """Get optimal eta for given (w, alpha, |Delta|).

        Parameters
        ----------
        w : float
            Prior weight in (0, 1)
        alpha : float
            Significance level in (0, 1)
        delta : float
            Prior-data conflict |Delta| in [0, inf)

        Returns
        -------
        eta_star : float
            Optimal tilting parameter
        """
        from .mlp_data import delta_transform, eta_inverse

        # Transform Delta -> Delta'
        delta_prime = delta_transform(delta)

        # Clip inputs to valid range
        w_clipped = np.clip(w, 0.01, 0.99)
        alpha_clipped = np.clip(alpha, 0.01, 0.99)
        delta_prime_clipped = np.clip(delta_prime, 0.0, 0.99)

        # Predict eta'*
        X = np.array([[w_clipped, alpha_clipped, delta_prime_clipped]])
        eta_prime_star = self.mono_mlp.predict(X)[0]

        # Transform eta' -> eta
        eta_star = eta_inverse(eta_prime_star, w)

        return float(eta_star)

    def get_optimal_eta_batch(
        self,
        w: np.ndarray,
        alpha: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Batch prediction for multiple points.

        Parameters
        ----------
        w, alpha, delta : ndarray
            Arrays of same length

        Returns
        -------
        eta_star : ndarray
            Array of optimal eta values
        """
        from .mlp_data import delta_transform, eta_inverse

        w = np.atleast_1d(w)
        alpha = np.atleast_1d(alpha)
        delta = np.atleast_1d(delta)

        delta_prime = delta_transform(delta)

        # Clip inputs
        w_clipped = np.clip(w, 0.01, 0.99)
        alpha_clipped = np.clip(alpha, 0.01, 0.99)
        delta_prime_clipped = np.clip(delta_prime, 0.0, 0.99)

        X = np.column_stack([w_clipped, alpha_clipped, delta_prime_clipped])
        eta_prime_star = self.mono_mlp.predict(X)

        return eta_inverse(eta_prime_star, w)

    @classmethod
    def from_file(cls, model_path: str) -> 'OptimalEtaPredictor':
        """Load predictor from model file.

        Parameters
        ----------
        model_path : str
            Path to saved MonotonicEtaMLP model

        Returns
        -------
        predictor : OptimalEtaPredictor
        """
        mono_mlp = MonotonicEtaMLP.load(model_path)
        return cls(mono_mlp)
