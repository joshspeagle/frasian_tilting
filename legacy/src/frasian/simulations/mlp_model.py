"""MLP model for width ratio prediction.

Train a neural network to predict E[W_tilted]/W_Wald from standardized
input coordinates (w, alpha, delta_prime, eta_prime).

Uses PyTorch with:
- GELU activations
- AdamW optimizer
- OneCycleLR scheduler
- Uniform distribution standardization
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt


# Uniform distribution statistics for [0, 1]
UNIFORM_MEAN = 0.5
UNIFORM_STD = 1.0 / np.sqrt(12)  # ≈ 0.2887


def standardize_uniform(x: np.ndarray) -> np.ndarray:
    """Standardize assuming uniform [0, 1] distribution."""
    return (x - UNIFORM_MEAN) / UNIFORM_STD


def destandardize_uniform(x_std: np.ndarray) -> np.ndarray:
    """Reverse standardization."""
    return x_std * UNIFORM_STD + UNIFORM_MEAN


class WidthRatioMLP:
    """PyTorch MLP for width ratio prediction with modern training."""

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (64, 64, 64),
        input_dim: int = 4,
    ):
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.model = None
        self.device = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _build_model(self):
        """Build PyTorch model with GELU activations."""
        import torch
        import torch.nn as nn

        layers = []
        in_features = self.input_dim

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.GELU())
            in_features = hidden_size

        # Output layer (single value: log width ratio)
        layers.append(nn.Linear(in_features, 1))

        return nn.Sequential(*layers)

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
        """Train the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 4)
            Input features [w, alpha, delta_prime, eta_prime]
        y : ndarray of shape (n_samples,)
            Target values (log width ratio)
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

        # Standardize inputs using uniform distribution statistics
        X_std = standardize_uniform(X)

        # Train/val split
        n_samples = len(X)
        n_val = int(n_samples * val_fraction)
        indices = np.random.RandomState(42).permutation(n_samples)
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

        # Optimizer and scheduler
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
            pct_start=0.1,  # 10% warmup
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

            # Print progress every 10 epochs or at milestones
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
        """Make predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 4)
            Input features (raw, will be standardized internally)

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
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

    def plot_loss_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot training and validation loss curves.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure

        Returns
        -------
        fig : matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        epochs = np.arange(1, len(self.train_losses) + 1)

        # Full loss curve
        ax1 = axes[0]
        ax1.plot(epochs, self.train_losses, label='Train', alpha=0.7)
        ax1.plot(epochs, self.val_losses, label='Validation', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Training Loss Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Zoomed to last 50%
        ax2 = axes[1]
        start_idx = len(epochs) // 2
        ax2.plot(epochs[start_idx:], self.train_losses[start_idx:], label='Train', alpha=0.7)
        ax2.plot(epochs[start_idx:], self.val_losses[start_idx:], label='Validation', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('Loss Curve (Last 50%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved loss curve to: {save_path}")

        return fig

    def save(self, model_path: str) -> None:
        """Save model to disk."""
        import torch

        state = {
            'model_state_dict': self.model.state_dict(),
            'hidden_sizes': self.hidden_sizes,
            'input_dim': self.input_dim,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(state, model_path)

    @classmethod
    def load(cls, model_path: str) -> 'WidthRatioMLP':
        """Load model from disk."""
        import torch

        state = torch.load(model_path, map_location='cpu')

        mlp = cls(
            hidden_sizes=state['hidden_sizes'],
            input_dim=state['input_dim'],
        )
        mlp.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlp.model = mlp._build_model().to(mlp.device)
        mlp.model.load_state_dict(state['model_state_dict'])
        mlp.train_losses = state.get('train_losses', [])
        mlp.val_losses = state.get('val_losses', [])

        return mlp


def train_pytorch_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: Tuple[int, ...] = (64, 64, 64),
    max_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> Tuple['WidthRatioMLP', Dict]:
    """Train PyTorch MLP with modern settings.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 4)
        Input features [w, alpha, delta_prime, eta_prime]
    y : ndarray of shape (n_samples,)
        Target values (log width ratio)
    hidden_sizes : tuple of int
        Hidden layer sizes
    max_epochs : int
        Number of training epochs
    batch_size : int
        Mini-batch size
    lr : float
        Peak learning rate
    weight_decay : float
        AdamW weight decay
    verbose : bool
        Print training progress

    Returns
    -------
    mlp : WidthRatioMLP
        Trained model
    history : dict
        Training history with losses
    """
    mlp = WidthRatioMLP(hidden_sizes=hidden_sizes, input_dim=X.shape[1])
    history = mlp.fit(
        X, y,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        verbose=verbose,
    )
    return mlp, history


def evaluate_model(
    mlp: WidthRatioMLP,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """Evaluate trained model.

    Parameters
    ----------
    mlp : WidthRatioMLP
        Trained model
    X : ndarray
        Input features
    y : ndarray
        True targets

    Returns
    -------
    metrics : dict with r2, rmse, mae
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y_pred = mlp.predict(X)

    return {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
    }


def create_predictor(mlp: WidthRatioMLP):
    """Create a prediction function.

    Parameters
    ----------
    mlp : WidthRatioMLP
        Trained model

    Returns
    -------
    predict_fn : callable
        Function that takes X (n_samples, 4) and returns predictions
    """
    def predict_fn(X: np.ndarray) -> np.ndarray:
        return mlp.predict(X)

    return predict_fn


# Keep sklearn version for backwards compatibility
def train_sklearn_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: Tuple[int, ...] = (64, 64, 64),
    max_iter: int = 1000,
    verbose: bool = True,
) -> Tuple:
    """Train MLP using sklearn's MLPRegressor (legacy).

    DEPRECATED: Use train_pytorch_mlp instead.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    print("WARNING: Using legacy sklearn MLP. Consider train_pytorch_mlp instead.")

    # Use uniform standardization for consistency
    scaler = StandardScaler()
    # Override with uniform stats
    scaler.mean_ = np.full(X.shape[1], UNIFORM_MEAN)
    scaler.scale_ = np.full(X.shape[1], UNIFORM_STD)
    scaler.var_ = np.full(X.shape[1], UNIFORM_STD**2)
    scaler.n_features_in_ = X.shape[1]

    X_scaled = scaler.transform(X)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        early_stopping=False,
        random_state=42,
        verbose=verbose,
    )
    mlp.fit(X_scaled, y)

    return mlp, scaler


def save_model(mlp, scaler_or_path, model_path: Optional[str] = None, scaler_path: Optional[str] = None) -> None:
    """Save model to disk.

    Handles both PyTorch and sklearn models.
    """
    if isinstance(mlp, WidthRatioMLP):
        # PyTorch model
        if isinstance(scaler_or_path, str):
            mlp.save(scaler_or_path)
        else:
            raise ValueError("For PyTorch model, provide single path")
    else:
        # sklearn model (legacy)
        import joblib
        model_path = Path(model_path) if model_path else Path(scaler_or_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(mlp, model_path)
        if scaler_path:
            joblib.dump(scaler_or_path, scaler_path)


def load_model(model_path: str, scaler_path: Optional[str] = None) -> Tuple:
    """Load model from disk.

    Automatically detects PyTorch vs sklearn format.
    """
    model_path = Path(model_path)

    # Try PyTorch first
    if model_path.suffix == '.pt' or model_path.suffix == '.pth':
        mlp = WidthRatioMLP.load(str(model_path))
        return mlp, None

    # Try to load as PyTorch anyway
    try:
        import torch
        state = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state:
            mlp = WidthRatioMLP.load(str(model_path))
            return mlp, None
    except:
        pass

    # Fall back to sklearn/joblib
    import joblib
    mlp = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    return mlp, scaler
