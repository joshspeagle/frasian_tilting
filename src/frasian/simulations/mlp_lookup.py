"""Generate and use lookup tables from trained MLP.

Post-process the trained MLP to create fast lookup tables for η*(w, α, Δ).
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator
import h5py
from pathlib import Path

from .mlp_data import delta_transform, delta_inverse, eta_transform, eta_inverse


def find_optimal_eta_prime(
    mlp_predict: Callable,
    w: float,
    alpha: float,
    delta_prime: float,
    log_target: bool = True,
    n_grid: int = 100,
) -> Tuple[float, float]:
    """Find η'* that minimizes E[W]/W_Wald for given (w, α, Δ').

    Uses grid search over η' for robust optimization.

    Parameters
    ----------
    mlp_predict : callable
        Function that takes X (n, 4) and returns predictions.
        If log_target=True, predictions are log(ratio).
    w : float
        Prior weight
    alpha : float
        Significance level
    delta_prime : float
        Transformed conflict Δ'
    log_target : bool
        If True, MLP predicts log(ratio), so we minimize that directly.
        The returned width_ratio is exp(prediction).
    n_grid : int
        Number of grid points for η' search (default 100)

    Returns
    -------
    eta_prime_star : float
        Optimal η' that minimizes width ratio
    min_width_ratio : float
        Width ratio at optimal η' (always in original scale, not log)
    """
    # Grid search over η' ∈ [0.01, 0.99]
    eta_prime_grid = np.linspace(0.01, 0.99, n_grid)

    # Vectorized prediction
    X = np.column_stack([
        np.full(n_grid, w),
        np.full(n_grid, alpha),
        np.full(n_grid, delta_prime),
        eta_prime_grid
    ])
    predictions = mlp_predict(X)

    # Find minimum
    min_idx = np.argmin(predictions)
    eta_prime_star = eta_prime_grid[min_idx]
    min_pred = predictions[min_idx]

    # Convert back from log scale if needed
    if log_target:
        width_ratio = np.exp(min_pred)
    else:
        width_ratio = min_pred

    return eta_prime_star, width_ratio


def generate_lookup_table(
    mlp_predict: Callable,
    w_grid: np.ndarray,
    alpha_grid: np.ndarray,
    delta_prime_grid: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """Generate lookup table: η'*(w, α, Δ') and corresponding E[W]/W_Wald.

    For each (w, α, Δ') triple, find the η' that minimizes the MLP-predicted
    width ratio.

    Parameters
    ----------
    mlp_predict : callable
        Function that takes X (n, 4) and returns predictions
    w_grid : ndarray
        Grid of w values
    alpha_grid : ndarray
        Grid of alpha values
    delta_prime_grid : ndarray
        Grid of Δ' values
    verbose : bool
        Show progress bar

    Returns
    -------
    lookup : dict with keys:
        - eta_prime_star: array (n_w, n_alpha, n_delta)
        - width_ratio: array (n_w, n_alpha, n_delta)
        - w_grid, alpha_grid, delta_prime_grid
    """
    from tqdm import tqdm

    n_w = len(w_grid)
    n_alpha = len(alpha_grid)
    n_delta = len(delta_prime_grid)

    eta_prime_star = np.zeros((n_w, n_alpha, n_delta))
    width_ratio = np.zeros((n_w, n_alpha, n_delta))

    total = n_w * n_alpha * n_delta
    iterator = tqdm(total=total, desc="Building lookup table") if verbose else None

    for i, w in enumerate(w_grid):
        for j, alpha in enumerate(alpha_grid):
            for k, delta_prime in enumerate(delta_prime_grid):
                eta_star, ratio = find_optimal_eta_prime(
                    mlp_predict, w, alpha, delta_prime
                )
                eta_prime_star[i, j, k] = eta_star
                width_ratio[i, j, k] = ratio

                if iterator:
                    iterator.update(1)

    if iterator:
        iterator.close()

    return {
        'eta_prime_star': eta_prime_star,
        'width_ratio': width_ratio,
        'w_grid': w_grid,
        'alpha_grid': alpha_grid,
        'delta_prime_grid': delta_prime_grid,
    }


def enforce_monotonicity(lookup: Dict) -> Dict:
    """Enforce monotonicity of η'* with respect to Δ' using cumulative maximum.

    For each (w, α) slice, applies cumulative max along the Δ' axis
    to ensure η'*(Δ') is monotonically increasing.

    Parameters
    ----------
    lookup : dict
        Lookup table from generate_lookup_table()

    Returns
    -------
    lookup_monotonic : dict
        Lookup table with monotonicity enforced
    """
    eta_prime_star = lookup['eta_prime_star'].copy()
    n_w, n_alpha, n_delta = eta_prime_star.shape

    for i in range(n_w):
        for j in range(n_alpha):
            # Apply cumulative max along Δ' axis
            eta_prime_star[i, j, :] = np.maximum.accumulate(eta_prime_star[i, j, :])

    return {
        **lookup,
        'eta_prime_star': eta_prime_star,
    }


def save_lookup_table(lookup: Dict, filepath: str) -> None:
    """Save lookup table to HDF5 file.

    Parameters
    ----------
    lookup : dict
        Lookup table from generate_lookup_table()
    filepath : str
        Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        for key, val in lookup.items():
            f.create_dataset(key, data=val, compression='gzip')


def load_lookup_table(filepath: str) -> Dict:
    """Load lookup table from HDF5 file.

    Parameters
    ----------
    filepath : str
        Input file path

    Returns
    -------
    lookup : dict
    """
    with h5py.File(filepath, 'r') as f:
        return {key: f[key][:] for key in f.keys()}


class OptimalEtaLookup:
    """Fast lookup for η*(w, α, |Δ|) using interpolation on precomputed table.

    This class provides efficient access to optimal tilting parameters
    by interpolating a precomputed lookup table.

    Parameters
    ----------
    lookup : dict
        Lookup table from generate_lookup_table() or load_lookup_table()

    Examples
    --------
    >>> lookup = OptimalEtaLookup.from_file("optimal_eta_lookup.h5")
    >>> eta_star = lookup.get_optimal_eta(w=0.5, alpha=0.05, delta=1.0)
    >>> ratio = lookup.get_width_ratio(w=0.5, alpha=0.05, delta=1.0)
    """

    def __init__(self, lookup: Dict):
        self.lookup = lookup
        self._w_grid = lookup['w_grid']
        self._alpha_grid = lookup['alpha_grid']
        self._delta_prime_grid = lookup['delta_prime_grid']

        # Build interpolator for η'*
        self._eta_interp = RegularGridInterpolator(
            (self._w_grid, self._alpha_grid, self._delta_prime_grid),
            lookup['eta_prime_star'],
            method='linear',
            bounds_error=False,
            fill_value=None,  # Extrapolate
        )

        # Build interpolator for width ratio
        self._ratio_interp = RegularGridInterpolator(
            (self._w_grid, self._alpha_grid, self._delta_prime_grid),
            lookup['width_ratio'],
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

    def get_optimal_eta(
        self,
        w: float,
        alpha: float,
        delta: float,
    ) -> float:
        """Get optimal η for given (w, α, |Δ|).

        Converts from (Δ, η') to (Δ', η) internally.

        Parameters
        ----------
        w : float
            Prior weight in (0, 1)
        alpha : float
            Significance level in (0, 1)
        delta : float
            Prior-data conflict |Δ| in [0, ∞)

        Returns
        -------
        eta_star : float
            Optimal tilting parameter in [η_min(w), 1]
        """
        # Transform Δ → Δ'
        delta_prime = delta_transform(delta)

        # Clip to grid bounds for safety
        w_clipped = np.clip(w, self._w_grid[0], self._w_grid[-1])
        alpha_clipped = np.clip(alpha, self._alpha_grid[0], self._alpha_grid[-1])
        delta_prime_clipped = np.clip(delta_prime, self._delta_prime_grid[0], self._delta_prime_grid[-1])

        # Lookup η'*
        eta_prime_star = self._eta_interp([[w_clipped, alpha_clipped, delta_prime_clipped]])[0]

        # Transform η' → η
        eta_star = eta_inverse(eta_prime_star, w)

        return float(eta_star)

    def get_width_ratio(
        self,
        w: float,
        alpha: float,
        delta: float,
    ) -> float:
        """Get expected E[W]/W_Wald at optimal η.

        Parameters
        ----------
        w : float
            Prior weight in (0, 1)
        alpha : float
            Significance level in (0, 1)
        delta : float
            Prior-data conflict |Δ| in [0, ∞)

        Returns
        -------
        width_ratio : float
            E[W_tilted(η*)]/W_Wald
        """
        delta_prime = delta_transform(delta)

        # Clip to grid bounds
        w_clipped = np.clip(w, self._w_grid[0], self._w_grid[-1])
        alpha_clipped = np.clip(alpha, self._alpha_grid[0], self._alpha_grid[-1])
        delta_prime_clipped = np.clip(delta_prime, self._delta_prime_grid[0], self._delta_prime_grid[-1])

        return float(self._ratio_interp([[w_clipped, alpha_clipped, delta_prime_clipped]])[0])

    def get_optimal_eta_batch(
        self,
        w: np.ndarray,
        alpha: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Batch lookup for multiple points.

        Parameters
        ----------
        w, alpha, delta : ndarray
            Arrays of same length

        Returns
        -------
        eta_star : ndarray
            Array of optimal η values
        """
        w = np.atleast_1d(w)
        alpha = np.atleast_1d(alpha)
        delta = np.atleast_1d(delta)

        delta_prime = delta_transform(delta)

        # Clip to grid bounds
        w_clipped = np.clip(w, self._w_grid[0], self._w_grid[-1])
        alpha_clipped = np.clip(alpha, self._alpha_grid[0], self._alpha_grid[-1])
        delta_prime_clipped = np.clip(delta_prime, self._delta_prime_grid[0], self._delta_prime_grid[-1])

        points = np.column_stack([w_clipped, alpha_clipped, delta_prime_clipped])
        eta_prime_star = self._eta_interp(points)

        return eta_inverse(eta_prime_star, w)

    @classmethod
    def from_file(cls, filepath: str) -> 'OptimalEtaLookup':
        """Load lookup from HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to lookup table HDF5 file

        Returns
        -------
        lookup : OptimalEtaLookup
        """
        return cls(load_lookup_table(filepath))

    @property
    def w_range(self) -> Tuple[float, float]:
        """Range of w values in lookup table."""
        return (float(self._w_grid[0]), float(self._w_grid[-1]))

    @property
    def alpha_range(self) -> Tuple[float, float]:
        """Range of alpha values in lookup table."""
        return (float(self._alpha_grid[0]), float(self._alpha_grid[-1]))

    @property
    def delta_range(self) -> Tuple[float, float]:
        """Range of |Δ| values in lookup table (original coordinates)."""
        return (
            float(delta_inverse(self._delta_prime_grid[0])),
            float(delta_inverse(self._delta_prime_grid[-1])),
        )


# =============================================================================
# Default Grid Configuration
# =============================================================================

LOOKUP_CONFIG = {
    'default': {
        'w_grid': np.linspace(0.01, 0.99, 21),
        'alpha_grid': np.linspace(0.01, 0.99, 21),
        'delta_prime_grid': np.linspace(0.0, 0.99, 51),
    },
    'fast': {
        'w_grid': np.array([0.2, 0.5, 0.8]),
        'alpha_grid': np.array([0.05, 0.10]),
        'delta_prime_grid': np.linspace(0, 0.9, 10),
    },
}
