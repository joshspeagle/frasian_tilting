"""Generate training data for MLP regression of optimal tilting.

This module generates training data in standardized 4D space:
- w: prior weight [0, 1]
- alpha: significance level [0, 1]
- delta_prime: transformed conflict Δ' = Δ/(1+Δ) [0, 1]
- eta_prime: transformed tilting η' = η(1-w) + w [0, 1]

Target: E[W_tilted]/W_Wald (width ratio)
"""

import numpy as np
from scipy.stats import qmc
from typing import Dict, Tuple, Optional
import h5py
from pathlib import Path


# =============================================================================
# Coordinate Transforms
# =============================================================================

def delta_transform(delta: np.ndarray) -> np.ndarray:
    """Transform Δ → Δ' = Δ/(1+Δ), maps [0, ∞) → [0, 1)."""
    delta = np.asarray(delta)
    return delta / (1 + delta)


def delta_inverse(delta_prime: np.ndarray) -> np.ndarray:
    """Transform Δ' → Δ = Δ'/(1-Δ'), maps [0, 1) → [0, ∞)."""
    delta_prime = np.asarray(delta_prime)
    return delta_prime / (1 - delta_prime)


def eta_transform(eta: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Transform η → η' = η(1-w) + w, maps [η_min(w), 1] → [0, 1].

    η_min(w) = -w/(1-w)
    """
    eta = np.asarray(eta)
    w = np.asarray(w)
    return eta * (1 - w) + w


def eta_inverse(eta_prime: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Transform η' → η = (η' - w)/(1-w), maps [0, 1] → [η_min(w), 1].

    η' = 0 → η = -w/(1-w) = η_min(w)
    η' = 1 → η = 1 (Wald)
    """
    eta_prime = np.asarray(eta_prime)
    w = np.asarray(w)
    return (eta_prime - w) / (1 - w)


def eta_min(w: float) -> float:
    """Compute minimum allowed η for given w.

    η > -w/(1-w) is required for tilted variance to be positive.
    """
    return -w / (1 - w)


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================

def generate_latin_hypercube_samples(
    n_samples: int = 10_000,
    seed: int = 42,
    eps: float = 0.01,
) -> np.ndarray:
    """Generate Latin Hypercube samples in standardized 4D space.

    All variables are sampled from [eps, 1-eps] to avoid boundary singularities.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility
    eps : float
        Padding from boundaries (default 0.01)

    Returns
    -------
    samples : ndarray of shape (n_samples, 4)
        Columns are [w, alpha, delta_prime, eta_prime], all in [eps, 1-eps]
    """
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples_unit = sampler.random(n=n_samples)  # [0,1]⁴

    # Scale to [eps, 1-eps]
    samples = eps + samples_unit * (1 - 2 * eps)

    return samples


# =============================================================================
# Width Ratio Computation
# =============================================================================

def compute_width_ratio_for_sample(
    w: float,
    alpha: float,
    delta_prime: float,
    eta_prime: float,
    n_mc: int = 100,
    mu0: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
) -> float:
    """Compute E[W_tilted]/W_Wald for a single (w, α, Δ', η') point.

    Parameters
    ----------
    w : float
        Prior weight in [0, 1]
    alpha : float
        Significance level in [0, 1]
    delta_prime : float
        Transformed conflict Δ' in [0, 1]
    eta_prime : float
        Transformed tilting η' in [0, 1]
    n_mc : int
        Number of Monte Carlo samples
    mu0 : float
        Prior mean (default 0)
    sigma : float
        Observation std (default 1)
    seed : int, optional
        Random seed

    Returns
    -------
    width_ratio : float
        E[W_tilted] / W_Wald, or np.nan if computation fails
    """
    from ..tilting import tilted_ci
    from ..waldo import wald_ci_width

    # Inverse transforms to original coordinates
    delta = delta_inverse(delta_prime)
    eta = eta_inverse(eta_prime, w)

    # Handle edge cases
    if w <= 0 or w >= 1:
        return np.nan

    # Compute σ₀ from w: w = σ₀²/(σ² + σ₀²) → σ₀ = σ√(w/(1-w))
    sigma0 = sigma * np.sqrt(w / (1 - w))

    # Find θ_true that gives E[Δ] = delta
    # Δ = (1-w)(μ₀ - D)/σ, so E[Δ] = (1-w)(μ₀ - θ_true)/σ when D ~ N(θ_true, σ²)
    # θ_true = μ₀ - delta * σ / (1-w)
    theta_true = mu0 - delta * sigma / (1 - w)

    rng = np.random.default_rng(seed)
    D_samples = rng.normal(theta_true, sigma, n_mc)

    # Compute tilted CI widths
    widths = []
    for D in D_samples:
        try:
            ci_low, ci_high = tilted_ci(D, mu0, sigma, sigma0, eta, alpha)
            width = ci_high - ci_low
            if np.isfinite(width) and width > 0:
                widths.append(width)
        except Exception:
            continue

    if len(widths) == 0:
        return np.nan

    # Wald width for comparison
    W_wald = wald_ci_width(sigma, alpha)

    ratio = np.mean(widths) / W_wald

    # Return log(ratio) for better MLP fitting
    # Log transform handles skewed distribution (ratio >> 1 for some configs)
    if ratio <= 0:
        return np.nan

    return np.log(ratio)


def _compute_one_sample(args):
    """Worker function for parallel computation."""
    i, X_row, n_mc, base_seed = args
    return compute_width_ratio_for_sample(
        w=X_row[0],
        alpha=X_row[1],
        delta_prime=X_row[2],
        eta_prime=X_row[3],
        n_mc=n_mc,
        seed=base_seed + i if base_seed is not None else None,
    )


# =============================================================================
# Training Data Generation
# =============================================================================

def generate_training_data(
    n_samples: int = 10_000,
    n_mc: int = 100,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Dict:
    """Generate full training dataset for MLP regression.

    Parameters
    ----------
    n_samples : int
        Number of Latin Hypercube samples (default 10,000)
    n_mc : int
        MC samples per point for width estimation (default 100)
    seed : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel workers (-1 for all cores)
    verbose : bool
        Show progress bar

    Returns
    -------
    data : dict with keys:
        - X: ndarray (n_valid, 4) of [w, alpha, delta_prime, eta_prime]
        - y: ndarray (n_valid,) of E[W]/W_Wald
        - metadata: dict with generation parameters
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    # Generate input samples
    X = generate_latin_hypercube_samples(n_samples, seed)

    # Prepare arguments for parallel computation
    args_list = [(i, X[i], n_mc, seed) for i in range(n_samples)]

    # Compute width ratios in parallel
    if verbose:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_compute_one_sample)(args)
            for args in tqdm(args_list, desc="Generating training data")
        )
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_compute_one_sample)(args) for args in args_list
        )

    y = np.array(results)

    # Filter out NaN values
    valid = ~np.isnan(y)
    X_valid = X[valid]
    y_valid = y[valid]

    return {
        'X': X_valid,
        'y': y_valid,
        'metadata': {
            'n_samples': n_samples,
            'n_mc': n_mc,
            'seed': seed,
            'n_valid': len(y_valid),
            'columns': ['w', 'alpha', 'delta_prime', 'eta_prime'],
        }
    }


# =============================================================================
# I/O Functions
# =============================================================================

def save_training_data(data: Dict, filepath: str) -> None:
    """Save training data to HDF5 file.

    Parameters
    ----------
    data : dict
        Training data from generate_training_data()
    filepath : str
        Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('X', data=data['X'], compression='gzip')
        f.create_dataset('y', data=data['y'], compression='gzip')

        for key, val in data['metadata'].items():
            if isinstance(val, (list, tuple)):
                f.attrs[key] = str(val)
            else:
                f.attrs[key] = val


def load_training_data(filepath: str) -> Dict:
    """Load training data from HDF5 file.

    Parameters
    ----------
    filepath : str
        Input file path

    Returns
    -------
    data : dict with X, y, metadata
    """
    with h5py.File(filepath, 'r') as f:
        return {
            'X': f['X'][:],
            'y': f['y'][:],
            'metadata': dict(f.attrs),
        }


# =============================================================================
# Configuration
# =============================================================================

MLP_DATA_CONFIG = {
    'default': {
        'n_samples': 10_000,
        'n_mc': 100,
        'seed': 42,
    },
    'fast': {
        'n_samples': 1_000,
        'n_mc': 50,
        'seed': 42,
    },
}
