"""
Raw Simulation Data Generation

This module generates the fundamental raw data for all simulations:
D ~ N(theta_true, sigma) samples. These are the only truly "raw" quantities
that need to be stored - everything else (CIs, coverage, widths) can be
recomputed from D with different parameters (alpha, method, eta).

Fixed canonical coordinates: mu0=0, sigma=1
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .storage import save_simulation, load_simulation, simulation_exists, SIMULATION_DIR


# ==============================================================================
# Directory Setup
# ==============================================================================

RAW_DIR = SIMULATION_DIR / "raw"
PROCESSED_DIR = SIMULATION_DIR / "processed"


def get_raw_dir() -> Path:
    """Get the raw simulations directory, creating if needed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR


def get_processed_dir() -> Path:
    """Get the processed cache directory, creating if needed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR


# ==============================================================================
# Raw D Sample Generation
# ==============================================================================

def generate_coverage_D_samples(
    theta_grid: np.ndarray,
    w_values: np.ndarray,
    n_reps: int,
    seed: int = 42,
    mu0: float = 0.0,
    sigma: float = 1.0,
) -> tuple[dict, dict]:
    """Generate raw D samples for coverage experiments.

    For each (theta, w) pair, generates n_reps samples of D ~ N(theta, sigma).

    Args:
        theta_grid: Array of true theta values to test
        w_values: Array of prior weight values
        n_reps: Number of MC replicates per cell
        seed: Random seed for reproducibility
        mu0: Prior mean (canonical: 0)
        sigma: Likelihood standard deviation (canonical: 1)

    Returns:
        Tuple of (data_dict, metadata_dict) suitable for save_simulation()

    Data structure:
        D_samples: float32[n_theta, n_w, n_reps] - the raw samples
        theta_grid: float64[n_theta]
        w_values: float64[n_w]
    """
    rng = np.random.default_rng(seed)

    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    w_values = np.asarray(w_values, dtype=np.float64)

    n_theta = len(theta_grid)
    n_w = len(w_values)

    # Generate D ~ N(theta_true, sigma) for each (theta, w) cell
    # Note: D doesn't depend on w, but we store it this way for consistency
    # with the coverage grid structure where we'll compute different sigma0 values
    D_samples = np.zeros((n_theta, n_w, n_reps), dtype=np.float32)

    for i, theta in enumerate(theta_grid):
        for j in range(n_w):
            # D ~ N(theta, sigma)
            D_samples[i, j, :] = rng.normal(theta, sigma, size=n_reps).astype(np.float32)

    data = {
        "D_samples": D_samples,
        "theta_grid": theta_grid,
        "w_values": w_values,
    }

    metadata = {
        "experiment": "coverage",
        "mu0": mu0,
        "sigma": sigma,
        "n_reps": n_reps,
        "n_theta": n_theta,
        "n_w": n_w,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
    }

    return data, metadata


def generate_distribution_D_samples(
    theta_values: np.ndarray,
    n_samples: int,
    w: float = 0.5,
    seed: int = 42,
    mu0: float = 0.0,
    sigma: float = 1.0,
) -> tuple[dict, dict]:
    """Generate raw D samples for distribution validation experiments.

    For each theta value, generates n_samples of D ~ N(theta, sigma).
    These are used to validate the distribution of (mu_n - theta) and tau_WALDO.

    Args:
        theta_values: Array of theta values to test
        n_samples: Number of samples per theta
        w: Prior weight (determines sigma0 = sigma * sqrt(w/(1-w)))
        seed: Random seed
        mu0: Prior mean (canonical: 0)
        sigma: Likelihood standard deviation (canonical: 1)

    Returns:
        Tuple of (data_dict, metadata_dict)

    Data structure:
        D_samples: float32[n_theta, n_samples]
        theta_values: float64[n_theta]
    """
    rng = np.random.default_rng(seed)

    theta_values = np.asarray(theta_values, dtype=np.float64)
    n_theta = len(theta_values)

    # Compute sigma0 from w: w = sigma0^2 / (sigma^2 + sigma0^2)
    # => sigma0 = sigma * sqrt(w / (1-w))
    sigma0 = sigma * np.sqrt(w / (1 - w))

    D_samples = np.zeros((n_theta, n_samples), dtype=np.float32)

    for i, theta in enumerate(theta_values):
        D_samples[i, :] = rng.normal(theta, sigma, size=n_samples).astype(np.float32)

    data = {
        "D_samples": D_samples,
        "theta_values": theta_values,
    }

    metadata = {
        "experiment": "distribution",
        "mu0": mu0,
        "sigma": sigma,
        "sigma0": sigma0,
        "w": w,
        "n_samples": n_samples,
        "n_theta": n_theta,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
    }

    return data, metadata


def generate_width_D_samples(
    theta_grid: np.ndarray,
    n_samples: int,
    w: float = 0.5,
    seed: int = 42,
    mu0: float = 0.0,
    sigma: float = 1.0,
) -> tuple[dict, dict]:
    """Generate raw D samples for CI width experiments.

    For each theta value, generates n_samples of D ~ N(theta, sigma).
    Width analysis will compute CI widths and Delta from the realized D values.

    Args:
        theta_grid: Array of theta values to sweep
        n_samples: Number of samples per theta
        w: Prior weight (determines sigma0)
        seed: Random seed
        mu0: Prior mean (canonical: 0)
        sigma: Likelihood standard deviation (canonical: 1)

    Returns:
        Tuple of (data_dict, metadata_dict)

    Data structure:
        D_samples: float32[n_theta, n_samples]
        theta_grid: float64[n_theta]
    """
    rng = np.random.default_rng(seed)

    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    n_theta = len(theta_grid)

    # Compute sigma0 from w
    sigma0 = sigma * np.sqrt(w / (1 - w))

    D_samples = np.zeros((n_theta, n_samples), dtype=np.float32)

    for i, theta in enumerate(theta_grid):
        D_samples[i, :] = rng.normal(theta, sigma, size=n_samples).astype(np.float32)

    data = {
        "D_samples": D_samples,
        "theta_grid": theta_grid,
    }

    metadata = {
        "experiment": "width",
        "mu0": mu0,
        "sigma": sigma,
        "sigma0": sigma0,
        "w": w,
        "n_samples": n_samples,
        "n_theta": n_theta,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
    }

    return data, metadata


# ==============================================================================
# Save/Load Raw Simulations
# ==============================================================================

def save_raw_simulation(
    name: str,
    data: dict,
    metadata: dict,
) -> Path:
    """Save raw simulation data to the raw/ directory.

    Args:
        name: Simulation name (e.g., "coverage_raw")
        data: Dictionary of numpy arrays
        metadata: Dictionary of metadata

    Returns:
        Path to saved file
    """
    return save_simulation(name, data, metadata, output_dir=get_raw_dir())


def load_raw_simulation(name: str) -> tuple[dict, dict]:
    """Load raw simulation data from the raw/ directory.

    Args:
        name: Simulation name (e.g., "coverage_raw")

    Returns:
        Tuple of (data_dict, metadata_dict)
    """
    return load_simulation(name, output_dir=get_raw_dir())


def raw_simulation_exists(name: str) -> bool:
    """Check if a raw simulation file exists."""
    return simulation_exists(name, output_dir=get_raw_dir())


def get_raw_simulation_path(name: str) -> Path:
    """Get path to a raw simulation file."""
    return get_raw_dir() / f"{name}.h5"


def get_raw_simulation_mtime(name: str) -> Optional[float]:
    """Get modification time of raw simulation file, or None if doesn't exist."""
    path = get_raw_simulation_path(name)
    if path.exists():
        return path.stat().st_mtime
    return None


# ==============================================================================
# Optimal Eta Computation (Numerical)
# ==============================================================================

def _compute_mean_width_for_eta(args):
    """Helper function for parallel computation of mean width at a given eta."""
    eta, D_samples, mu0, sigma, sigma0, alpha, tilted_ci_width = args
    widths = []
    for D in D_samples:
        try:
            width = tilted_ci_width(D, mu0, sigma, sigma0, eta, alpha)
            widths.append(width)
        except (ValueError, RuntimeError):
            continue
    if len(widths) > 0.5 * len(D_samples):
        return eta, np.mean(widths)
    return eta, np.inf


def _compute_optimal_eta_for_delta(args):
    """Helper function for computation of optimal eta at a given delta.

    Uses golden section search instead of grid search for efficiency.
    """
    (abs_delta, eta_bounds, n_sims, mu0, sigma, sigma0, w, alpha,
     seed_offset, compute_mean_width_fn) = args

    from scipy.optimize import minimize_scalar

    rng = np.random.default_rng(42 + seed_offset)

    # theta_true that produces expected |Delta| = abs_delta
    theta_true = mu0 + sigma * abs_delta / (1 - w)

    # Simulate D samples (shared across all eta evaluations)
    D_samples = rng.normal(theta_true, sigma, n_sims).astype(np.float32)

    # Extract bounds
    eta_min, eta_max = eta_bounds

    def objective(eta):
        """Objective function: mean CI width at this eta."""
        return compute_mean_width_fn(D_samples, mu0, sigma, sigma0, eta, alpha)

    # Use bounded minimization (Brent's method)
    result = minimize_scalar(
        objective,
        bounds=(eta_min, eta_max),
        method='bounded',
        options={'xatol': 0.01}  # Tolerance of 0.01 in eta
    )

    return result.x, result.fun


def generate_optimal_eta_grid(
    delta_grid: np.ndarray,
    eta_search_grid: np.ndarray = None,
    n_sims: int = 2000,
    w: float = 0.5,
    alpha: float = 0.05,
    seed: int = 42,
    mu0: float = 0.0,
    sigma: float = 1.0,
    verbose: bool = True,
    n_jobs: int = -1,  # Use all cores by default
) -> tuple[dict, dict]:
    """Numerically compute optimal tilting eta* for each |Delta| value.

    For each |Delta|, finds eta that minimizes expected CI width by:
    1. Setting theta_true to achieve target |Delta|
    2. Simulating D ~ N(theta_true, sigma)
    3. Computing mean CI width for each eta in search grid
    4. Selecting eta with minimum mean width

    Uses parallel processing for speedup.

    Args:
        delta_grid: Array of |Delta| values to compute eta* for
        eta_search_grid: Grid of eta values to search over (default: 0 to 1 by 0.05)
        n_sims: Number of D samples per (delta, eta) pair
        w: Prior weight
        alpha: Significance level
        seed: Random seed
        mu0: Prior mean (canonical: 0)
        sigma: Likelihood standard deviation (canonical: 1)
        verbose: Print progress
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Tuple of (data_dict, metadata_dict)

    Data structure:
        delta_grid: float64[n_delta] - the |Delta| values
        optimal_eta: float64[n_delta] - optimal eta* for each |Delta|
        expected_width_ratio: float64[n_delta] - E[W_eta*]/W_Wald for each |Delta|
    """
    # Import here to avoid circular imports
    from ..tilting import compute_mean_width_for_eta
    from ..waldo import wald_ci_width

    # Try to import tqdm for progress bars
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Try to import joblib for parallel processing
    try:
        from joblib import Parallel, delayed
        import os
        use_parallel = n_jobs != 1
        actual_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
    except ImportError:
        use_parallel = False
        actual_jobs = 1

    # Auto-compute eta bounds based on w
    # Constraint: σ_η² = σ²[w + η(1-w)] > 0  =>  η > -w/(1-w)
    eta_min = -w / (1 - w) + 0.01  # Small buffer above singularity
    eta_max = 1.0
    eta_bounds = (eta_min, eta_max)

    # eta_search_grid is kept for metadata but not used in optimization
    if eta_search_grid is None:
        eta_search_grid = np.array([eta_min, eta_max])  # Just store bounds

    delta_grid = np.asarray(delta_grid, dtype=np.float64)
    n_delta = len(delta_grid)

    # Compute sigma0 from w
    sigma0 = sigma * np.sqrt(w / (1 - w))

    # Reference Wald width
    w_wald = wald_ci_width(sigma, alpha)

    # Estimate ~10-15 function evals per delta with Brent's method
    est_evals_per_delta = 12
    est_total_cis = n_delta * est_evals_per_delta * n_sims
    if verbose:
        print(f"Computing optimal eta* for {n_delta} |Delta| values...")
        print(f"  eta bounds: [{eta_min:.2f}, {eta_max:.2f}] (using Brent optimization)")
        print(f"  {n_sims} simulations per optimization step")
        print(f"  Estimated CI computations: ~{est_total_cis:,}")
        if use_parallel:
            print(f"  Using {actual_jobs} parallel workers")
        print()

    if use_parallel:
        # Parallel computation with progress bar
        args_list = [
            (abs_delta, eta_bounds, n_sims, mu0, sigma, sigma0, w, alpha,
             i, compute_mean_width_for_eta)
            for i, abs_delta in enumerate(delta_grid)
        ]

        if use_tqdm and verbose:
            # Use tqdm with joblib
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_compute_optimal_eta_for_delta)(args)
                for args in tqdm(args_list, desc="Computing optimal eta*", unit="delta")
            )
        else:
            # Fallback: print progress manually
            print("  Starting parallel computation...")
            results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_compute_optimal_eta_for_delta)(args)
                for args in args_list
            )
            print("  Parallel computation complete.")

        optimal_eta = np.array([r[0] for r in results], dtype=np.float64)
        best_widths = np.array([r[1] for r in results], dtype=np.float64)
        expected_width_ratio = best_widths / w_wald
        expected_width_ratio[~np.isfinite(expected_width_ratio)] = 1.0
    else:
        # Sequential computation
        optimal_eta = np.zeros(n_delta, dtype=np.float64)
        expected_width_ratio = np.zeros(n_delta, dtype=np.float64)

        delta_iter = enumerate(delta_grid)
        if use_tqdm and verbose:
            delta_iter = tqdm(list(delta_iter), desc="Computing optimal eta*", unit="delta")

        for i, abs_delta in delta_iter:
            if not use_tqdm and verbose and (i % 10 == 0 or i == n_delta - 1):
                pct = 100 * (i + 1) / n_delta
                print(f"  [{pct:5.1f}%] |Delta| = {abs_delta:.2f} ({i+1}/{n_delta})")

            args = (abs_delta, eta_bounds, n_sims, mu0, sigma, sigma0, w, alpha,
                    i, compute_mean_width_for_eta)
            best_eta, best_mean_width = _compute_optimal_eta_for_delta(args)

            optimal_eta[i] = best_eta
            expected_width_ratio[i] = best_mean_width / w_wald if best_mean_width < np.inf else 1.0

    if verbose:
        print(f"\nOptimal eta* computation complete.")

    data = {
        "delta_grid": delta_grid,
        "optimal_eta": optimal_eta,
        "expected_width_ratio": expected_width_ratio,
        "eta_bounds": np.array([eta_min, eta_max]),
    }

    metadata = {
        "experiment": "optimal_eta",
        "mu0": mu0,
        "sigma": sigma,
        "sigma0": sigma0,
        "w": w,
        "alpha": alpha,
        "n_sims": n_sims,
        "n_delta": n_delta,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "optimization_method": "bounded_brent",
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
    }

    return data, metadata


def smooth_optimal_eta(delta_grid: np.ndarray, optimal_eta: np.ndarray,
                       window: int = 5) -> np.ndarray:
    """Apply smoothing to the optimal eta curve.

    Uses a Savitzky-Golay filter for smooth derivatives, or falls back
    to simple moving average if scipy is unavailable.

    Args:
        delta_grid: The |Delta| grid values
        optimal_eta: Raw optimal eta values
        window: Window size for smoothing (must be odd)

    Returns:
        Smoothed optimal eta values
    """
    try:
        from scipy.signal import savgol_filter
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        # Use polynomial order 2 for smooth but responsive fit
        return savgol_filter(optimal_eta, window, 2)
    except ImportError:
        # Fallback to simple moving average
        kernel = np.ones(window) / window
        # Pad edges to avoid shrinking
        padded = np.pad(optimal_eta, window // 2, mode='edge')
        return np.convolve(padded, kernel, mode='valid')


def get_optimal_eta_interpolator(fast: bool = False, smooth: bool = True,
                                  smooth_window: int = 11):
    """Get an interpolation function for optimal eta*.

    Loads precomputed optimal eta grid and returns a function that
    interpolates to any |Delta| value.

    Args:
        fast: If True, generate fast config if data doesn't exist
        smooth: If True, apply smoothing to the eta* curve
        smooth_window: Window size for Savitzky-Golay smoothing

    Returns:
        Function: abs_delta -> eta_star (handles scalar or array input)

    Example:
        get_eta = get_optimal_eta_interpolator()
        eta_star = get_eta(1.5)  # Get eta* for |Delta| = 1.5
    """
    from scipy.interpolate import interp1d

    # Ensure data exists
    if not raw_simulation_exists("optimal_eta"):
        config = FAST_CONFIG if fast else DEFAULT_CONFIG
        generate_all_raw_simulations(config=config, force=False, verbose=True)

    data, metadata = load_raw_simulation("optimal_eta")
    delta_grid = data["delta_grid"]
    optimal_eta = data["optimal_eta"]

    # Apply smoothing if requested
    if smooth and len(optimal_eta) > smooth_window:
        optimal_eta = smooth_optimal_eta(delta_grid, optimal_eta, smooth_window)

    # Create interpolator with extrapolation for values outside grid
    interp_func = interp1d(
        delta_grid, optimal_eta,
        kind='linear',
        bounds_error=False,
        fill_value=(optimal_eta[0], optimal_eta[-1])  # Extrapolate with edge values
    )

    def eta_star(abs_delta):
        """Get optimal eta* for given |Delta| value(s)."""
        # Interpolate - bounds are already respected in the precomputed grid
        return interp_func(np.abs(abs_delta))

    return eta_star


# Cache for the interpolator to avoid reloading
_optimal_eta_interpolator_cache = {}


def optimal_eta_empirical(abs_delta: float, fast: bool = False) -> float:
    """Get empirically computed optimal eta* for a given |Delta|.

    This uses the precomputed numerical grid with interpolation.
    Much faster than recomputing, and based on actual MC simulation.

    Args:
        abs_delta: Absolute value of scaled prior-data conflict
        fast: If True, use fast config for generation if needed

    Returns:
        Optimal tilting parameter eta*
    """
    cache_key = "fast" if fast else "default"

    if cache_key not in _optimal_eta_interpolator_cache:
        _optimal_eta_interpolator_cache[cache_key] = get_optimal_eta_interpolator(fast=fast)

    return float(_optimal_eta_interpolator_cache[cache_key](abs_delta))


# ==============================================================================
# Default Configurations
# ==============================================================================

# Production-quality configuration
DEFAULT_CONFIG = {
    "seed": 42,
    "mu0": 0.0,
    "sigma": 1.0,
    "coverage": {
        "theta_grid": np.linspace(-4, 6, 21),
        "w_values": np.array([0.2, 0.5, 0.8]),
        "n_reps": 10_000,
    },
    "distribution": {
        "theta_values": np.array([0.0, 1.0, 2.0, 3.0]),
        "w": 0.5,
        "n_samples": 10_000,
    },
    "width": {
        "theta_grid": np.linspace(-4, 6, 51),
        "w": 0.5,
        "n_samples": 5_000,
    },
    "optimal_eta": {
        "delta_grid": np.linspace(0, 5, 251),  # Very fine grid: 0, 0.02, 0.04, ..., 5.0
        "n_sims": 500,  # More sims for stability
        "w": 0.5,
        "alpha": 0.05,
    },
}

# Fast configuration for testing
FAST_CONFIG = {
    "seed": 42,
    "mu0": 0.0,
    "sigma": 1.0,
    "coverage": {
        "theta_grid": np.linspace(-4, 6, 11),
        "w_values": np.array([0.5]),  # Single w for speed
        "n_reps": 1_000,
    },
    "distribution": {
        "theta_values": np.array([0.0, 2.0]),
        "w": 0.5,
        "n_samples": 1_000,
    },
    "width": {
        "theta_grid": np.linspace(-4, 6, 21),
        "w": 0.5,
        "n_samples": 500,
    },
    "optimal_eta": {
        "delta_grid": np.linspace(0, 5, 11),  # Coarser grid for speed: 0, 0.5, 1.0, ..., 5.0
        "n_sims": 100,  # Fewer simulations for speed
        "w": 0.5,
        "alpha": 0.05,
    },
}


def generate_all_raw_simulations(
    config: Optional[dict] = None,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Generate all raw simulation data files.

    Args:
        config: Configuration dict (default: DEFAULT_CONFIG)
        force: If True, regenerate even if files exist
        verbose: Print progress messages

    Returns:
        Dictionary mapping simulation names to file paths
    """
    if config is None:
        config = DEFAULT_CONFIG

    paths = {}

    # Coverage raw data
    if force or not raw_simulation_exists("coverage_raw"):
        if verbose:
            print("Generating coverage raw D samples...")
        data, meta = generate_coverage_D_samples(
            theta_grid=config["coverage"]["theta_grid"],
            w_values=config["coverage"]["w_values"],
            n_reps=config["coverage"]["n_reps"],
            seed=config["seed"],
            mu0=config["mu0"],
            sigma=config["sigma"],
        )
        paths["coverage_raw"] = save_raw_simulation("coverage_raw", data, meta)
        if verbose:
            print(f"  Saved: {paths['coverage_raw']}")
    else:
        if verbose:
            print("Coverage raw data exists, skipping...")
        paths["coverage_raw"] = get_raw_simulation_path("coverage_raw")

    # Distribution raw data
    if force or not raw_simulation_exists("distribution_raw"):
        if verbose:
            print("Generating distribution raw D samples...")
        data, meta = generate_distribution_D_samples(
            theta_values=config["distribution"]["theta_values"],
            n_samples=config["distribution"]["n_samples"],
            w=config["distribution"]["w"],
            seed=config["seed"],
            mu0=config["mu0"],
            sigma=config["sigma"],
        )
        paths["distribution_raw"] = save_raw_simulation("distribution_raw", data, meta)
        if verbose:
            print(f"  Saved: {paths['distribution_raw']}")
    else:
        if verbose:
            print("Distribution raw data exists, skipping...")
        paths["distribution_raw"] = get_raw_simulation_path("distribution_raw")

    # Width raw data
    if force or not raw_simulation_exists("width_raw"):
        if verbose:
            print("Generating width raw D samples...")
        data, meta = generate_width_D_samples(
            theta_grid=config["width"]["theta_grid"],
            n_samples=config["width"]["n_samples"],
            w=config["width"]["w"],
            seed=config["seed"],
            mu0=config["mu0"],
            sigma=config["sigma"],
        )
        paths["width_raw"] = save_raw_simulation("width_raw", data, meta)
        if verbose:
            print(f"  Saved: {paths['width_raw']}")
    else:
        if verbose:
            print("Width raw data exists, skipping...")
        paths["width_raw"] = get_raw_simulation_path("width_raw")

    # Optimal eta grid (numerically computed)
    if force or not raw_simulation_exists("optimal_eta"):
        if verbose:
            print("Computing optimal eta* numerically...")
        eta_search = config["optimal_eta"].get("eta_search_grid", None)
        data, meta = generate_optimal_eta_grid(
            delta_grid=config["optimal_eta"]["delta_grid"],
            eta_search_grid=eta_search,
            n_sims=config["optimal_eta"]["n_sims"],
            w=config["optimal_eta"]["w"],
            alpha=config["optimal_eta"]["alpha"],
            seed=config["seed"],
            mu0=config["mu0"],
            sigma=config["sigma"],
            verbose=verbose,
        )
        paths["optimal_eta"] = save_raw_simulation("optimal_eta", data, meta)
        if verbose:
            print(f"  Saved: {paths['optimal_eta']}")
    else:
        if verbose:
            print("Optimal eta data exists, skipping...")
        paths["optimal_eta"] = get_raw_simulation_path("optimal_eta")

    return paths
