"""
Processed Results Caching

Caches computed results (coverage, widths) to avoid recomputation when the
same raw data and parameters are used multiple times. Cache is automatically
invalidated if the raw data file is newer than the cache file.

Cache keys are structured like:
    coverage_w0.5_alpha0.05_waldo.h5
    widths_alpha0.05_eta0.5.h5
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .storage import save_simulation, load_simulation, simulation_exists
from .raw import get_raw_dir, get_processed_dir, get_raw_simulation_mtime


# ==============================================================================
# Cache Key Generation
# ==============================================================================

def _format_float(x: float) -> str:
    """Format float for cache key (remove trailing zeros)."""
    return f"{x:.4f}".rstrip("0").rstrip(".")


def get_cache_key(
    raw_name: str,
    category: str,  # "coverage" or "widths"
    method: str,
    alpha: float,
    w: Optional[float] = None,
    eta: Optional[float] = None,
) -> str:
    """Generate a cache key for processed results.

    Args:
        raw_name: Name of the raw simulation file (e.g., "coverage_raw")
        category: "coverage" or "widths"
        method: CI method name
        alpha: Significance level
        w: Prior weight (for coverage results)
        eta: Tilting parameter (for tilted methods)

    Returns:
        Cache key string (used as filename without extension)
    """
    parts = [category]

    if w is not None:
        parts.append(f"w{_format_float(w)}")

    parts.append(f"alpha{_format_float(alpha)}")
    parts.append(method)

    if eta is not None:
        parts.append(f"eta{_format_float(eta)}")

    return "_".join(parts)


def get_cache_path(cache_key: str) -> Path:
    """Get full path to a cache file."""
    return get_processed_dir() / f"{cache_key}.h5"


# ==============================================================================
# Cache Validity Checking
# ==============================================================================

def is_cache_valid(cache_key: str, raw_name: str) -> bool:
    """Check if a cache file exists and is newer than the raw data.

    Args:
        cache_key: Cache key (filename without extension)
        raw_name: Name of the raw simulation file

    Returns:
        True if cache exists and is valid (raw data hasn't changed)
    """
    cache_path = get_cache_path(cache_key)

    if not cache_path.exists():
        return False

    raw_mtime = get_raw_simulation_mtime(raw_name)
    if raw_mtime is None:
        # Raw data doesn't exist - cache is invalid
        return False

    cache_mtime = cache_path.stat().st_mtime
    return cache_mtime > raw_mtime


def invalidate_cache(cache_key: str) -> bool:
    """Delete a cache file if it exists.

    Args:
        cache_key: Cache key

    Returns:
        True if file was deleted, False if it didn't exist
    """
    cache_path = get_cache_path(cache_key)
    if cache_path.exists():
        cache_path.unlink()
        return True
    return False


def clear_processed_cache() -> int:
    """Delete all processed cache files.

    Returns:
        Number of files deleted
    """
    processed_dir = get_processed_dir()
    if not processed_dir.exists():
        return 0

    count = 0
    for f in processed_dir.glob("*.h5"):
        f.unlink()
        count += 1

    return count


# ==============================================================================
# Cache Save/Load
# ==============================================================================

def save_cached_result(
    cache_key: str,
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    raw_name: str,
) -> Path:
    """Save processed results to the cache.

    Args:
        cache_key: Cache key
        data: Dictionary of numpy arrays
        metadata: Dictionary of metadata
        raw_name: Name of the raw data file (stored for reference)

    Returns:
        Path to saved cache file
    """
    # Add cache-specific metadata
    metadata = metadata.copy()
    metadata["_cache_key"] = cache_key
    metadata["_raw_name"] = raw_name
    metadata["_cached_at"] = datetime.now().isoformat()

    return save_simulation(cache_key, data, metadata, output_dir=get_processed_dir())


def load_cached_result(cache_key: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load processed results from the cache.

    Args:
        cache_key: Cache key

    Returns:
        Tuple of (data_dict, metadata_dict)

    Raises:
        FileNotFoundError: If cache file doesn't exist
    """
    return load_simulation(cache_key, output_dir=get_processed_dir())


def cached_result_exists(cache_key: str) -> bool:
    """Check if a cache file exists (doesn't check validity)."""
    return simulation_exists(cache_key, output_dir=get_processed_dir())


# ==============================================================================
# Get-or-Compute Pattern
# ==============================================================================

def get_or_compute(
    cache_key: str,
    raw_name: str,
    compute_fn: Callable[[], tuple[dict, dict]],
    force: bool = False,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """Get result from cache or compute and cache it.

    Args:
        cache_key: Cache key for this result
        raw_name: Name of the raw simulation file
        compute_fn: Function that returns (data_dict, metadata_dict)
        force: If True, recompute even if cache exists
        verbose: Print cache status messages

    Returns:
        Tuple of (data_dict, metadata_dict)
    """
    if not force and is_cache_valid(cache_key, raw_name):
        if verbose:
            print(f"  Loading from cache: {cache_key}")
        return load_cached_result(cache_key)

    if verbose:
        if cached_result_exists(cache_key):
            print(f"  Cache invalidated (raw data changed): {cache_key}")
        else:
            print(f"  Computing: {cache_key}")

    # Compute the result
    data, metadata = compute_fn()

    # Save to cache
    save_cached_result(cache_key, data, metadata, raw_name)

    if verbose:
        print(f"  Cached: {cache_key}")

    return data, metadata


# ==============================================================================
# High-Level Cache Functions for Common Operations
# ==============================================================================

def get_or_compute_coverage(
    D_samples: np.ndarray,
    theta_grid: np.ndarray,
    w: float,
    method: str,
    alpha: float = 0.05,
    eta: Optional[float] = None,
    mu0: float = 0.0,
    sigma: float = 1.0,
    raw_name: str = "coverage_raw",
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """Get or compute coverage results with caching.

    Args:
        D_samples: 2D array [n_theta, n_reps] for this w value
        theta_grid: Array of theta values
        w: Prior weight
        method: CI method
        alpha: Significance level
        eta: Tilting parameter (for "tilted" method)
        mu0: Prior mean
        sigma: Likelihood standard deviation
        raw_name: Name of raw data file
        force: Force recomputation
        verbose: Print progress

    Returns:
        Dictionary with coverage_rates, coverage_se, indicators
    """
    from .processing import compute_coverage_indicators, compute_sigma0_from_w

    cache_key = get_cache_key(raw_name, "coverage", method, alpha, w=w, eta=eta)

    def compute_fn():
        sigma0 = compute_sigma0_from_w(sigma, w)
        n_theta = len(theta_grid)
        n_reps = D_samples.shape[1]

        coverage_rates = np.zeros(n_theta)
        coverage_se = np.zeros(n_theta)
        indicators = np.zeros((n_theta, n_reps), dtype=bool)

        for i, theta in enumerate(theta_grid):
            D_row = D_samples[i, :]
            ind = compute_coverage_indicators(
                D_row, theta, mu0, sigma, sigma0, method, alpha, eta
            )
            indicators[i, :] = ind
            coverage_rates[i] = ind.mean()
            n = len(ind)
            p = coverage_rates[i]
            coverage_se[i] = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0

        data = {
            "coverage_rates": coverage_rates,
            "coverage_se": coverage_se,
            "indicators": indicators.astype(np.uint8),  # Save as uint8 to save space
            "theta_grid": theta_grid,
        }

        metadata = {
            "method": method,
            "alpha": alpha,
            "eta": eta,
            "w": w,
            "mu0": mu0,
            "sigma": sigma,
            "n_theta": n_theta,
            "n_reps": n_reps,
        }

        return data, metadata

    data, metadata = get_or_compute(cache_key, raw_name, compute_fn, force, verbose)

    # Convert indicators back to bool
    if "indicators" in data:
        data["indicators"] = data["indicators"].astype(bool)

    return data


def get_or_compute_widths(
    D_samples: np.ndarray,
    theta_grid: np.ndarray,
    method: str,
    alpha: float = 0.05,
    eta: Optional[float] = None,
    mu0: float = 0.0,
    sigma: float = 1.0,
    w: float = 0.5,
    raw_name: str = "width_raw",
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """Get or compute width results with caching.

    Args:
        D_samples: 2D array [n_theta, n_samples] of D samples
        theta_grid: Array of theta values
        method: CI method
        alpha: Significance level
        eta: Tilting parameter (for "tilted" method)
        mu0: Prior mean
        sigma: Likelihood standard deviation
        w: Prior weight
        raw_name: Name of raw data file
        force: Force recomputation
        verbose: Print progress

    Returns:
        Dictionary with mean_widths, width_se, width_samples, delta_samples
    """
    from .processing import (
        compute_width_samples,
        compute_delta_samples,
        compute_sigma0_from_w,
    )

    cache_key = get_cache_key(raw_name, "widths", method, alpha, eta=eta)

    def compute_fn():
        sigma0 = compute_sigma0_from_w(sigma, w)
        n_theta = len(theta_grid)
        n_samples = D_samples.shape[1]

        mean_widths = np.zeros(n_theta)
        width_se = np.zeros(n_theta)
        all_widths = np.zeros((n_theta, n_samples), dtype=np.float32)
        all_deltas = np.zeros((n_theta, n_samples), dtype=np.float32)

        for i, theta in enumerate(theta_grid):
            D_row = D_samples[i, :]
            widths = compute_width_samples(D_row, mu0, sigma, sigma0, method, alpha, eta)
            all_widths[i, :] = widths
            mean_widths[i] = widths.mean()
            width_se[i] = widths.std(ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0.0

            # Compute Delta for each D
            all_deltas[i, :] = compute_delta_samples(D_row, mu0, sigma, sigma0)

        data = {
            "mean_widths": mean_widths,
            "width_se": width_se,
            "width_samples": all_widths,
            "delta_samples": all_deltas,
            "theta_grid": theta_grid,
        }

        metadata = {
            "method": method,
            "alpha": alpha,
            "eta": eta,
            "w": w,
            "mu0": mu0,
            "sigma": sigma,
            "n_theta": n_theta,
            "n_samples": n_samples,
        }

        return data, metadata

    return get_or_compute(cache_key, raw_name, compute_fn, force, verbose)


# ==============================================================================
# Cache Info
# ==============================================================================

def list_cached_results() -> list[str]:
    """List all cached result keys."""
    processed_dir = get_processed_dir()
    if not processed_dir.exists():
        return []
    return [f.stem for f in processed_dir.glob("*.h5")]


def get_cache_info() -> dict:
    """Get information about the cache.

    Returns:
        Dictionary with:
            - "raw_files": list of raw simulation files
            - "cached_files": list of cached result files
            - "raw_size_bytes": total size of raw files
            - "cached_size_bytes": total size of cached files
    """
    raw_dir = get_raw_dir()
    processed_dir = get_processed_dir()

    raw_files = [f.stem for f in raw_dir.glob("*.h5")] if raw_dir.exists() else []
    cached_files = list_cached_results()

    raw_size = sum(f.stat().st_size for f in raw_dir.glob("*.h5")) if raw_dir.exists() else 0
    cached_size = sum(f.stat().st_size for f in processed_dir.glob("*.h5")) if processed_dir.exists() else 0

    return {
        "raw_files": raw_files,
        "cached_files": cached_files,
        "raw_size_bytes": raw_size,
        "cached_size_bytes": cached_size,
    }
