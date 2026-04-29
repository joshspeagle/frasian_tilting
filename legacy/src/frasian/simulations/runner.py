"""
Simulation Runner

Orchestrates raw data generation and optional pre-processing.

Three-layer architecture:
- Layer 0: Generate raw D samples
- Layer 1.5: Optionally precompute common processed results (coverage, widths)
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .raw import (
    generate_all_raw_simulations,
    load_raw_simulation,
    raw_simulation_exists,
    DEFAULT_CONFIG,
    FAST_CONFIG,
    RAW_DIR,
)
from .cache import (
    get_or_compute_coverage,
    get_or_compute_widths,
    clear_processed_cache,
    get_cache_info,
)
from .processing import compute_sigma0_from_w


ExperimentType = Literal["raw", "coverage", "distributions", "widths", "all"]


class SimulationRunner:
    """Orchestrates raw data generation and processing.

    The new architecture:
    1. Generate raw D samples once (stored permanently)
    2. Compute derived quantities (CIs, coverage, widths) on demand
    3. Cache processed results for common configurations
    """

    def __init__(
        self,
        fast: bool = False,
        config: Optional[dict] = None,
    ):
        """Initialize runner.

        Args:
            fast: If True, use reduced sample sizes for quick testing
            config: Custom configuration (overrides fast flag if provided)
        """
        if config is not None:
            self.config = config
        else:
            self.config = FAST_CONFIG if fast else DEFAULT_CONFIG

    def run_raw(
        self,
        force: bool = False,
        verbose: bool = True,
    ) -> dict[str, Path]:
        """Generate all raw D sample files.

        Args:
            force: If True, regenerate even if files exist
            verbose: Print progress messages

        Returns:
            Dictionary mapping simulation names to file paths
        """
        if verbose:
            print("=" * 60)
            print("GENERATING RAW D SAMPLES")
            print("=" * 60)
            print(f"Configuration: {'FAST' if self.config is FAST_CONFIG else 'DEFAULT'}")

        return generate_all_raw_simulations(
            config=self.config,
            force=force,
            verbose=verbose,
        )

    def precompute_coverage(
        self,
        methods: list[str] = None,
        alpha: float = 0.05,
        force: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Precompute coverage results for common configurations.

        This loads raw D samples and computes coverage for all (theta, w, method)
        combinations, caching the results.

        Args:
            methods: List of CI methods (default: ["wald", "posterior", "waldo"])
            alpha: Significance level
            force: Force recomputation even if cache exists
            verbose: Print progress

        Returns:
            Dictionary of processed results by (w, method)
        """
        if methods is None:
            methods = ["wald", "posterior", "waldo"]

        # Load raw data
        if not raw_simulation_exists("coverage_raw"):
            if verbose:
                print("Raw coverage data not found, generating...")
            self.run_raw(force=False, verbose=verbose)

        data, metadata = load_raw_simulation("coverage_raw")
        D_samples = data["D_samples"]  # [n_theta, n_w, n_reps]
        theta_grid = data["theta_grid"]
        w_values = data["w_values"]
        mu0 = metadata["mu0"]
        sigma = metadata["sigma"]

        if verbose:
            print("\nPrecomputing coverage results...")
            print(f"  theta grid: {len(theta_grid)} points")
            print(f"  w values: {w_values}")
            print(f"  methods: {methods}")

        results = {}
        for j, w in enumerate(w_values):
            D_w = D_samples[:, j, :]  # [n_theta, n_reps]
            for method in methods:
                if verbose:
                    print(f"  Processing w={w}, method={method}...")

                result = get_or_compute_coverage(
                    D_samples=D_w,
                    theta_grid=theta_grid,
                    w=w,
                    method=method,
                    alpha=alpha,
                    mu0=mu0,
                    sigma=sigma,
                    force=force,
                    verbose=False,
                )
                results[(w, method)] = result

        return results

    def precompute_widths(
        self,
        methods: list[str] = None,
        eta_values: list[float] = None,
        alpha: float = 0.05,
        force: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Precompute width results for common configurations.

        Args:
            methods: List of CI methods (default: ["wald", "waldo", "tilted_optimal"])
            eta_values: Eta values for tilted method (default: [0, 0.25, 0.5, 0.75, 1.0])
            alpha: Significance level
            force: Force recomputation
            verbose: Print progress

        Returns:
            Dictionary of processed results by method
        """
        if methods is None:
            methods = ["wald", "waldo", "tilted_optimal"]
        if eta_values is None:
            eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Load raw data
        if not raw_simulation_exists("width_raw"):
            if verbose:
                print("Raw width data not found, generating...")
            self.run_raw(force=False, verbose=verbose)

        data, metadata = load_raw_simulation("width_raw")
        D_samples = data["D_samples"]  # [n_theta, n_samples]
        theta_grid = data["theta_grid"]
        mu0 = metadata["mu0"]
        sigma = metadata["sigma"]
        w = metadata["w"]

        if verbose:
            print("\nPrecomputing width results...")
            print(f"  theta grid: {len(theta_grid)} points")
            print(f"  methods: {methods}")

        results = {}

        for method in methods:
            if method == "tilted":
                # For tilted, compute for each eta value
                for eta in eta_values:
                    if verbose:
                        print(f"  Processing method={method}, eta={eta}...")
                    result = get_or_compute_widths(
                        D_samples=D_samples,
                        theta_grid=theta_grid,
                        method=method,
                        alpha=alpha,
                        eta=eta,
                        mu0=mu0,
                        sigma=sigma,
                        w=w,
                        force=force,
                        verbose=False,
                    )
                    results[(method, eta)] = result
            else:
                if verbose:
                    print(f"  Processing method={method}...")
                result = get_or_compute_widths(
                    D_samples=D_samples,
                    theta_grid=theta_grid,
                    method=method,
                    alpha=alpha,
                    mu0=mu0,
                    sigma=sigma,
                    w=w,
                    force=force,
                    verbose=False,
                )
                results[(method, None)] = result

        return results

    def run_all(
        self,
        precompute: bool = True,
        force: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Run all simulations and optionally precompute processed results.

        Args:
            precompute: If True, also precompute common processed results
            force: Force regeneration
            verbose: Print progress

        Returns:
            Dictionary with paths and info
        """
        results = {
            "raw_paths": self.run_raw(force=force, verbose=verbose)
        }

        if precompute:
            if verbose:
                print("\n" + "=" * 60)
                print("PRECOMPUTING PROCESSED RESULTS")
                print("=" * 60)

            results["coverage"] = self.precompute_coverage(force=force, verbose=verbose)
            results["widths"] = self.precompute_widths(force=force, verbose=verbose)

        if verbose:
            info = get_cache_info()
            print("\n" + "=" * 60)
            print("SIMULATION SUMMARY")
            print("=" * 60)
            print(f"Raw files: {info['raw_files']}")
            print(f"Cached files: {len(info['cached_files'])}")
            print(f"Raw size: {info['raw_size_bytes'] / 1024:.1f} KB")
            print(f"Cache size: {info['cached_size_bytes'] / 1024:.1f} KB")

        return results


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def list_cache():
    """Print information about cached simulations."""
    info = get_cache_info()

    print("\n" + "=" * 60)
    print("SIMULATION CACHE STATUS")
    print("=" * 60)

    print("\nRaw data files (permanent):")
    if info["raw_files"]:
        for name in info["raw_files"]:
            path = RAW_DIR / f"{name}.h5"
            size = path.stat().st_size if path.exists() else 0
            print(f"  - {name}.h5 ({format_size(size)})")
    else:
        print("  (none)")

    print(f"\nTotal raw size: {format_size(info['raw_size_bytes'])}")

    print("\nProcessed cache files (regenerable):")
    if info["cached_files"]:
        for name in sorted(info["cached_files"]):
            print(f"  - {name}.h5")
    else:
        print("  (none)")

    print(f"\nTotal cache size: {format_size(info['cached_size_bytes'])}")


def clear_cache(confirm: bool = True) -> int:
    """Clear processed cache (keeps raw data).

    Args:
        confirm: If True, prompt for confirmation

    Returns:
        Number of files deleted
    """
    if confirm:
        info = get_cache_info()
        n_files = len(info["cached_files"])
        if n_files == 0:
            print("No processed cache files to delete.")
            return 0

        print(f"This will delete {n_files} processed cache files.")
        print("Raw data files will NOT be deleted.")
        response = input("Continue? [y/N] ").strip().lower()
        if response != "y":
            print("Cancelled.")
            return 0

    count = clear_processed_cache()
    print(f"Deleted {count} cache files.")
    return count
