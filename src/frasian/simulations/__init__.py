"""
Simulation Infrastructure for Frasian Inference

Three-layer architecture:
- Layer 0 (raw): Raw D ~ N(theta, sigma) samples only
- Layer 1 (processing): Compute CIs, coverage, widths from D samples
- Layer 1.5 (cache): Optional caching of processed results

Modules:
- raw: Generate and save raw D samples
- processing: Compute derived quantities from D samples
- cache: Processed results caching with auto-invalidation
- storage: HDF5 I/O utilities
"""

from .storage import (
    save_simulation,
    load_simulation,
    simulation_exists,
    get_simulation_metadata,
    SIMULATION_DIR,
)

from .raw import (
    generate_coverage_D_samples,
    generate_distribution_D_samples,
    generate_width_D_samples,
    generate_optimal_eta_grid,
    save_raw_simulation,
    load_raw_simulation,
    raw_simulation_exists,
    generate_all_raw_simulations,
    get_optimal_eta_interpolator,
    optimal_eta_empirical,
    smooth_optimal_eta,
    DEFAULT_CONFIG,
    FAST_CONFIG,
    RAW_DIR,
    PROCESSED_DIR,
)

from .processing import (
    compute_ci,
    compute_coverage_indicators,
    compute_coverage_rate,
    compute_width_samples,
    compute_mean_width,
    compute_posterior_samples,
    compute_sigma0_from_w,
    bootstrap_proportion,
    bootstrap_mean,
    bootstrap_statistic,
    process_coverage_grid,
    process_width_grid,
)

from .cache import (
    get_cache_key,
    is_cache_valid,
    invalidate_cache,
    clear_processed_cache,
    save_cached_result,
    load_cached_result,
    get_or_compute,
    get_or_compute_coverage,
    get_or_compute_widths,
    list_cached_results,
    get_cache_info,
)

__all__ = [
    # Storage
    "save_simulation",
    "load_simulation",
    "simulation_exists",
    "get_simulation_metadata",
    "SIMULATION_DIR",
    # Raw
    "generate_coverage_D_samples",
    "generate_distribution_D_samples",
    "generate_width_D_samples",
    "generate_optimal_eta_grid",
    "save_raw_simulation",
    "load_raw_simulation",
    "raw_simulation_exists",
    "generate_all_raw_simulations",
    "get_optimal_eta_interpolator",
    "optimal_eta_empirical",
    "DEFAULT_CONFIG",
    "FAST_CONFIG",
    "RAW_DIR",
    "PROCESSED_DIR",
    # Processing
    "compute_ci",
    "compute_coverage_indicators",
    "compute_coverage_rate",
    "compute_width_samples",
    "compute_mean_width",
    "compute_posterior_samples",
    "compute_sigma0_from_w",
    "bootstrap_proportion",
    "bootstrap_mean",
    "bootstrap_statistic",
    "process_coverage_grid",
    "process_width_grid",
    # Cache
    "get_cache_key",
    "is_cache_valid",
    "invalidate_cache",
    "clear_processed_cache",
    "save_cached_result",
    "load_cached_result",
    "get_or_compute",
    "get_or_compute_coverage",
    "get_or_compute_widths",
    "list_cached_results",
    "get_cache_info",
]
