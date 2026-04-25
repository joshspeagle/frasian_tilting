#!/usr/bin/env python3
"""
Generate Publication Tables

Generates tables 1-4:
- Table 1: Coverage Results (Section 7)
- Table 2: CI Widths (Section 6)
- Table 3: Optimal Tilting Reference (Section 10)
- Table 4: Method Comparison Summary

Output formats: CSV and LaTeX

Usage:
    python scripts/generate_tables.py [--fast]
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from frasian.core import posterior_params, scaled_conflict, prior_residual
from frasian.waldo import (
    confidence_interval, wald_ci, posterior_ci,
    confidence_interval_width, wald_ci_width, posterior_ci_width
)
from frasian.tilting import optimal_eta_approximation, tilted_ci
from frasian.figure_style import MC_CONFIG


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_params(w: float, sigma: float = 1.0, mu0: float = 0.0):
    """Get model parameters for a given weight."""
    sigma0 = sigma * np.sqrt(w / (1 - w))
    return mu0, sigma, sigma0


def data_for_conflict(delta: float, mu0: float, w: float, sigma: float) -> float:
    """Compute D value that produces a given Delta."""
    return mu0 - sigma * delta / (1 - w)


def compute_coverage(theta_true, mu0, sigma, sigma0, n_reps=1000, alpha=0.05):
    """Compute coverage rates for all methods."""
    wald_covers = 0
    post_covers = 0
    waldo_covers = 0

    for _ in range(n_reps):
        D = np.random.normal(theta_true, sigma)

        # Wald CI
        wald_lo, wald_hi = wald_ci(D, sigma, alpha)
        if wald_lo <= theta_true <= wald_hi:
            wald_covers += 1

        # Posterior CI
        post_lo, post_hi = posterior_ci(D, mu0, sigma, sigma0, alpha)
        if post_lo <= theta_true <= post_hi:
            post_covers += 1

        # WALDO CI
        waldo_lo, waldo_hi = confidence_interval(D, mu0, sigma, sigma0, alpha)
        if waldo_lo <= theta_true <= waldo_hi:
            waldo_covers += 1

    return wald_covers/n_reps, post_covers/n_reps, waldo_covers/n_reps


def ensure_output_dir():
    """Ensure output directory exists."""
    output_dir = Path(__file__).parent.parent / "output" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_table(df: pd.DataFrame, name: str, output_dir: Path):
    """Save table in CSV and LaTeX formats."""
    # CSV
    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # LaTeX
    tex_path = output_dir / f"{name}.tex"
    latex_str = df.to_latex(index=True, float_format="%.2f")
    with open(tex_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved: {tex_path}")


# =============================================================================
# Table 1: Coverage Results
# =============================================================================

def generate_table_1_coverage(n_reps: int = 5000) -> pd.DataFrame:
    """
    Generate Table 1: Coverage Results (Section 7).

    Reproduces the coverage table showing Wald, Posterior, WALDO coverage rates.
    """
    print("\n" + "="*60)
    print("Table 1: Coverage Results (Section 7)")
    print("="*60)

    np.random.seed(42)

    # Model parameters (from document)
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    # Theta values to test
    theta_values = [-3, -2, -1, 0, 1, 2, 3, 5]

    results = []
    for theta in tqdm(theta_values, desc="Computing coverage"):
        wald_cov, post_cov, waldo_cov = compute_coverage(
            theta, mu0, sigma, sigma0, n_reps=n_reps
        )
        results.append({
            'theta': theta,
            'Wald': wald_cov * 100,
            'Posterior': post_cov * 100,
            'WALDO': waldo_cov * 100,
        })

    df = pd.DataFrame(results).set_index('theta')
    df.index.name = 'theta_true'

    print("\nCoverage Results (%):")
    print(df.to_string())

    return df


# =============================================================================
# Table 2: CI Widths
# =============================================================================

def generate_table_2_ci_widths() -> pd.DataFrame:
    """
    Generate Table 2: CI Widths (Section 6).

    Reproduces the CI width table.
    """
    print("\n" + "="*60)
    print("Table 2: CI Widths (Section 6)")
    print("="*60)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    # Delta values from document
    delta_values = [0, -1, -2.5, -5]

    results = []
    for delta in delta_values:
        D = data_for_conflict(delta, mu0, w, sigma)

        wald_w = wald_ci_width(sigma)
        post_w = posterior_ci_width(sigma, sigma0)
        waldo_w = confidence_interval_width(D, mu0, sigma, sigma0)

        results.append({
            'Delta': delta,
            'W_Wald': wald_w,
            'W_Posterior': post_w,
            'W_WALDO': waldo_w,
        })

    df = pd.DataFrame(results).set_index('Delta')

    print("\nCI Widths:")
    print(df.to_string())

    return df


# =============================================================================
# Table 3: Optimal Tilting Reference
# =============================================================================

def generate_table_3_optimal_tilting(n_samples: int = 200) -> pd.DataFrame:
    """
    Generate Table 3: Optimal Tilting Reference (Section 10).

    Shows eta* and expected width ratio for various |Delta|.
    """
    print("\n" + "="*60)
    print("Table 3: Optimal Tilting Reference (Section 10)")
    print("="*60)

    np.random.seed(42)

    # Model parameters
    w = 0.5
    mu0, sigma, sigma0 = get_model_params(w)

    # |Delta| values
    delta_abs_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    results = []
    for delta_abs in tqdm(delta_abs_values, desc="Computing optimal tilting"):
        eta_star = optimal_eta_approximation(delta_abs)

        # Compute expected width ratio via MC
        D = data_for_conflict(-delta_abs, mu0, w, sigma)  # D > mu0
        wald_width = wald_ci_width(sigma)

        # Sample D values and compute average tilted CI width
        theta_true = D  # Center at MLE for this calculation
        tilted_widths = []
        for _ in range(n_samples):
            D_sample = np.random.normal(theta_true, sigma)
            try:
                lo, hi = tilted_ci(D_sample, mu0, sigma, sigma0, eta_star)
                tilted_widths.append(hi - lo)
            except:
                tilted_widths.append(wald_width)

        expected_width_ratio = np.mean(tilted_widths) / wald_width

        results.append({
            '|Delta|': delta_abs,
            'eta_star': eta_star,
            'E[W]/W_Wald': expected_width_ratio,
        })

    df = pd.DataFrame(results).set_index('|Delta|')

    print("\nOptimal Tilting Reference:")
    print(df.to_string())

    return df


# =============================================================================
# Table 4: Method Comparison Summary
# =============================================================================

def generate_table_4_method_summary() -> pd.DataFrame:
    """
    Generate Table 4: Method Comparison Summary.

    Qualitative comparison of all methods.
    """
    print("\n" + "="*60)
    print("Table 4: Method Comparison Summary")
    print("="*60)

    data = {
        'Method': ['Wald', 'Posterior', 'WALDO', 'Tilted(eta*)'],
        'Coverage': ['95% always', 'Variable (fails at |theta| >> mu0)', '95% always', '95% always'],
        'Mean_Width': ['Constant (widest)', 'Narrowest', 'Varies with Delta', 'Varies, <= Wald avg'],
        'Uses_Prior': ['No', 'Yes', 'Yes', 'Yes (adaptively)'],
        'Computation': ['Trivial', 'Trivial', 'Root-finding', 'Optimization'],
    }

    df = pd.DataFrame(data).set_index('Method')

    print("\nMethod Comparison Summary:")
    print(df.to_string())

    return df


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication tables")
    parser.add_argument("--fast", action="store_true", help="Use fewer MC samples")
    args = parser.parse_args()

    output_dir = ensure_output_dir()

    n_coverage = 1000 if args.fast else MC_CONFIG['n_coverage']
    n_tilting = 100 if args.fast else 500

    print("="*60)
    print("GENERATING PUBLICATION TABLES")
    print("="*60)

    # Table 1: Coverage
    df1 = generate_table_1_coverage(n_reps=n_coverage)
    save_table(df1, "table_1_coverage", output_dir)

    # Table 2: CI Widths
    df2 = generate_table_2_ci_widths()
    save_table(df2, "table_2_ci_widths", output_dir)

    # Table 3: Optimal Tilting
    df3 = generate_table_3_optimal_tilting(n_samples=n_tilting)
    save_table(df3, "table_3_optimal_tilting", output_dir)

    # Table 4: Method Summary
    df4 = generate_table_4_method_summary()
    save_table(df4, "table_4_method_summary", output_dir)

    print("\n" + "="*60)
    print("DONE - All tables generated")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
