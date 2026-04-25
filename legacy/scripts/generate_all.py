#!/usr/bin/env python3
"""
Master script to generate all publication figures.

Usage:
    python scripts/generate_all.py [--fast] [--category CATEGORY]

Categories:
    coverage        - Figures 3.1-3.3 (Coverage analysis)
    widths          - Figures 4.1-4.3 (CI width analysis)
    tilting         - Figures 5.1-5.5 (Tilting framework)
    dynamic_tilting - Figures 5.6-5.12 (Dynamic tilting & MLP methodology)
    theory          - Figures 1.1-1.3 (Core theory)
    estimators      - Figures 2.1-2.3 (Estimator properties)
    regimes         - Figures 6.1-6.2 (Three regimes)
    summary         - Figures 7.1-7.3 (Method comparison)
    tables          - Tables 1-4 (CSV + LaTeX)
    all             - All figures and tables (default)
"""

import sys
import argparse
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

SCRIPTS = {
    'coverage': 'plot_coverage.py',
    'widths': 'plot_ci_widths.py',
    'tilting': 'plot_tilting.py',
    'dynamic_tilting': 'plot_dynamic_tilting.py',
    'theory': 'plot_theory.py',
    'estimators': 'plot_estimators.py',
    'regimes': 'plot_regimes.py',
    'summary': 'plot_summary.py',
    'tables': 'generate_tables.py',
}

# Priority order for generation (following plan)
PRIORITY_ORDER = ['coverage', 'widths', 'tilting', 'dynamic_tilting', 'theory', 'estimators', 'regimes', 'summary', 'tables']


def run_script(script_name: str, fast: bool = False):
    """Run a plotting script."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if fast:
        cmd.append('--fast')

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)

    result = subprocess.run(cmd, cwd=SCRIPTS_DIR.parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Generate all publication figures")
    parser.add_argument("--fast", action="store_true",
                        help="Use fewer MC samples for faster generation")
    parser.add_argument("--category", type=str, default="all",
                        choices=['all'] + list(SCRIPTS.keys()),
                        help="Category of figures to generate")
    args = parser.parse_args()

    print("="*60)
    print("FRASIAN INFERENCE - PUBLICATION FIGURES")
    print("="*60)

    if args.category == 'all':
        categories = PRIORITY_ORDER
    else:
        categories = [args.category]

    results = {}
    for category in categories:
        script = SCRIPTS[category]
        success = run_script(script, args.fast)
        results[category] = 'SUCCESS' if success else 'FAILED'

    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    for category, status in results.items():
        status_symbol = "[OK]" if status == 'SUCCESS' else "[FAIL]"
        print(f"  {status_symbol} {category}: {SCRIPTS[category]}")

    # List generated figures
    output_dir = SCRIPTS_DIR.parent / "output" / "figures"
    if output_dir.exists():
        print("\nGenerated figures:")
        for subdir in sorted(output_dir.iterdir()):
            if subdir.is_dir():
                pngs = list(subdir.glob("*.png"))
                print(f"  {subdir.name}/: {len(pngs)} figures")

    # List generated tables
    tables_dir = SCRIPTS_DIR.parent / "output" / "tables"
    if tables_dir.exists():
        csvs = list(tables_dir.glob("*.csv"))
        texs = list(tables_dir.glob("*.tex"))
        print(f"\nGenerated tables: {len(csvs)} CSV, {len(texs)} LaTeX")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
