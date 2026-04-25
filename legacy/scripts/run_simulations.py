#!/usr/bin/env python3
"""
Master Simulation Runner

Generates raw D samples and optionally precomputes processed results.

Three-layer architecture:
  - Layer 0: Raw D samples (permanent storage)
  - Layer 1: Processing functions (compute on demand)
  - Layer 1.5: Processed cache (optional, regenerable)

Usage:
    # Generate raw D samples only
    python scripts/run_simulations.py --raw

    # Generate raw + precompute processed results
    python scripts/run_simulations.py --precompute

    # Quick test mode (reduced samples)
    python scripts/run_simulations.py --fast

    # Force regeneration
    python scripts/run_simulations.py --force

    # List cached simulations
    python scripts/run_simulations.py --list

    # Clear processed cache (keeps raw data)
    python scripts/run_simulations.py --clear-cache

    # Clear everything (raw + processed)
    python scripts/run_simulations.py --clear-all
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frasian.simulations.runner import SimulationRunner, list_cache, clear_cache, format_size
from frasian.simulations.raw import RAW_DIR, raw_simulation_exists, get_raw_simulation_path
from frasian.simulations.cache import get_cache_info, clear_processed_cache
from frasian.simulations.storage import SIMULATION_DIR


def clear_all_cache(confirm: bool = True):
    """Clear both raw and processed data."""
    info = get_cache_info()
    n_raw = len(info["raw_files"])
    n_processed = len(info["cached_files"])

    if n_raw == 0 and n_processed == 0:
        print("Cache is already empty.")
        return

    if confirm:
        print(f"This will delete:")
        print(f"  - {n_raw} raw data file(s)")
        print(f"  - {n_processed} processed cache file(s)")
        response = input("\nProceed? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Clear processed
    count_processed = clear_processed_cache()

    # Clear raw
    count_raw = 0
    for name in info["raw_files"]:
        path = RAW_DIR / f"{name}.h5"
        if path.exists():
            path.unlink()
            count_raw += 1

    print(f"Deleted {count_raw} raw file(s) and {count_processed} processed file(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Generate raw D samples and precompute processed results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode flags
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Generate raw D samples only (no preprocessing)"
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Also precompute common processed results (coverage, widths)"
    )

    # Configuration
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced sample sizes for quick testing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if files exist"
    )

    # Cache management
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cached simulations and exit"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear processed cache (keeps raw data) and exit"
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear both raw and processed data and exit"
    )

    args = parser.parse_args()

    # Handle info/management commands
    if args.list:
        list_cache()
        return

    if args.clear_cache:
        clear_cache()
        return

    if args.clear_all:
        clear_all_cache()
        return

    # Determine what to run
    # Default behavior: generate raw if not present
    if not args.raw and not args.precompute:
        # Default: just generate raw
        args.raw = True

    # Print header
    print("=" * 60)
    print("FRASIAN INFERENCE - SIMULATION GENERATOR")
    print("=" * 60)
    config_str = "FAST (reduced samples)" if args.fast else "PRODUCTION"
    print(f"\nConfiguration: {config_str}")
    print(f"Output directory: {SIMULATION_DIR}")
    if args.raw:
        print(f"  Raw data: {RAW_DIR}")
    print(f"Force regeneration: {args.force}")
    print()

    # Create runner
    runner = SimulationRunner(fast=args.fast)

    try:
        if args.precompute:
            # Run all including precomputation
            results = runner.run_all(
                precompute=True,
                force=args.force,
                verbose=True
            )
        else:
            # Just generate raw data
            results = {"raw_paths": runner.run_raw(force=args.force, verbose=True)}

        # Print summary
        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)

        info = get_cache_info()
        print(f"\nRaw files: {len(info['raw_files'])}")
        for name in info["raw_files"]:
            path = RAW_DIR / f"{name}.h5"
            size = path.stat().st_size if path.exists() else 0
            print(f"  - {name}.h5 ({format_size(size)})")

        if info["cached_files"]:
            print(f"\nProcessed cache files: {len(info['cached_files'])}")
            for name in sorted(info["cached_files"])[:5]:  # Show first 5
                print(f"  - {name}.h5")
            if len(info["cached_files"]) > 5:
                print(f"  ... and {len(info['cached_files']) - 5} more")

        print(f"\nTotal raw size: {format_size(info['raw_size_bytes'])}")
        print(f"Total cache size: {format_size(info['cached_size_bytes'])}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nMake sure h5py is installed: pip install h5py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
