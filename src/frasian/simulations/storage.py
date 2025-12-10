"""
HDF5 Storage for Simulation Results

Provides save/load utilities for caching Monte Carlo simulation results
in HDF5 format for efficient storage and retrieval.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# Default output directory for simulation results
# Path: storage.py -> simulations/ -> frasian/ -> src/ -> frasian_tilting/ -> output/simulations/
SIMULATION_DIR = Path(__file__).parent.parent.parent.parent / "output" / "simulations"


def _ensure_h5py():
    """Raise helpful error if h5py not installed."""
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for simulation storage. "
            "Install with: pip install h5py"
        )


def _serialize_metadata(metadata: dict) -> str:
    """Serialize metadata dict to JSON string for HDF5 attribute storage."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    converted = {k: convert(v) for k, v in metadata.items()}
    return json.dumps(converted)


def _deserialize_metadata(json_str: str) -> dict:
    """Deserialize JSON string back to metadata dict."""
    return json.loads(json_str)


def save_simulation(
    name: str,
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_dir: Optional[Path] = None
) -> Path:
    """Save simulation results to HDF5 file.

    Args:
        name: Simulation name (used as filename without extension)
        data: Dictionary of numpy arrays to store
        metadata: Dictionary of metadata (will be stored as JSON attribute)
        output_dir: Output directory (default: output/simulations/)

    Returns:
        Path to saved file

    Example:
        >>> data = {"coverage": np.array([0.95, 0.94]), "se": np.array([0.01, 0.01])}
        >>> metadata = {"seed": 42, "n_reps": 10000, "timestamp": datetime.now()}
        >>> save_simulation("coverage_grid", data, metadata)
    """
    _ensure_h5py()

    if output_dir is None:
        output_dir = SIMULATION_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{name}.h5"

    # Add timestamp to metadata
    metadata = metadata.copy()
    metadata["_saved_at"] = datetime.now().isoformat()
    metadata["_name"] = name

    with h5py.File(filepath, "w") as f:
        # Store arrays as datasets
        for key, array in data.items():
            f.create_dataset(key, data=array, compression="gzip")

        # Store metadata as JSON attribute on root group
        f.attrs["metadata"] = _serialize_metadata(metadata)

    return filepath


def load_simulation(
    name: str,
    output_dir: Optional[Path] = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load simulation results from HDF5 file.

    Args:
        name: Simulation name (filename without extension)
        output_dir: Directory to look in (default: output/simulations/)

    Returns:
        Tuple of (data dict, metadata dict)

    Raises:
        FileNotFoundError: If simulation file doesn't exist
    """
    _ensure_h5py()

    if output_dir is None:
        output_dir = SIMULATION_DIR

    filepath = Path(output_dir) / f"{name}.h5"

    if not filepath.exists():
        raise FileNotFoundError(f"Simulation not found: {filepath}")

    data = {}
    with h5py.File(filepath, "r") as f:
        # Load all datasets
        for key in f.keys():
            data[key] = f[key][:]

        # Load metadata
        metadata = _deserialize_metadata(f.attrs["metadata"])

    return data, metadata


def simulation_exists(
    name: str,
    output_dir: Optional[Path] = None
) -> bool:
    """Check if a simulation cache file exists.

    Args:
        name: Simulation name
        output_dir: Directory to check (default: output/simulations/)

    Returns:
        True if cache file exists
    """
    if output_dir is None:
        output_dir = SIMULATION_DIR

    filepath = Path(output_dir) / f"{name}.h5"
    return filepath.exists()


def get_simulation_metadata(
    name: str,
    output_dir: Optional[Path] = None
) -> dict[str, Any]:
    """Get metadata from a simulation file without loading all data.

    Args:
        name: Simulation name
        output_dir: Directory to look in (default: output/simulations/)

    Returns:
        Metadata dictionary

    Raises:
        FileNotFoundError: If simulation file doesn't exist
    """
    _ensure_h5py()

    if output_dir is None:
        output_dir = SIMULATION_DIR

    filepath = Path(output_dir) / f"{name}.h5"

    if not filepath.exists():
        raise FileNotFoundError(f"Simulation not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        metadata = _deserialize_metadata(f.attrs["metadata"])

    return metadata


def list_simulations(
    output_dir: Optional[Path] = None
) -> list[str]:
    """List all available simulation cache files.

    Args:
        output_dir: Directory to scan (default: output/simulations/)

    Returns:
        List of simulation names (without .h5 extension)
    """
    if output_dir is None:
        output_dir = SIMULATION_DIR

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    return [f.stem for f in output_dir.glob("*.h5")]


def delete_simulation(
    name: str,
    output_dir: Optional[Path] = None
) -> bool:
    """Delete a simulation cache file.

    Args:
        name: Simulation name
        output_dir: Directory (default: output/simulations/)

    Returns:
        True if file was deleted, False if it didn't exist
    """
    if output_dir is None:
        output_dir = SIMULATION_DIR

    filepath = Path(output_dir) / f"{name}.h5"

    if filepath.exists():
        filepath.unlink()
        return True
    return False


def get_cache_size(
    output_dir: Optional[Path] = None
) -> int:
    """Get total size of simulation cache in bytes.

    Args:
        output_dir: Directory to scan (default: output/simulations/)

    Returns:
        Total size in bytes
    """
    if output_dir is None:
        output_dir = SIMULATION_DIR

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0

    return sum(f.stat().st_size for f in output_dir.glob("*.h5"))
