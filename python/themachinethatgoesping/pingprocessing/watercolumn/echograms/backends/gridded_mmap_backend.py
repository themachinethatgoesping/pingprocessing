"""Backend for gridded echogram data on a regular grid using memory-mapped files.

Gridded mmap stores data downsampled onto a regular (x, y) grid where:
- Y-axis (depth/range) has fixed step size, with 0 as a cell center
- X-axis is sparse: only occupied bins are stored (empty bins omitted)
- All rows have the same number of y cells (no padding)

Grid alignment: cell i has center at i * step.  Edges at (i - 0.5) * step
to (i + 0.5) * step.  This makes grids from different time ranges combinable.

Trade-offs vs regular mmap:
- Smaller files (downsampled)
- Faster get_image (uniform y → single index computation)
- Loss of original resolution
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import json

import numpy as np

from .base import EchogramDataBackend
from .storage_mode import StorageAxisMode
from ..indexers import EchogramImageRequest


GRIDDED_MMAP_FORMAT_VERSION = "1.0"

# Supported averaging modes
AVERAGING_MODES = {
    "db_mean": "Average dB values directly",
    "linear_mean": "Average in linear domain (10^(dB/10)), convert back to dB",
    "min": "Minimum value per cell",
    "max": "Maximum value per cell",
    "median": "Median value per cell (higher memory usage)",
}


class GriddedMmapBackend(EchogramDataBackend):
    """Backend for gridded echogram data stored as memory-mapped files.

    All rows share the same y-grid (uniform depth/range bins), enabling a
    simplified and faster ``get_image`` implementation that computes the
    y-index mapping only once.

    Store layout::

        store_dir/
            wci_data.bin            # (n_x_bins, n_y_cells) float32
            metadata.json
            bin_x_coordinates.npy   # (n_x_bins,) float64
            ping_times.npy          # alias for bin_x_coordinates when x=time
            ping_counts.npy         # (n_x_bins,) int32  source pings per bin
            [latitudes.npy]
            [longitudes.npy]
            [ping_param_{name}_times.npy]
            [ping_param_{name}_values.npy]
    """

    def __init__(
        self,
        store_path: str,
        wci_mmap: np.memmap,
        metadata: Dict,
    ):
        self._store_path = store_path
        self._wci_mmap = wci_mmap
        self._metadata = metadata
        self._n_x_bins = wci_mmap.shape[0]
        self._n_y_cells = wci_mmap.shape[1]

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    def from_path(cls, path: str) -> "GriddedMmapBackend":
        """Load a GriddedMmapBackend from a store directory."""
        path = Path(path)

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        fmt_type = metadata.get("format_type", "")
        if fmt_type != "gridded_mmap":
            raise ValueError(
                f"Not a gridded mmap store (format_type={fmt_type!r})"
            )

        version = metadata.get("format_version", "unknown")
        if not version.startswith("1."):
            raise ValueError(
                f"Unsupported gridded mmap version: {version}. "
                f"Expected {GRIDDED_MMAP_FORMAT_VERSION}"
            )

        # Storage mode
        sm_dict = metadata.get("storage_mode")
        if sm_dict is not None:
            metadata["_storage_mode"] = StorageAxisMode.from_dict(sm_dict)
        else:
            metadata["_storage_mode"] = StorageAxisMode.default()

        # Per-bin arrays
        metadata["bin_x_coordinates"] = np.load(path / "bin_x_coordinates.npy")
        metadata["ping_times_arr"] = np.load(path / "ping_times.npy")

        if (path / "ping_counts.npy").exists():
            metadata["ping_counts"] = np.load(path / "ping_counts.npy")

        # Optional lat/lon
        if (path / "latitudes.npy").exists():
            metadata["latitudes"] = np.load(path / "latitudes.npy")
            metadata["longitudes"] = np.load(path / "longitudes.npy")

        # Ping parameters
        ping_params = {}
        for name in metadata.get("ping_param_names", []):
            timestamps = np.load(path / f"ping_param_{name}_times.npy")
            values = np.load(path / f"ping_param_{name}_values.npy")
            y_ref = metadata["ping_params_meta"][name]
            ping_params[name] = {
                "y_reference": y_ref,
                "timestamps": timestamps,
                "values": values,
            }
        metadata["ping_params"] = ping_params

        # Layers
        for name in metadata.get("layer_names", []):
            min_y_file = path / f"layer_{name}_min_y.npy"
            max_y_file = path / f"layer_{name}_max_y.npy"
            if min_y_file.exists():
                metadata[f"layer_{name}_min_y"] = np.load(min_y_file)
                metadata[f"layer_{name}_max_y"] = np.load(max_y_file)
        if metadata.get("has_main_layer", False):
            min_y_file = path / "layer_main_min_y.npy"
            max_y_file = path / "layer_main_max_y.npy"
            if min_y_file.exists():
                metadata["layer_main_min_y"] = np.load(min_y_file)
                metadata["layer_main_max_y"] = np.load(max_y_file)

        # Open memory-mapped WCI data
        n_x = metadata["n_x_bins"]
        n_y = metadata["n_y_cells"]
        wci_file = path / "wci_data.bin"
        if not wci_file.exists():
            raise FileNotFoundError(f"WCI data file not found: {wci_file}")
        wci_mmap = np.memmap(
            wci_file, dtype=np.float32, mode="r", shape=(n_x, n_y)
        )

        return cls(str(path), wci_mmap, metadata)

    # =========================================================================
    # Metadata properties
    # =========================================================================

    @property
    def n_pings(self) -> int:
        return self._n_x_bins

    @property
    def ping_times(self) -> np.ndarray:
        return self._metadata["ping_times_arr"]

    @property
    def max_sample_counts(self) -> np.ndarray:
        # All rows have the same number of y cells.
        # max_sample_counts + 1 = n_y_cells  (see mmap convention).
        return np.full(self._n_x_bins, self._n_y_cells - 1, dtype=np.int32)

    @property
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros(self._n_x_bins, dtype=np.int32)
        maxes = np.full(self._n_x_bins, self._n_y_cells - 1, dtype=np.int32)
        return zeros, maxes

    @property
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        y_axis = self._metadata.get("y_axis_type", "")
        if y_axis != "range":
            return None
        y_origin = self._metadata["y_origin"]
        y_step = self._metadata["y_step"]
        y_max = y_origin + (self._n_y_cells - 1) * y_step
        mins = np.full(self._n_x_bins, y_origin, dtype=np.float32)
        maxs = np.full(self._n_x_bins, y_max, dtype=np.float32)
        return mins, maxs

    @property
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        y_axis = self._metadata.get("y_axis_type", "")
        if y_axis != "depth":
            return None
        y_origin = self._metadata["y_origin"]
        y_step = self._metadata["y_step"]
        y_max = y_origin + (self._n_y_cells - 1) * y_step
        mins = np.full(self._n_x_bins, y_origin, dtype=np.float32)
        maxs = np.full(self._n_x_bins, y_max, dtype=np.float32)
        return mins, maxs

    @property
    def has_navigation(self) -> bool:
        return self._metadata.get("has_navigation", False)

    @property
    def latitudes(self) -> Optional[np.ndarray]:
        return self._metadata.get("latitudes", None)

    @property
    def longitudes(self) -> Optional[np.ndarray]:
        return self._metadata.get("longitudes", None)

    @property
    def wci_value(self) -> str:
        return self._metadata.get("wci_value", "sv")

    @property
    def linear_mean(self) -> bool:
        return self._metadata.get("linear_mean", True)

    @property
    def storage_mode(self) -> StorageAxisMode:
        return self._metadata.get("_storage_mode", StorageAxisMode.default())

    @property
    def store_path(self) -> str:
        return self._store_path

    # =========================================================================
    # Gridded-specific properties
    # =========================================================================

    @property
    def x_step(self) -> float:
        return self._metadata["x_step"]

    @property
    def y_step(self) -> float:
        return self._metadata["y_step"]

    @property
    def y_origin(self) -> float:
        return self._metadata["y_origin"]

    @property
    def n_y_cells(self) -> int:
        return self._n_y_cells

    @property
    def bin_x_coordinates(self) -> np.ndarray:
        return self._metadata["bin_x_coordinates"]

    @property
    def ping_counts(self) -> Optional[np.ndarray]:
        return self._metadata.get("ping_counts", None)

    @property
    def averaging(self) -> str:
        return self._metadata.get("averaging", "db_mean")

    # =========================================================================
    # Ping parameters
    # =========================================================================

    def get_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        params = {}
        for name, data in self._metadata.get("ping_params", {}).items():
            params[name] = (data["y_reference"], (data["timestamps"], data["values"]))
        return params

    # =========================================================================
    # Data access
    # =========================================================================

    def get_column(self, ping_index: int) -> np.ndarray:
        return self._wci_mmap[int(ping_index), :].copy()

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        return self.get_column(ping_index)

    def get_chunk(self, start_ping: int, end_ping: int) -> np.ndarray:
        return self._wci_mmap[start_ping:end_ping, :]

    # =========================================================================
    # Image generation (optimised for uniform y-grid)
    # =========================================================================

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        """Build echogram image exploiting the uniform y-grid.

        Since all rows share the same y-extent, the sample-index mapping is
        computed once and reused for every x-column.
        """
        image = np.full(
            (request.nx, request.ny), request.fill_value, dtype=np.float32
        )

        valid_x_mask = request.ping_indexer >= 0
        if not np.any(valid_x_mask):
            return image

        valid_x_indices = np.where(valid_x_mask)[0]
        pings = request.ping_indexer[valid_x_indices]

        # Uniform affine: use first valid ping's params (all identical)
        first_ping = pings[0]
        a = request.affine_a[first_ping]
        b = request.affine_b[first_ping]
        max_s = request.max_sample_indices[first_ping]

        sample_indices = np.round(a + b * request.y_coordinates).astype(np.int64)
        valid_y = (sample_indices >= 0) & (sample_indices < max_s)
        if not np.any(valid_y):
            return image

        valid_y_indices = np.where(valid_y)[0]
        valid_sample_indices = sample_indices[valid_y]

        # De-duplicate pings for efficient mmap access
        unique_pings, inverse = np.unique(pings, return_inverse=True)

        # Single fancy-index read from the mmap
        data = self._wci_mmap[unique_pings][:, valid_sample_indices]

        # Scatter into output
        image[np.ix_(valid_x_indices, valid_y_indices)] = data[inverse]

        return image
