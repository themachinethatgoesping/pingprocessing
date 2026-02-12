"""Backend for ultra-fast echogram data access using memory-mapped files.

Memory-mapped files provide near-instantaneous random access by letting the
OS handle paging. This is ideal for interactive visualization where scattered
ping access patterns are common.

Trade-off vs Zarr:
- Mmap: Uncompressed (larger files), but 10-100x faster random access
- Zarr: Compressed (smaller files), but slower due to decompression

Memory efficiency:
- WCI data is truly lazy-loaded via OS page cache
- get_image processes in chunks to avoid loading entire dataset
- Only the requested pages are loaded from disk
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import json

import numpy as np

from .base import EchogramDataBackend
from .storage_mode import StorageAxisMode
from ..indexers import EchogramImageRequest


# Mmap store format version (3.0 = adds storage_mode support)
MMAP_FORMAT_VERSION = "3.0"


class MmapDataBackend(EchogramDataBackend):
    """Backend using memory-mapped files for ultra-fast random access.
    
    This backend stores data in a simple directory structure:
    - wci_data.bin: Raw float32 array (n_pings × max_samples)
    - metadata.json: All metadata and ping parameters
    
    Memory-mapped access means:
    - Zero decompression overhead
    - OS handles caching via page cache (truly lazy loading)
    - Instant random access (10-100x faster than Zarr for scattered reads)
    - File size = raw data size (no compression)
    - Supports larger-than-memory files (only accessed pages are loaded)
    """

    def __init__(
        self,
        store_path: str,
        wci_mmap: np.memmap,
        metadata: Dict,
    ):
        """Initialize MmapDataBackend.
        
        Prefer using the `from_path` factory method or `EchogramBuilder.from_mmap()`.
        
        Args:
            store_path: Path to the mmap store directory.
            wci_mmap: Memory-mapped WCI data array (lazy-loaded by OS).
            metadata: Dictionary containing all metadata.
        """
        self._store_path = store_path
        self._wci_mmap = wci_mmap  # np.memmap is lazy - OS loads pages on demand
        self._metadata = metadata
        
        # Cache frequently accessed values (small scalars, not the data)
        self._n_pings = wci_mmap.shape[0]
        self._n_samples = wci_mmap.shape[1]

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def from_path(cls, path: str) -> "MmapDataBackend":
        """Load a MmapDataBackend from a store directory.
        
        The WCI data is memory-mapped with mode='r' (read-only), meaning:
        - No data is loaded into memory until accessed
        - OS page cache handles all caching efficiently
        - Supports files larger than available RAM
        
        Args:
            path: Path to the mmap store directory.
            
        Returns:
            MmapDataBackend instance with lazy-loaded data.
        """
        path = Path(path)
        
        # Load scalar metadata (small JSON)
        metadata_file = path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Verify format version (support 2.x and 3.x)
        version = metadata.get("format_version", "unknown")
        if not (version.startswith("2.") or version.startswith("3.")):
            raise ValueError(
                f"Unsupported mmap store format version: {version}. "
                f"Expected version 2.x or {MMAP_FORMAT_VERSION}"
            )
        
        # Load storage mode (new in 3.0, default for older versions)
        storage_mode_dict = metadata.get("storage_mode", None)
        if storage_mode_dict is not None:
            metadata["_storage_mode"] = StorageAxisMode.from_dict(storage_mode_dict)
        else:
            metadata["_storage_mode"] = StorageAxisMode.default()
        
        # Load array metadata from binary .npy files
        metadata["ping_times"] = np.load(path / "ping_times.npy")
        metadata["max_sample_counts"] = np.load(path / "max_sample_counts.npy")
        metadata["sample_nr_min"] = np.load(path / "sample_nr_min.npy")
        metadata["sample_nr_max"] = np.load(path / "sample_nr_max.npy")
        
        if (path / "range_min.npy").exists():
            metadata["range_min"] = np.load(path / "range_min.npy")
            metadata["range_max"] = np.load(path / "range_max.npy")
        
        if (path / "depth_min.npy").exists():
            metadata["depth_min"] = np.load(path / "depth_min.npy")
            metadata["depth_max"] = np.load(path / "depth_max.npy")
        
        # Lat/lon coordinates (optional)
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
        
        # Open memory-mapped data (lazy - no data loaded yet)
        wci_file = path / "wci_data.bin"
        if not wci_file.exists():
            raise FileNotFoundError(f"WCI data file not found: {wci_file}")
        
        shape = (metadata["n_pings"], metadata["n_samples"])
        # mode='r' = read-only, lazy-loaded via OS page cache
        wci_mmap = np.memmap(wci_file, dtype=np.float32, mode="r", shape=shape)
        
        return cls(str(path), wci_mmap, metadata)



    # =========================================================================
    # Metadata properties
    # =========================================================================

    @property
    def n_pings(self) -> int:
        return self._n_pings

    @property
    def ping_times(self) -> np.ndarray:
        return self._metadata["ping_times"]

    @property
    def max_sample_counts(self) -> np.ndarray:
        return self._metadata["max_sample_counts"]

    @property
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._metadata["sample_nr_min"], self._metadata["sample_nr_max"]

    @property
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if "range_min" not in self._metadata:
            return None
        return self._metadata["range_min"], self._metadata["range_max"]

    @property
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if "depth_min" not in self._metadata:
            return None
        return self._metadata["depth_min"], self._metadata["depth_max"]

    @property
    def has_navigation(self) -> bool:
        return self._metadata.get("has_navigation", False)

    @property
    def latitudes(self) -> Optional[np.ndarray]:
        """Latitude for each ping in degrees, or None if not available."""
        return self._metadata.get("latitudes", None)

    @property
    def longitudes(self) -> Optional[np.ndarray]:
        """Longitude for each ping in degrees, or None if not available."""
        return self._metadata.get("longitudes", None)

    @property
    def wci_value(self) -> str:
        return self._metadata.get("wci_value", "sv")

    @property
    def linear_mean(self) -> bool:
        return self._metadata.get("linear_mean", True)

    @property
    def storage_mode(self) -> StorageAxisMode:
        """Storage coordinate system for this backend."""
        return self._metadata.get("_storage_mode", StorageAxisMode.default())

    # =========================================================================
    # Ping parameters
    # =========================================================================

    def get_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        """Return pre-computed ping parameters.
        
        Returns:
            Dict mapping param name to (y_reference, (timestamps, values)).
        """
        params = {}
        for name, data in self._metadata.get("ping_params", {}).items():
            params[name] = (data["y_reference"], (data["timestamps"], data["values"]))
        return params

    # =========================================================================
    # Data access
    # =========================================================================

    def get_column(self, ping_index: int) -> np.ndarray:
        """Get beam-averaged water column data for a single ping."""
        return self._wci_mmap[int(ping_index), :].copy()

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        """Get raw water column data (same as get_column for mmap)."""
        return self.get_column(ping_index)

    def get_chunk(self, start_ping: int, end_ping: int) -> np.ndarray:
        """Get a chunk of WCI data for multiple consecutive pings.
        
        Optimized for MmapDataBackend - direct slice from memory-mapped file.
        
        Args:
            start_ping: First ping index (inclusive).
            end_ping: Last ping index (exclusive).
            
        Returns:
            2D array of shape (end_ping - start_ping, n_samples).
        """
        # Direct slice from mmap - OS handles page loading efficiently
        return self._wci_mmap[start_ping:end_ping, :]

    # =========================================================================
    # Image generation (memory-efficient with chunked mmap access)
    # =========================================================================

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        """Build a complete echogram image from a request.
        
        Uses memory-mapped scattered access for ultra-fast loading.
        Processes in chunks to limit memory usage for large datasets.
        
        Memory usage: O(chunk_size * ny) + O(nx * ny) for output
        
        Args:
            request: Image request with ping mapping and affine parameters.
            
        Returns:
            2D array of shape (nx, ny) with echogram data (ping, sample).
        """
        # Create output array
        image = np.full((request.nx, request.ny), request.fill_value, dtype=np.float32)
        
        # Find valid x indices (where ping_indexer >= 0)
        valid_x_mask = request.ping_indexer >= 0
        valid_x_indices = np.where(valid_x_mask)[0]
        
        if len(valid_x_indices) == 0:
            return image
        
        # Get unique pings and mapping
        unique_pings, inverse_indices = np.unique(
            request.ping_indexer[valid_x_mask], return_inverse=True
        )
        
        n_unique = len(unique_pings)
        ny = request.ny
        y_coords = request.y_coordinates
        
        # Process in chunks to limit memory usage
        # Each chunk loads at most chunk_size pings worth of data
        chunk_size = 1000  # ~4MB per chunk at 1000 samples float32
        
        # Pre-allocate result array for unique pings
        unique_results = np.empty((n_unique, ny), dtype=np.float32)
        
        for chunk_start in range(0, n_unique, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_unique)
            chunk_indices = slice(chunk_start, chunk_end)
            chunk_pings = unique_pings[chunk_indices]
            n_chunk = chunk_end - chunk_start
            
            # Load chunk of ping data from mmap (only these pages loaded)
            ping_data = self._wci_mmap[chunk_pings, :]
            
            # Compute sample indices for this chunk
            a_chunk = request.affine_a[chunk_pings]
            b_chunk = request.affine_b[chunk_pings]
            max_samples_chunk = request.max_sample_indices[chunk_pings]
            
            # Sample indices: shape (n_chunk, ny)
            sample_indices = np.rint(
                a_chunk[:, np.newaxis] + b_chunk[:, np.newaxis] * y_coords
            )
            nan_mask = np.isnan(sample_indices)
            sample_indices = np.where(nan_mask, -1, sample_indices).astype(np.int32)
            
            # Bounds checking
            valid_samples = (
                (sample_indices >= 0) &
                (sample_indices < max_samples_chunk[:, np.newaxis])
            )
            
            # Clip for safe indexing
            sample_indices_clipped = np.clip(sample_indices, 0, ping_data.shape[1] - 1)
            
            # Gather values using advanced indexing
            ping_idx_flat = np.repeat(np.arange(n_chunk), ny)
            sample_idx_flat = sample_indices_clipped.ravel()
            chunk_values = ping_data[ping_idx_flat, sample_idx_flat].reshape(n_chunk, ny)
            
            # Apply mask
            chunk_values = np.where(valid_samples, chunk_values, request.fill_value)
            
            # Store in result array
            unique_results[chunk_indices] = chunk_values
            
            # Explicitly release chunk data to help garbage collector
            del ping_data, chunk_values
        
        # Map unique results back to output image
        image[valid_x_indices] = unique_results[inverse_indices]
        
        return image

    # =========================================================================
    # Store info
    # =========================================================================

    @property
    def store_path(self) -> str:
        """Path to the mmap store directory."""
        return self._store_path

    def __repr__(self) -> str:
        return (
            f"MmapDataBackend(n_pings={self.n_pings}, "
            f"wci_value='{self.wci_value}', "
            f"store='{self._store_path}')"
        )
