"""Backend for reading echogram data from Zarr arrays with lazy loading.

Uses xarray with Dask for efficient chunked access and parallel loading.
"""

from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import json

import numpy as np

try:
    import zarr
    import xarray as xr
    import dask.array as da
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from .base import EchogramDataBackend
from ..indexers import EchogramImageRequest


# Zarr store format version
ZARR_FORMAT_VERSION = "1.0"


class ZarrDataBackend(EchogramDataBackend):
    """Backend that reads data from Zarr arrays with lazy loading.
    
    Zarr arrays provide efficient chunked storage and lazy loading via Dask,
    making them suitable for large datasets that don't fit in memory.
    
    The Zarr store contains:
    - wci_data: 2D array (n_pings, max_samples) of beam-averaged water column data
    - ping_times: 1D array of timestamps
    - max_sample_counts: 1D array of sample counts per ping
    - sample_nr_min/max: 1D arrays of sample number extents
    - range_min/max: 1D arrays of range extents (optional)
    - depth_min/max: 1D arrays of depth extents (optional)
    - ping_params/*: Ping parameter arrays (bottom, echosounder depth, etc.)
    - Attributes: wci_value, linear_mean, has_navigation, format_version
    """

    def __init__(
        self,
        ds: "xr.Dataset",
        store_path: Optional[str] = None,
        max_chunk_mb: float = 100.0,
    ):
        """Initialize ZarrDataBackend from an xarray Dataset.
        
        Prefer using the `from_zarr` factory method instead of this constructor.
        
        Args:
            ds: xarray Dataset backed by Dask arrays.
            store_path: Path to the Zarr store (for reference).
            max_chunk_mb: Maximum memory (MB) per contiguous load chunk.
                         Larger = faster I/O, smaller = less memory.
                         Default 100 MB is a good balance.
        """
        if not HAS_ZARR:
            raise ImportError(
                "zarr, xarray, and dask are required for ZarrDataBackend. "
                "Install with: pip install zarr xarray dask"
            )
        
        self._ds = ds
        self._store_path = store_path
        self._max_chunk_mb = max_chunk_mb
        
        # Direct zarr array reference for fast get_image (bypasses dask overhead)
        self._zarr_wci: Optional["zarr.Array"] = None
        
        # Chunk cache for fast consecutive get_column calls
        # Stores (chunk_start, chunk_end, chunk_data) for the last loaded chunk
        self._column_cache: Optional[Tuple[int, int, np.ndarray]] = None
        
        # Cache frequently accessed metadata (loaded eagerly to avoid Dask overhead)
        self._n_pings = ds.sizes["ping"]
        self._wci_value = ds.attrs.get("wci_value", "sv")
        self._linear_mean = ds.attrs.get("linear_mean", True)
        self._has_navigation = ds.attrs.get("has_navigation", False)
        
        # Eagerly load small 1D metadata arrays (avoid repeated Dask computation)
        self._ping_times = ds["ping_times"].values
        self._max_sample_counts = ds["max_sample_counts"].values
        self._sample_nr_min = ds["sample_nr_min"].values
        self._sample_nr_max = ds["sample_nr_max"].values
        
        # Optional extents
        self._range_min = ds["range_min"].values if "range_min" in ds else None
        self._range_max = ds["range_max"].values if "range_max" in ds else None
        self._depth_min = ds["depth_min"].values if "depth_min" in ds else None
        self._depth_max = ds["depth_max"].values if "depth_max" in ds else None
        
        # Optional lat/lon coordinates
        self._latitudes = ds["latitudes"].values if "latitudes" in ds else None
        self._longitudes = ds["longitudes"].values if "longitudes" in ds else None
        
        # Load ping params structure
        self._ping_params = self._load_ping_params()

    def _load_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        """Load ping parameters from dataset."""
        params = {}
        
        # Check for ping_params_meta attribute
        params_meta = self._ds.attrs.get("ping_params_meta", "{}")
        if isinstance(params_meta, str):
            params_meta = json.loads(params_meta)
        
        for name, y_ref in params_meta.items():
            times_var = f"ping_param_{name}_times"
            values_var = f"ping_param_{name}_values"
            if times_var in self._ds and values_var in self._ds:
                # Load eagerly - these are small 1D arrays
                times = self._ds[times_var].values
                values = self._ds[values_var].values
                params[name] = (y_ref, (times, values))
        
        return params

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def from_zarr(
        cls,
        path: str,
        chunks: Optional[Dict[str, int]] = None,
        max_chunk_mb: float = 100.0,
    ) -> "ZarrDataBackend":
        """Create a ZarrDataBackend from a Zarr store.
        
        Args:
            path: Path to the Zarr store (directory).
            chunks: Optional chunk sizes for loading. Default uses stored chunks.
                    Pass chunks={} to use stored chunking (lazy), or
                    chunks=None to load eagerly as numpy arrays.
            max_chunk_mb: Maximum memory (MB) per contiguous load chunk in get_image().
                         Larger = faster I/O, smaller = less memory.
                         Default 100 MB is a good balance for most systems.
            
        Returns:
            ZarrDataBackend instance with lazy-loaded data.
        """
        if not HAS_ZARR:
            raise ImportError(
                "zarr, xarray, and dask are required for ZarrDataBackend. "
                "Install with: pip install zarr xarray dask"
            )
        
        path = str(path)
        
        # Default to {} which uses stored chunks (lazy dask arrays)
        # None would load eagerly as numpy
        if chunks is None:
            chunks = {}
        
        # Open with xarray + dask for lazy loading
        # Try consolidated=True first (faster), fall back to False
        try:
            ds = xr.open_zarr(path, chunks=chunks, consolidated=True)
        except ValueError:
            # Consolidated metadata not found, try without
            ds = xr.open_zarr(path, chunks=chunks, consolidated=False)
        
        # Verify format version
        version = ds.attrs.get("format_version", "unknown")
        if version != ZARR_FORMAT_VERSION:
            import warnings
            warnings.warn(
                f"Zarr store format version {version} may not be compatible "
                f"with current version {ZARR_FORMAT_VERSION}"
            )
        
        backend = cls(ds, store_path=path, max_chunk_mb=max_chunk_mb)
        
        # Open zarr directly for fast get_image (bypasses dask/xarray overhead)
        z = zarr.open(path, mode='r')
        backend._zarr_wci = z['wci_data']
        
        return backend

    # =========================================================================
    # Metadata properties
    # =========================================================================

    @property
    def n_pings(self) -> int:
        return self._n_pings

    @property
    def ping_times(self) -> np.ndarray:
        return self._ping_times

    @property
    def max_sample_counts(self) -> np.ndarray:
        return self._max_sample_counts

    @property
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._sample_nr_min, self._sample_nr_max)

    @property
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._range_min is None:
            return None
        return (self._range_min, self._range_max)

    @property
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._depth_min is None:
            return None
        return (self._depth_min, self._depth_max)

    @property
    def has_navigation(self) -> bool:
        return self._has_navigation

    @property
    def latitudes(self) -> Optional[np.ndarray]:
        """Latitude for each ping in degrees, or None if not available."""
        return self._latitudes

    @property
    def longitudes(self) -> Optional[np.ndarray]:
        """Longitude for each ping in degrees, or None if not available."""
        return self._longitudes

    @property
    def wci_value(self) -> str:
        return self._wci_value

    @property
    def linear_mean(self) -> bool:
        return self._linear_mean

    def get_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        return self._ping_params

    # =========================================================================
    # Data access methods
    # =========================================================================

    def get_column(self, ping_index: int) -> np.ndarray:
        """Get column data for a ping.
        
        Uses chunk caching for fast consecutive reads. When a column is 
        requested, the entire zarr chunk containing it is loaded and cached.
        Subsequent requests for columns in the same chunk are instant.
        
        Args:
            ping_index: Index of the ping to retrieve.
            
        Returns:
            1D array of shape (n_samples,) with processed values.
        """
        # Get the number of valid samples for this ping
        n_samples = int(self.max_sample_counts[ping_index]) + 1
        
        # Check if ping is in cached chunk
        if self._column_cache is not None:
            cache_start, cache_end, cache_data = self._column_cache
            if cache_start <= ping_index < cache_end:
                # Cache hit - return from cache
                return cache_data[ping_index - cache_start, :n_samples].copy()
        
        # Cache miss - load the chunk containing this ping
        if self._zarr_wci is not None:
            # Use direct zarr access (faster)
            # Determine chunk boundaries from zarr chunks
            chunk_size = self._zarr_wci.chunks[0]
            chunk_idx = ping_index // chunk_size
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, self._n_pings)
            
            # Load entire chunk
            chunk_data = self._zarr_wci[chunk_start:chunk_end, :]
            self._column_cache = (chunk_start, chunk_end, chunk_data)
            
            return chunk_data[ping_index - chunk_start, :n_samples].copy()
        else:
            # Fallback to xarray/dask (slower, no caching)
            column = self._ds["wci_data"].isel(ping=ping_index, sample=slice(0, n_samples))
            return column.values

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        """Get full-resolution column data for a ping.
        
        For ZarrDataBackend, this is the same as get_column since
        we store beam-averaged data.
        """
        return self.get_column(ping_index)

    def get_chunk(self, start_ping: int, end_ping: int) -> np.ndarray:
        """Get a chunk of WCI data for multiple consecutive pings.
        
        Optimized for ZarrDataBackend using direct zarr array access.
        
        Args:
            start_ping: First ping index (inclusive).
            end_ping: Last ping index (exclusive).
            
        Returns:
            2D array of shape (end_ping - start_ping, n_samples).
        """
        if self._zarr_wci is not None:
            # Fast path: direct zarr array access
            return self._zarr_wci[start_ping:end_ping, :]
        else:
            # Fallback to base class implementation
            return super().get_chunk(start_ping, end_ping)

    # =========================================================================
    # Image generation (direct zarr for speed)
    # =========================================================================

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        """Build a complete echogram image from a request.
        
        Uses direct zarr access with scattered (fancy) indexing to load only
        the unique pings needed. This is memory-efficient and fast for all
        access patterns.
        
        Args:
            request: Image request with ping mapping and affine parameters.
            
        Returns:
            2D array of shape (nx, ny) with echogram data (ping, sample).
        """
        # Use direct zarr access if available (2-3x faster than dask)
        if self._zarr_wci is not None:
            wci_data = self._zarr_wci
        else:
            wci_data = self._ds["wci_data"].data
        
        # Create output array
        image = np.full((request.nx, request.ny), request.fill_value, dtype=np.float32)
        
        # Find valid x indices (where ping_indexer >= 0)
        valid_x_mask = request.ping_indexer >= 0
        valid_x_indices = np.where(valid_x_mask)[0]
        
        if len(valid_x_indices) == 0:
            return image
        
        # Get unique pings (already sorted by np.unique)
        unique_pings, inverse_indices = np.unique(request.ping_indexer[valid_x_mask], return_inverse=True)
        
        # Load only the unique pings we need (memory-efficient)
        ping_data = wci_data[unique_pings, :]
        if hasattr(ping_data, 'compute'):
            ping_data = ping_data.compute()
        else:
            ping_data = np.asarray(ping_data)
        
        # Pre-compute sample indices for all unique pings
        y_coords = request.y_coordinates  # shape (ny,)
        a_unique = request.affine_a[unique_pings]  # shape (n_unique,)
        b_unique = request.affine_b[unique_pings]  # shape (n_unique,)
        max_samples_unique = request.max_sample_indices[unique_pings]  # shape (n_unique,)
        
        # Vectorized sample index computation: (n_unique, ny)
        unique_sample_indices = np.rint(
            a_unique[:, np.newaxis] + b_unique[:, np.newaxis] * y_coords
        ).astype(np.int32)
        
        # Mark out-of-bounds as invalid
        for i in range(len(unique_pings)):
            max_s = int(max_samples_unique[i])
            invalid = (unique_sample_indices[i] < 0) | (unique_sample_indices[i] >= max_s)
            unique_sample_indices[i, invalid] = -1
        
        # Fill image (ping_data[i] corresponds to unique_pings[i])
        for i, x_idx in enumerate(valid_x_indices):
            u_idx = inverse_indices[i]
            sample_indices = unique_sample_indices[u_idx]
            valid_mask = sample_indices >= 0
            if np.any(valid_mask):
                image[x_idx, valid_mask] = ping_data[u_idx, sample_indices[valid_mask]]
        
        return image

    # =========================================================================
    # Store info
    # =========================================================================

    @property
    def store_path(self) -> Optional[str]:
        """Path to the Zarr store, if available."""
        return self._store_path

    def __repr__(self) -> str:
        return (
            f"ZarrDataBackend(n_pings={self.n_pings}, "
            f"wci_value='{self.wci_value}', "
            f"store='{self._store_path}')"
        )
