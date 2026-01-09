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
    ):
        """Initialize ZarrDataBackend from an xarray Dataset.
        
        Prefer using the `from_zarr` factory method instead of this constructor.
        
        Args:
            ds: xarray Dataset backed by Dask arrays.
            store_path: Path to the Zarr store (for reference).
        """
        if not HAS_ZARR:
            raise ImportError(
                "zarr, xarray, and dask are required for ZarrDataBackend. "
                "Install with: pip install zarr xarray dask"
            )
        
        self._ds = ds
        self._store_path = store_path
        
        # Cache frequently accessed metadata
        self._n_pings = ds.sizes["ping"]
        self._wci_value = ds.attrs.get("wci_value", "sv")
        self._linear_mean = ds.attrs.get("linear_mean", True)
        self._has_navigation = ds.attrs.get("has_navigation", False)
        
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
    ) -> "ZarrDataBackend":
        """Create a ZarrDataBackend from a Zarr store.
        
        Args:
            path: Path to the Zarr store (directory).
            chunks: Optional chunk sizes for loading. Default uses stored chunks.
                    Pass chunks={} to use stored chunking (lazy), or
                    chunks=None to load eagerly as numpy arrays.
            
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
        # consolidated=True reads all metadata from single .zmetadata file (fast)
        # Falls back gracefully if not consolidated
        ds = xr.open_zarr(path, chunks=chunks, consolidated=True)
        
        # Verify format version
        version = ds.attrs.get("format_version", "unknown")
        if version != ZARR_FORMAT_VERSION:
            import warnings
            warnings.warn(
                f"Zarr store format version {version} may not be compatible "
                f"with current version {ZARR_FORMAT_VERSION}"
            )
        
        return cls(ds, store_path=path)

    # =========================================================================
    # Metadata properties
    # =========================================================================

    @property
    def n_pings(self) -> int:
        return self._n_pings

    @property
    def ping_times(self) -> np.ndarray:
        return self._ds["ping_times"].values

    @property
    def max_sample_counts(self) -> np.ndarray:
        return self._ds["max_sample_counts"].values

    @property
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self._ds["sample_nr_min"].values,
            self._ds["sample_nr_max"].values,
        )

    @property
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if "range_min" not in self._ds:
            return None
        return (
            self._ds["range_min"].values,
            self._ds["range_max"].values,
        )

    @property
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if "depth_min" not in self._ds or not self._has_navigation:
            return None
        return (
            self._ds["depth_min"].values,
            self._ds["depth_max"].values,
        )

    @property
    def has_navigation(self) -> bool:
        return self._has_navigation

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
        
        Loads the data from Zarr lazily and computes only what's needed.
        
        Args:
            ping_index: Index of the ping to retrieve.
            
        Returns:
            1D array of shape (n_samples,) with processed values.
        """
        # Get the number of valid samples for this ping
        n_samples = int(self.max_sample_counts[ping_index]) + 1
        
        # Slice and compute - Dask will only load the needed chunk
        column = self._ds["wci_data"].isel(ping=ping_index, sample=slice(0, n_samples))
        return column.values

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        """Get full-resolution column data for a ping.
        
        For ZarrDataBackend, this is the same as get_column since
        we store beam-averaged data.
        """
        return self.get_column(ping_index)

    # =========================================================================
    # Image generation (vectorized with Dask)
    # =========================================================================

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        """Build a complete echogram image from a request.
        
        Uses vectorized operations with Dask for efficient parallel loading.
        
        Args:
            request: Image request with ping mapping and affine parameters.
            
        Returns:
            2D array of shape (nx, ny) with echogram data (ping, sample).
        """
        # Get the underlying array (may be dask or numpy)
        wci_data = self._ds["wci_data"].data
        
        # Pre-compute all sample indices (vectorized)
        all_sample_indices = request.compute_all_sample_indices()  # (n_pings, ny)
        
        # Create output array
        image = np.full((request.nx, request.ny), request.fill_value, dtype=np.float32)
        
        # Find unique pings we need to load
        valid_x_mask = request.ping_indexer >= 0
        valid_x_indices = np.where(valid_x_mask)[0]
        
        if len(valid_x_indices) == 0:
            return image
        
        # Get unique pings and their x positions
        unique_pings = np.unique(request.ping_indexer[valid_x_mask])
        
        # Load all needed ping data at once
        # Handle both dask arrays (have .compute()) and numpy arrays
        ping_data_slice = wci_data[unique_pings, :]
        if hasattr(ping_data_slice, 'compute'):
            ping_data = ping_data_slice.compute()  # Dask array
        else:
            ping_data = np.asarray(ping_data_slice)  # Already numpy or similar
        
        # Create mapping from ping index to loaded data index
        ping_to_loaded = {p: i for i, p in enumerate(unique_pings)}
        
        # Fill image
        for x_idx in valid_x_indices:
            ping_idx = request.ping_indexer[x_idx]
            loaded_idx = ping_to_loaded[ping_idx]
            
            sample_indices = all_sample_indices[ping_idx]  # (ny,)
            valid_mask = sample_indices >= 0
            
            if np.any(valid_mask):
                image[x_idx, valid_mask] = ping_data[loaded_idx, sample_indices[valid_mask]]
        
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
