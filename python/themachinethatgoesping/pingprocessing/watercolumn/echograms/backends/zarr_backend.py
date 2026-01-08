"""Backend for reading echogram data from Zarr arrays (stub implementation)."""

from typing import Dict, Optional, Tuple

import numpy as np

from .base import EchogramDataBackend


class ZarrDataBackend(EchogramDataBackend):
    """Backend that reads data from Zarr arrays with lazy loading.
    
    This is a stub implementation for future development.
    
    Zarr arrays provide efficient chunked storage and lazy loading,
    making them suitable for large datasets that don't fit in memory.
    """

    def __init__(
        self,
        zarr_group,
        data_variable: str = "sv",
        time_variable: str = "ping_time",
        depth_variable: Optional[str] = "depth",
        range_variable: Optional[str] = "range",
    ):
        """Initialize ZarrDataBackend.
        
        Args:
            zarr_group: Zarr group containing the data arrays.
            data_variable: Name of the data variable to read (e.g., 'sv', 'av').
            time_variable: Name of the time coordinate variable.
            depth_variable: Name of the depth coordinate variable, or None.
            range_variable: Name of the range coordinate variable, or None.
        """
        raise NotImplementedError(
            "ZarrDataBackend is not yet implemented. "
            "This is a stub for future development."
        )

    @classmethod
    def from_zarr_store(
        cls,
        path: str,
        data_variable: str = "sv",
        **kwargs,
    ) -> "ZarrDataBackend":
        """Create a ZarrDataBackend from a Zarr store path.
        
        Args:
            path: Path to the Zarr store (directory or .zarr file).
            data_variable: Name of the data variable to read.
            **kwargs: Additional arguments passed to constructor.
            
        Returns:
            ZarrDataBackend instance.
        """
        raise NotImplementedError(
            "ZarrDataBackend is not yet implemented. "
            "This is a stub for future development."
        )

    # =========================================================================
    # Metadata properties (stubs)
    # =========================================================================

    @property
    def n_pings(self) -> int:
        raise NotImplementedError()

    @property
    def ping_times(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def max_sample_counts(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @property
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    @property
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    @property
    def has_navigation(self) -> bool:
        raise NotImplementedError()

    @property
    def wci_value(self) -> str:
        raise NotImplementedError()

    @property
    def linear_mean(self) -> bool:
        raise NotImplementedError()

    def get_ping_params(self) -> Dict[str, Tuple[str, np.ndarray]]:
        raise NotImplementedError()

    def get_range_stack_column(self, ping_index: int) -> np.ndarray:
        raise NotImplementedError()

    def get_depth_stack_column(self, ping_index: int, y_gridder) -> np.ndarray:
        raise NotImplementedError()

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        raise NotImplementedError()
