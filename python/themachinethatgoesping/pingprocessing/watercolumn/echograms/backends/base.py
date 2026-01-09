"""Abstract base class for echogram data backends."""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional, Tuple, Iterable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..indexers import EchogramImageRequest


class EchogramDataBackend(ABC):
    """Abstract base class for echogram data sources.
    
    Backends are responsible for:
    - Providing metadata about the data (ping times, extents, etc.)
    - Reading water column data for individual pings
    - Managing data caching if needed
    
    Backends receive sample indices from the EchogramBuilder and return
    the corresponding data. Coordinate conversion is handled by EchogramBuilder.
    """

    # =========================================================================
    # Metadata properties (read-only)
    # =========================================================================

    @property
    @abstractmethod
    def n_pings(self) -> int:
        """Number of pings in the dataset."""
        ...

    @property
    @abstractmethod
    def ping_times(self) -> np.ndarray:
        """Timestamps for each ping (Unix timestamps as float64)."""
        ...

    @property
    @abstractmethod
    def max_sample_counts(self) -> np.ndarray:
        """Maximum number of samples for each ping (int array)."""
        ...

    @property
    @abstractmethod
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample number extents as (min_sample_nrs, max_sample_nrs) arrays."""
        ...

    @property
    @abstractmethod
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Range extents as (min_ranges, max_ranges) arrays, or None if not available."""
        ...

    @property
    @abstractmethod
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Depth extents as (min_depths, max_depths) arrays, or None if no navigation."""
        ...

    @property
    @abstractmethod
    def has_navigation(self) -> bool:
        """Whether depth information is available (requires navigation data)."""
        ...

    @property
    @abstractmethod
    def wci_value(self) -> str:
        """The water column image value type (e.g., 'sv', 'av', 'pv')."""
        ...

    @property
    @abstractmethod
    def linear_mean(self) -> bool:
        """Whether to use linear mean for beam averaging."""
        ...

    # =========================================================================
    # Ping parameters (bottom, echosounder depth, etc.)
    # =========================================================================

    @abstractmethod
    def get_ping_params(self) -> Dict[str, Tuple[str, np.ndarray]]:
        """Return pre-computed ping parameters.
        
        Returns:
            Dictionary mapping parameter names (e.g., 'bottom', 'minslant', 'echosounder')
            to tuples of (y_reference, values) where y_reference is one of
            'Depth (m)', 'Range (m)', 'Sample number', 'Y indice'.
        """
        ...

    # =========================================================================
    # Data access methods (sample-indexed)
    # =========================================================================

    @abstractmethod
    def get_column(self, ping_index: int) -> np.ndarray:
        """Get column data for a ping.
        
        Returns beam-averaged water column data. The processing method
        (range stack vs depth stack) is determined by the backend's configuration.
        Both modes return data of the same shape (n_samples for the ping).
        
        Args:
            ping_index: Index of the ping to retrieve.
            
        Returns:
            1D array of shape (n_samples,) with processed values.
        """
        ...

    # =========================================================================
    # Raw data access (for layers, non-downsampled)
    # =========================================================================

    @abstractmethod
    def get_raw_column(self, ping_index: int) -> np.ndarray:
        """Get full-resolution beam-averaged column data for a ping.
        
        Unlike get_column, this always returns range-stacked data regardless
        of the backend's stacking mode. Used for layer extraction where
        sample indices need to be preserved.
        
        Args:
            ping_index: Index of the ping to retrieve.
            
        Returns:
            1D array with all samples for the ping.
        """
        ...

    def iterate_raw_columns(
        self, ping_indices: Iterable[int]
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over raw columns for given ping indices.
        
        Args:
            ping_indices: Indices of pings to iterate over.
            
        Yields:
            Tuples of (ping_index, raw_column_data).
        """
        for idx in ping_indices:
            yield idx, self.get_raw_column(idx)

    # =========================================================================
    # Optional: beam sample selection access (for advanced use cases)
    # =========================================================================

    def get_beam_sample_selection(self, ping_index: int):
        """Get the beam sample selection for a ping, if available.
        
        Not all backends support this. Override in subclasses that do.
        
        Args:
            ping_index: Index of the ping.
            
        Returns:
            BeamSampleSelection object or None if not supported.
        """
        return None

    # =========================================================================
    # Optional: cache management
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear any cached data. Override in subclasses that implement caching."""
        pass

    # =========================================================================
    # Image generation (vectorized)
    # =========================================================================

    @abstractmethod
    def get_image(self, request: "EchogramImageRequest") -> np.ndarray:
        """Build a complete echogram image from a request.
        
        Each backend implements this differently based on its data storage format.
        
        Args:
            request: Image request with ping mapping and affine parameters.
            
        Returns:
            2D array of shape (nx, ny) with echogram data (ping, sample).
        """
        ...
