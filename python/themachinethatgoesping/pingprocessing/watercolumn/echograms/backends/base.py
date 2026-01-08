"""Abstract base class for echogram data backends."""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional, Tuple, Iterable

import numpy as np


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
    def get_range_stack_column(self, ping_index: int) -> np.ndarray:
        """Get beam-averaged column data for a ping (range-stacked).
        
        Returns the water column data averaged across beams, indexed by sample number.
        
        Args:
            ping_index: Index of the ping to retrieve.
            
        Returns:
            1D array of shape (n_samples,) with beam-averaged values.
        """
        ...

    @abstractmethod
    def get_depth_stack_column(self, ping_index: int, y_gridder) -> np.ndarray:
        """Get depth-gridded column data for a ping.
        
        Transforms water column data from sample-indexed to depth-indexed coordinates
        using beam geometry.
        
        Args:
            ping_index: Index of the ping to retrieve.
            y_gridder: ForwardGridder1D instance defining the depth grid.
            
        Returns:
            1D array of shape (n_depth_bins,) with depth-gridded values.
        """
        ...

    # =========================================================================
    # Raw data access (for layers, non-downsampled)
    # =========================================================================

    @abstractmethod
    def get_raw_column(self, ping_index: int) -> np.ndarray:
        """Get full-resolution beam-averaged column data for a ping.
        
        Unlike get_range_stack_column, this always returns the complete data
        without any sample selection applied by viewing parameters.
        
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
