"""Backend for concatenating multiple echograms along the time/ping axis.

This backend provides a virtual view over multiple backends, allowing them
to be treated as a single continuous echogram. Useful for combining data
from multiple files or acquisition sessions.
"""

from typing import Dict, List, Optional, Tuple, Callable
from bisect import bisect_right
import numpy as np

from .base import EchogramDataBackend
from .storage_mode import StorageAxisMode
from ..indexers import EchogramImageRequest


class ConcatBackend(EchogramDataBackend):
    """Virtual backend that concatenates multiple backends along X (ping) axis.
    
    This backend does not store any data itself - it delegates to sub-backends
    and combines their results. It provides:
    
    - Lazy access: data is only loaded from sub-backends when requested
    - O(log n) lookup for mapping global ping index to sub-backend
    - Efficient image generation that only queries relevant sub-backends
    - Support for both "preserve" (real time gaps) and "continuous" (no gaps) modes
    
    Example usage:
        >>> backends = [backend1, backend2, backend3]  # From different files
        >>> concat = ConcatBackend(backends)
        >>> builder = EchogramBuilder(concat)
        >>> image, extent = builder.build_image()
    """

    def __init__(
        self,
        backends: List[EchogramDataBackend],
        gap_handling: str = "preserve",
    ):
        """Initialize ConcatBackend.
        
        Args:
            backends: List of backends to concatenate, in temporal order.
                      Must have compatible metadata (same wci_value, etc.).
            gap_handling: How to handle gaps between backends:
                - "preserve": Keep real time gaps (x-axis shows true times)
                - "continuous": Virtual continuous (ignore gaps between files)
        
        Raises:
            ValueError: If backends list is empty or has incompatible metadata.
        """
        if not backends:
            raise ValueError("ConcatBackend requires at least one backend")
        
        if gap_handling not in ("preserve", "continuous"):
            raise ValueError(f"gap_handling must be 'preserve' or 'continuous', got '{gap_handling}'")
        
        self._backends = backends
        self._gap_handling = gap_handling
        
        # Validate and collect metadata
        self._validate_backends()
        
        # Build index structures for fast lookup
        self._build_index()
        
        # Combine metadata from all backends
        self._compute_combined_metadata()

    def _validate_backends(self):
        """Validate that backends are compatible for concatenation."""
        first = self._backends[0]
        
        for i, backend in enumerate(self._backends[1:], 1):
            # Check wci_value compatibility
            if backend.wci_value != first.wci_value:
                import warnings
                warnings.warn(
                    f"Backend {i} has different wci_value ('{backend.wci_value}') "
                    f"than first backend ('{first.wci_value}'). Using first backend's value."
                )
            
            # Check linear_mean compatibility
            if backend.linear_mean != first.linear_mean:
                import warnings
                warnings.warn(
                    f"Backend {i} has different linear_mean ({backend.linear_mean}) "
                    f"than first backend ({first.linear_mean}). Using first backend's value."
                )

    def _build_index(self):
        """Build index structures for O(log n) ping lookup."""
        # Cumulative ping counts for global→local index conversion
        # cumulative_pings[i] = total pings in backends 0..i-1
        self._cumulative_pings = [0]
        for backend in self._backends:
            self._cumulative_pings.append(self._cumulative_pings[-1] + backend.n_pings)
        
        # Total pings
        self._n_pings = self._cumulative_pings[-1]
        
        # Time ranges for each backend (for efficient range queries)
        self._time_ranges = []
        for backend in self._backends:
            times = backend.ping_times
            if len(times) > 0:
                self._time_ranges.append((times[0], times[-1]))
            else:
                self._time_ranges.append((np.nan, np.nan))

    def _compute_combined_metadata(self):
        """Compute combined metadata from all backends."""
        # Concatenate ping times
        all_times = [b.ping_times for b in self._backends]
        self._ping_times = np.concatenate(all_times)
        
        # Concatenate sample counts
        all_counts = [b.max_sample_counts for b in self._backends]
        self._max_sample_counts = np.concatenate(all_counts)
        
        # Concatenate sample_nr extents
        all_min_s = [b.sample_nr_extents[0] for b in self._backends]
        all_max_s = [b.sample_nr_extents[1] for b in self._backends]
        self._sample_nr_min = np.concatenate(all_min_s)
        self._sample_nr_max = np.concatenate(all_max_s)
        
        # Concatenate range extents (if all backends have them)
        if all(b.range_extents is not None for b in self._backends):
            all_min_r = [b.range_extents[0] for b in self._backends]
            all_max_r = [b.range_extents[1] for b in self._backends]
            self._range_min = np.concatenate(all_min_r)
            self._range_max = np.concatenate(all_max_r)
        else:
            self._range_min = None
            self._range_max = None
        
        # Concatenate depth extents (if all backends have them)
        if all(b.depth_extents is not None for b in self._backends):
            all_min_d = [b.depth_extents[0] for b in self._backends]
            all_max_d = [b.depth_extents[1] for b in self._backends]
            self._depth_min = np.concatenate(all_min_d)
            self._depth_max = np.concatenate(all_max_d)
        else:
            self._depth_min = None
            self._depth_max = None
        
        # Concatenate lat/lon (if all backends have them)
        if all(b.has_latlon for b in self._backends):
            all_lats = [b.latitudes for b in self._backends]
            all_lons = [b.longitudes for b in self._backends]
            self._latitudes = np.concatenate(all_lats)
            self._longitudes = np.concatenate(all_lons)
        else:
            self._latitudes = None
            self._longitudes = None
        
        # Use first backend's settings for scalar metadata
        first = self._backends[0]
        self._wci_value = first.wci_value
        self._linear_mean = first.linear_mean
        self._has_navigation = all(b.has_navigation for b in self._backends)
        
        # Concatenate ping parameters
        self._ping_params = self._concat_ping_params()

    def _concat_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        """Concatenate ping parameters from all backends."""
        # Collect all param names
        all_names = set()
        for backend in self._backends:
            all_names.update(backend.get_ping_params().keys())
        
        result = {}
        for name in all_names:
            # Collect times and values from all backends that have this param
            all_times = []
            all_values = []
            y_ref = None
            
            for backend in self._backends:
                params = backend.get_ping_params()
                if name in params:
                    y_ref_b, (times, values) = params[name]
                    if y_ref is None:
                        y_ref = y_ref_b
                    all_times.append(times)
                    all_values.append(values)
            
            if all_times and y_ref is not None:
                combined_times = np.concatenate(all_times)
                combined_values = np.concatenate(all_values)
                result[name] = (y_ref, (combined_times, combined_values))
        
        return result

    def _global_to_local(self, global_ping: int) -> Tuple[int, int]:
        """Convert global ping index to (backend_index, local_ping_index).
        
        Uses binary search for O(log n) lookup.
        """
        if global_ping < 0 or global_ping >= self._n_pings:
            raise IndexError(f"Ping index {global_ping} out of range [0, {self._n_pings})")
        
        # Binary search: find which backend contains this ping
        backend_idx = bisect_right(self._cumulative_pings, global_ping) - 1
        local_ping = global_ping - self._cumulative_pings[backend_idx]
        
        return backend_idx, local_ping

    def _local_to_global(self, backend_idx: int, local_ping: int) -> int:
        """Convert (backend_index, local_ping_index) to global ping index."""
        return self._cumulative_pings[backend_idx] + local_ping

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
        return self._latitudes

    @property
    def longitudes(self) -> Optional[np.ndarray]:
        return self._longitudes

    @property
    def wci_value(self) -> str:
        return self._wci_value

    @property
    def linear_mean(self) -> bool:
        return self._linear_mean

    @property
    def storage_mode(self) -> StorageAxisMode:
        """Return storage mode from first backend."""
        return self._backends[0].storage_mode

    @property
    def gap_handling(self) -> str:
        """Gap handling mode: 'preserve' or 'continuous'."""
        return self._gap_handling

    @property
    def num_backends(self) -> int:
        """Number of sub-backends."""
        return len(self._backends)

    def get_backend(self, index: int) -> EchogramDataBackend:
        """Get a specific sub-backend by index."""
        return self._backends[index]

    # =========================================================================
    # Ping parameters
    # =========================================================================

    def get_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        return self._ping_params

    # =========================================================================
    # Data access
    # =========================================================================

    def get_column(self, ping_index: int) -> np.ndarray:
        """Get column data for a ping by delegating to the appropriate sub-backend."""
        backend_idx, local_ping = self._global_to_local(ping_index)
        return self._backends[backend_idx].get_column(local_ping)

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        """Get raw column data for a ping."""
        backend_idx, local_ping = self._global_to_local(ping_index)
        return self._backends[backend_idx].get_raw_column(local_ping)

    def get_chunk(self, start_ping: int, end_ping: int) -> np.ndarray:
        """Get a chunk of WCI data spanning potentially multiple backends."""
        # Find which backends are involved
        start_backend, start_local = self._global_to_local(start_ping)
        end_backend, end_local = self._global_to_local(end_ping - 1)
        
        n_pings = end_ping - start_ping
        max_samples = int(self._max_sample_counts[start_ping:end_ping].max()) + 1
        
        result = np.full((n_pings, max_samples), np.nan, dtype=np.float32)
        
        # Gather from each backend
        result_offset = 0
        for bi in range(start_backend, end_backend + 1):
            backend = self._backends[bi]
            
            # Determine local range for this backend
            if bi == start_backend:
                local_start = start_local
            else:
                local_start = 0
            
            if bi == end_backend:
                local_end = end_local + 1
            else:
                local_end = backend.n_pings
            
            # Get chunk from this backend
            chunk = backend.get_chunk(local_start, local_end)
            n_chunk = local_end - local_start
            n_cols = min(chunk.shape[1], max_samples)
            
            result[result_offset:result_offset + n_chunk, :n_cols] = chunk[:, :n_cols]
            result_offset += n_chunk
        
        return result

    # =========================================================================
    # Image generation
    # =========================================================================

    def _compute_backend_affines(self, backend: EchogramDataBackend, local_pings: np.ndarray):
        """Compute affine parameters for a specific backend's depth-to-sample mapping.
        
        The affine mapping is: sample_idx = round(a + b * y)
        where y is the depth/range coordinate.
        
        Args:
            backend: The backend to compute affines for.
            local_pings: Array of local ping indices within this backend.
            
        Returns:
            Tuple of (affine_a, affine_b) arrays for the local pings.
        """
        if backend.depth_extents is None:
            return None, None
        
        min_depths, max_depths = backend.depth_extents
        max_samples = backend.max_sample_counts
        
        # Get values for requested pings
        local_pings = np.asarray(local_pings)
        n = len(local_pings)
        
        affine_a = np.full(n, np.nan, dtype=np.float32)
        affine_b = np.full(n, np.nan, dtype=np.float32)
        
        valid_mask = (local_pings >= 0) & (local_pings < len(min_depths))
        valid_local = local_pings[valid_mask]
        
        if len(valid_local) == 0:
            return affine_a, affine_b
        
        # Resolution per ping: (max_depth - min_depth) / n_samples
        with np.errstate(divide='ignore', invalid='ignore'):
            resolutions = (max_depths[valid_local] - min_depths[valid_local]) / (max_samples[valid_local] + 1)
            resolutions = np.where(max_samples[valid_local] > 0, resolutions, np.nan)
            
            # Affine params: sample = a + b * depth
            b_vals = 1.0 / resolutions
            a_vals = -min_depths[valid_local] / resolutions
        
        affine_a[valid_mask] = a_vals
        affine_b[valid_mask] = b_vals
        
        return affine_a, affine_b

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        """Build image by delegating to sub-backends.
        
        Efficiently determines which backends have data in the requested range
        and only queries those backends. When in depth mode, computes proper
        affine parameters for each backend.
        """
        # Create output array
        image = np.full((request.nx, request.ny), request.fill_value, dtype=np.float32)
        
        # Find valid x indices
        valid_x_mask = request.ping_indexer >= 0
        if not np.any(valid_x_mask):
            return image
        
        valid_x_indices = np.where(valid_x_mask)[0]
        valid_pings = request.ping_indexer[valid_x_mask]
        
        # Determine if we're in depth mode
        y_coords = request.y_coordinates
        is_depth_mode = (np.nanmax(np.abs(y_coords)) > 10 or 
                         not np.allclose(y_coords, y_coords.astype(int)))
        
        # Group by backend for efficient access
        # For each backend, collect (global_ping, x_index) pairs
        backend_requests = {i: [] for i in range(len(self._backends))}
        
        for x_idx, global_ping in zip(valid_x_indices, valid_pings):
            backend_idx, local_ping = self._global_to_local(global_ping)
            backend_requests[backend_idx].append((global_ping, local_ping, x_idx))
        
        # Process each backend that has data in the request
        for backend_idx, requests in backend_requests.items():
            if not requests:
                continue
            
            backend = self._backends[backend_idx]
            
            # Build a sub-request for this backend
            global_pings = np.array([r[0] for r in requests])
            local_pings = np.array([r[1] for r in requests])
            x_indices = np.array([r[2] for r in requests])
            
            # Compute backend-specific affines if in depth mode
            if is_depth_mode:
                backend_affine_a, backend_affine_b = self._compute_backend_affines(
                    backend, local_pings
                )
            else:
                backend_affine_a = backend_affine_b = None
            
            # Get unique local pings for this backend
            unique_local, inverse = np.unique(local_pings, return_inverse=True)
            
            # Fetch data for unique pings
            for i, local_ping in enumerate(unique_local):
                column = backend.get_column(local_ping)
                global_ping = self._local_to_global(backend_idx, local_ping)
                
                # Use backend-specific affines in depth mode, or global affines otherwise
                if is_depth_mode and backend_affine_a is not None:
                    # Find the first occurrence of this local_ping in our arrays
                    first_idx = np.where(local_pings == local_ping)[0][0]
                    a = backend_affine_a[first_idx]
                    b = backend_affine_b[first_idx]
                else:
                    a = request.affine_a[global_ping]
                    b = request.affine_b[global_ping]
                
                if np.isnan(a) or np.isnan(b):
                    continue
                
                sample_indices = np.rint(a + b * request.y_coordinates).astype(np.int32)
                max_sample = len(column)
                
                valid_samples = (sample_indices >= 0) & (sample_indices < max_sample)
                
                # Fill all x positions that use this ping
                x_positions = x_indices[inverse == i]
                for x_pos in x_positions:
                    image[x_pos, valid_samples] = column[sample_indices[valid_samples]]
        
        return image

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"ConcatBackend(n_backends={len(self._backends)}, "
            f"n_pings={self._n_pings}, "
            f"gap_handling='{self._gap_handling}')"
        )
