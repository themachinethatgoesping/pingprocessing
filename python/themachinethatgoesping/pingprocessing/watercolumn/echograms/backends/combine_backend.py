"""Backend for combining multiple echograms with mathematical operations.

This backend provides a virtual view that combines multiple backends using
a user-specified function (mean, median, sum, etc.). Useful for combining
data from different frequencies or different acquisition systems.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np

from .base import EchogramDataBackend
from .storage_mode import StorageAxisMode
from ..indexers import EchogramImageRequest


# Built-in combine functions - suppress "Mean of empty slice" warnings
# which occur frequently when combining sparse echograms
def nanmean(stack: np.ndarray, axis: int) -> np.ndarray:
    """Mean ignoring NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stack, axis=axis)


def nanmedian(stack: np.ndarray, axis: int) -> np.ndarray:
    """Median ignoring NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(stack, axis=axis)


def nansum(stack: np.ndarray, axis: int) -> np.ndarray:
    """Sum ignoring NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nansum(stack, axis=axis)


def nanmax(stack: np.ndarray, axis: int) -> np.ndarray:
    """Maximum ignoring NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(stack, axis=axis)


def nanmin(stack: np.ndarray, axis: int) -> np.ndarray:
    """Minimum ignoring NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(stack, axis=axis)


def nanstd(stack: np.ndarray, axis: int) -> np.ndarray:
    """Standard deviation ignoring NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(stack, axis=axis)


def mean(stack: np.ndarray, axis: int) -> np.ndarray:
    """Mean (NaN propagates)."""
    return np.mean(stack, axis=axis)


def first_valid(stack: np.ndarray, axis: int) -> np.ndarray:
    """Return first non-NaN value along axis (priority order)."""
    result = stack[0].copy()
    for i in range(1, stack.shape[0]):
        mask = np.isnan(result)
        result[mask] = stack[i][mask]
    return result


# Registry of built-in functions
COMBINE_FUNCTIONS = {
    "nanmean": nanmean,
    "nanmedian": nanmedian,
    "nansum": nansum,
    "nanmax": nanmax,
    "nanmin": nanmin,
    "nanstd": nanstd,
    "mean": mean,
    "first_valid": first_valid,
}


class CombineBackend(EchogramDataBackend):
    """Virtual backend that combines multiple backends with a reduce function.
    
    Combination happens AFTER downsampling for efficiency:
    1. Each sub-backend produces its downsampled image via get_image()
    2. Images are stacked (handling different extents)
    3. combine_func reduces the stack to a single image
    
    Alignment modes:
    - x_align="ping_index": Backends are aligned by ping index (ping N with ping N)
    - x_align="time": Backends are aligned by timestamp (find closest ping by time)
    - y_align="sample_index": Backends are aligned by sample index
    - y_align="depth": Backends are aligned by depth coordinates
    
    Linear mode (linear=True):
    - Data is assumed to be in dB (e.g., Sv values)
    - Before combining: converts from dB to linear using 10^(0.1 * dB)
    - Applies combine function (e.g., nanmean) in linear domain
    - After combining: converts back to dB using 10 * log10(linear)
    - This gives physically correct averaging of acoustic intensities
    
    Memory usage: O(n_backends * nx * ny) - all downsampled images must fit in RAM.
    This is typically fine since downsampled images are small (e.g., 4096 x 1024).
    
    Example usage:
        >>> backends = [backend_18khz, backend_38khz, backend_120khz]
        >>> combine = CombineBackend(backends, combine_func=np.nanmean)
        >>> builder = EchogramBuilder(combine)
        >>> image, extent = builder.build_image()
        >>>
        >>> # For acoustically correct averaging in linear domain:
        >>> combine = CombineBackend(backends, combine_func="nanmean", linear=True)
    
    Custom combine functions must have signature:
        func(stack: np.ndarray, axis: int) -> np.ndarray
    where stack has shape (n_backends, nx, ny) and the function reduces along axis=0.
    """

    def __init__(
        self,
        backends: List[EchogramDataBackend],
        combine_func: Union[str, Callable[[np.ndarray, int], np.ndarray]] = "nanmean",
        name: str = "combined",
        x_align: str = "ping_index",
        y_align: str = "sample_index",
        linear: bool = True,
    ):
        """Initialize CombineBackend.
        
        Args:
            backends: List of backends to combine. Should have overlapping time ranges.
            combine_func: Function to combine stacked images, either:
                - String name of built-in: "nanmean", "nanmedian", "nansum", 
                  "nanmax", "nanmin", "nanstd", "mean", "first_valid"
                - Callable with signature (stack, axis) -> result
                  Stack has shape (n_backends, nx, ny), reduce along axis=0.
            name: Name for the combined echogram (used in repr).
            x_align: How to align backends on x-axis:
                - "ping_index": Align by ping index (ping N with ping N)
                - "time": Align by timestamp (find closest ping by time)
            y_align: How to align backends on y-axis:
                - "sample_index": Align by sample index
                - "depth": Align by depth coordinates (requires depth extents)
                - "range": Align by range coordinates (requires range extents)
            linear: If True (default), convert dB data to linear domain before
                combining, then convert back to dB. This gives acoustically
                correct averaging of intensities. Set to False to combine
                directly in dB domain.
        
        Raises:
            ValueError: If backends list is empty or invalid align mode.
        """
        if not backends:
            raise ValueError("CombineBackend requires at least one backend")
        
        if x_align not in ("ping_index", "time"):
            raise ValueError(f"x_align must be 'ping_index' or 'time', got '{x_align}'")
        if y_align not in ("sample_index", "depth", "range"):
            raise ValueError(f"y_align must be 'sample_index', 'depth', or 'range', got '{y_align}'")
        
        self._backends = backends
        self._name = name
        self._x_align = x_align
        self._y_align = y_align
        self._linear = linear
        
        # Resolve combine function
        if isinstance(combine_func, str):
            if combine_func not in COMBINE_FUNCTIONS:
                raise ValueError(
                    f"Unknown combine function '{combine_func}'. "
                    f"Available: {list(COMBINE_FUNCTIONS.keys())}"
                )
            self._combine_func = COMBINE_FUNCTIONS[combine_func]
            self._combine_func_name = combine_func
        else:
            self._combine_func = combine_func
            self._combine_func_name = getattr(combine_func, "__name__", "custom")
        
        # Compute combined metadata
        self._compute_combined_metadata()

    @property
    def x_align(self) -> str:
        """X-axis alignment mode."""
        return self._x_align
    
    @x_align.setter
    def x_align(self, value: str):
        """Set X-axis alignment mode."""
        if value not in ("ping_index", "time"):
            raise ValueError(f"x_align must be 'ping_index' or 'time', got '{value}'")
        self._x_align = value
    
    @property
    def y_align(self) -> str:
        """Y-axis alignment mode."""
        return self._y_align
    
    @y_align.setter
    def y_align(self, value: str):
        """Set Y-axis alignment mode."""
        if value not in ("sample_index", "depth", "range"):
            raise ValueError(f"y_align must be 'sample_index', 'depth', or 'range', got '{value}'")
        self._y_align = value

    @property
    def linear(self) -> bool:
        """Whether to combine in linear domain."""
        return self._linear
    
    @linear.setter
    def linear(self, value: bool):
        """Set whether to combine in linear domain."""
        self._linear = bool(value)

    def _db_to_linear(self, data: np.ndarray) -> np.ndarray:
        """Convert dB values to linear scale: 10^(0.1 * dB)."""
        return np.power(10.0, 0.1 * data)

    def _linear_to_db(self, data: np.ndarray) -> np.ndarray:
        """Convert linear values to dB scale: 10 * log10(linear)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = 10.0 * np.log10(data)
        return result

    def _compute_combined_metadata(self):
        """Compute metadata that spans all backends."""
        # Use first backend as reference for most metadata
        first = self._backends[0]
        
        # For combined echograms, we need the union of all time ranges
        # Use the first backend's ping structure as the "master"
        # Other backends will be resampled to match during get_image
        self._n_pings = first.n_pings
        self._ping_times = first.ping_times.copy()
        self._max_sample_counts = first.max_sample_counts.copy()
        
        # Sample extents from first backend
        self._sample_nr_min, self._sample_nr_max = first.sample_nr_extents
        self._sample_nr_min = self._sample_nr_min.copy()
        self._sample_nr_max = self._sample_nr_max.copy()
        
        # Range extents - use union (min of mins, max of maxes) across backends
        if all(b.range_extents is not None for b in self._backends):
            all_min_r = np.stack([b.range_extents[0][:self._n_pings] for b in self._backends 
                                  if len(b.range_extents[0]) >= self._n_pings], axis=0)
            all_max_r = np.stack([b.range_extents[1][:self._n_pings] for b in self._backends
                                  if len(b.range_extents[1]) >= self._n_pings], axis=0)
            if len(all_min_r) > 0:
                self._range_min = np.nanmin(all_min_r, axis=0)
                self._range_max = np.nanmax(all_max_r, axis=0)
            else:
                self._range_min = first.range_extents[0].copy()
                self._range_max = first.range_extents[1].copy()
        else:
            self._range_min = None
            self._range_max = None
        
        # Depth extents - use union across backends
        if all(b.depth_extents is not None for b in self._backends):
            all_min_d = np.stack([b.depth_extents[0][:self._n_pings] for b in self._backends
                                  if len(b.depth_extents[0]) >= self._n_pings], axis=0)
            all_max_d = np.stack([b.depth_extents[1][:self._n_pings] for b in self._backends
                                  if len(b.depth_extents[1]) >= self._n_pings], axis=0)
            if len(all_min_d) > 0:
                self._depth_min = np.nanmin(all_min_d, axis=0)
                self._depth_max = np.nanmax(all_max_d, axis=0)
            else:
                self._depth_min = first.depth_extents[0].copy()
                self._depth_max = first.depth_extents[1].copy()
        else:
            self._depth_min = None
            self._depth_max = None
        
        # Lat/lon from first backend that has it
        self._latitudes = None
        self._longitudes = None
        for backend in self._backends:
            if backend.has_latlon:
                self._latitudes = backend.latitudes
                self._longitudes = backend.longitudes
                break
        
        # Scalar metadata from first backend
        self._wci_value = first.wci_value
        self._linear_mean = first.linear_mean
        self._has_navigation = any(b.has_navigation for b in self._backends)
        
        # Combine ping parameters
        self._ping_params = self._combine_ping_params()

    def _combine_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        """Combine ping parameters from all backends.
        
        Strategy:
        1. Keep individual params as "name_0", "name_1", etc. for transparency
        2. Create combined param "name" by interpolating all to common times and combining
        """
        # Collect all param names across backends
        all_names = set()
        for backend in self._backends:
            all_names.update(backend.get_ping_params().keys())
        
        result = {}
        
        for name in all_names:
            # First, keep individual backend params with suffix
            for i, backend in enumerate(self._backends):
                params = backend.get_ping_params()
                if name in params:
                    y_ref, (times, values) = params[name]
                    result[f"{name}_{i}"] = (y_ref, (times.copy(), values.copy()))
            
            # Now create combined param
            # Collect all (times, values) pairs for this param
            all_data = []
            y_ref = None
            for backend in self._backends:
                params = backend.get_ping_params()
                if name in params:
                    y_ref_b, (times, values) = params[name]
                    if y_ref is None:
                        y_ref = y_ref_b
                    all_data.append((times, values))
            
            if not all_data or y_ref is None:
                continue
            
            # Create combined time grid (union of all times)
            all_times = np.concatenate([d[0] for d in all_data])
            unique_times = np.unique(all_times)
            unique_times = unique_times[np.isfinite(unique_times)]
            
            if len(unique_times) == 0:
                continue
            
            # Interpolate each backend's values to the combined time grid
            from themachinethatgoesping import tools
            
            interpolated_values = []
            for times, values in all_data:
                # Filter valid points
                valid = np.isfinite(times) & np.isfinite(values)
                if np.sum(valid) < 2:
                    # Not enough points, fill with NaN
                    interpolated_values.append(np.full(len(unique_times), np.nan))
                    continue
                
                try:
                    interp = tools.vectorinterpolators.LinearInterpolator(
                        times[valid], values[valid], extrapolation_mode="nearest"
                    )
                    interpolated_values.append(interp(unique_times))
                except Exception:
                    interpolated_values.append(np.full(len(unique_times), np.nan))
            
            # Stack and combine
            stacked = np.stack(interpolated_values, axis=0)
            
            # Apply the same combine function used for images
            combined_values = self._combine_func(stacked, axis=0)
            
            result[name] = (y_ref, (unique_times, combined_values))
        
        return result

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
    def combine_func_name(self) -> str:
        """Name of the combine function."""
        return self._combine_func_name

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
    # Data access (per-column)
    # =========================================================================

    def get_column(self, ping_index: int) -> np.ndarray:
        """Get combined column data for a ping.
        
        Fetches data from all backends, aligns by sample index, and combines.
        Note: This is slower than get_image() due to per-column overhead.
        For building full images, get_image() is preferred.
        """
        # Collect columns from all backends
        columns = []
        max_len = 0
        
        for backend in self._backends:
            if ping_index < backend.n_pings:
                col = backend.get_column(ping_index)
                columns.append(col)
                max_len = max(max_len, len(col))
            else:
                columns.append(None)
        
        if max_len == 0:
            return np.array([], dtype=np.float32)
        
        # Pad columns to same length and stack
        padded = []
        for col in columns:
            if col is None:
                padded.append(np.full(max_len, np.nan, dtype=np.float32))
            elif len(col) < max_len:
                pad = np.full(max_len, np.nan, dtype=np.float32)
                pad[:len(col)] = col
                padded.append(pad)
            else:
                padded.append(col[:max_len])
        
        stacked = np.stack(padded, axis=0)  # (n_backends, max_len)
        
        # Combine along backend axis (optionally in linear domain)
        if self._linear:
            stacked = self._db_to_linear(stacked)
            result = self._combine_func(stacked, axis=0)
            result = self._linear_to_db(result)
        else:
            result = self._combine_func(stacked, axis=0)
        
        return result.astype(np.float32)

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        """Get combined raw column data."""
        return self.get_column(ping_index)

    # =========================================================================
    # Image generation
    # =========================================================================

    def _compute_backend_affines(self, backend: EchogramDataBackend, n_pings: int):
        """Compute affine parameters for a specific backend's depth-to-sample mapping.
        
        The affine mapping is: sample_idx = round(a + b * y)
        where y is the depth/range coordinate.
        
        For depth mode:
            depth = depth_min + sample_idx * resolution
            => sample_idx = (depth - depth_min) / resolution
            => a = -depth_min / resolution, b = 1 / resolution
        """
        if backend.depth_extents is None:
            return None, None
        
        min_depths, max_depths = backend.depth_extents
        max_samples = backend.max_sample_counts
        
        # Ensure arrays are long enough
        n = min(n_pings, len(min_depths), len(max_depths), len(max_samples))
        
        # Resolution per ping: (max_depth - min_depth) / n_samples
        with np.errstate(divide='ignore', invalid='ignore'):
            resolutions = (max_depths[:n] - min_depths[:n]) / (max_samples[:n] + 1)
            resolutions = np.where(max_samples[:n] > 0, resolutions, np.nan)
        
        # Affine params: sample = a + b * depth
        # where a = -depth_min / resolution, b = 1 / resolution
        with np.errstate(divide='ignore', invalid='ignore'):
            affine_b = 1.0 / resolutions
            affine_a = -min_depths[:n] / resolutions
        
        # Pad to full length if needed
        if n < n_pings:
            a_full = np.full(n_pings, np.nan, dtype=np.float32)
            b_full = np.full(n_pings, np.nan, dtype=np.float32)
            a_full[:n] = affine_a
            b_full[:n] = affine_b
            return a_full, b_full
        
        return affine_a.astype(np.float32), affine_b.astype(np.float32)

    def _create_time_aligned_ping_indexer(
        self, 
        request_ping_indexer: np.ndarray, 
        backend: EchogramDataBackend,
        time_tolerance: float = 0.5
    ) -> np.ndarray:
        """Create a ping indexer for a backend aligned by time.
        
        The request's ping_indexer maps x-positions to ping indices of the 
        COMBINED backend (first backend). This method translates those to
        the equivalent ping indices in a sub-backend based on matching times.
        
        Args:
            request_ping_indexer: Ping indices from request (into combined/first backend).
            backend: The sub-backend to create indexer for.
            time_tolerance: Maximum time difference (seconds) to consider a match.
            
        Returns:
            New ping indexer mapping x-positions to this backend's ping indices.
            -1 indicates no matching ping in this backend.
        """
        # Get times for the combined backend (self)
        combined_times = self._ping_times
        
        # Get times for the sub-backend
        backend_times = backend.ping_times
        
        # Create output indexer
        backend_indexer = np.full_like(request_ping_indexer, -1)
        
        # For each valid ping in request, find matching ping in backend by time
        valid_mask = request_ping_indexer >= 0
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0 or len(backend_times) == 0:
            return backend_indexer
        
        # Get the combined ping indices we need to map
        combined_ping_indices = request_ping_indexer[valid_mask]
        
        # Filter out any indices that are out of range for combined_times
        in_range = combined_ping_indices < len(combined_times)
        if not np.any(in_range):
            return backend_indexer
        
        # Get times for those pings
        query_times = combined_times[combined_ping_indices[in_range]]
        
        # Use searchsorted to find nearest ping in backend for each query time
        # This is O(n log m) instead of O(n*m)
        insert_positions = np.searchsorted(backend_times, query_times)
        
        # For each query, check the ping at insert_pos and insert_pos-1 to find closest
        result_indices = np.full(len(query_times), -1, dtype=np.int64)
        
        for i, (query_time, pos) in enumerate(zip(query_times, insert_positions)):
            best_idx = -1
            best_diff = time_tolerance
            
            # Check position before
            if pos > 0:
                diff = abs(backend_times[pos - 1] - query_time)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = pos - 1
            
            # Check position at/after
            if pos < len(backend_times):
                diff = abs(backend_times[pos] - query_time)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = pos
            
            result_indices[i] = best_idx
        
        # Map back to output array
        valid_indices_in_range = valid_indices[in_range]
        backend_indexer[valid_indices_in_range] = result_indices
        
        return backend_indexer

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        """Build combined image by getting from each backend and combining.
        
        This is the main entry point for efficient image generation.
        Each backend produces its own downsampled image using its own
        depth-to-sample mapping, then images are stacked and combined.
        
        IMPORTANT: When combining echograms in depth mode, each backend
        may have different sample-to-depth relationships. This method
        computes the correct affine parameters for each backend.
        
        IMPORTANT: Backends are aligned by TIME, not by ping index. This is
        critical when combining different frequencies that may have different
        ping rates or timing.
        
        Handles backends with different coverage (different number of pings)
        by masking out-of-range pings as invalid for each backend.
        """
        # Determine if we're in depth mode by checking if y_coordinates
        # look like depth values (not sample indices)
        y_coords = request.y_coordinates
        
        # Heuristic: if y values are > 10 or fractional, likely depth/range mode
        is_depth_mode = (np.nanmax(np.abs(y_coords)) > 10 or 
                         not np.allclose(y_coords, y_coords.astype(int)))
        
        # Collect images from all backends
        images = []
        
        for backend_idx, backend in enumerate(self._backends):
            backend_n_pings = backend.n_pings
            
            # Create a TIME-ALIGNED ping indexer for this backend
            # The first backend uses the original indexer (it's the reference)
            if backend_idx == 0:
                backend_ping_indexer = request.ping_indexer.copy()
                # Just mark out-of-range as invalid
                out_of_range = backend_ping_indexer >= backend_n_pings
                backend_ping_indexer[out_of_range] = -1
            else:
                # For other backends, align by time
                backend_ping_indexer = self._create_time_aligned_ping_indexer(
                    request.ping_indexer, backend
                )
            
            # Determine if we need depth-aligned y-axis based on y_align setting
            use_depth_affines = self._y_align in ("depth", "range")
            
            if use_depth_affines:
                # Compute this backend's specific affine parameters
                backend_affine_a, backend_affine_b = self._compute_backend_affines(
                    backend, backend_n_pings
                )
                
                if backend_affine_a is not None and backend_affine_b is not None:
                    # Pad max_sample_indices
                    backend_max_samples = backend.max_sample_counts.astype(np.int64) + 1
                    
                    # Create modified request with this backend's affines
                    backend_request = EchogramImageRequest(
                        nx=request.nx,
                        ny=request.ny,
                        y_coordinates=request.y_coordinates,
                        ping_indexer=backend_ping_indexer,
                        affine_a=backend_affine_a,
                        affine_b=backend_affine_b,
                        max_sample_indices=backend_max_samples,
                        fill_value=request.fill_value,
                    )
                    img = backend.get_image(backend_request)
                else:
                    # Fallback - still need safe ping indexer and appropriate affines
                    # For fallback, use original request affines but only where valid
                    n_request = len(request.affine_a)
                    fallback_affine_a = np.full(backend_n_pings, np.nan, dtype=np.float32)
                    fallback_affine_b = np.full(backend_n_pings, np.nan, dtype=np.float32)
                    
                    # Copy affines for valid ping mappings
                    for x_idx, backend_ping in enumerate(backend_ping_indexer):
                        if backend_ping >= 0:
                            orig_ping = request.ping_indexer[x_idx]
                            if orig_ping >= 0 and orig_ping < n_request:
                                fallback_affine_a[backend_ping] = request.affine_a[orig_ping]
                                fallback_affine_b[backend_ping] = request.affine_b[orig_ping]
                    
                    backend_max_samples = backend.max_sample_counts.astype(np.int64) + 1
                    
                    backend_request = EchogramImageRequest(
                        nx=request.nx,
                        ny=request.ny,
                        y_coordinates=request.y_coordinates,
                        ping_indexer=backend_ping_indexer,
                        affine_a=fallback_affine_a,
                        affine_b=fallback_affine_b,
                        max_sample_indices=backend_max_samples,
                        fill_value=request.fill_value,
                    )
                    img = backend.get_image(backend_request)
            else:
                # Sample index mode - use original affines but with correct ping indexer
                backend_max_samples = backend.max_sample_counts.astype(np.int64) + 1
                
                # For sample index mode, we need to map the affines if x_align is "time"
                if self._x_align == "time" and backend_idx > 0:
                    n_request = len(request.affine_a)
                    backend_affine_a = np.full(backend_n_pings, np.nan, dtype=np.float32)
                    backend_affine_b = np.full(backend_n_pings, np.nan, dtype=np.float32)
                    
                    # Copy affines for valid ping mappings
                    for x_idx, backend_ping in enumerate(backend_ping_indexer):
                        if backend_ping >= 0:
                            orig_ping = request.ping_indexer[x_idx]
                            if orig_ping >= 0 and orig_ping < n_request:
                                backend_affine_a[backend_ping] = request.affine_a[orig_ping]
                                backend_affine_b[backend_ping] = request.affine_b[orig_ping]
                    
                    backend_request = EchogramImageRequest(
                        nx=request.nx,
                        ny=request.ny,
                        y_coordinates=request.y_coordinates,
                        ping_indexer=backend_ping_indexer,
                        affine_a=backend_affine_a,
                        affine_b=backend_affine_b,
                        max_sample_indices=backend_max_samples,
                        fill_value=request.fill_value,
                    )
                else:
                    # ping_index mode or first backend - use original affines
                    # Just need to ensure ping indexer is valid for this backend
                    backend_request = EchogramImageRequest(
                        nx=request.nx,
                        ny=request.ny,
                        y_coordinates=request.y_coordinates,
                        ping_indexer=backend_ping_indexer,
                        affine_a=request.affine_a,
                        affine_b=request.affine_b,
                        max_sample_indices=backend_max_samples,
                        fill_value=request.fill_value,
                    )
                
                img = backend.get_image(backend_request)
            
            images.append(img)
        
        # Stack: shape (n_backends, nx, ny)
        stacked = np.stack(images, axis=0)
        
        # Apply combine function along backend axis (optionally in linear domain)
        if self._linear:
            stacked = self._db_to_linear(stacked)
            result = self._combine_func(stacked, axis=0)
            result = self._linear_to_db(result)
        else:
            result = self._combine_func(stacked, axis=0)
        
        return result.astype(np.float32)

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"CombineBackend(name='{self._name}', "
            f"n_backends={len(self._backends)}, "
            f"combine_func='{self._combine_func_name}', "
            f"x_align='{self._x_align}', y_align='{self._y_align}', "
            f"linear={self._linear})"
        )
