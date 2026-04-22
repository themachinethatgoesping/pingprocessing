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


def nandiff(stack: np.ndarray, axis: int) -> np.ndarray:
    """Difference: first backend minus all others (NaN-aware).
    
    Computes stack[0] - stack[1] - stack[2] - ... ignoring NaN.
    """
    result = stack[0].copy()
    for i in range(1, stack.shape[0]):
        result = result - stack[i]
    return result


# Registry of built-in functions
COMBINE_FUNCTIONS = {
    "nanmean": nanmean,
    "nanmedian": nanmedian,
    "nansum": nansum,
    "nanmax": nanmax,
    "nanmin": nanmin,
    "nanstd": nanstd,
    "nandiff": nandiff,
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
        data_transforms: list = None,
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
            data_transforms: Optional list of (factor, offset, transform_func) tuples,
                one per backend. If provided, applied to each backend's image before
                combining. None entries or None list means no transforms.
        
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
        self._data_transforms = data_transforms
        
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
        # Use first backend as scalar metadata fallback
        first = self._backends[0]

        # Build combined time grid as the sorted union of all backend ping times.
        # This allows combining temporally disjoint echograms into one timeline.
        all_times = []
        for backend in self._backends:
            bt = np.asarray(backend.ping_times, dtype=np.float64)
            if len(bt) == 0:
                continue
            all_times.append(bt[np.isfinite(bt)])
        if all_times:
            self._ping_times = np.unique(np.concatenate(all_times))
        else:
            self._ping_times = np.array([], dtype=np.float64)
        self._n_pings = len(self._ping_times)

        # Sample-index extents across all backends, aligned to combined timeline.
        aligned_min_s = []
        aligned_max_s = []
        for i, b in enumerate(self._backends):
            min_s, max_s = b.sample_nr_extents
            aligned_min_s.append(self._align_backend_array_to_combined(min_s, i))
            aligned_max_s.append(self._align_backend_array_to_combined(max_s, i))
        all_min_s = np.stack(aligned_min_s, axis=0)
        all_max_s = np.stack(aligned_max_s, axis=0)
        self._sample_nr_min = np.nanmin(all_min_s, axis=0)
        self._sample_nr_max = np.nanmax(all_max_s, axis=0)
        
        # Range extents - use union (min of mins, max of maxes) across backends
        # Time-aligned: each backend's extents placed at correct combined indices
        if all(b.range_extents is not None for b in self._backends):
            aligned_min_r = []
            aligned_max_r = []
            for i, b in enumerate(self._backends):
                min_r, max_r = b.range_extents
                aligned_min_r.append(self._align_backend_array_to_combined(min_r, i))
                aligned_max_r.append(self._align_backend_array_to_combined(max_r, i))
            all_min_r = np.stack(aligned_min_r, axis=0)
            all_max_r = np.stack(aligned_max_r, axis=0)
            self._range_min = np.nanmin(all_min_r, axis=0)
            self._range_max = np.nanmax(all_max_r, axis=0)
        else:
            self._range_min = None
            self._range_max = None
        
        # Depth extents - use union across backends
        # Time-aligned: each backend's extents placed at correct combined indices
        if all(b.depth_extents is not None for b in self._backends):
            aligned_min_d = []
            aligned_max_d = []
            for i, b in enumerate(self._backends):
                min_d, max_d = b.depth_extents
                aligned_min_d.append(self._align_backend_array_to_combined(min_d, i))
                aligned_max_d.append(self._align_backend_array_to_combined(max_d, i))
            all_min_d = np.stack(aligned_min_d, axis=0)
            all_max_d = np.stack(aligned_max_d, axis=0)
            self._depth_min = np.nanmin(all_min_d, axis=0)
            self._depth_max = np.nanmax(all_max_d, axis=0)
        else:
            self._depth_min = None
            self._depth_max = None
        
        # Compute max_sample_counts from the finest resolution across all
        # contributing backends per-ping.  This ensures that get_column()
        # (and hence the layer path) uses a combined depth grid that
        # preserves the finest backend resolution.
        self._max_sample_counts = self._compute_combined_max_sample_counts()
        
        # Lat/lon from first backend that has it, time-aligned to combined grid
        self._latitudes = None
        self._longitudes = None
        for i, backend in enumerate(self._backends):
            if backend.has_latlon:
                self._latitudes = self._align_backend_array_to_combined(backend.latitudes, i)
                self._longitudes = self._align_backend_array_to_combined(backend.longitudes, i)
                break
        
        # Scalar metadata from first backend
        self._wci_value = first.wci_value
        self._linear_mean = first.linear_mean
        self._has_navigation = any(b.has_navigation for b in self._backends)
        
        # Combine ping parameters
        self._ping_params = self._combine_ping_params()

    @staticmethod
    def _pad_to_length(arr: np.ndarray, target_length: int) -> np.ndarray:
        """Pad array to target length with NaN if shorter, or truncate if longer."""
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) >= target_length:
            return arr[:target_length]
        padded = np.full(target_length, np.nan, dtype=np.float64)
        padded[:len(arr)] = arr
        return padded

    def _align_backend_array_to_combined(self, arr: np.ndarray, backend_idx: int) -> np.ndarray:
        """Align a per-ping array from a sub-backend to combined ping indices.

        Maps each COMBINED ping to the nearest BACKEND
        ping by time.  Only combined pings within the backend's time range
        receive a value; all others remain NaN.  This gives dense coverage
        across the entire overlap region (every combined ping gets a value
        from the nearest backend ping).
        """
        backend = self._backends[backend_idx]
        backend_times = backend.ping_times
        combined_times = self._ping_times
        result = np.full(self._n_pings, np.nan, dtype=np.float64)

        if len(backend_times) == 0 or len(combined_times) == 0:
            return result

        arr = np.asarray(arr, dtype=np.float64)
        n_backend = len(backend_times)
        n = min(len(arr), n_backend)
        if n <= 0:
            return result

        # For each combined ping, find the nearest backend ping
        insert_pos = np.searchsorted(backend_times, combined_times)

        cand_left = np.clip(insert_pos - 1, 0, n_backend - 1)
        cand_right = np.clip(insert_pos, 0, n_backend - 1)

        diff_left = np.abs(backend_times[cand_left] - combined_times)
        diff_right = np.abs(backend_times[cand_right] - combined_times)

        use_right = diff_right <= diff_left
        best_backend_idx = np.where(use_right, cand_right, cand_left)

        # Only assign values for combined pings within the backend's time range
        t_min = backend_times[0]
        t_max = backend_times[-1]
        # Use a tolerance of half the backend's median ping interval
        if n_backend > 1:
            half_interval = 0.5 * np.median(np.abs(np.diff(backend_times)))
        else:
            half_interval = 0.5
        in_range = (combined_times >= t_min - half_interval) & (combined_times <= t_max + half_interval)

        # Clip backend indices to valid array range
        safe_idx = np.clip(best_backend_idx, 0, n - 1)
        result[in_range] = arr[safe_idx[in_range]]

        return result

    def _compute_combined_max_sample_counts(self) -> np.ndarray:
        """Compute max_sample_counts that preserves the finest backend resolution.

        For each combined ping the depth-per-sample resolution of every
        contributing backend is evaluated. The finest (smallest) resolution
        is chosen, and the combined sample count is set so that the full
        combined depth range is covered at that resolution:
            n_samples = round((depth_max - depth_min) / finest_resolution)

        When depth extents are unavailable the first backend's counts are
        used as fallback.
        """
        first = self._backends[0]
        fallback = first.max_sample_counts.copy()

        # Need depth extents to compute resolution-based counts
        if self._depth_min is None or self._depth_max is None:
            return self._pad_to_length(fallback, self._n_pings).astype(np.float32)

        resolutions = []
        for i, b in enumerate(self._backends):
            if b.depth_extents is None:
                continue
            b_min, b_max = b.depth_extents
            b_samples = b.max_sample_counts
            a_min = self._align_backend_array_to_combined(b_min, i)
            a_max = self._align_backend_array_to_combined(b_max, i)
            a_samp = self._align_backend_array_to_combined(b_samples, i)
            with np.errstate(divide='ignore', invalid='ignore'):
                res = (a_max - a_min) / a_samp
            resolutions.append(res)

        if not resolutions:
            return self._pad_to_length(fallback, self._n_pings).astype(np.float32)

        all_res = np.stack(resolutions, axis=0)
        finest_res = np.nanmin(all_res, axis=0)

        combined_range = self._depth_max - self._depth_min
        with np.errstate(divide='ignore', invalid='ignore'):
            counts = np.round(combined_range / finest_res)

        invalid = ~np.isfinite(counts) | (counts <= 0)
        # Fallback to first backend where the resolution-based count is invalid
        fb = self._pad_to_length(fallback, self._n_pings)
        counts[invalid] = fb[invalid]

        return counts.astype(np.float32)

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
                        times[valid], values[valid], extrapolation_mode="nan"
                    )
                    interpolated_values.append(interp(unique_times))
                except Exception:
                    interpolated_values.append(np.full(len(unique_times), np.nan))
            
            # Stack and combine
            stacked = np.stack(interpolated_values, axis=0)
            
            # Always use nanmean for combining ping parameters (e.g., bottom depth).
            # Using the image combine function (e.g., nandiff) would produce
            # meaningless reference values.
            combined_values = nanmean(stacked, axis=0)
            
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

    def _find_backend_ping_index(self, backend_idx: int, combined_ping_index: int) -> int:
        """Find the ping index in a sub-backend that matches the combined ping's time.

        We find the nearest ping
        by timestamp, respecting an adaptive time tolerance.

        Returns -1 if no matching ping exists in the backend.
        """
        backend = self._backends[backend_idx]
        if combined_ping_index >= len(self._ping_times):
            return -1
        
        query_time = self._ping_times[combined_ping_index]
        backend_times = backend.ping_times
        
        if len(backend_times) == 0:
            return -1
        
        # Compute tolerance from this backend's own cadence only.
        # Using the combined union timeline here can over-inflate tolerance
        # when files are temporally disjoint.
        if len(backend_times) > 1:
            backend_intervals = np.abs(np.diff(backend_times))
            tolerance = float(np.nanquantile(backend_intervals, 0.95))
            if not np.isfinite(tolerance) or tolerance <= 0:
                tolerance = 0.5
        else:
            tolerance = 0.5

        # Fast reject if query time is outside this backend's time extent.
        if query_time < (backend_times[0] - tolerance) or query_time > (backend_times[-1] + tolerance):
            return -1
        
        # Find nearest ping by time
        idx = int(np.searchsorted(backend_times, query_time))
        best_idx = -1
        best_diff = np.inf
        for candidate in (idx - 1, idx):
            if 0 <= candidate < len(backend_times):
                diff = abs(backend_times[candidate] - query_time)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = candidate
        
        return best_idx if best_diff <= tolerance else -1

    def get_column(self, ping_index: int) -> np.ndarray:
        """Get combined column data for a ping.
        
        When y_align is "depth" or "range", each backend's column is
        resampled onto a common depth/range grid before combining.  This
        ensures that the same depth maps to the same output sample index
        regardless of which backend the data comes from.
        
        In sample-index mode the original pad-and-stack behaviour is used.
        """
        if self._y_align in ("depth", "range"):
            return self._get_column_depth_aligned(ping_index)
        return self._get_column_sample_aligned(ping_index)

    def _get_column_sample_aligned(self, ping_index: int) -> np.ndarray:
        """Original combine-by-sample-index path."""
        columns = []
        max_len = 0
        
        for backend_idx, backend in enumerate(self._backends):
            mapped_ping = self._find_backend_ping_index(backend_idx, ping_index)
            if mapped_ping >= 0:
                col = backend.get_column(mapped_ping)
                columns.append(col)
                max_len = max(max_len, len(col))
            else:
                columns.append(None)
        
        if max_len == 0:
            return np.array([], dtype=np.float32)
        
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
        
        stacked = np.stack(padded, axis=0)
        if self._linear:
            stacked = self._db_to_linear(stacked)
            result = self._combine_func(stacked, axis=0)
            result = self._linear_to_db(result)
        else:
            result = self._combine_func(stacked, axis=0)
        return result.astype(np.float32)

    def _get_column_depth_aligned(self, ping_index: int) -> np.ndarray:
        """Combine backends by resampling each to a common depth grid.

        The common grid spans self._depth_min[ping] to self._depth_max[ping]
        with self._max_sample_counts[ping]+1 points, matching the affine
        that the coordinate system will use to display the data.
        """
        extents_attr = "depth_extents" if self._y_align == "depth" else "range_extents"

        # Combined depth grid for this ping
        if ping_index >= self._n_pings:
            return np.array([], dtype=np.float32)

        if self._y_align == "depth":
            c_min = self._depth_min[ping_index]
            c_max = self._depth_max[ping_index]
        else:
            c_min = self._range_min[ping_index]
            c_max = self._range_max[ping_index]

        n_samples = int(self._max_sample_counts[ping_index]) + 1
        if not np.isfinite(c_min) or not np.isfinite(c_max) or c_max <= c_min or n_samples <= 1:
            return np.array([], dtype=np.float32)

        combined_depths = np.linspace(c_min, c_max, n_samples)

        resampled = []
        for backend_idx, backend in enumerate(self._backends):
            nan_col = np.full(n_samples, np.nan, dtype=np.float32)

            mapped_ping = self._find_backend_ping_index(backend_idx, ping_index)
            if mapped_ping < 0:
                resampled.append(nan_col)
                continue

            col = backend.get_column(mapped_ping)
            if len(col) <= 1:
                resampled.append(nan_col)
                continue

            b_ext = getattr(backend, extents_attr, None)
            if b_ext is None:
                resampled.append(nan_col)
                continue

            b_min_arr, b_max_arr = b_ext
            if mapped_ping >= len(b_min_arr):
                resampled.append(nan_col)
                continue

            b_min = float(b_min_arr[mapped_ping])
            b_max = float(b_max_arr[mapped_ping])
            if not np.isfinite(b_min) or not np.isfinite(b_max) or b_max <= b_min:
                resampled.append(nan_col)
                continue

            # Map combined depth values to fractional sample indices in this backend
            n_col = len(col)
            with np.errstate(divide='ignore', invalid='ignore'):
                frac_idx = (combined_depths - b_min) / (b_max - b_min) * (n_col - 1)

            indices = np.round(frac_idx).astype(np.int64)
            valid = (indices >= 0) & (indices < n_col)
            result = nan_col.copy()
            result[valid] = col[indices[valid]]
            resampled.append(result)

        if not resampled:
            return np.array([], dtype=np.float32)

        stacked = np.stack(resampled, axis=0)
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
        
        # Resolution per ping: (max_depth - min_depth) / max_samples
        # max_samples is the max valid sample index (n_samples - 1), matching
        # the coordinate system's convention: value = min + (max-min)/n * sample_idx
        min_d = np.asarray(min_depths[:n], dtype=np.float64)
        max_d = np.asarray(max_depths[:n], dtype=np.float64)
        ms = np.asarray(max_samples[:n], dtype=np.float64)
        
        # Mark invalid pings: non-finite depths, zero/negative range, zero samples
        invalid = ~(
            np.isfinite(min_d) & np.isfinite(max_d)
            & (max_d > min_d) & (ms > 0)
        )
        
        with np.errstate(divide='ignore', invalid='ignore'):
            resolutions = (max_d - min_d) / ms
        
        # Affine params: sample = a + b * depth
        # where a = -depth_min / resolution, b = 1 / resolution
        with np.errstate(divide='ignore', invalid='ignore'):
            affine_b = 1.0 / resolutions
            affine_a = -min_d / resolutions
        
        affine_a[invalid] = np.nan
        affine_b[invalid] = np.nan
        
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
        time_tolerance: Optional[float] = None,
    ) -> np.ndarray:
        """Create a ping indexer for a backend aligned by time.

        The request's ping_indexer maps x-positions to ping indices of the
        COMBINED backend. This method translates those to
        the equivalent ping indices in a sub-backend based on matching times.
        
        Args:
            request_ping_indexer: Ping indices from request (into combined backend).
            backend: The sub-backend to create indexer for.
            time_tolerance: Maximum time difference (seconds) to consider a match.
                If None, automatically computed as the median ping interval of the
                backend (ensures every ping in the backend can be matched).
            
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
        
        # Auto-compute tolerance from this backend's own ping cadence.
        # Do not use the combined union timeline here: large gaps between
        # disjoint files would inflate tolerance and cause false matches.
        if time_tolerance is None:
            if len(backend_times) > 1:
                backend_intervals = np.abs(np.diff(backend_times))
                time_tolerance = float(np.nanquantile(backend_intervals, 0.95))
                if not np.isfinite(time_tolerance) or time_tolerance <= 0:
                    time_tolerance = 0.5
            else:
                time_tolerance = 0.5
        
        # Get the combined ping indices we need to map
        combined_ping_indices = request_ping_indexer[valid_mask]
        
        # Filter out any indices that are out of range for combined_times
        in_range = combined_ping_indices < len(combined_times)
        if not np.any(in_range):
            return backend_indexer
        
        # Get times for those pings
        query_times = combined_times[combined_ping_indices[in_range]]

        # Fast short-circuit: if the visible request window does not overlap this
        # backend's time extent (within tolerance), no ping can match.
        backend_t_min = backend_times[0]
        backend_t_max = backend_times[-1]
        if query_times[-1] < (backend_t_min - time_tolerance) or query_times[0] > (backend_t_max + time_tolerance):
            return backend_indexer

        # Restrict matching work to query times that can plausibly hit this backend.
        # This avoids unnecessary searchsorted work for far-away disjoint windows.
        plausible = (query_times >= (backend_t_min - time_tolerance)) & (query_times <= (backend_t_max + time_tolerance))
        if not np.any(plausible):
            return backend_indexer
        query_times = query_times[plausible]
        
        # Use searchsorted to find nearest ping in backend for each query time
        # This is O(n log m) instead of O(n*m)
        insert_positions = np.searchsorted(backend_times, query_times)
        
        # Vectorised nearest-neighbour: check candidates at pos-1 and pos
        n_query = len(query_times)
        n_backend = len(backend_times)
        
        # Candidate indices (clip to valid range)
        cand_left = np.clip(insert_positions - 1, 0, n_backend - 1)
        cand_right = np.clip(insert_positions, 0, n_backend - 1)
        
        diff_left = np.abs(backend_times[cand_left] - query_times)
        diff_right = np.abs(backend_times[cand_right] - query_times)
        
        # Pick the closer candidate
        use_right = diff_right <= diff_left
        best_indices = np.where(use_right, cand_right, cand_left)
        best_diffs = np.where(use_right, diff_right, diff_left)
        
        # Apply tolerance
        best_indices[best_diffs > time_tolerance] = -1
        
        # Map back to output array
        valid_indices_in_range = valid_indices[in_range][plausible]
        backend_indexer[valid_indices_in_range] = best_indices
        
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
        # Collect images from all backends
        images = []
        
        for backend_idx, backend in enumerate(self._backends):
            backend_n_pings = backend.n_pings

            # Create backend ping indexer according to selected x alignment mode.
            if self._x_align == "time":
                backend_ping_indexer = self._create_time_aligned_ping_indexer(
                    request.ping_indexer, backend
                )
            else:
                backend_ping_indexer = request.ping_indexer.copy()
                out_of_range = backend_ping_indexer >= backend_n_pings
                backend_ping_indexer[out_of_range] = -1

            # If this backend has no mapped pings for the current view/request,
            # skip expensive backend image generation entirely.
            if not np.any(backend_ping_indexer >= 0):
                images.append(np.full((request.nx, request.ny), request.fill_value, dtype=np.float32))
                continue
            
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
                
                # For sample index mode in time alignment, map request affines
                # from combined ping indices to this backend's ping indices.
                if self._x_align == "time":
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
            
            # Apply per-backend data transforms (factor, offset, transform)
            if self._data_transforms is not None and self._data_transforms[backend_idx] is not None:
                factor, offset, transform_func = self._data_transforms[backend_idx]
                if factor != 1.0 or offset != 0.0:
                    img = img * factor + offset
                if transform_func is not None:
                    img = transform_func(img)

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
