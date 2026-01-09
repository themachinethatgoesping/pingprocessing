"""Coordinate system management for echograms.

This module provides the EchogramCoordinateSystem class which handles all
coordinate transformations between different axis types (ping index, time,
datetime for X; sample index, sample number, depth, range for Y).
"""

import numpy as np
import datetime as dt
from typing import Optional, List, Tuple, Callable
from copy import deepcopy

import themachinethatgoesping as theping
from themachinethatgoesping import tools
from themachinethatgoesping.algorithms.gridding import ForwardGridder1D
from themachinethatgoesping.algorithms_nanopy.featuremapping import NearestFeatureMapper

from themachinethatgoesping.pingprocessing.core.asserts import assert_valid_argument
from .indexers import EchogramImageRequest


class EchogramCoordinateSystem:
    """Manages coordinate systems and transformations for echogram display.
    
    This class handles:
    - X-axis systems (ping index, ping time, datetime)
    - Y-axis systems (sample index, sample number, depth, range)
    - Interpolators for converting between coordinate systems
    - Extent management (min/max for each axis type)
    - Ping parameters (additional per-ping data like bottom depth)
    
    The coordinate system is independent of data storage, allowing it to be
    reused across different data sources.
    """

    def __init__(
        self,
        n_pings: int,
        max_number_of_samples: np.ndarray,
        ping_times: np.ndarray,
        ping_numbers: Optional[np.ndarray] = None,
        time_zone: dt.timezone = dt.timezone.utc,
    ):
        """Initialize coordinate system.
        
        Args:
            n_pings: Number of pings.
            max_number_of_samples: Array of max sample counts per ping.
            ping_times: Array of ping timestamps.
            ping_numbers: Optional array of ping numbers. If None, uses 0..n_pings-1.
            time_zone: Timezone for datetime display.
        """
        if n_pings == 0:
            raise RuntimeError("ERROR[EchogramCoordinateSystem]: n_pings must be > 0")

        self._n_pings = n_pings
        self.max_number_of_samples = np.asarray(max_number_of_samples)
        self.time_zone = time_zone
        self.mp_cores = 1

        # Feature mapper for coordinate lookups
        self.feature_mapper = NearestFeatureMapper()

        # Set up ping numbers and times
        if ping_numbers is None:
            ping_numbers = np.arange(n_pings)
        self.set_ping_numbers(ping_numbers)
        self.set_ping_times(ping_times)

        # Ping parameters (e.g., bottom depth)
        self.param = {}

        # Initialize extent flags
        self.has_ranges = False
        self.has_depths = False
        self.has_sample_nrs = False

        # Initialize axis state
        self.x_axis_name = None
        self.y_axis_name = None
        self._x_kwargs = {}
        self._y_kwargs = {}
        self._x_axis_function = None
        self._y_axis_function = None
        self._initialized = False

        # Interpolator lists (per-ping)
        self._init_interpolators()

    @property
    def n_pings(self) -> int:
        """Number of pings."""
        return self._n_pings

    @property
    def initialized(self) -> bool:
        """Whether coordinate system is fully initialized."""
        return self._initialized

    def _init_interpolators(self):
        """Initialize interpolator lists."""
        n = self._n_pings
        self.y_coordinate_indice_interpolator = [None] * n
        self.y_indice_to_y_coordinate_interpolator = [None] * n
        self.depth_to_y_coordinate_interpolator = [None] * n
        self.range_to_y_coordinate_interpolator = [None] * n
        self.sample_nr_to_y_coordinate_interpolator = [None] * n
        self.y_indice_to_depth_interpolator = [None] * n
        self.y_indice_to_range_interpolator = [None] * n
        self.y_indice_to_sample_nr_interpolator = [None] * n

    # =========================================================================
    # Ping numbers and times
    # =========================================================================

    def set_ping_numbers(self, ping_numbers: np.ndarray):
        """Set ping numbers for x-axis indexing."""
        assert len(ping_numbers) == self._n_pings, \
            f"ping_numbers length ({len(ping_numbers)}) must match n_pings ({self._n_pings})"
        self.feature_mapper.set_feature("Ping index", ping_numbers)
        self.ping_numbers = np.asarray(ping_numbers)
        self._initialized = False

    def set_ping_times(self, ping_times: np.ndarray, time_zone: Optional[dt.timezone] = None):
        """Set ping times for x-axis time display."""
        assert len(ping_times) == self._n_pings, \
            f"ping_times length ({len(ping_times)}) must match n_pings ({self._n_pings})"
        self.feature_mapper.set_feature("Ping time", ping_times)
        self.feature_mapper.set_feature("Date time", ping_times)
        self.ping_times = np.asarray(ping_times)
        if time_zone is not None:
            self.time_zone = time_zone
        self._initialized = False

    # =========================================================================
    # Extent setters
    # =========================================================================

    def set_range_extent(self, min_ranges: np.ndarray, max_ranges: np.ndarray):
        """Set range extents (per-ping min/max range in meters)."""
        assert len(min_ranges) == len(max_ranges) == self._n_pings, \
            f"range arrays must have length {self._n_pings}"
        self.min_ranges = np.asarray(min_ranges, dtype=np.float32)
        self.max_ranges = np.asarray(max_ranges, dtype=np.float32)
        self.res_ranges = ((self.max_ranges - self.min_ranges) / self.max_number_of_samples).astype(np.float32)
        self.has_ranges = True
        self._initialized = False

    def set_depth_extent(self, min_depths: np.ndarray, max_depths: np.ndarray):
        """Set depth extents (per-ping min/max depth in meters)."""
        assert len(min_depths) == len(max_depths) == self._n_pings, \
            f"depth arrays must have length {self._n_pings}"
        self.min_depths = np.asarray(min_depths, dtype=np.float32)
        self.max_depths = np.asarray(max_depths, dtype=np.float32)
        self.res_depths = ((self.max_depths - self.min_depths) / self.max_number_of_samples).astype(np.float32)
        self.has_depths = True
        self._initialized = False

    def set_sample_nr_extent(self, min_sample_nrs: np.ndarray, max_sample_nrs: np.ndarray):
        """Set sample number extents (per-ping min/max sample numbers)."""
        assert len(min_sample_nrs) == len(max_sample_nrs) == self._n_pings, \
            f"sample_nr arrays must have length {self._n_pings}"
        self.min_sample_nrs = np.asarray(min_sample_nrs, dtype=np.float32)
        self.max_sample_nrs = np.asarray(max_sample_nrs, dtype=np.float32)
        self.res_sample_nrs = ((self.max_sample_nrs - self.min_sample_nrs) / self.max_number_of_samples).astype(np.float32)
        self.has_sample_nrs = True
        self._initialized = False

    # =========================================================================
    # Ping parameters
    # =========================================================================

    def add_ping_param(self, name: str, x_reference: str, y_reference: str, 
                       vec_x_val: np.ndarray, vec_y_val: np.ndarray):
        """Add a ping parameter (e.g., bottom depth, layer boundary).
        
        Args:
            name: Parameter name (e.g., 'bottom', 'minslant').
            x_reference: X reference type ('Ping index', 'Ping time', 'Date time').
            y_reference: Y reference type ('Y indice', 'Sample number', 'Depth (m)', 'Range (m)').
            vec_x_val: X values (timestamps or indices).
            vec_y_val: Y values (depths, ranges, etc.).
        """
        assert_valid_argument("add_ping_param", x_reference, ["Ping index", "Ping time", "Date time"])
        assert_valid_argument("add_ping_param", y_reference, ["Y indice", "Sample number", "Depth (m)", "Range (m)"])

        # convert datetimes to timestamps
        if len(vec_x_val) > 0 and isinstance(vec_x_val[0], dt.datetime):
            vec_x_val = [x.timestamp() for x in vec_x_val]

        # convert to numpy arrays
        vec_x_val = np.array(vec_x_val)
        vec_y_val = np.array(vec_y_val)

        # filter nans and infs
        arg = np.where(np.isfinite(vec_x_val))[0]
        vec_x_val = vec_x_val[arg]
        vec_y_val = vec_y_val[arg]
        arg = np.where(np.isfinite(vec_y_val))[0]
        vec_x_val = vec_x_val[arg]
        vec_y_val = vec_y_val[arg]

        if len(vec_x_val) == 0:
            return  # No valid data

        match x_reference:
            case "Ping index":
                comp_vec_x_val = self.ping_numbers
            case "Ping time" | "Date time":
                comp_vec_x_val = self.ping_times

        # average vec_y_val for all duplicate vec_x_vals using vectorized numpy
        unique_x_vals, indices = np.unique(vec_x_val, return_inverse=True)
        # Use bincount to sum y values per unique x, then divide by counts
        sums = np.bincount(indices, weights=vec_y_val)
        counts = np.bincount(indices)
        averaged_y_vals = sums / counts

        vec_x_val = unique_x_vals
        vec_y_val = averaged_y_vals

        # convert to represent indices
        vec_y_val = tools.vectorinterpolators.LinearInterpolator(
            vec_x_val, vec_y_val, extrapolation_mode="nearest"
        )(comp_vec_x_val)

        self.param[name] = (y_reference, vec_y_val)

    def get_ping_param(self, name: str, use_x_coordinates: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get a ping parameter's values in current coordinate system.
        
        Args:
            name: Parameter name.
            use_x_coordinates: If True, use current x coordinates instead of all pings.
            
        Returns:
            Tuple of (x_coordinates, y_values).
        """
        self.reinit()
        assert name in self.param.keys(), f"ERROR[get_ping_param]: name '{name}' not registered"
        
        if use_x_coordinates:
            x_coordinates = np.array(self.feature_mapper.get_feature_values("X coordinate"))
            x_indices = np.array(self.feature_mapper.feature_to_index(
                self.x_axis_name, x_coordinates, mp_cores=self.mp_cores
            ))
        else:
            x_indices = np.arange(self._n_pings)
            x_coordinates = self.feature_mapper.get_feature_values(self.x_axis_name)

        reference, param = self.param[name]
        param = np.array(param)[x_indices]

        return_param = np.empty(len(param))
        return_param.fill(np.nan)

        match reference:
            case "Y indice":
                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.y_indice_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case "Sample number":
                assert self.has_sample_nrs, "ERROR: Sample nr values not initialized"
                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.sample_nr_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case "Depth (m)":
                assert self.has_depths, "ERROR: Depths values not initialized"
                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.depth_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case "Range (m)":
                assert self.has_ranges, "ERROR: Ranges values not initialized"
                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.range_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case _:
                raise RuntimeError(f"Invalid reference '{reference}'")

        if self.x_axis_name == "Date time":
            x_coordinates = [dt.datetime.fromtimestamp(t, self.time_zone) for t in x_coordinates]

        return x_coordinates, return_param

    # =========================================================================
    # Coordinate system initialization
    # =========================================================================

    def reinit(self):
        """Reinitialize coordinate systems if needed."""
        if self._initialized:
            return
        if self._y_axis_function is not None:
            self._y_axis_function(**self._y_kwargs)
        if self._x_axis_function is not None:
            self._x_axis_function(**self._x_kwargs)

    def get_x_kwargs(self) -> dict:
        """Get current X-axis configuration."""
        return deepcopy(self._x_kwargs)

    def get_y_kwargs(self) -> dict:
        """Get current Y-axis configuration."""
        return deepcopy(self._y_kwargs)

    # =========================================================================
    # Y-coordinate setup (internal)
    # =========================================================================

    def _set_y_coordinates(self, name: str, y_coordinates: np.ndarray, 
                           vec_min_y: np.ndarray, vec_max_y: np.ndarray,
                           layer_update_callback: Optional[Callable] = None):
        """Set Y coordinates for the display grid.
        
        Args:
            name: Axis name ('Y indice', 'Sample number', 'Depth (m)', 'Range (m)').
            y_coordinates: Array of Y coordinate values.
            vec_min_y: Per-ping minimum Y values.
            vec_max_y: Per-ping maximum Y values.
            layer_update_callback: Optional callback to update layers.
        """
        n_pings = self._n_pings
        assert len(vec_min_y) == len(vec_max_y) == n_pings, \
            f"min/max y vectors must have length {n_pings}"
            
        self.y_axis_name = name
        self.y_coordinates = y_coordinates
        self.y_resolution = y_coordinates[1] - y_coordinates[0]
        self.y_extent = [
            self.y_coordinates[-1] + self.y_resolution / 2,
            self.y_coordinates[0] - self.y_resolution / 2,
        ]
        self.vec_min_y = vec_min_y
        self.vec_max_y = vec_max_y

        self.y_gridder = ForwardGridder1D.from_res(
            self.y_resolution, self.y_coordinates[0], self.y_coordinates[-1]
        )
        
        # Call layer update callback if provided
        if layer_update_callback is not None:
            layer_update_callback()

        # Initialize interpolators
        self._init_interpolators()

        for nr in range(n_pings):
            y1, y2 = vec_min_y[nr], vec_max_y[nr]
            n_samples = self.max_number_of_samples[nr] + 1
            
            try:
                # Skip pings with invalid y values or no samples
                if not (np.isfinite(y1) and np.isfinite(y2)):
                    continue
                if y1 >= y2:
                    continue
                if n_samples <= 1:
                    continue
                    
                I = tools.vectorinterpolators.LinearInterpolatorF([y1, y2], [0, n_samples - 1])
                self.y_coordinate_indice_interpolator[nr] = I

                I = tools.vectorinterpolators.LinearInterpolatorF([0, n_samples - 1], [y1, y2])
                self.y_indice_to_y_coordinate_interpolator[nr] = I

                if self.has_depths:
                    d1, d2 = self.min_depths[nr], self.max_depths[nr]
                    if np.isfinite(d1) and np.isfinite(d2) and d1 < d2:
                        I = tools.vectorinterpolators.LinearInterpolatorF([d1, d2], [y1, y2])
                        self.depth_to_y_coordinate_interpolator[nr] = I
                        I = tools.vectorinterpolators.LinearInterpolatorF([0, n_samples - 1], [d1, d2])
                        self.y_indice_to_depth_interpolator[nr] = I

                if self.has_ranges:
                    r1, r2 = self.min_ranges[nr], self.max_ranges[nr]
                    if np.isfinite(r1) and np.isfinite(r2) and r1 < r2:
                        I = tools.vectorinterpolators.LinearInterpolatorF([r1, r2], [y1, y2])
                        self.range_to_y_coordinate_interpolator[nr] = I
                        I = tools.vectorinterpolators.LinearInterpolatorF([0, n_samples - 1], [r1, r2])
                        self.y_indice_to_range_interpolator[nr] = I

                if self.has_sample_nrs:
                    s1, s2 = self.min_sample_nrs[nr], self.max_sample_nrs[nr]
                    if np.isfinite(s1) and np.isfinite(s2) and s1 < s2:
                        I = tools.vectorinterpolators.LinearInterpolatorF([s1, s2], [y1, y2])
                        self.sample_nr_to_y_coordinate_interpolator[nr] = I
                        I = tools.vectorinterpolators.LinearInterpolatorF([0, n_samples - 1], [s1, s2])
                        self.y_indice_to_sample_nr_interpolator[nr] = I

            except Exception as e:
                message = f"{e}\n- nr {nr}\n- y1 {y1}\n- y2 {y2}\n- n_samples {n_samples}"
                raise RuntimeError(message)

    def _set_x_coordinates(self, name: str, x_coordinates: np.ndarray, x_interpolation_limit: float):
        """Set X coordinates for the display grid.
        
        Args:
            name: Axis name ('Ping index', 'Ping time', 'Date time').
            x_coordinates: Array of X coordinate values.
            x_interpolation_limit: Max distance for interpolation.
        """
        if len(x_coordinates) < 2:
            raise RuntimeError("ERROR: x_coordinates must contain at least two values")
        
        self.x_axis_name = name
        
        x_resolution = x_coordinates[1] - x_coordinates[0]
        
        self.x_interpolation_limit = x_interpolation_limit
        self.x_extent = [
            x_coordinates[0] - x_resolution / 2,
            x_coordinates[-1] + x_resolution / 2,
        ]

        self.feature_mapper.set_feature("X coordinate", x_coordinates)

    # =========================================================================
    # Y-axis setters
    # =========================================================================

    def set_y_axis_y_indice(
        self,
        min_sample_nr: float = 0,
        max_sample_nr: float = np.nan,
        max_steps: int = 1024,
        layer_update_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """Set Y axis to sample indices.
        
        Args:
            min_sample_nr: Minimum sample index to display.
            max_sample_nr: Maximum sample index to display (nan = auto).
            max_steps: Maximum number of Y pixels.
            layer_update_callback: Callback to update layers after axis change.
        """
        n_pings = self._n_pings
        vec_min_y = np.zeros(n_pings).astype(np.float32)
        vec_max_y = self.max_number_of_samples.astype(np.float32)

        y_kwargs = {"min_sample_nr": min_sample_nr, "max_sample_nr": max_sample_nr, "max_steps": max_steps}
        self._y_axis_function = self.set_y_axis_y_indice
        if self.y_axis_name == "Y indice" and self._y_kwargs == y_kwargs:
            return
        self._y_kwargs = y_kwargs

        y_coordinates = theping.algorithms.gridding.functions.compute_resampled_coordinates(
            values_min=vec_min_y,
            values_max=vec_max_y,
            values_res=np.ones(n_pings).astype(np.float32),
            grid_min=min_sample_nr,
            grid_max=max_sample_nr,
            max_steps=max_steps,
        )

        self._set_y_coordinates("Y indice", y_coordinates, vec_min_y, vec_max_y, layer_update_callback)
        self._initialized = True

    def set_y_axis_depth(
        self,
        min_depth: float = np.nan,
        max_depth: float = np.nan,
        max_steps: int = 1024,
        layer_update_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """Set Y axis to depth in meters.
        
        Args:
            min_depth: Minimum depth to display (nan = auto).
            max_depth: Maximum depth to display (nan = auto).
            max_steps: Maximum number of Y pixels.
            layer_update_callback: Callback to update layers after axis change.
        """
        assert self.has_depths, "ERROR: Depths values not initialized, call set_depth_extent first"

        y_kwargs = {"min_depth": min_depth, "max_depth": max_depth, "max_steps": max_steps}
        self._y_axis_function = self.set_y_axis_depth
        if self.y_axis_name == "Depth (m)" and self._y_kwargs == y_kwargs:
            return
        self._y_kwargs = y_kwargs

        y_coordinates = theping.algorithms.gridding.functions.compute_resampled_coordinates(
            values_min=self.min_depths,
            values_max=self.max_depths,
            values_res=self.res_depths,
            grid_min=min_depth,
            grid_max=max_depth,
            max_steps=max_steps,
        )

        self._set_y_coordinates("Depth (m)", y_coordinates, self.min_depths, self.max_depths, layer_update_callback)
        self._initialized = True

    def set_y_axis_range(
        self,
        min_range: float = np.nan,
        max_range: float = np.nan,
        max_steps: int = 1024,
        layer_update_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """Set Y axis to range in meters.
        
        Args:
            min_range: Minimum range to display (nan = auto).
            max_range: Maximum range to display (nan = auto).
            max_steps: Maximum number of Y pixels.
            layer_update_callback: Callback to update layers after axis change.
        """
        assert self.has_ranges, "ERROR: Range values not initialized, call set_range_extent first"

        y_kwargs = {"min_range": min_range, "max_range": max_range, "max_steps": max_steps}
        self._y_axis_function = self.set_y_axis_range
        if self.y_axis_name == "Range (m)" and self._y_kwargs == y_kwargs:
            return
        self._y_kwargs = y_kwargs

        y_coordinates = theping.algorithms.gridding.functions.compute_resampled_coordinates(
            values_min=self.min_ranges,
            values_max=self.max_ranges,
            values_res=self.res_ranges,
            grid_min=min_range,
            grid_max=max_range,
            max_steps=max_steps,
        )

        self._set_y_coordinates("Range (m)", y_coordinates, self.min_ranges, self.max_ranges, layer_update_callback)
        self._initialized = True

    def set_y_axis_sample_nr(
        self,
        min_sample_nr: float = 0,
        max_sample_nr: float = np.nan,
        max_steps: int = 1024,
        layer_update_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """Set Y axis to sample numbers.
        
        Args:
            min_sample_nr: Minimum sample number to display.
            max_sample_nr: Maximum sample number to display (nan = auto).
            max_steps: Maximum number of Y pixels.
            layer_update_callback: Callback to update layers after axis change.
        """
        assert self.has_sample_nrs, "ERROR: Sample nr values not initialized, call set_sample_nr_extent first"

        y_kwargs = {"min_sample_nr": min_sample_nr, "max_sample_nr": max_sample_nr, "max_steps": max_steps}
        self._y_axis_function = self.set_y_axis_sample_nr
        if self.y_axis_name == "Sample number" and self._y_kwargs == y_kwargs:
            return
        self._y_kwargs = y_kwargs

        y_coordinates = theping.algorithms.gridding.functions.compute_resampled_coordinates(
            values_min=self.min_sample_nrs,
            values_max=self.max_sample_nrs,
            values_res=self.res_sample_nrs,
            grid_min=min_sample_nr,
            grid_max=max_sample_nr,
            max_steps=max_steps,
        )

        self._set_y_coordinates("Sample number", y_coordinates, self.min_sample_nrs, self.max_sample_nrs, layer_update_callback)
        self._initialized = True

    # =========================================================================
    # X-axis setters
    # =========================================================================

    def set_x_axis_ping_index(
        self,
        min_ping_index: float = 0,
        max_ping_index: float = np.nan,
        max_steps: int = 4096,
        **kwargs,
    ):
        """Set X axis to ping index.
        
        Args:
            min_ping_index: Minimum ping index to display.
            max_ping_index: Maximum ping index to display (nan = auto).
            max_steps: Maximum number of X pixels.
        """
        x_kwargs = {
            "min_ping_index": min_ping_index,
            "max_ping_index": max_ping_index,
            "max_steps": max_steps,
        }

        self._x_axis_function = self.set_x_axis_ping_index
        if self.x_axis_name == "Ping index" and self._x_kwargs == x_kwargs:
            return

        self._x_kwargs = x_kwargs

        if not np.isfinite(max_ping_index):
            max_ping_index = np.max(self.ping_numbers)

        n_pings = self._n_pings
        npings = int(max_ping_index - min_ping_index) + 1

        if npings > n_pings:
            npings = n_pings

        if npings > max_steps:
            npings = max_steps

        x_coordinates = np.linspace(min_ping_index, max_ping_index, npings)

        self._set_x_coordinates("Ping index", x_coordinates, 1)
        self._initialized = True

    def set_x_axis_ping_time(
        self,
        min_timestamp: float = np.nan,
        max_timestamp: float = np.nan,
        time_resolution: float = np.nan,
        time_interpolation_limit: float = np.nan,
        max_steps: int = 4096,
        **kwargs,
    ):
        """Set X axis to ping time (Unix timestamp).
        
        Args:
            min_timestamp: Minimum timestamp to display (nan = auto).
            max_timestamp: Maximum timestamp to display (nan = auto).
            time_resolution: Time resolution in seconds (nan = auto).
            time_interpolation_limit: Max time gap for interpolation (nan = auto).
            max_steps: Maximum number of X pixels.
        """
        x_kwargs = {
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
            "time_resolution": time_resolution,
            "time_interpolation_limit": time_interpolation_limit,
            "max_steps": max_steps,
        }

        self._x_axis_function = self.set_x_axis_ping_time
        if self.x_axis_name == "Ping time" and self._x_kwargs == x_kwargs:
            return

        self._x_kwargs = x_kwargs

        if not np.isfinite(min_timestamp):
            min_timestamp = np.min(self.ping_times)

        if not np.isfinite(max_timestamp):
            max_timestamp = np.max(self.ping_times)

        ping_delta_t = np.array(self.ping_times[1:] - self.ping_times[:-1])
        if len(ping_delta_t[ping_delta_t < 0]) > 0:
            raise RuntimeError("ERROR: ping times are not sorted in ascending order!")

        # Handle zero time differences
        zero_time_diff = np.where(abs(ping_delta_t) < 0.000001)[0]
        while len(zero_time_diff) > 0:
            self.ping_times[zero_time_diff + 1] += 0.0001
            ping_delta_t = np.array(self.ping_times[1:] - self.ping_times[:-1])
            zero_time_diff = np.where(abs(ping_delta_t) < 0.0001)[0]

        if not np.isfinite(time_resolution):
            time_resolution = np.nanquantile(ping_delta_t, 0.05)

        if not np.isfinite(time_interpolation_limit):
            time_interpolation_limit = np.nanquantile(ping_delta_t, 0.95)

        try:
            arange = False
            if (max_timestamp + time_resolution - min_timestamp) / time_resolution + 1 <= max_steps:
                x_coordinates = np.arange(min_timestamp, max_timestamp + time_resolution, time_resolution)
            else:
                arange = True

            if arange or len(x_coordinates) > max_steps:
                x_coordinates = np.linspace(min_timestamp, max_timestamp, max_steps)
        except Exception as e:
            message = f"{e}\n -min_timestamp: {min_timestamp}\n -max_timestamp: {max_timestamp}\n -time_resolution: {time_resolution}\n -max_steps: {max_steps}"
            raise RuntimeError(message)

        self._set_x_coordinates("Ping time", x_coordinates, time_interpolation_limit)
        self._initialized = True

    def set_x_axis_date_time(
        self,
        min_ping_time: float = np.nan,
        max_ping_time: float = np.nan,
        time_resolution: float = np.nan,
        time_interpolation_limit: float = np.nan,
        max_steps: int = 4096,
        **kwargs,
    ):
        """Set X axis to datetime.
        
        Args:
            min_ping_time: Minimum time (timestamp or datetime, nan = auto).
            max_ping_time: Maximum time (timestamp or datetime, nan = auto).
            time_resolution: Time resolution (seconds or timedelta, nan = auto).
            time_interpolation_limit: Max time gap (seconds or timedelta, nan = auto).
            max_steps: Maximum number of X pixels.
        """
        x_kwargs = {
            "min_ping_time": min_ping_time,
            "max_ping_time": max_ping_time,
            "time_resolution": time_resolution,
            "time_interpolation_limit": time_interpolation_limit,
            "max_steps": max_steps,
        }

        self._x_axis_function = self.set_x_axis_date_time
        if self.x_axis_name == "Date time" and self._x_kwargs == x_kwargs:
            return

        if isinstance(min_ping_time, dt.datetime):
            min_ping_time = min_ping_time.timestamp()

        if isinstance(max_ping_time, dt.datetime):
            max_ping_time = max_ping_time.timestamp()

        if isinstance(time_resolution, dt.timedelta):
            time_resolution = time_resolution.total_seconds()

        if isinstance(time_interpolation_limit, dt.timedelta):
            time_interpolation_limit = time_interpolation_limit.total_seconds()

        self.set_x_axis_ping_time(
            min_timestamp=min_ping_time,
            max_timestamp=max_ping_time,
            time_resolution=time_resolution,
            time_interpolation_limit=time_interpolation_limit,
            max_steps=max_steps,
        )

        self.x_extent[0] = dt.datetime.fromtimestamp(self.x_extent[0], self.time_zone)
        self.x_extent[1] = dt.datetime.fromtimestamp(self.x_extent[1], self.time_zone)
        self.x_axis_name = "Date time"
        self._x_axis_function = self.set_x_axis_date_time
        self._x_kwargs = x_kwargs

    # =========================================================================
    # Index mapping
    # =========================================================================

    def get_y_indices(self, wci_nr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get Y indices mapping image coordinates to data indices.
        
        Args:
            wci_nr: Ping/column number.
            
        Returns:
            Tuple of (image_indices, data_indices) arrays.
        """
        interpolator = self.y_coordinate_indice_interpolator[wci_nr]
        if interpolator is None:
            return np.array([], dtype=int), np.array([], dtype=int)
            
        n_samples = int(self.max_number_of_samples[wci_nr]) + 1
        y_indices_image = np.arange(len(self.y_coordinates))
        y_indices_wci = np.round(interpolator(self.y_coordinates)).astype(int)

        valid_coordinates = np.where(np.logical_and(y_indices_wci >= 0, y_indices_wci < n_samples))[0]

        return y_indices_image[valid_coordinates], y_indices_wci[valid_coordinates]

    def get_x_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get X indices mapping image coordinates to ping indices.
        
        Returns:
            Tuple of (image_indices, ping_indices) arrays.
        """
        x_coordinates = np.array(self.feature_mapper.get_feature_values("X coordinate"))
        vec_x_val = np.array(self.feature_mapper.get_feature_values(self.x_axis_name))
        
        image_index = np.array(self.feature_mapper.get_feature_indices("X coordinate"))
        wci_index = self.feature_mapper.feature_to_index(
            self.x_axis_name, x_coordinates,
            mp_cores=self.mp_cores
        )

        delta_x = np.abs(vec_x_val[wci_index] - x_coordinates)
        valid = np.where(delta_x < self.x_interpolation_limit)[0]
        return image_index[valid], wci_index[valid]

    # =========================================================================
    # Axis copying
    # =========================================================================

    def copy_xy_axis_to(self, other: "EchogramCoordinateSystem"):
        """Copy X/Y axis settings to another coordinate system.
        
        Args:
            other: Target coordinate system.
        """
        match self.x_axis_name:
            case "Date time":
                other.set_x_axis_date_time(**self._x_kwargs)
            case "Ping time":
                other.set_x_axis_ping_time(**self._x_kwargs)
            case "Ping index":
                other.set_x_axis_ping_index(**self._x_kwargs)
            case _:
                raise RuntimeError(f"ERROR: unknown x axis name '{self.x_axis_name}'")

        match self.y_axis_name:
            case "Depth (m)":
                other.set_y_axis_depth(**self._y_kwargs)
            case "Range (m)":
                other.set_y_axis_range(**self._y_kwargs)
            case "Sample number":
                other.set_y_axis_sample_nr(**self._y_kwargs)
            case "Y indice":
                other.set_y_axis_y_indice(**self._y_kwargs)
            case _:
                raise RuntimeError(f"ERROR: unknown y axis name '{self.y_axis_name}'")

    # =========================================================================
    # Image request generation
    # =========================================================================

    def _estimate_affine_y_to_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate affine parameters for y→sample index mapping per ping.
        
        The y_coordinate_indice_interpolator is a LinearInterpolator, so
        it's exactly: sample_idx = a + b * y
        
        Returns:
            Tuple of (a, b) arrays, each shape (n_pings,).
            Values are NaN where interpolator is not defined.
        """
        n_pings = self._n_pings
        a = np.full(n_pings, np.nan, dtype=np.float32)
        b = np.full(n_pings, np.nan, dtype=np.float32)
        
        y = np.asarray(self.y_coordinates, dtype=np.float32)
        if y.size < 2:
            return a, b
        
        # Pick two well-separated points for numerical stability
        y0 = float(y[0])
        y1 = float(y[-1])
        if y1 == y0:
            y1 = y0 + 1.0
        
        for p in range(n_pings):
            interpolator = self.y_coordinate_indice_interpolator[p]
            if interpolator is None:
                continue
            
            # Evaluate at two points to extract slope and intercept
            i0 = float(interpolator(y0))
            i1 = float(interpolator(y1))
            
            bp = (i1 - i0) / (y1 - y0)
            ap = i0 - bp * y0
            
            a[p] = ap
            b[p] = bp
        
        return a, b

    def make_image_request(self) -> EchogramImageRequest:
        """Create a backend-ready request for building the current image.
        
        This method generates all the indexing information needed for a backend
        to produce a downsampled (nx, ny) echogram image without needing to
        know about the coordinate system internals.
        
        The request includes:
        - ping_indexer: which ping to use for each output x column
        - affine params (a, b): for computing sample indices from y coordinates
        - max_sample_indices: for bounds checking
        
        Returns:
            EchogramImageRequest with all necessary indexing information.
        """
        self.reinit()
        
        # Get x mapping
        x_coords = np.array(self.feature_mapper.get_feature_values("X coordinate"))
        nx = len(x_coords)
        
        image_indices, wci_indices = self.get_x_indices()
        
        # Build dense ping indexer: -1 means no valid ping
        ping_indexer = np.full(nx, -1, dtype=np.int64)
        ping_indexer[np.asarray(image_indices, dtype=np.int64)] = np.asarray(wci_indices, dtype=np.int64)
        
        # Get affine params
        affine_a, affine_b = self._estimate_affine_y_to_sample()
        
        # Get y coordinates
        y_coords = np.asarray(self.y_coordinates, dtype=np.float32)
        ny = len(y_coords)
        
        # Max sample indices for bounds checking
        max_sample_idx = self.max_number_of_samples.astype(np.int64) + 1
        
        return EchogramImageRequest(
            nx=nx,
            ny=ny,
            y_coordinates=y_coords,
            ping_indexer=ping_indexer,
            affine_a=affine_a,
            affine_b=affine_b,
            max_sample_indices=max_sample_idx,
            fill_value=np.nan,
        )
