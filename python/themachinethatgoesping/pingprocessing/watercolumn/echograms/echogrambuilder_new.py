import numpy as np

from typing import Optional, TYPE_CHECKING
from copy import deepcopy
import datetime as dt

import warnings

# external Ping packages
import themachinethatgoesping as theping
from themachinethatgoesping import echosounders
from themachinethatgoesping import tools

# internal Ping.pingprocessing packages
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.pingprocessing.core.asserts import assert_length, assert_valid_argument
from themachinethatgoesping.algorithms.gridding import ForwardGridder1D

from themachinethatgoesping.algorithms_nanopy.featuremapping import NearestFeatureMapper

# backends
from .backends import EchogramDataBackend, PingDataBackend

# subpackages
from .layers.echolayer import EchoLayer, PingData

if TYPE_CHECKING:
    from .backends import EchogramDataBackend


class EchogramBuilder:
    """Builder for echogram images with coordinate system and layer management.
    
    The EchogramBuilder controls:
    - Coordinate systems (x-axis: ping index/time/datetime, y-axis: sample/range/depth)
    - Sampling/viewing parameters (max_steps, extents)
    - Layer management for region selection
    - Image building using data from a backend
    
    Data access is delegated to an EchogramDataBackend, allowing different
    data sources (pings, Zarr, NetCDF, etc.).
    """

    def __init__(
        self,
        backend: "EchogramDataBackend",
        ping_numbers: Optional[np.ndarray] = None,
    ):
        """Initialize EchogramBuilder with a data backend.
        
        Args:
            backend: Data backend providing access to echogram data.
            ping_numbers: Optional array of ping numbers. If None, uses 0..n_pings-1.
        """
        if backend.n_pings == 0:
            raise RuntimeError("ERROR[EchogramBuilder]: trying to initialize with empty data (no valid pings)")

        self._backend = backend
        self.feature_mapper = NearestFeatureMapper()

        self.layers = {}
        self.main_layer = None
        
        # Initialize from backend metadata
        self.max_number_of_samples = backend.max_sample_counts
        
        # Set up ping numbers and times
        if ping_numbers is None:
            ping_numbers = np.arange(backend.n_pings)
        self.set_ping_numbers(ping_numbers)
        self.set_ping_times(backend.ping_times)

        self.param = {}

        # Initialize extent flags
        self.has_ranges = backend.range_extents is not None
        self.has_depths = backend.has_navigation
        self.has_sample_nrs = True  # Always have sample numbers
        
        # Set extents from backend
        min_s, max_s = backend.sample_nr_extents
        self._set_sample_nr_extent_internal(min_s, max_s)
        
        if backend.range_extents is not None:
            min_r, max_r = backend.range_extents
            self._set_range_extent_internal(min_r, max_r)
            
        if backend.depth_extents is not None:
            min_d, max_d = backend.depth_extents
            self._set_depth_extent_internal(min_d, max_d)

        # Initialize ping params from backend
        self._init_ping_params_from_backend()

        # Set default axes
        self.x_axis_name = None
        self.y_axis_name = None
        self.set_y_axis_y_indice()
        self.set_x_axis_ping_index()
        self.initialized = True
        
        self.mp_cores = 1

    def _init_ping_params_from_backend(self):
        """Initialize ping parameters from backend's pre-computed values."""
        ping_params = self._backend.get_ping_params()
        for name, (y_reference, (times, values)) in ping_params.items():
            self.add_ping_param(name, "Ping time", y_reference, times, values)

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def from_pings(
        cls,
        pings,
        pss=None,
        wci_value: str = "sv/av/pv/rv",
        linear_mean: bool = True,
        no_navigation: bool = False,
        apply_pss_to_bottom: bool = False,
        force_angle: Optional[float] = None,
        depth_stack: bool = False,
        verbose: bool = True,
        mp_cores: int = 1,
    ) -> "EchogramBuilder":
        """Create an EchogramBuilder from a list of pings.
        
        Args:
            pings: List of ping objects.
            pss: PingSampleSelector for beam/sample selection. If None, uses default.
            wci_value: Water column image value type (e.g., 'sv', 'av', 'sv/av/pv/rv').
            linear_mean: Whether to use linear mean for beam averaging.
            no_navigation: If True, skip depth calculations.
            apply_pss_to_bottom: Whether to apply PSS to bottom detection.
            force_angle: Force a specific angle for depth projection (degrees).
            depth_stack: If True, use depth stacking mode (requires navigation).
            verbose: Whether to show progress bar.
            mp_cores: Number of cores for parallel processing.
            
        Returns:
            EchogramBuilder instance.
        """
        if pss is None:
            pss = echosounders.pingtools.PingSampleSelector()
            
        backend = PingDataBackend.from_pings(
            pings=pings,
            pss=pss,
            wci_value=wci_value,
            linear_mean=linear_mean,
            no_navigation=no_navigation,
            apply_pss_to_bottom=apply_pss_to_bottom,
            force_angle=force_angle,
            depth_stack=depth_stack,
            verbose=verbose,
            mp_cores=mp_cores,
        )
        
        builder = cls(backend=backend)
        builder.verbose = verbose
        builder.mp_cores = mp_cores
        return builder

    @classmethod
    def from_backend(
        cls,
        backend: "EchogramDataBackend",
    ) -> "EchogramBuilder":
        """Create an EchogramBuilder from an existing backend.
        
        Args:
            backend: Data backend instance.
            
        Returns:
            EchogramBuilder instance.
        """
        return cls(backend=backend)

    # =========================================================================
    # Backend access (for backward compatibility and advanced use)
    # =========================================================================

    @property
    def backend(self) -> "EchogramDataBackend":
        """Access the data backend."""
        return self._backend

    @property
    def pings(self):
        """Direct access to pings (for backward compatibility).
        
        Only works with PingDataBackend.
        """
        if hasattr(self._backend, 'pings'):
            return self._backend.pings
        raise AttributeError("Backend does not provide direct ping access")

    @property
    def beam_sample_selections(self):
        """Direct access to beam sample selections (for backward compatibility).
        
        Only works with PingDataBackend.
        """
        if hasattr(self._backend, 'beam_sample_selections'):
            return self._backend.beam_sample_selections
        raise AttributeError("Backend does not provide beam sample selections")

    @property
    def wci_value(self) -> str:
        """Water column image value type from backend."""
        return self._backend.wci_value

    @property
    def linear_mean(self) -> bool:
        """Whether linear mean is used for beam averaging."""
        return self._backend.linear_mean

    # =========================================================================
    # Data access methods (delegate to backend)
    # =========================================================================

    def get_column(self, nr):
        """Get column data for a ping from backend."""
        return self._backend.get_column(nr)

    # =========================================================================
    # Ping numbers and times
    # =========================================================================

    def set_ping_numbers(self, ping_numbers):
        """Set ping numbers for x-axis indexing."""
        assert len(ping_numbers) == self._backend.n_pings, \
            f"ping_numbers length ({len(ping_numbers)}) must match n_pings ({self._backend.n_pings})"
        self.feature_mapper.set_feature("Ping index", ping_numbers)
        self.ping_numbers = ping_numbers
        self.initialized = False

    def set_ping_times(self, ping_times, time_zone=dt.timezone.utc):
        """Set ping times for x-axis time display."""
        assert len(ping_times) == self._backend.n_pings, \
            f"ping_times length ({len(ping_times)}) must match n_pings ({self._backend.n_pings})"
        self.feature_mapper.set_feature("Ping time", ping_times)
        self.feature_mapper.set_feature("Date time", ping_times)
        self.ping_times = ping_times
        self.time_zone = time_zone
        self.initialized = False

    # =========================================================================
    # Extent setters (internal, from backend)
    # =========================================================================

    def _set_range_extent_internal(self, min_ranges, max_ranges):
        """Set range extents from arrays (internal use)."""
        self.min_ranges = np.asarray(min_ranges, dtype=np.float32)
        self.max_ranges = np.asarray(max_ranges, dtype=np.float32)
        self.res_ranges = ((self.max_ranges - self.min_ranges) / self.max_number_of_samples).astype(np.float32)
        self.has_ranges = True
        self.initialized = False

    def _set_depth_extent_internal(self, min_depths, max_depths):
        """Set depth extents from arrays (internal use)."""
        self.min_depths = np.asarray(min_depths, dtype=np.float32)
        self.max_depths = np.asarray(max_depths, dtype=np.float32)
        self.res_depths = ((self.max_depths - self.min_depths) / self.max_number_of_samples).astype(np.float32)
        self.has_depths = True
        self.initialized = False

    def _set_sample_nr_extent_internal(self, min_sample_nrs, max_sample_nrs):
        """Set sample number extents from arrays (internal use)."""
        self.min_sample_nrs = np.asarray(min_sample_nrs, dtype=np.float32)
        self.max_sample_nrs = np.asarray(max_sample_nrs, dtype=np.float32)
        self.res_sample_nrs = ((self.max_sample_nrs - self.min_sample_nrs) / self.max_number_of_samples).astype(np.float32)
        self.has_sample_nrs = True
        self.initialized = False

    # Public extent setters (for backward compatibility)
    def set_range_extent(self, min_ranges, max_ranges):
        """Set range extents."""
        assert len(min_ranges) == len(max_ranges) == self._backend.n_pings
        self._set_range_extent_internal(min_ranges, max_ranges)

    def set_depth_extent(self, min_depths, max_depths):
        """Set depth extents."""
        assert len(min_depths) == len(max_depths) == self._backend.n_pings
        self._set_depth_extent_internal(min_depths, max_depths)

    def set_sample_nr_extent(self, min_sample_nrs, max_sample_nrs):
        """Set sample number extents."""
        assert len(min_sample_nrs) == len(max_sample_nrs) == self._backend.n_pings
        self._set_sample_nr_extent_internal(min_sample_nrs, max_sample_nrs)

    # =========================================================================
    # Ping parameters
    # =========================================================================

    def add_ping_param(self, name, x_reference, y_reference, vec_x_val, vec_y_val):
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
            case "Ping time":
                comp_vec_x_val = self.ping_times
            case "Date time":
                comp_vec_x_val = self.ping_times

        # average vec_y_val for all double vec_x_vals
        unique_x_vals, indices = np.unique(vec_x_val, return_inverse=True)
        averaged_y_vals = np.zeros(len(unique_x_vals))
        counts = np.bincount(indices)
        for i in range(len(unique_x_vals)):
            start_index = np.where(indices == i)[0][0]
            end_index = start_index + counts[i]
            averaged_y_vals[i] = np.mean(vec_y_val[start_index:end_index])

        vec_x_val = unique_x_vals
        vec_y_val = averaged_y_vals

        # convert to to represent indices
        vec_y_val = tools.vectorinterpolators.LinearInterpolator(vec_x_val, vec_y_val, extrapolation_mode="nearest")(
            comp_vec_x_val
        )

        self.param[name] = y_reference, vec_y_val

    def get_ping_param(self, name, use_x_coordinates=False):
        """Get a ping parameter's values in current coordinate system."""
        self.reinit()
        assert name in self.param.keys(), f"ERROR[get_ping_param]: name '{name}' not registered"
        if use_x_coordinates:
            x_coordinates = np.array(self.feature_mapper.get_feature_values("X coordinate"))
            x_indices = np.array(self.feature_mapper.feature_to_index(self.x_axis_name, x_coordinates, mp_cores=self.mp_cores))
        else:
            x_indices = np.arange(self._backend.n_pings)
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
                assert self.has_sample_nrs, \
                    "ERROR: Sample nr values not initialized, call set_sample_nr_extent method"

                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.sample_nr_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case "Depth (m)":
                assert self.has_depths, \
                    "ERROR: Depths values not initialized, call set_depth_extent method"

                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.depth_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case "Range (m)":
                assert self.has_ranges, \
                    "ERROR: Ranges values not initialized, call set_range_extent method"

                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.range_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case _:
                raise RuntimeError(f"Invalid reference '{reference}'. This should not happen, please report")

        if self.x_axis_name == "Date time":
            x_coordinates = [dt.datetime.fromtimestamp(t, self.time_zone) for t in x_coordinates]

        return x_coordinates, return_param

    def get_x_kwargs(self):
        return deepcopy(self.__x_kwargs)

    def get_y_kwargs(self):
        return deepcopy(self.__y_kwargs)

    # =========================================================================
    # Coordinate system management
    # =========================================================================

    def reinit(self):
        """Reinitialize coordinate systems if needed."""
        if self.initialized:
            return
        self.y_axis_function(**self.__y_kwargs)
        self.x_axis_function(**self.__x_kwargs)

    def set_y_coordinates(self, name, y_coordinates, vec_min_y, vec_max_y):
        """Set Y coordinates for the display grid."""
        n_pings = self._backend.n_pings
        assert len(vec_min_y) == len(vec_max_y) == n_pings, \
            f"ERROR min/max y vectors must have the same length as n_pings ({n_pings})"
            
        self.y_axis_name = name
        self.y_coordinates = y_coordinates
        self.y_resolution = y_coordinates[1] - y_coordinates[0]
        self.y_extent = [
            self.y_coordinates[-1] + self.y_resolution / 2,
            self.y_coordinates[0] - self.y_resolution / 2,
        ]
        self.vec_min_y = vec_min_y
        self.vec_max_y = vec_max_y

        self.y_gridder = ForwardGridder1D.from_res(self.y_resolution, self.y_coordinates[0], self.y_coordinates[-1])
        if self.main_layer is not None:
            self.main_layer.update_y_gridder()
        for layer in self.layers.values():
            layer.update_y_gridder()

        # Initialize interpolators
        self.y_coordinate_indice_interpolator = [None for _ in range(n_pings)]
        self.y_indice_to_y_coordinate_interpolator = [None for _ in range(n_pings)]
        self.depth_to_y_coordinate_interpolator = [None for _ in range(n_pings)]
        self.range_to_y_coordinate_interpolator = [None for _ in range(n_pings)]
        self.sample_nr_to_y_coordinate_interpolator = [None for _ in range(n_pings)]
        self.y_indice_to_depth_interpolator = [None for _ in range(n_pings)]
        self.y_indice_to_range_interpolator = [None for _ in range(n_pings)]
        self.y_indice_to_sample_nr_interpolator = [None for _ in range(n_pings)]

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
                message = f"{e}\n- nr {nr}\n- y1 {y1}\n -y2 {y2}\n -n_samples {n_samples}"
                raise RuntimeError(message)

    def set_x_coordinates(self, name, x_coordinates, x_interpolation_limit):
        """Set X coordinates for the display grid."""
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
        min_sample_nr=0,
        max_sample_nr=np.nan,
        max_steps=1024,
        **kwargs,
    ):
        """Set Y axis to sample indices."""
        n_pings = self._backend.n_pings
        vec_min_y = np.zeros(n_pings).astype(np.float32)
        vec_max_y = self.max_number_of_samples.astype(np.float32)

        y_kwargs = {"min_sample_nr": min_sample_nr, "max_sample_nr": max_sample_nr, "max_steps": max_steps}
        self.y_axis_function = self.set_y_axis_y_indice
        if self.y_axis_name == "Y indice" and self.__y_kwargs == y_kwargs:
            return
        self.__y_kwargs = y_kwargs

        y_coordinates = theping.algorithms.gridding.functions.compute_resampled_coordinates(
            values_min=vec_min_y,
            values_max=vec_max_y,
            values_res=np.ones(n_pings).astype(np.float32),
            grid_min=min_sample_nr,
            grid_max=max_sample_nr,
            max_steps=max_steps,
        )

        self.set_y_coordinates("Y indice", y_coordinates, vec_min_y, vec_max_y)

    def set_y_axis_depth(
        self,
        min_depth=np.nan,
        max_depth=np.nan,
        max_steps=1024,
        **kwargs,
    ):
        """Set Y axis to depth in meters."""
        assert self.has_depths, "ERROR: Depths values not initialized, call set_depth_extent method"

        y_kwargs = {"min_depth": min_depth, "max_depth": max_depth, "max_steps": max_steps}
        self.y_axis_function = self.set_y_axis_depth
        if self.y_axis_name == "Depth (m)" and self.__y_kwargs == y_kwargs:
            return
        self.__y_kwargs = y_kwargs

        y_coordinates = theping.algorithms.gridding.functions.compute_resampled_coordinates(
            values_min=self.min_depths,
            values_max=self.max_depths,
            values_res=self.res_depths,
            grid_min=min_depth,
            grid_max=max_depth,
            max_steps=max_steps,
        )

        self.set_y_coordinates("Depth (m)", y_coordinates, self.min_depths, self.max_depths)

    def set_y_axis_range(
        self,
        min_range=np.nan,
        max_range=np.nan,
        max_steps=1024,
        **kwargs,
    ):
        """Set Y axis to range in meters."""
        assert self.has_ranges, "ERROR: Range values not initialized, call set_range_extent method"

        y_kwargs = {"min_range": min_range, "max_range": max_range, "max_steps": max_steps}
        self.y_axis_function = self.set_y_axis_range
        if self.y_axis_name == "Range (m)" and self.__y_kwargs == y_kwargs:
            return
        self.__y_kwargs = y_kwargs

        y_coordinates = theping.algorithms.gridding.functions.compute_resampled_coordinates(
            values_min=self.min_ranges,
            values_max=self.max_ranges,
            values_res=self.res_ranges,
            grid_min=min_range,
            grid_max=max_range,
            max_steps=max_steps,
        )

        self.set_y_coordinates("Range (m)", y_coordinates, self.min_ranges, self.max_ranges)

    def set_y_axis_sample_nr(
        self,
        min_sample_nr=0,
        max_sample_nr=np.nan,
        max_steps=1024,
        **kwargs,
    ):
        """Set Y axis to sample numbers."""
        assert self.has_sample_nrs, \
            "ERROR: Sample nr values not initialized, call set_sample_nr_extent method"

        y_kwargs = {"min_sample_nr": min_sample_nr, "max_sample_nr": max_sample_nr, "max_steps": max_steps}
        self.y_axis_function = self.set_y_axis_sample_nr
        if self.y_axis_name == "Sample number" and self.__y_kwargs == y_kwargs:
            return
        self.__y_kwargs = y_kwargs

        y_coordinates = theping.algorithms.gridding.functions.compute_resampled_coordinates(
            values_min=self.min_sample_nrs,
            values_max=self.max_sample_nrs,
            values_res=self.res_sample_nrs,
            grid_min=min_sample_nr,
            grid_max=max_sample_nr,
            max_steps=max_steps,
        )

        self.set_y_coordinates("Sample number", y_coordinates, self.min_sample_nrs, self.max_sample_nrs)

    # =========================================================================
    # X-axis setters
    # =========================================================================

    def set_x_axis_ping_index(
        self,
        min_ping_index=0,
        max_ping_index=np.nan,
        max_steps=4096,
        **kwargs,
    ):
        """Set X axis to ping index."""
        x_kwargs = {
            "min_ping_index": min_ping_index,
            "max_ping_index": max_ping_index,
            "max_steps": max_steps,
        }

        self.x_axis_function = self.set_x_axis_ping_index
        if self.x_axis_name == "Ping index" and self.__x_kwargs == x_kwargs:
            return

        self.__x_kwargs = x_kwargs

        if not np.isfinite(max_ping_index):
            max_ping_index = np.max(self.ping_numbers)

        n_pings = self._backend.n_pings
        npings = int(max_ping_index - min_ping_index) + 1

        if npings > n_pings:
            npings = n_pings

        if npings > max_steps:
            npings = max_steps

        x_coordinates = np.linspace(min_ping_index, max_ping_index, npings)

        self.set_x_coordinates("Ping index", x_coordinates, 1)

    def set_x_axis_ping_time(
        self,
        min_timestamp=np.nan,
        max_timestamp=np.nan,
        time_resolution=np.nan,
        time_interpolation_limit=np.nan,
        max_steps=4096,
        **kwargs,
    ):
        """Set X axis to ping time (Unix timestamp)."""
        x_kwargs = {
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
            "time_resolution": time_resolution,
            "time_interpolation_limit": time_interpolation_limit,
            "max_steps": max_steps,
        }

        self.x_axis_function = self.set_x_axis_ping_time
        if self.x_axis_name == "Ping time" and self.__x_kwargs == x_kwargs:
            return

        self.__x_kwargs = x_kwargs

        if not np.isfinite(min_timestamp):
            min_timestamp = np.min(self.ping_times)

        if not np.isfinite(max_timestamp):
            max_timestamp = np.max(self.ping_times)

        ping_delta_t = np.array(self.ping_times[1:] - self.ping_times[:-1])
        if len(ping_delta_t[ping_delta_t < 0]) > 0:
            raise RuntimeError("ERROR: ping times are not sorted in ascending order!")

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

        self.set_x_coordinates("Ping time", x_coordinates, time_interpolation_limit)

    def set_x_axis_date_time(
        self,
        min_ping_time=np.nan,
        max_ping_time=np.nan,
        time_resolution=np.nan,
        time_interpolation_limit=np.nan,
        max_steps=4096,
        **kwargs,
    ):
        """Set X axis to datetime."""
        x_kwargs = {
            "min_ping_time": min_ping_time,
            "max_ping_time": max_ping_time,
            "time_resolution": time_resolution,
            "time_interpolation_limit": time_interpolation_limit,
            "max_steps": max_steps,
        }

        self.x_axis_function = self.set_x_axis_date_time
        if self.x_axis_name == "Date time" and self.__x_kwargs == x_kwargs:
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
        self.x_axis_function = self.set_x_axis_date_time
        self.__x_kwargs = x_kwargs

    def copy_xy_axis(self, other):
        """Copy X/Y axis settings to another EchogramBuilder."""
        match self.x_axis_name:
            case "Date time":
                other.set_x_axis_date_time(**self.__x_kwargs)
            case "Ping time":
                other.set_x_axis_ping_time(**self.__x_kwargs)
            case "Ping index":
                other.set_x_axis_ping_index(**self.__x_kwargs)
            case _:
                raise RuntimeError(f"ERROR: unknown x axis name '{self.x_axis_name}'")

        match self.y_axis_name:
            case "Depth (m)":
                other.set_y_axis_depth(**self.__y_kwargs)
            case "Range (m)":
                other.set_y_axis_range(**self.__y_kwargs)
            case "Sample number":
                other.set_y_axis_sample_nr(**self.__y_kwargs)
            case "Y indice":
                other.set_y_axis_y_indice(**self.__y_kwargs)
            case _:
                raise RuntimeError(f"ERROR: unknown y axis name '{self.y_axis_name}'")

    # =========================================================================
    # Index mapping
    # =========================================================================

    def get_y_indices(self, wci_nr):
        """Get Y indices mapping image coordinates to data indices.
        
        Returns:
            Tuple of (image_indices, data_indices) arrays.
        """
        # Handle pings with no valid data (interpolator is None)
        interpolator = self.y_coordinate_indice_interpolator[wci_nr]
        if interpolator is None:
            return np.array([], dtype=int), np.array([], dtype=int)
            
        n_samples = int(self.max_number_of_samples[wci_nr]) + 1
        y_indices_image = np.arange(len(self.y_coordinates))
        y_indices_wci = np.round(interpolator(self.y_coordinates)).astype(int)

        valid_coordinates = np.where(np.logical_and(y_indices_wci >= 0, y_indices_wci < n_samples))[0]

        return y_indices_image[valid_coordinates], y_indices_wci[valid_coordinates]

    def get_x_indices(self):
        """Get X indices mapping image coordinates to ping indices."""
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
    # Image building
    # =========================================================================

    def build_image(self, progress=None):
        """Build the echogram image.
        
        Returns:
            Tuple of (image, extent) where image is a 2D numpy array
            and extent is [x_min, x_max, y_max, y_min].
        """
        self.reinit()
        ny = len(self.y_coordinates)
        nx = len(self.feature_mapper.get_feature_values("X coordinate"))

        image = np.empty((nx, ny), dtype=np.float32)
        image.fill(np.nan)

        image_indices, wci_indices = self.get_x_indices()
        image_indices = get_progress_iterator(image_indices, progress, desc="Building echogram image")

        for image_index, wci_index in zip(image_indices, wci_indices):
            wci = self.get_column(wci_index)
            if len(wci) > 1:
                y1, y2 = self.get_y_indices(wci_index)
                if len(y1) > 0:
                    image[image_index, y1] = wci[y2]

        extent = deepcopy(self.x_extent)
        extent.extend(self.y_extent)

        return image, extent

    def build_image_and_layer_image(self, progress=None):
        """Build echogram image and combined layer image.
        
        Returns:
            Tuple of (image, layer_image, extent).
        """
        self.reinit()
        ny = len(self.y_coordinates)
        nx = len(self.feature_mapper.get_feature_values("X coordinate"))

        image = np.empty((nx, ny), dtype=np.float32)
        image.fill(np.nan)
        layer_image = image.copy()

        image_indices, wci_indices = self.get_x_indices()
        image_indices = get_progress_iterator(image_indices, progress, desc="Building echogram image")

        for image_index, wci_index in zip(image_indices, wci_indices):
            wci = self.get_column(wci_index)
            if len(wci) > 1:
                if self.main_layer is None:
                    y1, y2 = self.get_y_indices(wci_index)
                    if len(y1) > 0:
                        image[image_index, y1] = wci[y2]
                else:
                    y1, y2 = self.main_layer.get_y_indices(wci_index)
                    if y1 is not None and len(y1) > 0:
                        image[image_index, y1] = wci[y2]

                for k, layer in self.layers.items():
                    y1_layer, y2_layer = layer.get_y_indices(wci_index)
                    if y1_layer is not None and len(y1_layer) > 0:
                        layer_image[image_index, y1_layer] = wci[y2_layer]

        extent = deepcopy(self.x_extent)
        extent.extend(self.y_extent)

        return image, layer_image, extent

    def build_image_and_layer_images(self, progress=None):
        """Build echogram image and individual layer images.
        
        Returns:
            Tuple of (image, layer_images_dict, extent).
        """
        self.reinit()
        ny = len(self.y_coordinates)
        nx = len(self.feature_mapper.get_feature_values("X coordinate"))

        image = np.empty((nx, ny), dtype=np.float32)
        image.fill(np.nan)

        layer_images = {}
        for key in self.layers.keys():
            layer_images[key] = image.copy()

        image_indices, wci_indices = self.get_x_indices()
        image_indices = get_progress_iterator(image_indices, progress, desc="Building echogram image")

        for image_index, wci_index in zip(image_indices, wci_indices):
            wci = self.get_column(wci_index)
            if len(wci) > 1:
                if self.main_layer is None:
                    y1, y2 = self.get_y_indices(wci_index)
                    if len(y1) > 0:
                        image[image_index, y1] = wci[y2]
                else:
                    y1, y2 = self.main_layer.get_y_indices(wci_index)
                    if y1 is not None and len(y1) > 0:
                        image[image_index, y1] = wci[y2]

                for key, layer in self.layers.items():
                    y1_layer, y2_layer = layer.get_y_indices(wci_index)
                    if y1_layer is not None and len(y1_layer) > 0:
                        layer_images[key][image_index, y1_layer] = wci[y2_layer]

        extent = deepcopy(self.x_extent)
        extent.extend(self.y_extent)

        return image, layer_images, extent

    # =========================================================================
    # Layer management
    # =========================================================================

    def get_wci_layers(self, nr):
        """Get WCI data split by layers."""
        wci = self.get_column(nr)

        wci_layers = {}
        for key, layer in self.layers.items():
            wci_layers[key] = wci[layer.i0[nr] : layer.i1[nr]]

        return wci_layers

    def get_extent_layers(self, nr, axis_name=None):
        """Get extents for each layer at a given ping."""
        if axis_name is None:
            axis_name = self.y_axis_name
        extents = {}

        for key, layer in self.layers.items():
            match axis_name:
                case "Y indice":
                    extents[key] = layer.i0[nr] - 0.5, layer.i1[nr] - 0.5

                case "Sample number":
                    assert self.has_sample_nrs, \
                        "ERROR: Sample nr values not initialized"
                    extents[key] = self.y_indice_to_sample_nr_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case "Depth (m)":
                    assert self.has_depths, \
                        "ERROR: Depths values not initialized"
                    extents[key] = self.y_indice_to_depth_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case "Range (m)":
                    assert self.has_ranges, \
                        "ERROR: Ranges values not initialized"
                    extents[key] = self.y_indice_to_range_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case _:
                    raise RuntimeError(f"Invalid axis_name '{axis_name}'")

        return extents

    def get_limits_layers(self, nr, axis_name=None):
        """Get limits for each layer at a given ping."""
        if axis_name is None:
            axis_name = self.y_axis_name
        extents = {}

        for key, layer in self.layers.items():
            match axis_name:
                case "Y indice":
                    extents[key] = layer.i0[nr] - 0.5, layer.i1[nr] - 0.5

                case "Sample number":
                    assert self.has_sample_nrs, \
                        "ERROR: Sample nr values not initialized"
                    extents[key] = self.y_indice_to_sample_nr_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case "Depth (m)":
                    assert self.has_depths, \
                        "ERROR: Depths values not initialized"
                    extents[key] = self.y_indice_to_depth_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case "Range (m)":
                    assert self.has_ranges, \
                        "ERROR: Ranges values not initialized"
                    extents[key] = self.y_indice_to_range_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case _:
                    raise RuntimeError(f"Invalid axis_name '{axis_name}'")

        return extents

    def __set_layer__(self, name, layer):
        """Internal method to set or combine layers."""
        if name == "main":
            if self.main_layer is not None:
                self.main_layer.combine(layer)
            else:
                self.main_layer = layer
        else:
            if name in self.layers.keys():
                self.layers[name].combine(layer)
            else:
                self.layers[name] = layer

    def add_layer(self, name, vec_x_val, vec_min_y, vec_max_y):
        """Add a layer with explicit boundaries."""
        layer = EchoLayer(self, vec_x_val, vec_min_y, vec_max_y)
        self.__set_layer__(name, layer)

    def add_layer_from_static_layer(self, name, min_y, max_y):
        """Add a layer with static boundaries."""
        layer = EchoLayer.from_static_layer(self, min_y, max_y)
        self.__set_layer__(name, layer)

    def add_layer_from_ping_param_offsets_absolute(self, name, ping_param_name, offset_0, offset_1):
        """Add a layer based on absolute offsets from a ping parameter."""
        layer = EchoLayer.from_ping_param_offsets_absolute(self, ping_param_name, offset_0, offset_1)
        self.__set_layer__(name, layer)

    def add_layer_from_ping_param_offsets_relative(self, name, ping_param_name, offset_0, offset_1):
        """Add a layer based on relative offsets from a ping parameter."""
        layer = EchoLayer.from_ping_param_offsets_relative(self, ping_param_name, offset_0, offset_1)
        self.__set_layer__(name, layer)

    def remove_layer(self, name):
        """Remove a layer by name."""
        if name == "main":
            self.main_layer = None
        elif name in self.layers.keys():
            self.layers.pop(name)

    def clear_layers(self):
        """Remove all layers except main."""
        self.layers = {}

    def clear_main_layer(self):
        """Remove the main layer."""
        self.main_layer = None

    def iterate_ping_data(self, keep_to_xlimits=True):
        """Iterate over ping data objects."""
        if keep_to_xlimits:
            xcoord = self.get_x_indices()[1]
            nrs = np.arange(xcoord[0], xcoord[-1] + 1)
        else:
            nrs = range(self._backend.n_pings)

        return [PingData(self, nr) for nr in nrs]

    # =========================================================================
    # Raw data access (for layers, non-downsampled)
    # =========================================================================

    def get_raw_layer_data(self, layer_name, ping_indices=None):
        """Get raw (non-downsampled) data for a specific layer.
        
        Args:
            layer_name: Name of the layer to extract data from.
            ping_indices: Optional list of ping indices. If None, uses visible x range.
            
        Yields:
            Tuples of (ping_index, raw_data, (sample_start, sample_end)).
        """
        if layer_name not in self.layers:
            raise KeyError(f"Layer '{layer_name}' not found")
        
        layer = self.layers[layer_name]
        
        if ping_indices is None:
            # Use visible x range
            _, ping_indices = self.get_x_indices()
        
        for ping_idx in ping_indices:
            raw_column = self._backend.get_raw_column(ping_idx)
            sample_start = layer.i0[ping_idx]
            sample_end = layer.i1[ping_idx]
            
            if sample_start < len(raw_column) and sample_end > sample_start:
                layer_data = raw_column[sample_start:sample_end]
                yield ping_idx, layer_data, (sample_start, sample_end)

    def get_raw_data_at_coordinates(self, x_coord, y_start, y_end):
        """Get raw (non-downsampled) data at specific coordinates.
        
        Args:
            x_coord: X coordinate (ping time, index, or datetime).
            y_start: Start Y coordinate (depth, range, or sample number).
            y_end: End Y coordinate.
            
        Returns:
            Tuple of (raw_data, (sample_start, sample_end)) or None if not found.
        """
        self.reinit()
        
        # Convert x_coord to ping index
        if isinstance(x_coord, dt.datetime):
            x_coord = x_coord.timestamp()
        
        x_coordinates = np.array([x_coord])
        ping_indices = self.feature_mapper.feature_to_index(
            self.x_axis_name, x_coordinates, mp_cores=self.mp_cores
        )
        
        if len(ping_indices) == 0:
            return None
            
        ping_idx = ping_indices[0]
        
        # Convert y coordinates to sample indices
        interpolator = self.y_coordinate_indice_interpolator[ping_idx]
        if interpolator is None:
            return None
            
        sample_start = int(interpolator(y_start) + 0.5)
        sample_end = int(interpolator(y_end) + 0.5)
        
        # Get raw data
        raw_column = self._backend.get_raw_column(ping_idx)
        
        # Clamp to valid range
        sample_start = max(0, sample_start)
        sample_end = min(len(raw_column), sample_end)
        
        if sample_end <= sample_start:
            return None
            
        return raw_column[sample_start:sample_end], (sample_start, sample_end)
