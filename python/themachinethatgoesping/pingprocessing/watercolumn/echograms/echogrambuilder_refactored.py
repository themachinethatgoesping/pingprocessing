"""Builder for echogram images with coordinate system and layer management.

This module provides the EchogramBuilder class which handles:
- Image building from data backends
- Layer management for region selection
- Coordinate system delegation to EchogramCoordinateSystem
"""

import numpy as np
from typing import Optional
from copy import deepcopy
from pathlib import Path
import datetime as dt

# external Ping packages
from themachinethatgoesping import echosounders

# internal Ping.pingprocessing packages
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

# Local imports
from .coordinate_system import EchogramCoordinateSystem
from .backends import EchogramDataBackend, PingDataBackend
from .layers.echolayer import EchoLayer, PingData


class EchogramBuilder:
    """Builder for echogram images with coordinate system and layer management.
    
    The EchogramBuilder controls:
    - Image building using data from a backend
    - Layer management for region selection
    
    Coordinate systems are delegated to an EchogramCoordinateSystem instance.
    Data access is delegated to an EchogramDataBackend.
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
        
        # Create coordinate system
        self._coord_system = EchogramCoordinateSystem(
            n_pings=backend.n_pings,
            max_number_of_samples=backend.max_sample_counts,
            ping_times=backend.ping_times,
            ping_numbers=ping_numbers,
        )

        # Layer management
        self.layers = {}
        self.main_layer = None

        # Set extents from backend
        min_s, max_s = backend.sample_nr_extents
        self._coord_system.set_sample_nr_extent(min_s, max_s)
        
        if backend.range_extents is not None:
            min_r, max_r = backend.range_extents
            self._coord_system.set_range_extent(min_r, max_r)
            
        if backend.depth_extents is not None:
            min_d, max_d = backend.depth_extents
            self._coord_system.set_depth_extent(min_d, max_d)

        # Initialize ping params from backend
        self._init_ping_params_from_backend()

        # Set default axes
        self._coord_system.set_y_axis_y_indice(layer_update_callback=self._update_layers)
        self._coord_system.set_x_axis_ping_index()
        
        self.mp_cores = 1
        self.verbose = True

    def _init_ping_params_from_backend(self):
        """Initialize ping parameters from backend's pre-computed values."""
        ping_params = self._backend.get_ping_params()
        for name, (y_reference, (times, values)) in ping_params.items():
            self._coord_system.add_ping_param(name, "Ping time", y_reference, times, values)

    def _update_layers(self):
        """Update all layers after coordinate system change."""
        if self.main_layer is not None:
            self.main_layer.update_y_gridder()
        for layer in self.layers.values():
            layer.update_y_gridder()

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
    # Coordinate system access
    # =========================================================================

    @property
    def coord_system(self) -> EchogramCoordinateSystem:
        """Access the coordinate system.
        
        All coordinate-related properties are available through this object:
        - x_axis_name, y_axis_name: Current axis names
        - x_extent, y_extent: Data extents
        - y_coordinates, y_resolution, y_gridder: Y-axis grid info
        - feature_mapper: X-axis mapping
        - ping_times, ping_numbers: Ping information
        - time_zone: Timezone for datetime conversions
        - max_number_of_samples: Per-ping sample counts
        - has_depths, has_ranges, has_sample_nrs: Available coordinate types
        - Various interpolators for coordinate transformations
        """
        return self._coord_system

    # =========================================================================
    # Backend access
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
    # Coordinate system methods (delegating to coord_system)
    # =========================================================================

    def reinit(self):
        """Reinitialize coordinate systems if needed."""
        self._coord_system.reinit()

    def get_x_kwargs(self):
        return self._coord_system.get_x_kwargs()

    def get_y_kwargs(self):
        return self._coord_system.get_y_kwargs()

    # Y-axis setters
    def set_y_axis_y_indice(self, min_sample_nr=0, max_sample_nr=np.nan, max_steps=1024, **kwargs):
        """Set Y axis to sample indices."""
        self._coord_system.set_y_axis_y_indice(
            min_sample_nr=min_sample_nr,
            max_sample_nr=max_sample_nr,
            max_steps=max_steps,
            layer_update_callback=self._update_layers,
            **kwargs
        )

    def set_y_axis_depth(self, min_depth=np.nan, max_depth=np.nan, max_steps=1024, **kwargs):
        """Set Y axis to depth in meters."""
        self._coord_system.set_y_axis_depth(
            min_depth=min_depth,
            max_depth=max_depth,
            max_steps=max_steps,
            layer_update_callback=self._update_layers,
            **kwargs
        )

    def set_y_axis_range(self, min_range=np.nan, max_range=np.nan, max_steps=1024, **kwargs):
        """Set Y axis to range in meters."""
        self._coord_system.set_y_axis_range(
            min_range=min_range,
            max_range=max_range,
            max_steps=max_steps,
            layer_update_callback=self._update_layers,
            **kwargs
        )

    def set_y_axis_sample_nr(self, min_sample_nr=0, max_sample_nr=np.nan, max_steps=1024, **kwargs):
        """Set Y axis to sample numbers."""
        self._coord_system.set_y_axis_sample_nr(
            min_sample_nr=min_sample_nr,
            max_sample_nr=max_sample_nr,
            max_steps=max_steps,
            layer_update_callback=self._update_layers,
            **kwargs
        )

    # X-axis setters
    def set_x_axis_ping_index(self, min_ping_index=0, max_ping_index=np.nan, max_steps=4096, **kwargs):
        """Set X axis to ping index."""
        self._coord_system.set_x_axis_ping_index(
            min_ping_index=min_ping_index,
            max_ping_index=max_ping_index,
            max_steps=max_steps,
            **kwargs
        )

    def set_x_axis_ping_time(self, min_timestamp=np.nan, max_timestamp=np.nan,
                             time_resolution=np.nan, time_interpolation_limit=np.nan,
                             max_steps=4096, **kwargs):
        """Set X axis to ping time (Unix timestamp)."""
        self._coord_system.set_x_axis_ping_time(
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
            time_resolution=time_resolution,
            time_interpolation_limit=time_interpolation_limit,
            max_steps=max_steps,
            **kwargs
        )

    def set_x_axis_date_time(self, min_ping_time=np.nan, max_ping_time=np.nan,
                             time_resolution=np.nan, time_interpolation_limit=np.nan,
                             max_steps=4096, **kwargs):
        """Set X axis to datetime."""
        self._coord_system.set_x_axis_date_time(
            min_ping_time=min_ping_time,
            max_ping_time=max_ping_time,
            time_resolution=time_resolution,
            time_interpolation_limit=time_interpolation_limit,
            max_steps=max_steps,
            **kwargs
        )

    def copy_xy_axis(self, other: "EchogramBuilder"):
        """Copy X/Y axis settings to another EchogramBuilder."""
        self._coord_system.copy_xy_axis_to(other._coord_system)

    # Ping numbers/times setters
    def set_ping_numbers(self, ping_numbers):
        """Set ping numbers for x-axis indexing."""
        self._coord_system.set_ping_numbers(ping_numbers)

    def set_ping_times(self, ping_times, time_zone=dt.timezone.utc):
        """Set ping times for x-axis time display."""
        self._coord_system.set_ping_times(ping_times, time_zone)

    # Extent setters
    def set_range_extent(self, min_ranges, max_ranges):
        """Set range extents."""
        self._coord_system.set_range_extent(min_ranges, max_ranges)

    def set_depth_extent(self, min_depths, max_depths):
        """Set depth extents."""
        self._coord_system.set_depth_extent(min_depths, max_depths)

    def set_sample_nr_extent(self, min_sample_nrs, max_sample_nrs):
        """Set sample number extents."""
        self._coord_system.set_sample_nr_extent(min_sample_nrs, max_sample_nrs)

    # Ping parameters
    def add_ping_param(self, name, x_reference, y_reference, vec_x_val, vec_y_val):
        """Add a ping parameter (e.g., bottom depth, layer boundary)."""
        self._coord_system.add_ping_param(name, x_reference, y_reference, vec_x_val, vec_y_val)

    def get_ping_param(self, name, use_x_coordinates=False):
        """Get a ping parameter's values in current coordinate system."""
        return self._coord_system.get_ping_param(name, use_x_coordinates)

    # Index mapping
    def get_y_indices(self, wci_nr):
        """Get Y indices mapping image coordinates to data indices."""
        return self._coord_system.get_y_indices(wci_nr)

    def get_x_indices(self):
        """Get X indices mapping image coordinates to ping indices."""
        return self._coord_system.get_x_indices()

    # =========================================================================
    # Data access methods (delegate to backend)
    # =========================================================================

    def get_column(self, nr):
        """Get column data for a ping from backend."""
        return self._backend.get_column(nr)

    # =========================================================================
    # Image building
    # =========================================================================

    def build_image(self, progress=None):
        """Build the echogram image.
        
        Uses the backend's get_image() method with affine indexing for efficiency.
        Backends can override get_image() for vectorized implementations (e.g., Zarr/Dask).
        
        Args:
            progress: Optional progress bar or None (not currently used).
            
        Returns:
            Tuple of (image, extent) where image is a 2D numpy array of shape (nx, ny)
            and extent is [x_min, x_max, y_max, y_min].
        """
        self.reinit()
        cs = self._coord_system
        
        # Create image request with affine parameters
        request = cs.make_image_request()
        
        # Use backend's get_image() method (may be overridden for Dask/Zarr)
        # Backend returns (nx, ny) - ping, sample
        image = self._backend.get_image(request)
        
        extent = deepcopy(cs.x_extent)
        extent.extend(cs.y_extent)

        return image, extent

    def build_image_and_layer_image(self, progress=None):
        """Build echogram image and combined layer image.
        
        Uses fast vectorized get_image() for the main echogram when no main_layer
        is set. Falls back to per-column iteration only for layer processing.
        
        Returns:
            Tuple of (image, layer_image, extent).
        """
        self.reinit()
        cs = self._coord_system
        ny = len(cs.y_coordinates)
        nx = len(cs.feature_mapper.get_feature_values("X coordinate"))

        # Fast path: use vectorized get_image for main echogram if no main_layer
        if self.main_layer is None:
            request = cs.make_image_request()
            image = self._backend.get_image(request)
        else:
            # Slow path: need per-column iteration for main_layer
            image = np.full((nx, ny), np.nan, dtype=np.float32)
            image_indices, wci_indices = self.get_x_indices()
            for image_index, wci_index in zip(image_indices, wci_indices):
                wci = self.get_column(wci_index)
                if len(wci) > 1:
                    y1, y2 = self.main_layer.get_y_indices(wci_index)
                    if y1 is not None and len(y1) > 0:
                        image[image_index, y1] = wci[y2]

        # Build layer image (requires per-column iteration)
        layer_image = np.full((nx, ny), np.nan, dtype=np.float32)
        if len(self.layers) > 0:
            image_indices, wci_indices = self.get_x_indices()
            image_indices = get_progress_iterator(image_indices, progress, desc="Building layer image")
            
            for image_index, wci_index in zip(image_indices, wci_indices):
                wci = self.get_column(wci_index)
                if len(wci) > 1:
                    for k, layer in self.layers.items():
                        y1_layer, y2_layer = layer.get_y_indices(wci_index)
                        if y1_layer is not None and len(y1_layer) > 0:
                            layer_image[image_index, y1_layer] = wci[y2_layer]

        extent = deepcopy(cs.x_extent)
        extent.extend(cs.y_extent)

        return image, layer_image, extent

    def build_image_and_layer_images(self, progress=None):
        """Build echogram image and individual layer images.
        
        Uses fast vectorized get_image() for the main echogram when no main_layer
        is set. Falls back to per-column iteration only for layer processing.
        
        Returns:
            Tuple of (image, layer_images_dict, extent).
        """
        self.reinit()
        cs = self._coord_system
        ny = len(cs.y_coordinates)
        nx = len(cs.feature_mapper.get_feature_values("X coordinate"))

        # Fast path: use vectorized get_image for main echogram if no main_layer
        if self.main_layer is None:
            request = cs.make_image_request()
            image = self._backend.get_image(request)
        else:
            # Slow path: need per-column iteration for main_layer
            image = np.full((nx, ny), np.nan, dtype=np.float32)
            image_indices, wci_indices = self.get_x_indices()
            for image_index, wci_index in zip(image_indices, wci_indices):
                wci = self.get_column(wci_index)
                if len(wci) > 1:
                    y1, y2 = self.main_layer.get_y_indices(wci_index)
                    if y1 is not None and len(y1) > 0:
                        image[image_index, y1] = wci[y2]

        # Build layer images (requires per-column iteration)
        layer_images = {}
        for key in self.layers.keys():
            layer_images[key] = np.full((nx, ny), np.nan, dtype=np.float32)

        if len(self.layers) > 0:
            image_indices, wci_indices = self.get_x_indices()
            image_indices = get_progress_iterator(image_indices, progress, desc="Building layer images")

            for image_index, wci_index in zip(image_indices, wci_indices):
                wci = self.get_column(wci_index)
                if len(wci) > 1:
                    for key, layer in self.layers.items():
                        y1_layer, y2_layer = layer.get_y_indices(wci_index)
                        if y1_layer is not None and len(y1_layer) > 0:
                            layer_images[key][image_index, y1_layer] = wci[y2_layer]

        extent = deepcopy(cs.x_extent)
        extent.extend(cs.y_extent)

        return image, layer_images, extent

    # =========================================================================
    # Layer management
    # =========================================================================

    def get_wci_layers(self, nr):
        """Get WCI data split by layers."""
        wci = self.get_column(nr)

        wci_layers = {}
        for key, layer in self.layers.items():
            wci_layers[key] = wci[layer.i0[nr]:layer.i1[nr]]

        return wci_layers

    def get_extent_layers(self, nr, axis_name=None):
        """Get extents for each layer at a given ping."""
        cs = self._coord_system
        if axis_name is None:
            axis_name = cs.y_axis_name
        extents = {}

        for key, layer in self.layers.items():
            match axis_name:
                case "Y indice":
                    extents[key] = layer.i0[nr] - 0.5, layer.i1[nr] - 0.5

                case "Sample number":
                    assert cs.has_sample_nrs, "ERROR: Sample nr values not initialized"
                    extents[key] = cs.y_indice_to_sample_nr_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case "Depth (m)":
                    assert cs.has_depths, "ERROR: Depths values not initialized"
                    extents[key] = cs.y_indice_to_depth_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case "Range (m)":
                    assert cs.has_ranges, "ERROR: Ranges values not initialized"
                    extents[key] = cs.y_indice_to_range_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case _:
                    raise RuntimeError(f"Invalid axis_name '{axis_name}'")

        return extents

    def get_limits_layers(self, nr, axis_name=None):
        """Get limits for each layer at a given ping."""
        cs = self._coord_system
        if axis_name is None:
            axis_name = cs.y_axis_name
        extents = {}

        for key, layer in self.layers.items():
            match axis_name:
                case "Y indice":
                    extents[key] = layer.i0[nr] - 0.5, layer.i1[nr] - 0.5

                case "Sample number":
                    assert cs.has_sample_nrs, "ERROR: Sample nr values not initialized"
                    extents[key] = cs.y_indice_to_sample_nr_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case "Depth (m)":
                    assert cs.has_depths, "ERROR: Depths values not initialized"
                    extents[key] = cs.y_indice_to_depth_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case "Range (m)":
                    assert cs.has_ranges, "ERROR: Ranges values not initialized"
                    extents[key] = cs.y_indice_to_range_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case _:
                    raise RuntimeError(f"Invalid axis_name '{axis_name}'")

        return extents

    def _set_layer(self, name, layer):
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

    # Backward compatibility alias
    __set_layer__ = _set_layer

    def add_layer(self, name, vec_x_val, vec_min_y, vec_max_y):
        """Add a layer with explicit boundaries."""
        layer = EchoLayer(self, vec_x_val, vec_min_y, vec_max_y)
        self._set_layer(name, layer)

    def add_layer_from_static_layer(self, name, min_y, max_y):
        """Add a layer with static boundaries."""
        layer = EchoLayer.from_static_layer(self, min_y, max_y)
        self._set_layer(name, layer)

    def add_layer_from_ping_param_offsets_absolute(self, name, ping_param_name, offset_0, offset_1):
        """Add a layer based on absolute offsets from a ping parameter."""
        layer = EchoLayer.from_ping_param_offsets_absolute(self, ping_param_name, offset_0, offset_1)
        self._set_layer(name, layer)

    def add_layer_from_ping_param_offsets_relative(self, name, ping_param_name, offset_0, offset_1):
        """Add a layer based on relative offsets from a ping parameter."""
        layer = EchoLayer.from_ping_param_offsets_relative(self, ping_param_name, offset_0, offset_1)
        self._set_layer(name, layer)

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
        
        cs = self._coord_system
        x_coordinates = np.array([x_coord])
        ping_indices = cs.feature_mapper.feature_to_index(
            cs.x_axis_name, x_coordinates, mp_cores=self.mp_cores
        )
        
        if len(ping_indices) == 0:
            return None
            
        ping_idx = ping_indices[0]
        
        # Convert y coordinates to sample indices using affine transform
        if cs._affine_y_to_sample is None:
            return None
        
        a_inv, b_inv = cs._affine_y_to_sample
        a, b = a_inv[ping_idx], b_inv[ping_idx]
        
        if not np.isfinite(a) or not np.isfinite(b):
            return None
            
        sample_start = int(a + b * y_start + 0.5)
        sample_end = int(a + b * y_end + 0.5)
        
        # Get raw data
        raw_column = self._backend.get_raw_column(ping_idx)
        
        # Clamp to valid range
        sample_start = max(0, sample_start)
        sample_end = min(len(raw_column), sample_end)
        
        if sample_end <= sample_start:
            return None
            
        return raw_column[sample_start:sample_end], (sample_start, sample_end)

    # =========================================================================
    # Zarr export
    # =========================================================================

    def to_zarr(
        self,
        path: str,
        chunks: tuple = (64, -1),
        compressor: str = "zstd",
        compression_level: int = 3,
        progress: bool = True,
    ) -> str:
        """Export echogram data to a Zarr store for fast lazy loading.
        
        Reads and writes data in chunks for memory efficiency and speed.
        Each chunk of pings is read, assembled in memory, and written at once.
        
        Args:
            path: Path for the Zarr store (directory, will be created).
            chunks: Chunk sizes as (ping_chunk, sample_chunk). 
                    Use -1 for full dimension. Default (64, -1) = 64 pings per chunk.
            compressor: Compression algorithm ('zstd', 'lz4', 'zlib', 'none').
            compression_level: Compression level (1-22 for zstd, higher = smaller/slower).
            progress: Whether to show progress bar.
            
        Returns:
            Path to the created Zarr store.
        """
        try:
            import zarr
            import json
            from tqdm.auto import tqdm
        except ImportError:
            raise ImportError(
                "zarr is required for to_zarr(). Install with: pip install zarr"
            )
        
        from .backends.zarr_backend import ZARR_FORMAT_VERSION
        
        # Get dimensions
        n_pings = self._backend.n_pings
        max_samples = int(self._backend.max_sample_counts.max()) + 1
        
        # Configure chunks
        ping_chunk = chunks[0] if chunks[0] > 0 else n_pings
        sample_chunk = chunks[1] if chunks[1] > 0 else max_samples
        
        # Configure compressor
        if compressor == "none":
            comp = None
        elif compressor == "zstd":
            comp = zarr.codecs.ZstdCodec(level=compression_level)
        elif compressor == "lz4":
            # LZ4 is available via BloscCodec in Zarr v3
            comp = zarr.codecs.BloscCodec(cname="lz4", clevel=compression_level)
        elif compressor == "zlib":
            comp = zarr.codecs.GzipCodec(level=compression_level)
        elif compressor == "blosc":
            comp = zarr.codecs.BloscCodec(cname="zstd", clevel=compression_level)
        else:
            raise ValueError(f"Unknown compressor: {compressor}. Options: none, zstd, lz4, zlib, blosc")
        
        # Create Zarr store (v3)
        store = zarr.open_group(path, mode="w")
        
        # Create the main data array with dimension names for xarray compatibility
        wci_data = store.create_array(
            "wci_data",
            shape=(n_pings, max_samples),
            chunks=(ping_chunk, sample_chunk),
            dtype=np.float32,
            fill_value=np.nan,
            compressors=[comp] if comp else None,
            dimension_names=["ping", "sample"],
        )
        
        # Write in chunks for efficiency
        # Each iteration: read ping_chunk columns, assemble in RAM, write as one block
        n_chunks = (n_pings + ping_chunk - 1) // ping_chunk
        
        chunk_iter = range(n_chunks)
        if progress:
            # Show pings/second in progress bar
            chunk_iter = tqdm(
                chunk_iter, 
                desc="Writing WCI data", 
                delay=1,
                unit="chunk",
                postfix={"pings": 0},
            )
        
        pings_written = 0
        for chunk_idx in chunk_iter:
            chunk_start = chunk_idx * ping_chunk
            chunk_end = min(chunk_start + ping_chunk, n_pings)
            chunk_size = chunk_end - chunk_start
            
            # Allocate buffer for this chunk
            chunk_buffer = np.full((chunk_size, max_samples), np.nan, dtype=np.float32)
            
            # Read columns into buffer
            for i, ping_idx in enumerate(range(chunk_start, chunk_end)):
                column = self._backend.get_column(ping_idx)
                chunk_buffer[i, :len(column)] = column
            
            # Write entire chunk at once
            wci_data[chunk_start:chunk_end, :] = chunk_buffer
            
            pings_written += chunk_size
            if progress:
                # Update progress with pings/sec
                elapsed = chunk_iter.format_dict.get('elapsed', 1) or 1
                pings_per_sec = pings_written / elapsed
                chunk_iter.set_postfix({"pings/s": f"{pings_per_sec:.0f}"})
        
        # Write metadata arrays with dimension names for xarray compatibility
        store.create_array("ping_times", data=self._backend.ping_times.astype(np.float64), dimension_names=["ping"])
        store.create_array("max_sample_counts", data=self._backend.max_sample_counts.astype(np.int32), dimension_names=["ping"])
        
        # Sample number extents
        min_s, max_s = self._backend.sample_nr_extents
        store.create_array("sample_nr_min", data=min_s.astype(np.float32), dimension_names=["ping"])
        store.create_array("sample_nr_max", data=max_s.astype(np.float32), dimension_names=["ping"])
        
        # Range extents (optional)
        range_ext = self._backend.range_extents
        if range_ext is not None:
            min_r, max_r = range_ext
            store.create_array("range_min", data=min_r.astype(np.float32), dimension_names=["ping"])
            store.create_array("range_max", data=max_r.astype(np.float32), dimension_names=["ping"])
        
        # Depth extents (optional)
        depth_ext = self._backend.depth_extents
        if depth_ext is not None:
            min_d, max_d = depth_ext
            store.create_array("depth_min", data=min_d.astype(np.float32), dimension_names=["ping"])
            store.create_array("depth_max", data=max_d.astype(np.float32), dimension_names=["ping"])
        
        # Ping parameters (these are per-param, not per-ping, so use different dim name)
        ping_params = self._backend.get_ping_params()
        params_meta = {}
        for name, (y_ref, (times, values)) in ping_params.items():
            params_meta[name] = y_ref
            store.create_array(f"ping_param_{name}_times", data=np.asarray(times, dtype=np.float64), dimension_names=[f"param_{name}"])
            store.create_array(f"ping_param_{name}_values", data=np.asarray(values, dtype=np.float32), dimension_names=[f"param_{name}"])
        
        # Store attributes
        store.attrs["format_version"] = ZARR_FORMAT_VERSION
        store.attrs["wci_value"] = self._backend.wci_value
        store.attrs["linear_mean"] = self._backend.linear_mean
        store.attrs["has_navigation"] = self._backend.has_navigation
        store.attrs["ping_params_meta"] = json.dumps(params_meta)
        store.attrs["n_pings"] = n_pings
        store.attrs["max_samples"] = max_samples
        
        # Consolidate metadata for faster reads (single file instead of many)
        zarr.consolidate_metadata(path)
        
        return path

    @classmethod
    def from_zarr(
        cls,
        path: str,
        chunks: dict = None,
    ) -> "EchogramBuilder":
        """Load an EchogramBuilder from a Zarr store.
        
        Args:
            path: Path to the Zarr store (directory).
            chunks: Optional chunk sizes for Dask loading.
            
        Returns:
            EchogramBuilder with ZarrDataBackend.
        """
        from .backends import ZarrDataBackend
        
        backend = ZarrDataBackend.from_zarr(path, chunks=chunks)
        return cls(backend)

    # =========================================================================
    # Mmap export (ultra-fast random access)
    # =========================================================================

    def to_mmap(
        self,
        path: str,
        progress: bool = True,
        chunk_mb: float = 10.0,
    ) -> str:
        """Export echogram data to a memory-mapped store for ultra-fast access.
        
        Memory-mapped files provide near-instantaneous random access, making
        them ideal for interactive visualization (zooming, panning). Trade-off:
        files are uncompressed and larger than Zarr stores.
        
        Memory efficiency:
        - Writes in chunks based on chunk_mb (default 10MB)
        - Chunk size adapts to data dimensions
        - Peak memory: ~chunk_mb + output metadata
        - Supports exporting larger-than-memory datasets
        
        Performance comparison (75K pings × 379 samples):
        - Full view: Mmap 3x faster than Zarr
        - Scattered access (100 pings): Mmap 100x faster
        - Thumbnail (100×100): Mmap 180x faster
        
        Args:
            path: Path for the mmap store (directory, will be created).
            progress: Whether to show progress bar.
            chunk_mb: Chunk size in megabytes for writing (default 10MB).
                      Larger chunks = faster export but more memory.
            
        Returns:
            Path to the created mmap store.
        """
        import json
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None
            progress = False
        
        from .backends.mmap_backend import MMAP_FORMAT_VERSION
        
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get dimensions
        n_pings = self._backend.n_pings
        max_samples = int(self._backend.max_sample_counts.max()) + 1
        
        # Calculate chunk size based on MB
        bytes_per_ping = max_samples * 4  # float32 = 4 bytes
        chunk_size = max(1, int(chunk_mb * 1024 * 1024 / bytes_per_ping))
        chunk_size = min(chunk_size, n_pings)  # Don't exceed total pings
        
        # Create memory-mapped file for WCI data
        wci_file = output_path / "wci_data.bin"
        wci_mmap = np.memmap(
            wci_file, dtype=np.float32, mode="w+", 
            shape=(n_pings, max_samples)
        )
        
        # Fill with NaN initially (in chunks)
        nan_chunk = np.full((chunk_size, max_samples), np.nan, dtype=np.float32)
        for chunk_start in range(0, n_pings, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_pings)
            wci_mmap[chunk_start:chunk_end, :] = nan_chunk[:chunk_end - chunk_start, :]
        del nan_chunk
        
        n_chunks = (n_pings + chunk_size - 1) // chunk_size
        
        chunk_iter = range(n_chunks)
        if progress and tqdm is not None:
            chunk_iter = tqdm(
                chunk_iter,
                desc=f"Exporting ({chunk_mb:.0f}MB chunks)",
                delay=0.5,
                unit="chunk",
                total=n_chunks,
                postfix={"pings": 0},
            )
        
        pings_written = 0
        for chunk_idx in chunk_iter:
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_pings)
            
            # Use generic get_chunk - backends optimize this internally
            chunk_data = self._backend.get_chunk(chunk_start, chunk_end)
            wci_mmap[chunk_start:chunk_end, :chunk_data.shape[1]] = chunk_data
            del chunk_data
            
            pings_written += chunk_end - chunk_start
            if progress and tqdm is not None:
                elapsed = chunk_iter.format_dict.get('elapsed', 1) or 1
                pings_per_sec = pings_written / elapsed
                chunk_iter.set_postfix({"pings/s": f"{pings_per_sec:.0f}"})
        
        # Flush and close the mmap
        wci_mmap.flush()
        del wci_mmap
        
        # Save array metadata as binary .npy files (much faster than JSON)
        sample_min, sample_max = self._backend.sample_nr_extents
        np.save(output_path / "ping_times.npy", self._backend.ping_times.astype(np.float64))
        np.save(output_path / "max_sample_counts.npy", self._backend.max_sample_counts.astype(np.int32))
        np.save(output_path / "sample_nr_min.npy", sample_min.astype(np.int32))
        np.save(output_path / "sample_nr_max.npy", sample_max.astype(np.int32))
        
        # Optional extents
        range_ext = self._backend.range_extents
        if range_ext is not None:
            min_r, max_r = range_ext
            np.save(output_path / "range_min.npy", min_r.astype(np.float32))
            np.save(output_path / "range_max.npy", max_r.astype(np.float32))
        
        depth_ext = self._backend.depth_extents
        if depth_ext is not None:
            min_d, max_d = depth_ext
            np.save(output_path / "depth_min.npy", min_d.astype(np.float32))
            np.save(output_path / "depth_max.npy", max_d.astype(np.float32))
        
        # Ping parameters (binary .npy files)
        ping_params = self._backend.get_ping_params()
        ping_params_meta = {}  # y_reference for each param (stored in JSON)
        ping_param_names = []
        for name, (y_ref, (timestamps, values)) in ping_params.items():
            ping_param_names.append(name)
            ping_params_meta[name] = y_ref
            np.save(output_path / f"ping_param_{name}_times.npy", np.asarray(timestamps, dtype=np.float64))
            np.save(output_path / f"ping_param_{name}_values.npy", np.asarray(values, dtype=np.float32))
        
        # Small scalar metadata as JSON (fast to load)
        metadata = {
            "format_version": MMAP_FORMAT_VERSION,
            "n_pings": n_pings,
            "n_samples": max_samples,
            "wci_value": self._backend.wci_value,
            "linear_mean": self._backend.linear_mean,
            "has_navigation": self._backend.has_navigation,
            "ping_param_names": ping_param_names,
            "ping_params_meta": ping_params_meta,
        }
        
        # Write small JSON metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        
        return str(path)
    
    @classmethod
    def from_mmap(
        cls,
        path: str,
    ) -> "EchogramBuilder":
        """Load an EchogramBuilder from a mmap store.
        
        The WCI data is memory-mapped and lazy-loaded:
        - No data loaded into memory until accessed
        - OS page cache handles caching efficiently
        - Supports files larger than available RAM
        
        Args:
            path: Path to the mmap store (directory).
            
        Returns:
            EchogramBuilder with MmapDataBackend.
        """
        from .backends import MmapDataBackend
        
        backend = MmapDataBackend.from_path(path)
        return cls(backend)

