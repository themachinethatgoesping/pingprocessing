"""Builder for echogram images with coordinate system and layer management.

This module provides the EchogramBuilder class which handles:
- Image building from data backends
- Layer management for region selection
- Coordinate system delegation to EchogramCoordinateSystem
"""

import numpy as np
from typing import Optional, Tuple
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
        
        # Value offset (applied to all data values)
        self._offset = 0.0
        
        # Oversampling settings
        self._x_oversampling = 1
        self._y_oversampling = 1
        self._oversampling_mode = "linear_mean"  # "linear_mean" or "db_mean"

    def _init_ping_params_from_backend(self):
        """Initialize ping parameters from backend's pre-computed values."""
        ping_params = self._backend.get_ping_params()
        for name, (y_reference, (times, values)) in ping_params.items():
            self._coord_system.add_ping_param(name, "Ping time", y_reference, times, values)

    def _get_all_ping_params(self):
        """Collect all ping parameters from backend and coordinate system.
        
        Returns all params in the backend format:
            {name: (y_reference, (timestamps, values))}
        
        Backend params are included first, then any additional params from
        the coordinate system that were added via add_ping_param() after
        construction are appended using ping_times as their timestamps.
        """
        # Start with backend params
        result = dict(self._backend.get_ping_params())
        
        # Add coord system params not already in backend
        for name, param_data in self._coord_system.param.items():
            if name not in result:
                y_reference, dense_values = param_data
                # dense_values may be a tuple (sparse format) or ndarray (dense)
                if isinstance(dense_values, tuple) and len(dense_values) == 2:
                    # Already in sparse (timestamps, values) format
                    result[name] = (y_reference, dense_values)
                else:
                    # Dense format: one value per ping -> use ping_times as x
                    result[name] = (y_reference, (self._backend.ping_times, np.asarray(dense_values, dtype=np.float64)))
        
        return result

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

    @classmethod
    def from_pings_dict(
        cls,
        pings_dict: dict,
        progress: bool = True,
        **kwargs,
    ) -> dict:
        """Create multiple EchogramBuilders from a dictionary of ping lists.
        
        Convenience method for creating multiple echograms at once, e.g., from
        pings grouped by frequency or channel.
        
        Args:
            pings_dict: Dictionary mapping keys (e.g., frequency) to ping lists.
            progress: Show progress bar for each echogram. Default True.
            **kwargs: Additional arguments passed to from_pings() for each echogram.
                Common kwargs: pss, wci_value, linear_mean, depth_stack, mp_cores.
            
        Returns:
            Dictionary mapping the same keys to EchogramBuilder instances.
            
        Examples:
            >>> pings_by_freq = {18000: pings_18k, 38000: pings_38k, 120000: pings_120k}
            >>> echograms = EchogramBuilder.from_pings_dict(
            ...     pings_by_freq,
            ...     pss=pss,
            ...     depth_stack=True,
            ...     progress=True
            ... )
            >>> # echograms = {18000: EchogramBuilder, 38000: EchogramBuilder, ...}
            >>> 
            >>> # Access individual echograms
            >>> echogram_38k = echograms[38000]
        """
        result = {}
        for key, pings in pings_dict.items():
            result[key] = cls.from_pings(
                pings,
                verbose=progress,
                **kwargs,
            )
        return result

    @classmethod
    def concat(
        cls,
        builders_or_backends: list,
        gap_handling: str = "preserve",
    ) -> "EchogramBuilder":
        """Concatenate multiple echograms along the time/ping axis.
        
        Creates a virtual echogram that spans all input echograms. Data is
        loaded lazily from the original backends as needed.
        
        Args:
            builders_or_backends: List of EchogramBuilder or EchogramDataBackend instances.
                Must be in temporal order.
            gap_handling: How to handle gaps between echograms:
                - "preserve": Keep real time gaps (x-axis shows true times)
                - "continuous": Virtual continuous (ignore gaps between files)
        
        Returns:
            EchogramBuilder with ConcatBackend.
            
        Examples:
            >>> # Concatenate echograms from multiple files
            >>> echograms = [EchogramBuilder.from_zarr(f) for f in file_list]
            >>> combined = EchogramBuilder.concat(echograms)
            >>> 
            >>> # Build full timeline image
            >>> image, extent = combined.build_image()
        """
        from .backends import ConcatBackend
        
        # Extract backends from builders if needed
        backends = []
        for item in builders_or_backends:
            if isinstance(item, EchogramBuilder):
                backends.append(item.backend)
            elif isinstance(item, EchogramDataBackend):
                backends.append(item)
            else:
                raise TypeError(
                    f"Expected EchogramBuilder or EchogramDataBackend, got {type(item)}"
                )
        
        concat_backend = ConcatBackend(backends, gap_handling=gap_handling)
        return cls(concat_backend)

    @classmethod
    def combine(
        cls,
        builders_or_backends,
        combine_func: str = "nanmean",
        name: str = "combined",
        linear: bool = True,
    ) -> "EchogramBuilder":
        """Combine multiple echograms with a mathematical operation.
        
        Creates a virtual echogram that combines all input echograms using
        the specified function (mean, median, sum, etc.). Useful for combining
        different frequencies or averaging multiple acquisitions.
        
        Combination happens AFTER downsampling for efficiency - each backend
        produces its downsampled image, then they are combined.
        
        The view settings (x_axis, y_axis) from the first EchogramBuilder are
        preserved in the combined result.
        
        Args:
            builders_or_backends: List or dict of EchogramBuilder or EchogramDataBackend
                instances. If a dict, uses dict.values(). Should have overlapping 
                time ranges.
            combine_func: Function to combine images, either:
                - String name: "nanmean", "nanmedian", "nansum", "nanmax", 
                  "nanmin", "nanstd", "mean", "first_valid"
                - Callable with signature (stack, axis) -> result
                  Stack has shape (n_backends, nx, ny), reduce along axis=0.
            name: Name for the combined echogram.
            linear: If True (default), convert dB data to linear domain before
                combining, then convert back to dB. This gives acoustically
                correct averaging of intensities. Set to False to combine
                directly in dB domain.
        
        Returns:
            EchogramBuilder with CombineBackend.
            
        Examples:
            >>> # Combine different frequencies with mean (linear domain)
            >>> echograms = {18000: echo_18k, 38000: echo_38k, 120000: echo_120k}
            >>> combined = EchogramBuilder.combine(echograms)  # dict works directly
            >>> 
            >>> # Use median instead of mean
            >>> combined = EchogramBuilder.combine(echograms, combine_func="nanmedian")
            >>> 
            >>> # Combine directly in dB domain (not acoustically correct)
            >>> combined = EchogramBuilder.combine(echograms, linear=False)
            >>> 
            >>> # Custom RMS combination
            >>> def rms(stack, axis):
            ...     return np.sqrt(np.nanmean(stack**2, axis=axis))
            >>> combined = EchogramBuilder.combine(echograms, combine_func=rms, linear=False)
        """
        # Handle dict input - extract values
        if isinstance(builders_or_backends, dict):
            builders_or_backends = list(builders_or_backends.values())
        from .backends import CombineBackend
        
        # Track the first builder for copying view settings
        first_builder = None
        
        # Extract backends from builders if needed
        backends = []
        for item in builders_or_backends:
            if isinstance(item, EchogramBuilder):
                if first_builder is None:
                    first_builder = item
                backends.append(item.backend)
            elif isinstance(item, EchogramDataBackend):
                backends.append(item)
            else:
                raise TypeError(
                    f"Expected EchogramBuilder or EchogramDataBackend, got {type(item)}"
                )
        
        # Determine alignment modes from first builder's coordinate system
        x_align = "ping_index"  # Default: align by ping index
        y_align = "sample_index"  # Default: align by sample index
        
        if first_builder is not None:
            cs = first_builder.coord_system
            # Determine x alignment from x_axis_name
            if cs.x_axis_name in ("Ping time", "Date time"):
                x_align = "time"
            # else: "Ping index" → ping_index (default)
            
            # Determine y alignment from y_axis_name  
            if cs.y_axis_name == "Depth (m)":
                y_align = "depth"
            elif cs.y_axis_name == "Range (m)":
                y_align = "range"
            # else: "Y indice", "Sample number" → sample_index (default)
        
        combine_backend = CombineBackend(
            backends, combine_func=combine_func, name=name,
            x_align=x_align, y_align=y_align, linear=linear
        )
        result = cls(combine_backend)
        
        # Copy view settings from first builder if available
        if first_builder is not None:
            first_builder._copy_view_settings(result)
        
        return result

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

    @property
    def offset(self) -> float:
        """Value offset applied to all data.
        
        This offset is added to all echogram values when building images
        (build_image, build_image_and_layer_image, build_image_and_layer_images)
        and permanently applied when saving (to_mmap, to_zarr).
        
        Returns:
            Current offset value (default 0.0).
        """
        return self._offset
    
    @offset.setter
    def offset(self, value: float):
        """Set the value offset applied to all data.
        
        Args:
            value: Offset to add to all echogram values (e.g., calibration correction).
        """
        self._offset = float(value)

    # =========================================================================
    # Navigation/track access
    # =========================================================================

    def get_track(
        self,
        start_ping: Optional[int] = None,
        end_ping: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the navigation track (latitudes, longitudes) for this echogram.
        
        Returns the lat/lon coordinates stored in the backend, which represent
        the ship position for each ping.
        
        Args:
            start_ping: Optional start ping index. Default: 0.
            end_ping: Optional end ping index (exclusive). Default: n_pings.
            
        Returns:
            Tuple of (latitudes, longitudes) arrays in degrees, or None if
            navigation data is not available.
            
        Example:
            >>> lats, lons = echogram.get_track()
            >>> # Get track for visible range
            >>> lats_visible, lons_visible = echogram.get_track(100, 500)
        """
        if not self._backend.has_latlon:
            return None
        
        if start_ping is None:
            start_ping = 0
        if end_ping is None:
            end_ping = self._backend.n_pings
            
        return (
            self._backend.latitudes[start_ping:end_ping],
            self._backend.longitudes[start_ping:end_ping],
        )
    
    @property
    def has_track(self) -> bool:
        """Check if navigation track data is available."""
        return self._backend.has_latlon

    def _copy_view_settings(self, other: "EchogramBuilder"):
        """Copy view settings (x_axis, y_axis, offset) to another EchogramBuilder.
        
        This is used when combining echograms to preserve the view from
        the first echogram in the list.
        """
        self.copy_xy_axis(other)
        other._offset = self._offset

    def _apply_axis_type(self, x_axis_name: str, y_axis_name: str):
        """Apply axis type settings without specific zoom parameters.
        
        This restores the type of axis (e.g., "Date time", "Depth (m)")
        when loading from saved data, using default extents.
        
        Args:
            x_axis_name: Saved x-axis name ("Ping index", "Ping time", "Date time")
            y_axis_name: Saved y-axis name ("Y indice", "Sample number", "Depth (m)", "Range (m)")
        """
        # Apply y-axis type
        if y_axis_name == "Depth (m)":
            self.set_y_axis_depth()
        elif y_axis_name == "Range (m)":
            self.set_y_axis_range()
        elif y_axis_name == "Sample number":
            self.set_y_axis_sample_nr()
        elif y_axis_name in ("Y indice", None):
            self.set_y_axis_y_indice()
        
        # Apply x-axis type
        if x_axis_name == "Date time":
            self.set_x_axis_date_time()
        elif x_axis_name == "Ping time":
            self.set_x_axis_ping_time()
        elif x_axis_name in ("Ping index", None):
            self.set_x_axis_ping_index()

    def _save_layers_to_dir(self, output_path: Path, metadata: dict):
        """Save layer boundary arrays to a directory and update metadata dict.
        
        Stores per-ping vec_min_y/vec_max_y for each named layer and the
        main layer (if set) as .npy files. Adds layer_names and has_main_layer
        to the metadata dict.
        
        Args:
            output_path: Directory to write .npy files into.
            metadata: Metadata dict to update with layer info.
        """
        layer_names = []
        for layer_name, layer in self.layers.items():
            layer_names.append(layer_name)
            np.save(output_path / f"layer_{layer_name}_min_y.npy", layer.vec_min_y.astype(np.float32))
            np.save(output_path / f"layer_{layer_name}_max_y.npy", layer.vec_max_y.astype(np.float32))
        if self.main_layer is not None:
            np.save(output_path / "layer_main_min_y.npy", self.main_layer.vec_min_y.astype(np.float32))
            np.save(output_path / "layer_main_max_y.npy", self.main_layer.vec_max_y.astype(np.float32))
        metadata["layer_names"] = layer_names
        metadata["has_main_layer"] = self.main_layer is not None

    def _restore_layers_from_store(self, layer_names_json, has_main_layer, get_array):
        """Restore layers from stored boundary arrays.
        
        Reconstructs EchoLayer objects from saved vec_min_y/vec_max_y arrays
        using ping times as x-coordinates.
        
        Args:
            layer_names_json: JSON string of layer name list, or a list.
            has_main_layer: Whether a main layer was stored.
            get_array: Callable(name) -> np.ndarray or None. Retrieves a stored
                       array by name (e.g. "layer_bottom_min_y").
        """
        import json as json_mod
        
        if isinstance(layer_names_json, str):
            layer_names = json_mod.loads(layer_names_json)
        else:
            layer_names = list(layer_names_json)
        
        ping_times = self._backend.ping_times
        
        for name in layer_names:
            min_y = get_array(f"layer_{name}_min_y")
            max_y = get_array(f"layer_{name}_max_y")
            if min_y is not None and max_y is not None:
                try:
                    self.add_layer(name, ping_times, min_y, max_y)
                except Exception:
                    pass  # Skip layers that fail to reconstruct
        
        if has_main_layer:
            min_y = get_array("layer_main_min_y")
            max_y = get_array("layer_main_max_y")
            if min_y is not None and max_y is not None:
                try:
                    layer = EchoLayer(self, ping_times, min_y, max_y)
                    self._set_layer("main", layer)
                except Exception:
                    pass  # Skip main layer if it fails to reconstruct

    # =========================================================================
    # Coordinate system methods (delegating to coord_system)
    # =========================================================================

    def reinit(self):
        """Reinitialize coordinate systems if needed."""
        self._coord_system.reinit()

    def get_x_axis_name(self):
        return self._coord_system.x_axis_name

    def get_y_axis_name(self):
        return self._coord_system.y_axis_name

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
    # Oversampling configuration
    # =========================================================================

    def set_oversampling(self, x_oversampling=1, y_oversampling=1, mode="linear_mean"):
        """Configure oversampling for image building.
        
        When oversampling > 1, build_image() will request a higher-resolution
        image from the backend (max_steps * oversampling) and then block-average
        it down to the original output resolution. This reduces aliasing artifacts
        from nearest-neighbor sampling.
        
        Args:
            x_oversampling: Integer oversampling factor for X axis (pings). Default 1.
            y_oversampling: Integer oversampling factor for Y axis (samples). Default 1.
            mode: Averaging mode for block downsampling.
                - 'linear_mean': Convert dB to linear (power(10, 0.1*v)), average,
                  convert back (10*log10). Correct for dB-domain data.
                - 'db_mean': Average directly in dB domain (geometric mean in
                  linear domain). Faster but less physically correct.
        """
        if x_oversampling < 1 or y_oversampling < 1:
            raise ValueError("Oversampling factors must be >= 1")
        if mode not in ("linear_mean", "db_mean"):
            raise ValueError(f"Invalid oversampling mode '{mode}'. Use 'linear_mean' or 'db_mean'.")
        
        self._x_oversampling = int(x_oversampling)
        self._y_oversampling = int(y_oversampling)
        self._oversampling_mode = mode
        # Force reinit so coordinate system recomputes with new oversampling
        self._coord_system._initialized = False

    @property
    def _has_oversampling(self):
        """Whether any oversampling is active."""
        return self._x_oversampling > 1 or self._y_oversampling > 1

    def _downsample_image(self, oversampled_image, target_nx, target_ny):
        """Block-average an oversampled image down to target resolution.
        
        Computes block sizes dynamically from the actual oversampled image
        shape vs target shape. This handles cases where the oversampled grid
        was clamped to native resolution (fewer pixels than requested).
        
        Args:
            oversampled_image: Array of shape (os_nx, os_ny) with oversampled data.
            target_nx: Desired number of X pixels in output.
            target_ny: Desired number of Y pixels in output.
            
        Returns:
            Array of shape (target_nx, target_ny) with block-averaged data.
        """
        os_nx, os_ny = oversampled_image.shape
        
        # If oversampled image is same size as target, no averaging needed
        if os_nx == target_nx and os_ny == target_ny:
            return oversampled_image
        
        # Compute actual block sizes from image dimensions
        bx = os_nx // target_nx
        by = os_ny // target_ny
        
        # Clamp to at least 1
        bx = max(bx, 1)
        by = max(by, 1)
        
        # Usable pixels = target * block_size (trim remainder)
        actual_nx = min(target_nx, os_nx // bx)
        actual_ny = min(target_ny, os_ny // by)
        
        if actual_nx == 0 or actual_ny == 0:
            return np.full((target_nx, target_ny), np.nan, dtype=np.float32)
        
        # Reshape into blocks: (actual_nx, bx, actual_ny, by)
        blocked = oversampled_image[:actual_nx * bx, :actual_ny * by].reshape(
            actual_nx, bx, actual_ny, by
        )
        
        if self._oversampling_mode == "linear_mean":
            # dB → linear → nanmean → dB
            # Use float64 for precision in power conversion
            with np.errstate(invalid='ignore'):
                linear = np.power(10.0, np.float64(blocked) * 0.1)
                mean_linear = np.nanmean(linear, axis=(1, 3))
                result = (10.0 * np.log10(mean_linear)).astype(np.float32)
        else:
            # db_mean: average directly in dB domain
            result = np.nanmean(blocked, axis=(1, 3)).astype(np.float32)
        
        # If target is larger than what we computed (edge case), pad with NaN
        if actual_nx < target_nx or actual_ny < target_ny:
            padded = np.full((target_nx, target_ny), np.nan, dtype=np.float32)
            padded[:actual_nx, :actual_ny] = result
            return padded
        
        return result

    # =========================================================================
    # Image building
    # =========================================================================

    def build_image(self, progress=None):
        """Build the echogram image.
        
        Uses the backend's get_image() method with affine indexing for efficiency.
        Backends can override get_image() for vectorized implementations (e.g., Zarr/Dask).
        
        When oversampling is configured (via set_oversampling()), requests a higher-
        resolution image and block-averages it down for anti-aliasing.
        
        Args:
            progress: Optional progress bar or None (not currently used).
            
        Returns:
            Tuple of (image, extent) where image is a 2D numpy array of shape (nx, ny)
            and extent is [x_min, x_max, y_max, y_min].
        """
        self.reinit()
        cs = self._coord_system
        
        if self._has_oversampling:
            # Oversampled path: request larger image, then block-average down
            target_nx = len(cs.feature_mapper.get_feature_values("X coordinate"))
            target_ny = len(cs.y_coordinates)
            
            request = cs.make_oversampled_image_request(
                x_oversampling=self._x_oversampling,
                y_oversampling=self._y_oversampling,
            )
            oversampled_image = self._backend.get_image(request)
            
            # Apply offset before averaging (offset is additive, order doesn't matter)
            if self._offset != 0.0:
                oversampled_image = oversampled_image + self._offset
            
            image = self._downsample_image(oversampled_image, target_nx, target_ny)
        else:
            # Standard path: no oversampling
            request = cs.make_image_request()
            image = self._backend.get_image(request)
            
            if self._offset != 0.0:
                image = image + self._offset
        
        extent = deepcopy(cs.x_extent)
        extent.extend(cs.y_extent)

        return image, extent

    def build_image_and_layer_image(self, progress=None):
        """Build echogram image and combined layer image.
        
        Uses fast vectorized get_image() for the main echogram when no main_layer
        is set. Falls back to per-column iteration only for layer processing.
        
        Note: Oversampling is applied to the main echogram image only.
        Layer images are built at native resolution (per-column iteration).
        
        Returns:
            Tuple of (image, layer_image, extent).
        """
        self.reinit()
        cs = self._coord_system
        ny = len(cs.y_coordinates)
        nx = len(cs.feature_mapper.get_feature_values("X coordinate"))

        # Fast path: use vectorized get_image for main echogram if no main_layer
        if self.main_layer is None:
            if self._has_oversampling:
                request = cs.make_oversampled_image_request(
                    x_oversampling=self._x_oversampling,
                    y_oversampling=self._y_oversampling,
                )
                oversampled_image = self._backend.get_image(request)
                if self._offset != 0.0:
                    oversampled_image = oversampled_image + self._offset
                image = self._downsample_image(oversampled_image, nx, ny)
            else:
                request = cs.make_image_request()
                image = self._backend.get_image(request)
                if self._offset != 0.0:
                    image = image + self._offset
        else:
            # Slow path: need per-column iteration for main_layer
            # (oversampling not supported for main_layer path)
            image = np.full((nx, ny), np.nan, dtype=np.float32)
            image_indices, wci_indices = self.get_x_indices()
            for image_index, wci_index in zip(image_indices, wci_indices):
                wci = self.get_column(wci_index)
                if len(wci) > 1:
                    y1, y2 = self.main_layer.get_y_indices(wci_index)
                    if y1 is not None and len(y1) > 0:
                        image[image_index, y1] = wci[y2]
            if self._offset != 0.0:
                image = image + self._offset

        # Build layer image (requires per-column iteration, no oversampling)
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

        if self._offset != 0.0:
            layer_image = layer_image + self._offset

        extent = deepcopy(cs.x_extent)
        extent.extend(cs.y_extent)

        return image, layer_image, extent

    def build_image_and_layer_images(self, progress=None):
        """Build echogram image and individual layer images.
        
        Uses fast vectorized get_image() for the main echogram when no main_layer
        is set. Falls back to per-column iteration only for layer processing.
        
        Note: Oversampling is applied to the main echogram image only.
        Layer images are built at native resolution (per-column iteration).
        
        Returns:
            Tuple of (image, layer_images_dict, extent).
        """
        self.reinit()
        cs = self._coord_system
        ny = len(cs.y_coordinates)
        nx = len(cs.feature_mapper.get_feature_values("X coordinate"))

        # Fast path: use vectorized get_image for main echogram if no main_layer
        if self.main_layer is None:
            if self._has_oversampling:
                request = cs.make_oversampled_image_request(
                    x_oversampling=self._x_oversampling,
                    y_oversampling=self._y_oversampling,
                )
                oversampled_image = self._backend.get_image(request)
                if self._offset != 0.0:
                    oversampled_image = oversampled_image + self._offset
                image = self._downsample_image(oversampled_image, nx, ny)
            else:
                request = cs.make_image_request()
                image = self._backend.get_image(request)
                if self._offset != 0.0:
                    image = image + self._offset
        else:
            # Slow path: need per-column iteration for main_layer
            # (oversampling not supported for main_layer path)
            image = np.full((nx, ny), np.nan, dtype=np.float32)
            image_indices, wci_indices = self.get_x_indices()
            for image_index, wci_index in zip(image_indices, wci_indices):
                wci = self.get_column(wci_index)
                if len(wci) > 1:
                    y1, y2 = self.main_layer.get_y_indices(wci_index)
                    if y1 is not None and len(y1) > 0:
                        image[image_index, y1] = wci[y2]
            if self._offset != 0.0:
                image = image + self._offset

        # Build layer images (requires per-column iteration, no oversampling)
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

        if self._offset != 0.0:
            for key in layer_images:
                layer_images[key] = layer_images[key] + self._offset

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
            wci_layers[key] = wci[int(layer.i0[nr]):int(layer.i1[nr])]

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
            sample_start = int(layer.i0[ping_idx])
            sample_end = int(layer.i1[ping_idx])
            
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
    # Update store metadata (no WCI data rewrite)
    # =========================================================================

    def update_store(self, path: str = None):
        """Write current axis, oversampling, ping-param and layer settings to
        an existing store **and** synchronise the in-memory backend state.
        
        Only the metadata of the mmap / zarr store is touched — the
        (potentially huge) WCI data array is left untouched.  Call this after
        changing view settings on a loaded echogram, e.g.::
        
            builder = EchogramBuilder.from_mmap("my_store.mmap")
            builder.set_y_axis_depth()
            builder.set_oversampling(x_oversampling=2, y_oversampling=2)
            builder.add_ping_param("my_line", "Ping time", "Depth (m)", ts, vals)
            builder.add_layer_from_static_layer("roi", 10, 50)
            builder.update_store()          # writes back to the same store
            
            # …or save to a *different* store's metadata
            builder.update_store("other_store.mmap")
        
        After the on-disk store is updated the backend's cached data
        structures are refreshed so that subsequent calls to e.g.
        ``backend.get_ping_params()`` return the new values.
        
        Args:
            path: Path to the store directory / zarr group.  If *None*, writes
                back to the store the current backend was loaded from
                (requires a backend with a ``store_path`` attribute).
                
        Raises:
            ValueError: If *path* is None and the backend has no ``store_path``.
            FileNotFoundError: If the target path does not exist.
        """
        import json
        from pathlib import Path as _Path
        
        # Resolve target path
        if path is None:
            if not hasattr(self._backend, "store_path") or self._backend.store_path is None:
                raise ValueError(
                    "Cannot infer store path from backend. "
                    "Pass the path explicitly: update_store('/path/to/store')"
                )
            path = self._backend.store_path
        
        target = _Path(path)
        
        if not target.exists():
            raise FileNotFoundError(f"Store path does not exist: {target}")
        
        # Detect store type
        is_zarr = (target / ".zgroup").exists() or (target / "zarr.json").exists() or (target / ".zmetadata").exists()
        is_mmap = (target / "metadata.json").exists() and (target / "wci_data.bin").exists()
        
        if is_zarr:
            self._save_settings_zarr(target)
        elif is_mmap:
            self._save_settings_mmap(target)
        else:
            raise ValueError(
                f"Cannot determine store type at '{target}'. "
                "Expected a zarr store or mmap directory."
            )
        
        # Keep the in-memory backend in sync with what we just wrote
        self._sync_backend_state()

    def _sync_backend_state(self):
        """Refresh the backend's in-memory caches to match current settings.
        
        Called automatically at the end of :meth:`update_store` so that
        ``backend.get_ping_params()`` and similar accessors reflect any
        additions made via :meth:`add_ping_param`, :meth:`add_layer*`, etc.
        """
        from .backends.mmap_backend import MmapDataBackend
        from .backends.zarr_backend import ZarrDataBackend
        
        ping_params = self._get_all_ping_params()
        
        if isinstance(self._backend, MmapDataBackend):
            # MmapDataBackend stores params in
            #   _metadata["ping_params"][name] = {y_reference, timestamps, values}
            mmap_params = {}
            param_names = []
            params_meta = {}
            for name, (y_ref, (times, values)) in ping_params.items():
                param_names.append(name)
                params_meta[name] = y_ref
                mmap_params[name] = {
                    "y_reference": y_ref,
                    "timestamps": np.asarray(times, dtype=np.float64),
                    "values": np.asarray(values, dtype=np.float32),
                }
            self._backend._metadata["ping_params"] = mmap_params
            self._backend._metadata["ping_param_names"] = param_names
            self._backend._metadata["ping_params_meta"] = params_meta
        
        elif isinstance(self._backend, ZarrDataBackend):
            # ZarrDataBackend caches params in
            #   _ping_params[name] = (y_ref, (times, values))
            self._backend._ping_params = {
                name: (y_ref, (np.asarray(times, dtype=np.float64),
                               np.asarray(values, dtype=np.float32)))
                for name, (y_ref, (times, values)) in ping_params.items()
            }

    def _save_settings_zarr(self, target):
        """Write metadata-only update to an existing Zarr store."""
        import zarr
        import json
        
        store = zarr.open_group(str(target), mode="r+")
        
        # Axis settings
        store.attrs["x_axis_name"] = self._coord_system.x_axis_name
        store.attrs["y_axis_name"] = self._coord_system.y_axis_name
        
        # Oversampling
        store.attrs["x_oversampling"] = self._x_oversampling
        store.attrs["y_oversampling"] = self._y_oversampling
        store.attrs["oversampling_mode"] = self._oversampling_mode
        
        # Ping parameters – remove old ones, write all current ones
        old_params_meta = store.attrs.get("ping_params_meta", "{}")
        if isinstance(old_params_meta, str):
            old_params_meta = json.loads(old_params_meta)
        for old_name in old_params_meta:
            for suffix in ("_times", "_values"):
                arr_name = f"ping_param_{old_name}{suffix}"
                if arr_name in store:
                    del store[arr_name]
        
        ping_params = self._get_all_ping_params()
        params_meta = {}
        for name, (y_ref, (times, values)) in ping_params.items():
            params_meta[name] = y_ref
            store.create_array(
                f"ping_param_{name}_times",
                data=np.asarray(times, dtype=np.float64),
                dimension_names=[f"param_{name}"],
                overwrite=True,
            )
            store.create_array(
                f"ping_param_{name}_values",
                data=np.asarray(values, dtype=np.float32),
                dimension_names=[f"param_{name}"],
                overwrite=True,
            )
        store.attrs["ping_params_meta"] = json.dumps(params_meta)
        
        # Layers – remove old ones, write all current ones
        old_layer_names = store.attrs.get("layer_names", "[]")
        if isinstance(old_layer_names, str):
            old_layer_names = json.loads(old_layer_names)
        for old_name in old_layer_names:
            for suffix in ("_min_y", "_max_y"):
                arr_name = f"layer_{old_name}{suffix}"
                if arr_name in store:
                    del store[arr_name]
        for suffix in ("_min_y", "_max_y"):
            arr_name = f"layer_main{suffix}"
            if arr_name in store:
                del store[arr_name]
        
        layer_names = []
        for layer_name, layer in self.layers.items():
            layer_names.append(layer_name)
            store.create_array(
                f"layer_{layer_name}_min_y",
                data=layer.vec_min_y.astype(np.float32),
                dimension_names=["ping"],
                overwrite=True,
            )
            store.create_array(
                f"layer_{layer_name}_max_y",
                data=layer.vec_max_y.astype(np.float32),
                dimension_names=["ping"],
                overwrite=True,
            )
        if self.main_layer is not None:
            store.create_array(
                "layer_main_min_y",
                data=self.main_layer.vec_min_y.astype(np.float32),
                dimension_names=["ping"],
                overwrite=True,
            )
            store.create_array(
                "layer_main_max_y",
                data=self.main_layer.vec_max_y.astype(np.float32),
                dimension_names=["ping"],
                overwrite=True,
            )
        store.attrs["layer_names"] = json.dumps(layer_names)
        store.attrs["has_main_layer"] = self.main_layer is not None
        
        # Re-consolidate metadata
        zarr.consolidate_metadata(str(target))

    def _save_settings_mmap(self, target):
        """Write metadata-only update to an existing mmap store."""
        import json
        from pathlib import Path as _Path
        
        metadata_file = target / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Axis settings
        metadata["x_axis_name"] = self._coord_system.x_axis_name
        metadata["y_axis_name"] = self._coord_system.y_axis_name
        
        # Oversampling
        metadata["x_oversampling"] = self._x_oversampling
        metadata["y_oversampling"] = self._y_oversampling
        metadata["oversampling_mode"] = self._oversampling_mode
        
        # Ping parameters – remove old .npy files, write all current ones
        for old_name in metadata.get("ping_param_names", []):
            for suffix in ("_times", "_values"):
                f = target / f"ping_param_{old_name}{suffix}.npy"
                if f.exists():
                    f.unlink()
        
        ping_params = self._get_all_ping_params()
        ping_param_names = []
        ping_params_meta = {}
        for name, (y_ref, (timestamps, values)) in ping_params.items():
            ping_param_names.append(name)
            ping_params_meta[name] = y_ref
            np.save(target / f"ping_param_{name}_times.npy", np.asarray(timestamps, dtype=np.float64))
            np.save(target / f"ping_param_{name}_values.npy", np.asarray(values, dtype=np.float32))
        metadata["ping_param_names"] = ping_param_names
        metadata["ping_params_meta"] = ping_params_meta
        
        # Layers – remove old .npy files, write all current ones
        for old_name in metadata.get("layer_names", []):
            for suffix in ("_min_y", "_max_y"):
                f = target / f"layer_{old_name}{suffix}.npy"
                if f.exists():
                    f.unlink()
        for suffix in ("_min_y", "_max_y"):
            f = target / f"layer_main{suffix}.npy"
            if f.exists():
                f.unlink()
        
        self._save_layers_to_dir(target, metadata)
        
        # Write updated metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

    # =========================================================================
    # Zarr export
    # =========================================================================

    def to_zarr(
        self,
        path: str,
        mode: str = "native",
        chunks: tuple = (64, -1),
        compressor: str = "zstd",
        compression_level: int = 3,
        progress: bool = True,
        resolution: float = None,
        interpolation: str = "nearest",
    ) -> str:
        """Export echogram data to a Zarr store for fast lazy loading.
        
        Reads and writes data in chunks for memory efficiency and speed.
        Each chunk of pings is read, assembled in memory, and written at once.
        
        Args:
            path: Path for the Zarr store (directory, will be created).
            mode: Storage mode:
                - "native": Store raw sample indices (fastest, smallest overhead)
                - "view": Transform to match current axis settings (depth/range)
            chunks: Chunk sizes as (ping_chunk, sample_chunk). 
                    Use -1 for full dimension. Default (64, -1) = 64 pings per chunk.
            compressor: Compression algorithm ('zstd', 'lz4', 'zlib', 'none').
            compression_level: Compression level (1-22 for zstd, higher = smaller/slower).
            progress: Whether to show progress bar.
            resolution: Y-axis resolution in meters (auto-detected if None).
                        Only used when mode="view" with depth/range axis.
            interpolation: Resampling method ("nearest" or "linear").
            
        Returns:
            Path to the created Zarr store.
            
        Examples:
            >>> # Default: store in native sample indices
            >>> builder.to_zarr("output.zarr")
            >>> 
            >>> # First set view to depth mode, then save in those coordinates
            >>> builder.set_y_axis_depth()
            >>> builder.to_zarr("output.zarr", mode="view")
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
        from .backends.storage_mode import StorageAxisMode
        
        # Validate parameters
        if mode not in ("native", "view"):
            raise ValueError(f"mode must be 'native' or 'view', got '{mode}'")
        if interpolation not in ("nearest", "linear"):
            raise ValueError(f"interpolation must be 'nearest' or 'linear', got '{interpolation}'")
        
        # Derive transformation settings from current axis settings
        y_coords = None
        if mode == "view":
            y_axis_name = self._coord_system.y_axis_name
            if y_axis_name == "Depth (m)":
                y_coords = "depth"
            elif y_axis_name == "Range (m)":
                y_coords = "range"
            # "Sample number" and "Y indice" stay as native
        
        # Check if we need coordinate transformation
        needs_transform = y_coords is not None
        
        if needs_transform:
            # Check for required extents
            if y_coords == "depth" and self._backend.depth_extents is None:
                raise ValueError("Backend does not have depth extents. Cannot transform to depth coordinates.")
            if y_coords == "range" and self._backend.range_extents is None:
                raise ValueError("Backend does not have range extents. Cannot transform to range coordinates.")
            
            y_extents = None
            if y_coords == "depth":
                y_extents = self._backend.depth_extents
            elif y_coords == "range":
                y_extents = self._backend.range_extents
            
            # Compute output grid
            ping_times, y_grid, storage_mode = self._compute_output_grid(
                y_coords=y_coords,
                y_resolution=resolution,
                y_extents=y_extents,
                x_strategy="primary",
                x_resolution=None,
                time_tolerance=0.5,
            )
            
            n_pings = len(ping_times)
            max_samples = len(y_grid) if y_grid is not None else int(self._backend.max_sample_counts.max()) + 1
        else:
            ping_times = self._backend.ping_times
            y_grid = None
            storage_mode = self._backend.storage_mode
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
        n_chunks = (n_pings + ping_chunk - 1) // ping_chunk
        
        chunk_iter = range(n_chunks)
        if progress:
            chunk_iter = tqdm(
                chunk_iter, 
                desc="Writing WCI data" + (f" ({y_coords} coords)" if needs_transform else ""), 
                delay=1,
                unit="chunk",
                postfix={"pings": 0},
            )
        
        pings_written = 0
        for chunk_idx in chunk_iter:
            chunk_start = chunk_idx * ping_chunk
            chunk_end = min(chunk_start + ping_chunk, n_pings)
            chunk_size = chunk_end - chunk_start
            
            if needs_transform:
                # Use transformation method
                chunk_buffer = self._transform_chunk(
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    output_ping_times=ping_times,
                    y_grid=y_grid,
                    y_coords=y_coords,
                    x_strategy=x_strategy,
                    time_tolerance=time_tolerance,
                    interpolation=interpolation,
                )
            else:
                # Original behavior: read columns directly
                chunk_buffer = np.full((chunk_size, max_samples), np.nan, dtype=np.float32)
                for i, ping_idx in enumerate(range(chunk_start, chunk_end)):
                    column = self._backend.get_column(ping_idx)
                    chunk_buffer[i, :len(column)] = column
            
            # Apply offset if set
            if self._offset != 0.0:
                chunk_buffer = chunk_buffer + self._offset
            
            # Write entire chunk at once
            wci_data[chunk_start:chunk_end, :] = chunk_buffer
            
            pings_written += chunk_size
            if progress:
                # Update progress with pings/sec
                elapsed = chunk_iter.format_dict.get('elapsed', 1) or 1
                pings_per_sec = pings_written / elapsed
                chunk_iter.set_postfix({"pings/s": f"{pings_per_sec:.0f}"})
        
        # Write metadata arrays - use transformed data if applicable
        if needs_transform:
            store.create_array("ping_times", data=ping_times.astype(np.float64), dimension_names=["ping"])
            # For transformed data, max_sample_counts is uniform
            max_sample_counts = np.full(n_pings, max_samples - 1, dtype=np.int32)
            store.create_array("max_sample_counts", data=max_sample_counts, dimension_names=["ping"])
            
            # Sample number extents
            store.create_array("sample_nr_min", data=np.zeros(n_pings, dtype=np.float32), dimension_names=["ping"])
            store.create_array("sample_nr_max", data=np.full(n_pings, max_samples - 1, dtype=np.float32), dimension_names=["ping"])
            
            # For depth/range transformed data, store uniform extents
            if y_coords == "depth" and y_grid is not None:
                store.create_array("depth_min", data=np.full(n_pings, y_grid[0], dtype=np.float32), dimension_names=["ping"])
                store.create_array("depth_max", data=np.full(n_pings, y_grid[-1], dtype=np.float32), dimension_names=["ping"])
            elif y_coords == "range" and y_grid is not None:
                store.create_array("range_min", data=np.full(n_pings, y_grid[0], dtype=np.float32), dimension_names=["ping"])
                store.create_array("range_max", data=np.full(n_pings, y_grid[-1], dtype=np.float32), dimension_names=["ping"])
            
            # Lat/lon - interpolate to new ping times
            if self._backend.has_latlon:
                new_lats, new_lons = self._interpolate_latlon_to_times(ping_times)
                store.create_array("latitudes", data=new_lats.astype(np.float64), dimension_names=["ping"])
                store.create_array("longitudes", data=new_lons.astype(np.float64), dimension_names=["ping"])
        else:
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
            
            # Lat/lon coordinates (optional)
            if self._backend.has_latlon:
                store.create_array("latitudes", data=self._backend.latitudes.astype(np.float64), dimension_names=["ping"])
                store.create_array("longitudes", data=self._backend.longitudes.astype(np.float64), dimension_names=["ping"])
        
        # Ping parameters (these are per-param, not per-ping, so use different dim name)
        ping_params = self._get_all_ping_params()
        params_meta = {}
        for name, (y_ref, (times, values)) in ping_params.items():
            # Update y_ref to match the new coordinate system if transformed
            if needs_transform and y_coords == "depth":
                y_ref = "Depth (m)"
            elif needs_transform and y_coords == "range":
                y_ref = "Range (m)"
            params_meta[name] = y_ref
            store.create_array(f"ping_param_{name}_times", data=np.asarray(times, dtype=np.float64), dimension_names=[f"param_{name}"])
            store.create_array(f"ping_param_{name}_values", data=np.asarray(values, dtype=np.float32), dimension_names=[f"param_{name}"])
        
        # Determine axis names for saved metadata
        if needs_transform:
            if y_coords == "depth":
                saved_y_axis_name = "Depth (m)"
            elif y_coords == "range":
                saved_y_axis_name = "Range (m)"
            else:
                saved_y_axis_name = self._coord_system.y_axis_name
            
            if x_strategy == "regular":
                saved_x_axis_name = "Ping time"
            else:
                saved_x_axis_name = self._coord_system.x_axis_name or "Date time"
        else:
            saved_x_axis_name = self._coord_system.x_axis_name
            saved_y_axis_name = self._coord_system.y_axis_name
        
        # Store attributes
        store.attrs["format_version"] = ZARR_FORMAT_VERSION
        store.attrs["wci_value"] = self._backend.wci_value
        store.attrs["linear_mean"] = self._backend.linear_mean
        store.attrs["has_navigation"] = self._backend.has_navigation
        store.attrs["has_latlon"] = self._backend.has_latlon
        store.attrs["ping_params_meta"] = json.dumps(params_meta)
        store.attrs["n_pings"] = n_pings
        store.attrs["max_samples"] = max_samples
        store.attrs["storage_mode"] = json.dumps(storage_mode.to_dict())
        # View axis settings (type only, not zoom level)
        store.attrs["x_axis_name"] = saved_x_axis_name
        store.attrs["y_axis_name"] = saved_y_axis_name
        # Oversampling settings
        store.attrs["x_oversampling"] = self._x_oversampling
        store.attrs["y_oversampling"] = self._y_oversampling
        store.attrs["oversampling_mode"] = self._oversampling_mode
        
        # Layer data
        layer_names = []
        for layer_name, layer in self.layers.items():
            layer_names.append(layer_name)
            store.create_array(f"layer_{layer_name}_min_y", data=layer.vec_min_y.astype(np.float32), dimension_names=["ping"])
            store.create_array(f"layer_{layer_name}_max_y", data=layer.vec_max_y.astype(np.float32), dimension_names=["ping"])
        if self.main_layer is not None:
            store.create_array("layer_main_min_y", data=self.main_layer.vec_min_y.astype(np.float32), dimension_names=["ping"])
            store.create_array("layer_main_max_y", data=self.main_layer.vec_max_y.astype(np.float32), dimension_names=["ping"])
        store.attrs["layer_names"] = json.dumps(layer_names)
        store.attrs["has_main_layer"] = self.main_layer is not None
        
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
        
        The axis type settings (e.g., "Date time", "Depth (m)") are restored
        from the saved metadata, but zoom levels are not preserved.
        
        Args:
            path: Path to the Zarr store (directory).
            chunks: Optional chunk sizes for Dask loading.
            
        Returns:
            EchogramBuilder with ZarrDataBackend.
        """
        import zarr
        import json as json_mod
        from .backends import ZarrDataBackend
        
        backend = ZarrDataBackend.from_zarr(path, chunks=chunks)
        builder = cls(backend)
        
        # Restore axis settings, oversampling, and layers from zarr attributes
        try:
            store = zarr.open_group(path, mode="r")
            x_axis_name = store.attrs.get("x_axis_name")
            y_axis_name = store.attrs.get("y_axis_name")
            
            if x_axis_name or y_axis_name:
                builder._apply_axis_type(x_axis_name, y_axis_name)
            
            # Restore oversampling settings
            builder._x_oversampling = int(store.attrs.get("x_oversampling", 1))
            builder._y_oversampling = int(store.attrs.get("y_oversampling", 1))
            builder._oversampling_mode = store.attrs.get("oversampling_mode", "linear_mean")
            
            # Restore layers
            builder._restore_layers_from_store(
                layer_names_json=store.attrs.get("layer_names", "[]"),
                has_main_layer=store.attrs.get("has_main_layer", False),
                get_array=lambda name: store[name][:] if name in store else None,
            )
        except Exception:
            pass  # Ignore errors loading settings from older files
        
        return builder

    # =========================================================================
    # Mmap export (ultra-fast random access)
    # =========================================================================

    def to_mmap(
        self,
        path: str,
        mode: str = "native",
        progress: bool = True,
        chunk_mb: float = 10.0,
        resolution: float = None,
        interpolation: str = "nearest",
    ) -> str:
        """Export echogram data to a memory-mapped store for ultra-fast access.
        
        Memory-mapped files provide near-instantaneous random access, making
        them ideal for interactive visualization (zooming, panning). Trade-off:
        files are uncompressed and larger than Zarr stores.
        
        Args:
            path: Path for the mmap store (directory, will be created).
            mode: Storage mode:
                - "native": Store raw sample indices (fastest, smallest overhead)
                - "view": Transform to match current axis settings (depth/range)
            progress: Whether to show progress bar.
            chunk_mb: Chunk size in megabytes for writing (default 10MB).
            resolution: Y-axis resolution in meters (auto-detected if None).
                        Only used when mode="view" with depth/range axis.
            interpolation: Resampling method ("nearest" or "linear").
            
        Returns:
            Path to the created mmap store.
            
        Examples:
            >>> # Default: store in native sample indices (fastest)
            >>> builder.to_mmap("output.mmap")
            >>> 
            >>> # First set view to depth mode, then save in those coordinates
            >>> builder.set_y_axis_depth()
            >>> builder.to_mmap("output.mmap", mode="view")
            >>> 
            >>> # Store with specific resolution
            >>> builder.set_y_axis_depth()
            >>> builder.to_mmap("output.mmap", mode="view", resolution=0.1)
        """
        import json
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None
            progress = False
        
        from .backends.mmap_backend import MMAP_FORMAT_VERSION
        from .backends.storage_mode import StorageAxisMode
        
        # Validate mode
        if mode not in ("native", "view"):
            raise ValueError(f"mode must be 'native' or 'view', got '{mode}'")
        if interpolation not in ("nearest", "linear"):
            raise ValueError(f"interpolation must be 'nearest' or 'linear', got '{interpolation}'")
        
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Derive transformation settings from current axis settings
        y_coords = None
        if mode == "view":
            y_axis_name = self._coord_system.y_axis_name
            if y_axis_name == "Depth (m)":
                y_coords = "depth"
            elif y_axis_name == "Range (m)":
                y_coords = "range"
            # "Sample number" and "Y indice" stay as native
        
        # Determine if we need coordinate transformation
        needs_transform = y_coords is not None
        
        if not needs_transform:
            # Simple case: no transformation needed
            return self._to_mmap_simple(output_path, progress, chunk_mb)
        
        # Complex case: coordinate transformation required
        return self._to_mmap_transformed(
            output_path, progress, chunk_mb,
            y_coords=y_coords,
            y_resolution=resolution,
            x_strategy="primary",
            x_resolution=None,
            time_tolerance=0.5,
            interpolation=interpolation,
        )

    def _to_mmap_simple(self, output_path: Path, progress: bool, chunk_mb: float) -> str:
        """Export to mmap without coordinate transformation (original behavior)."""
        import json
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None
            progress = False
        
        from .backends.mmap_backend import MMAP_FORMAT_VERSION
        
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
            
            # Apply offset if set
            if self._offset != 0.0:
                chunk_data = chunk_data + self._offset
            
            # Truncate to mmap size if chunk is larger (can happen due to rounding in sample counts)
            n_cols = min(chunk_data.shape[1], max_samples)
            wci_mmap[chunk_start:chunk_end, :n_cols] = chunk_data[:, :n_cols]
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
        
        # Lat/lon coordinates (optional)
        if self._backend.has_latlon:
            np.save(output_path / "latitudes.npy", self._backend.latitudes.astype(np.float64))
            np.save(output_path / "longitudes.npy", self._backend.longitudes.astype(np.float64))
        
        # Ping parameters (binary .npy files)
        ping_params = self._get_all_ping_params()
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
            "has_latlon": self._backend.has_latlon,
            "ping_param_names": ping_param_names,
            "ping_params_meta": ping_params_meta,
            "storage_mode": self._backend.storage_mode.to_dict(),
            # View axis settings (type only, not zoom level)
            "x_axis_name": self._coord_system.x_axis_name,
            "y_axis_name": self._coord_system.y_axis_name,
            # Oversampling settings
            "x_oversampling": self._x_oversampling,
            "y_oversampling": self._y_oversampling,
            "oversampling_mode": self._oversampling_mode,
        }
        
        # Save layer data as .npy files
        self._save_layers_to_dir(output_path, metadata)
        
        # Write small JSON metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        
        return str(output_path)

    def _to_mmap_transformed(
        self,
        output_path: Path,
        progress: bool,
        chunk_mb: float,
        y_coords: str,
        y_resolution: float,
        x_strategy: str,
        x_resolution: float,
        time_tolerance: float,
        interpolation: str,
    ) -> str:
        """Export to mmap with coordinate transformation.
        
        This handles:
        - Y-axis transformation from sample indices to depth/range
        - X-axis strategies (primary, union, regular)
        - Resampling with nearest-neighbor or linear interpolation
        """
        import json
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None
            progress = False
        
        from .backends.mmap_backend import MMAP_FORMAT_VERSION
        from .backends.storage_mode import StorageAxisMode
        from .backends import CombineBackend
        
        # Check if backend has the required extents
        if y_coords == "depth":
            if self._backend.depth_extents is None:
                raise ValueError("Backend does not have depth extents. Cannot transform to depth coordinates.")
            y_extents = self._backend.depth_extents
        elif y_coords == "range":
            if self._backend.range_extents is None:
                raise ValueError("Backend does not have range extents. Cannot transform to range coordinates.")
            y_extents = self._backend.range_extents
        else:
            # y_coords is None, but we're here because x_strategy != "primary"
            y_extents = None
        
        # Determine output dimensions and grid
        ping_times, y_grid, storage_mode = self._compute_output_grid(
            y_coords=y_coords,
            y_resolution=y_resolution,
            y_extents=y_extents,
            x_strategy=x_strategy,
            x_resolution=x_resolution,
            time_tolerance=time_tolerance,
        )
        
        n_pings = len(ping_times)
        n_samples = len(y_grid) if y_grid is not None else int(self._backend.max_sample_counts.max()) + 1
        
        # Calculate chunk size based on MB
        bytes_per_ping = n_samples * 4  # float32 = 4 bytes
        chunk_size = max(1, int(chunk_mb * 1024 * 1024 / bytes_per_ping))
        chunk_size = min(chunk_size, n_pings)
        
        # Create memory-mapped file for WCI data
        wci_file = output_path / "wci_data.bin"
        wci_mmap = np.memmap(
            wci_file, dtype=np.float32, mode="w+",
            shape=(n_pings, n_samples)
        )
        
        # Fill with NaN initially
        wci_mmap[:] = np.nan
        
        n_chunks = (n_pings + chunk_size - 1) // chunk_size
        
        pbar = None
        if progress and tqdm is not None:
            pbar = tqdm(
                total=n_pings,
                desc=f"Transforming to {y_coords or 'native'} coords",
                delay=0.5,
                unit="pings",
            )
        
        # Process each chunk
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_pings)
            actual_chunk = chunk_end - chunk_start
            
            chunk_data = self._transform_chunk(
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                output_ping_times=ping_times,
                y_grid=y_grid,
                y_coords=y_coords,
                x_strategy=x_strategy,
                time_tolerance=time_tolerance,
                interpolation=interpolation,
            )
            
            # Apply offset if set
            if self._offset != 0.0:
                chunk_data = chunk_data + self._offset
            
            wci_mmap[chunk_start:chunk_end, :] = chunk_data
            del chunk_data
            
            if pbar is not None:
                pbar.update(actual_chunk)
        
        if pbar is not None:
            pbar.close()
        
        # Flush and close
        wci_mmap.flush()
        del wci_mmap
        
        # Save metadata
        np.save(output_path / "ping_times.npy", ping_times.astype(np.float64))
        
        # For transformed data, max_sample_counts is uniform
        max_sample_counts = np.full(n_pings, n_samples - 1, dtype=np.int32)
        np.save(output_path / "max_sample_counts.npy", max_sample_counts)
        
        # Sample number extents (for transformed data, this is the y_grid indices)
        np.save(output_path / "sample_nr_min.npy", np.zeros(n_pings, dtype=np.int32))
        np.save(output_path / "sample_nr_max.npy", np.full(n_pings, n_samples - 1, dtype=np.int32))
        
        # For depth/range transformed data, we store the grid as uniform extents
        if y_coords == "depth":
            depth_min = np.full(n_pings, y_grid[0], dtype=np.float32)
            depth_max = np.full(n_pings, y_grid[-1], dtype=np.float32)
            np.save(output_path / "depth_min.npy", depth_min)
            np.save(output_path / "depth_max.npy", depth_max)
        elif y_coords == "range":
            range_min = np.full(n_pings, y_grid[0], dtype=np.float32)
            range_max = np.full(n_pings, y_grid[-1], dtype=np.float32)
            np.save(output_path / "range_min.npy", range_min)
            np.save(output_path / "range_max.npy", range_max)
        
        # Lat/lon - interpolate to new ping times if needed
        if self._backend.has_latlon:
            new_lats, new_lons = self._interpolate_latlon_to_times(ping_times)
            np.save(output_path / "latitudes.npy", new_lats.astype(np.float64))
            np.save(output_path / "longitudes.npy", new_lons.astype(np.float64))
        
        # Ping parameters - interpolate to new ping times
        ping_params = self._get_all_ping_params()
        ping_params_meta = {}
        ping_param_names = []
        for name, (y_ref, (timestamps, values)) in ping_params.items():
            ping_param_names.append(name)
            # Update y_ref to match the new coordinate system
            if y_coords == "depth":
                new_y_ref = "Depth (m)"
            elif y_coords == "range":
                new_y_ref = "Range (m)"
            else:
                new_y_ref = y_ref
            ping_params_meta[name] = new_y_ref
            np.save(output_path / f"ping_param_{name}_times.npy", np.asarray(timestamps, dtype=np.float64))
            np.save(output_path / f"ping_param_{name}_values.npy", np.asarray(values, dtype=np.float32))
        
        # Determine axis names for saved metadata
        if y_coords == "depth":
            y_axis_name = "Depth (m)"
        elif y_coords == "range":
            y_axis_name = "Range (m)"
        else:
            y_axis_name = self._coord_system.y_axis_name
        
        if x_strategy == "regular":
            x_axis_name = "Ping time"  # Regular grid implies time-based
        else:
            x_axis_name = self._coord_system.x_axis_name or "Date time"
        
        metadata = {
            "format_version": MMAP_FORMAT_VERSION,
            "n_pings": n_pings,
            "n_samples": n_samples,
            "wci_value": self._backend.wci_value,
            "linear_mean": self._backend.linear_mean,
            "has_navigation": self._backend.has_navigation,
            "has_latlon": self._backend.has_latlon,
            "ping_param_names": ping_param_names,
            "ping_params_meta": ping_params_meta,
            "storage_mode": storage_mode.to_dict(),
            "x_axis_name": x_axis_name,
            "y_axis_name": y_axis_name,
            # Oversampling settings
            "x_oversampling": self._x_oversampling,
            "y_oversampling": self._y_oversampling,
            "oversampling_mode": self._oversampling_mode,
        }
        
        # Save layer data as .npy files
        self._save_layers_to_dir(output_path, metadata)
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        
        return str(output_path)

    def _compute_output_grid(
        self,
        y_coords: str,
        y_resolution: float,
        y_extents: tuple,
        x_strategy: str,
        x_resolution: float,
        time_tolerance: float,
    ) -> tuple:
        """Compute output grid dimensions and storage mode.
        
        Returns:
            (ping_times, y_grid, storage_mode)
        """
        from .backends.storage_mode import StorageAxisMode
        from .backends import CombineBackend
        
        # Compute Y grid
        if y_coords is not None and y_extents is not None:
            y_min_arr, y_max_arr = y_extents
            y_min = float(np.nanmin(y_min_arr))
            y_max = float(np.nanmax(y_max_arr))
            
            # Auto-detect resolution if not specified
            if y_resolution is None:
                # Estimate from per-ping extents and sample counts
                max_samples = self._backend.max_sample_counts
                y_ranges = y_max_arr - y_min_arr
                sample_intervals = y_ranges / (max_samples + 1)
                y_resolution = float(np.nanmedian(sample_intervals[sample_intervals > 0]))
                # Round to nice value
                y_resolution = self._round_to_nice_value(y_resolution)
            
            n_y_bins = int(np.ceil((y_max - y_min) / y_resolution)) + 1
            y_grid = np.linspace(y_min, y_min + (n_y_bins - 1) * y_resolution, n_y_bins)
        else:
            y_grid = None
            y_resolution = 1.0
            y_min = 0.0
        
        # Compute X grid (ping times)
        backend = self._backend
        
        if x_strategy == "primary":
            # Use backend's native ping times
            ping_times = backend.ping_times.copy()
            x_origin = None
            x_res = None
        elif x_strategy == "union":
            # For CombineBackend, collect all ping times from sub-backends
            if isinstance(backend, CombineBackend):
                all_times = []
                for i in range(backend.num_backends):
                    sub = backend.get_backend(i)
                    all_times.append(sub.ping_times)
                ping_times = np.unique(np.concatenate(all_times))
                ping_times = ping_times[np.isfinite(ping_times)]
                ping_times = np.sort(ping_times)
            else:
                ping_times = backend.ping_times.copy()
            x_origin = None
            x_res = None
        elif x_strategy == "regular":
            # Create regular time grid
            t_min = float(np.nanmin(backend.ping_times))
            t_max = float(np.nanmax(backend.ping_times))
            n_time_bins = int(np.ceil((t_max - t_min) / x_resolution)) + 1
            ping_times = np.linspace(t_min, t_min + (n_time_bins - 1) * x_resolution, n_time_bins)
            x_origin = t_min
            x_res = x_resolution
        else:
            raise ValueError(f"Unknown x_strategy: {x_strategy}")
        
        # Create storage mode
        if y_coords == "depth":
            storage_mode = StorageAxisMode(
                x_axis="ping_time",
                y_axis="depth",
                x_resolution=x_res,
                x_origin=x_origin,
                y_resolution=y_resolution,
                y_origin=y_min if y_grid is not None else 0.0,
            )
        elif y_coords == "range":
            storage_mode = StorageAxisMode(
                x_axis="ping_time",
                y_axis="range",
                x_resolution=x_res,
                x_origin=x_origin,
                y_resolution=y_resolution,
                y_origin=y_min if y_grid is not None else 0.0,
            )
        else:
            storage_mode = StorageAxisMode.default()
        
        return ping_times, y_grid, storage_mode

    def _round_to_nice_value(self, value: float) -> float:
        """Round a resolution value to a nice number (0.01, 0.02, 0.05, 0.1, etc.)."""
        if value <= 0:
            return 0.1
        
        # Find order of magnitude
        exponent = np.floor(np.log10(value))
        mantissa = value / (10 ** exponent)
        
        # Round mantissa to 1, 2, or 5
        if mantissa < 1.5:
            mantissa = 1
        elif mantissa < 3.5:
            mantissa = 2
        elif mantissa < 7.5:
            mantissa = 5
        else:
            mantissa = 10
        
        return float(mantissa * (10 ** exponent))

    def _transform_chunk(
        self,
        chunk_start: int,
        chunk_end: int,
        output_ping_times: np.ndarray,
        y_grid: np.ndarray,
        y_coords: str,
        x_strategy: str,
        time_tolerance: float,
        interpolation: str,
    ) -> np.ndarray:
        """Transform a chunk of data to the output coordinate system.
        
        Optimized for speed using vectorized operations where possible.
        """
        from .backends import CombineBackend
        
        chunk_size = chunk_end - chunk_start
        n_y = len(y_grid) if y_grid is not None else int(self._backend.max_sample_counts.max()) + 1
        
        result = np.full((chunk_size, n_y), np.nan, dtype=np.float32)
        
        backend = self._backend
        output_times_chunk = output_ping_times[chunk_start:chunk_end]
        
        # For primary strategy, pre-compute all nearest indices at once
        if x_strategy == "primary" and not isinstance(backend, CombineBackend):
            backend_times = backend.ping_times
            # Vectorized: find nearest ping for all output times in chunk
            # Using broadcasting to compute all pairwise differences
            time_diffs = np.abs(backend_times[np.newaxis, :] - output_times_chunk[:, np.newaxis])
            nearest_indices = np.argmin(time_diffs, axis=1)
            min_diffs = time_diffs[np.arange(chunk_size), nearest_indices]
            valid_mask = min_diffs <= time_tolerance
            
            for i in range(chunk_size):
                if not valid_mask[i]:
                    continue
                    
                ping_idx = nearest_indices[i]
                column_data = backend.get_column(ping_idx)
                
                if column_data is None or len(column_data) == 0:
                    continue
                
                if y_grid is not None and y_coords is not None:
                    source_y = self._get_y_coordinates_for_ping(backend, ping_idx, y_coords)
                    if source_y is not None:
                        result[i, :] = self._resample_column(
                            column_data, source_y, y_grid, interpolation
                        )
                else:
                    n_copy = min(len(column_data), n_y)
                    result[i, :n_copy] = column_data[:n_copy]
        
        elif isinstance(backend, CombineBackend):
            # CombineBackend: need to get data with proper y-alignment
            for i in range(chunk_size):
                out_time = output_times_chunk[i]
                
                # Find best matching ping across all sub-backends
                column_data, source_y = self._get_column_and_y_for_time(
                    backend, out_time, y_coords, time_tolerance
                )
                
                if column_data is None or len(column_data) == 0:
                    continue
                
                if y_grid is not None and y_coords is not None and source_y is not None:
                    result[i, :] = self._resample_column(
                        column_data, source_y, y_grid, interpolation
                    )
                else:
                    n_copy = min(len(column_data), n_y)
                    result[i, :n_copy] = column_data[:n_copy]
        else:
            # Regular strategy or other cases
            backend_times = backend.ping_times
            for i in range(chunk_size):
                out_time = output_times_chunk[i]
                time_diffs = np.abs(backend_times - out_time)
                nearest_idx = np.argmin(time_diffs)
                
                if time_diffs[nearest_idx] > time_tolerance:
                    continue
                
                column_data = backend.get_column(nearest_idx)
                
                if column_data is None or len(column_data) == 0:
                    continue
                
                if y_grid is not None and y_coords is not None:
                    source_y = self._get_y_coordinates_for_ping(backend, nearest_idx, y_coords)
                    if source_y is not None:
                        result[i, :] = self._resample_column(
                            column_data, source_y, y_grid, interpolation
                        )
                else:
                    n_copy = min(len(column_data), n_y)
                    result[i, :n_copy] = column_data[:n_copy]
        
        return result

    def _get_column_and_y_for_time(
        self,
        backend,  # CombineBackend
        target_time: float,
        y_coords: str,
        time_tolerance: float,
    ) -> tuple:
        """Get column and Y coordinates from CombineBackend for a specific time.
        
        Returns:
            (column_data, source_y) tuple, or (None, None) if no match found.
        """
        from .backends import CombineBackend
        
        best_diff = float('inf')
        best_column = None
        best_source_y = None
        
        for i in range(backend.num_backends):
            sub = backend.get_backend(i)
            sub_times = sub.ping_times
            time_diffs = np.abs(sub_times - target_time)
            nearest_idx = np.argmin(time_diffs)
            diff = time_diffs[nearest_idx]
            
            if diff <= time_tolerance and diff < best_diff:
                best_diff = diff
                best_column = sub.get_column(nearest_idx)
                if y_coords is not None:
                    best_source_y = self._get_y_coordinates_for_ping(sub, nearest_idx, y_coords)
        
        return best_column, best_source_y

    def _get_y_coordinates_for_ping(
        self,
        backend,
        ping_idx: int,
        y_coords: str,
    ) -> np.ndarray:
        """Get Y coordinates (depth/range) for a specific ping."""
        if y_coords == "depth":
            extents = backend.depth_extents
        elif y_coords == "range":
            extents = backend.range_extents
        else:
            return None
        
        if extents is None:
            return None
        
        y_min, y_max = extents
        n_samples = int(backend.max_sample_counts[ping_idx]) + 1
        
        # Linear interpolation from min to max depth/range
        return np.linspace(y_min[ping_idx], y_max[ping_idx], n_samples)

    def _resample_column(
        self,
        source_data: np.ndarray,
        source_y: np.ndarray,
        target_y: np.ndarray,
        interpolation: str,
    ) -> np.ndarray:
        """Resample a column from source Y coordinates to target Y coordinates.
        
        Optimized for speed using vectorized numpy operations.
        """
        if len(source_data) == 0 or len(source_y) == 0:
            return np.full(len(target_y), np.nan, dtype=np.float32)
        
        # Ensure same length
        n = min(len(source_data), len(source_y))
        source_data = source_data[:n]
        source_y = source_y[:n]
        
        if interpolation == "nearest":
            # Vectorized nearest-neighbor interpolation
            # Use searchsorted for all target values at once
            indices = np.searchsorted(source_y, target_y)
            
            # Clip to valid range
            indices = np.clip(indices, 0, len(source_y) - 1)
            
            # For values between bins, check which neighbor is closer
            # (only needed for indices not at boundaries)
            mask_check = (indices > 0) & (indices < len(source_y))
            
            # Compute distances to left and right neighbors
            left_indices = np.clip(indices - 1, 0, len(source_y) - 1)
            left_dist = np.abs(target_y - source_y[left_indices])
            right_dist = np.abs(target_y - source_y[indices])
            
            # Use left neighbor where it's closer
            use_left = left_dist < right_dist
            final_indices = np.where(use_left, left_indices, indices)
            
            # Get values
            result = source_data[final_indices]
            
            # Mark values outside source range as NaN
            result = result.astype(np.float32)
            result[target_y < source_y[0]] = np.nan
            result[target_y > source_y[-1]] = np.nan
            
            return result
        
        elif interpolation == "linear":
            # Linear interpolation (already vectorized)
            result = np.interp(target_y, source_y, source_data, left=np.nan, right=np.nan)
            return result.astype(np.float32)
        
        return np.full(len(target_y), np.nan, dtype=np.float32)

    def _interpolate_latlon_to_times(self, target_times: np.ndarray) -> tuple:
        """Interpolate lat/lon coordinates to new ping times."""
        source_times = self._backend.ping_times
        source_lats = self._backend.latitudes
        source_lons = self._backend.longitudes
        
        # Simple linear interpolation
        new_lats = np.interp(target_times, source_times, source_lats)
        new_lons = np.interp(target_times, source_times, source_lons)
        
        return new_lats, new_lons
    
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
        
        The axis type settings (e.g., "Date time", "Depth (m)") are restored
        from the saved metadata, but zoom levels are not preserved.
        
        Args:
            path: Path to the mmap store (directory).
            
        Returns:
            EchogramBuilder with MmapDataBackend.
        """
        import json
        from pathlib import Path
        from .backends import MmapDataBackend
        
        backend = MmapDataBackend.from_path(path)
        builder = cls(backend)
        
        # Restore axis settings, oversampling, and layers from metadata
        metadata_file = Path(path) / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            x_axis_name = metadata.get("x_axis_name")
            y_axis_name = metadata.get("y_axis_name")
            
            # Apply saved axis type (without specific zoom parameters)
            builder._apply_axis_type(x_axis_name, y_axis_name)
            
            # Restore oversampling settings
            builder._x_oversampling = int(metadata.get("x_oversampling", 1))
            builder._y_oversampling = int(metadata.get("y_oversampling", 1))
            builder._oversampling_mode = metadata.get("oversampling_mode", "linear_mean")
            
            # Restore layers
            store_path = Path(path)
            builder._restore_layers_from_store(
                layer_names_json=json.dumps(metadata.get("layer_names", [])),
                has_main_layer=metadata.get("has_main_layer", False),
                get_array=lambda name: np.load(store_path / f"{name}.npy") if (store_path / f"{name}.npy").exists() else None,
            )
        
        return builder

