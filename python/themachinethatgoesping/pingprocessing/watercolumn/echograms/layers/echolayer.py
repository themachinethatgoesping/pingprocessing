# This is an internal class used by the echogram class to represent a layer in the echogram.
"""Layer classes for echogram region selection.

EchoLayer represents a region of interest within an echogram, defined by
upper and lower bounds that can vary across pings. Layers are used to
isolate specific water column regions for analysis.

PingData provides a convenient wrapper for accessing per-ping data including
layer-specific extractions.
"""

import datetime as dt
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, Dict

# external Ping packages
from themachinethatgoesping import tools

# internal pingprocessing imports
from themachinethatgoesping.pingprocessing.core.asserts import assert_length

if TYPE_CHECKING:
    from ..echogrambuilder_refactored import EchogramBuilder
    from ..coordinate_system import EchogramCoordinateSystem


class EchoLayer:
    """Represents a region of interest within an echogram.
    
    A layer defines upper and lower bounds in Y coordinates that can vary
    across pings (X axis). This is useful for isolating specific water column
    regions like the bottom layer, surface layer, or acoustic layers.
    
    Attributes:
        echodata: Reference to the parent EchogramBuilder.
        i0: Array of start sample indices per ping.
        i1: Array of end sample indices per ping (exclusive).
        y0: Array of start grid indices per ping.
        y1: Array of end grid indices per ping.
        vec_min_y: Array of minimum Y coordinates per ping.
        vec_max_y: Array of maximum Y coordinates per ping.
    """

    def __init__(self, echodata: "EchogramBuilder", vec_x_val, vec_min_y, vec_max_y):
        """Create a layer with explicit boundaries.
        
        Args:
            echodata: Parent EchogramBuilder instance.
            vec_x_val: X values (timestamps, indices, or datetimes) for boundary points.
            vec_min_y: Minimum Y values (depths, ranges, etc.) at each X point.
            vec_max_y: Maximum Y values at each X point.
        """
        self.echodata = echodata
        cs = self._cs  # coordinate system shortcut
        
        if vec_min_y is None:
            vec_min_y = np.zeros(len(vec_x_val))

        if vec_max_y is None:
            vec_max_y = np.empty(len(vec_x_val))
            vec_max_y.fill(cs.feature_mapper.get_feature_values('X coordinate')[-1])
            
        assert_length("EchoLayer.__init__", vec_x_val, [vec_min_y, vec_max_y])
        
        # convert datetimes to timestamps
        if len(vec_x_val) > 0:
            first_val = vec_x_val.iloc[0] if hasattr(vec_x_val, 'iloc') else vec_x_val[0]
            if isinstance(first_val, dt.datetime):
                vec_x_val = [x.timestamp() for x in vec_x_val]
        
        # convert to numpy arrays
        vec_x_val = np.array(vec_x_val).astype(float)
        vec_min_y = np.array(vec_min_y).astype(float)
        vec_max_y = np.array(vec_max_y).astype(float)
        
        # filter nans and infs
        valid = np.isfinite(vec_x_val) & np.isfinite(vec_min_y) & np.isfinite(vec_max_y)
        vec_x_val = vec_x_val[valid]
        vec_min_y = vec_min_y[valid]
        vec_max_y = vec_max_y[valid]
        
        # Interpolate to all pings
        x_values = cs.feature_mapper.get_feature_values(cs.x_axis_name)
        vec_min_y = tools.vectorinterpolators.LinearInterpolator(
            vec_x_val, vec_min_y, extrapolation_mode='nearest'
        )(x_values)
        vec_max_y = tools.vectorinterpolators.LinearInterpolator(
            vec_x_val, vec_max_y, extrapolation_mode='nearest'
        )(x_values)
        
        # Get n_pings from backend or pings list
        n_pings = self._get_n_pings()
        
        # Create layer indices using affine transforms (vectorized)
        # sample_index = a_inv + b_inv * y_coord (inverse of y = a + b * sample)
        if cs._affine_y_to_sample is not None:
            a_inv, b_inv = cs._affine_y_to_sample
            i0 = np.round(a_inv + b_inv * vec_min_y).astype(int)
            i1 = np.round(a_inv + b_inv * vec_max_y).astype(int) + 1
        else:
            # Fallback if affine not set
            i0 = np.zeros(n_pings, dtype=int)
            i1 = np.zeros(n_pings, dtype=int)

        self.set_indices(i0, i1, vec_min_y, vec_max_y)

    @property
    def _cs(self) -> "EchogramCoordinateSystem":
        """Get coordinate system from echodata (supports both old and new builders)."""
        if hasattr(self.echodata, '_coord_system'):
            return self.echodata._coord_system
        # Fallback for old builder - return echodata itself (it has the same interface)
        return self.echodata

    def _get_n_pings(self) -> int:
        """Get number of pings from echodata."""
        if hasattr(self.echodata, '_backend'):
            return self.echodata._backend.n_pings
        return len(self.echodata.pings)

    def _get_max_samples(self, wci_nr: int) -> int:
        """Get max sample count for a specific ping."""
        cs = self._cs
        if hasattr(cs, 'max_number_of_samples'):
            return int(cs.max_number_of_samples[wci_nr]) + 1
        return self.echodata.beam_sample_selections[wci_nr].get_number_of_samples_ensemble()

    def set_indices(self, i0, i1, vec_min_y, vec_max_y):
        """Set layer sample indices and update grid indices.
        
        Args:
            i0: Start sample indices per ping.
            i1: End sample indices per ping (exclusive).
            vec_min_y: Minimum Y coordinates per ping.
            vec_max_y: Maximum Y coordinates per ping.
        """
        n_pings = self._get_n_pings()
        assert len(i0) == len(i1) == n_pings, \
            f"Index arrays must have length {n_pings}"

        self.i0 = np.array(i0)
        self.i1 = np.array(i1)

        # Clamp to valid range
        self.i0 = np.maximum(self.i0, 0)
        self.i1 = np.maximum(self.i1, self.i0)
        
        # Clamp to max samples per ping
        max_samples = self._cs.max_number_of_samples
        self.i1 = np.minimum(self.i1, max_samples + 1)
        
        self.vec_min_y = np.array(vec_min_y).astype(float)
        self.vec_max_y = np.array(vec_max_y).astype(float)
        self.update_y_gridder()
        
    def update_y_gridder(self):
        """Update grid indices after coordinate system change."""
        cs = self._cs
        y0 = cs.y_gridder.get_x_index(self.vec_min_y)
        y1 = cs.y_gridder.get_x_index(self.vec_max_y) + 1
        self.y0 = np.array(y0)
        self.y1 = np.array(y1)
        self.y0 = np.maximum(self.y0, 0)
        self.y1 = np.maximum(self.y1, self.y0)
        self.y1 = np.minimum(self.y1, cs.y_gridder.get_nx())

    @classmethod
    def from_static_layer(cls, echodata: "EchogramBuilder", min_y: float, max_y: float) -> "EchoLayer":
        """Create a layer with static (constant) boundaries.
        
        Args:
            echodata: Parent EchogramBuilder instance.
            min_y: Constant minimum Y value across all pings.
            max_y: Constant maximum Y value across all pings.
            
        Returns:
            New EchoLayer instance.
        """
        # Get coord_system - support both old and new builders
        cs = echodata._coord_system if hasattr(echodata, '_coord_system') else echodata
        vec_x_val = cs.feature_mapper.get_feature_values(cs.x_axis_name)
        min_y_arr = [min_y, min_y] if min_y is not None else None
        max_y_arr = [max_y, max_y] if max_y is not None else None
        return cls(echodata, [vec_x_val[0], vec_x_val[-1]], min_y_arr, max_y_arr)

    @classmethod
    def from_ping_param_offsets_absolute(
        cls, echodata: "EchogramBuilder", ping_param_name: str,
        offset_0: Optional[float], offset_1: Optional[float]
    ) -> "EchoLayer":
        """Create a layer based on absolute offsets from a ping parameter.
        
        Args:
            echodata: Parent EchogramBuilder instance.
            ping_param_name: Name of the ping parameter (e.g., 'bottom').
            offset_0: Absolute offset for upper bound (added to param).
            offset_1: Absolute offset for lower bound (added to param).
            
        Returns:
            New EchoLayer instance.
        """
        x, y = echodata.get_ping_param(ping_param_name)
        y0 = np.array(y) + offset_0 if offset_0 is not None else None
        y1 = np.array(y) + offset_1 if offset_1 is not None else None
        return cls(echodata, x, y0, y1)

    @classmethod
    def from_ping_param_offsets_relative(
        cls, echodata: "EchogramBuilder", ping_param_name: str,
        offset_0: Optional[float], offset_1: Optional[float]
    ) -> "EchoLayer":
        """Create a layer based on relative offsets from a ping parameter.
        
        Args:
            echodata: Parent EchogramBuilder instance.
            ping_param_name: Name of the ping parameter (e.g., 'bottom').
            offset_0: Relative multiplier for upper bound.
            offset_1: Relative multiplier for lower bound.
            
        Returns:
            New EchoLayer instance.
        """
        x, y = echodata.get_ping_param(ping_param_name)
        y0 = np.array(y) * offset_0 if offset_0 is not None else None
        y1 = np.array(y) * offset_1 if offset_1 is not None else None
        return cls(echodata, x, y0, y1)

    def get_y_indices(self, wci_nr: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get Y indices constrained to layer bounds.
        
        Uses precomputed affine coefficients for speed.
        
        Args:
            wci_nr: Ping/column number.
        
        Returns:
            Tuple of (image_indices, data_indices) arrays, or (None, None) if no valid range.
        """
        cs = self._cs
        n_samples = self._get_max_samples(wci_nr)
        
        # Use affine transform: sample_index = a + b * y_coord
        if cs._affine_y_to_sample is None:
            return None, None
        
        a_inv, b_inv = cs._affine_y_to_sample
        a, b = a_inv[wci_nr], b_inv[wci_nr]
        
        if not np.isfinite(a) or not np.isfinite(b):
            return None, None
            
        y_indices_image = np.arange(len(cs.y_coordinates))
        y_indices_wci = np.round(a + b * cs.y_coordinates).astype(int)

        start_y = max(0, self.i0[wci_nr])
        end_y = min(n_samples, self.i1[wci_nr])

        if start_y >= end_y:
            return None, None
            
        valid_coordinates = np.where(
            (y_indices_wci >= start_y) & (y_indices_wci < end_y)
        )[0]

        return y_indices_image[valid_coordinates], y_indices_wci[valid_coordinates]

    def combine(self, other: "EchoLayer"):
        """Combine this layer with another by taking the intersection.
        
        The resulting layer will have the more restrictive bounds from both layers.
        
        Args:
            other: Another EchoLayer to combine with.
        """
        assert len(self.i0) == len(other.i0), "Layers must have the same number of pings"
        
        i0 = np.maximum(self.i0, other.i0)
        i1 = np.minimum(self.i1, other.i1)
        vec_min_y = np.maximum(self.vec_min_y, other.vec_min_y)
        vec_max_y = np.minimum(self.vec_max_y, other.vec_max_y)

        self.set_indices(i0, i1, vec_min_y, vec_max_y)


class PingData:
    """Wrapper for accessing per-ping data from an EchogramBuilder.
    
    Provides convenient access to water column data and layer extractions
    for a specific ping.
    
    Attributes:
        echodata: Reference to the parent EchogramBuilder.
        nr: Ping index.
    """

    def __init__(self, echodata: "EchogramBuilder", nr: int):
        """Initialize PingData for a specific ping.
        
        Args:
            echodata: Parent EchogramBuilder instance.
            nr: Ping index.
        """
        self.echodata = echodata
        self.nr = nr

    @property
    def _cs(self) -> "EchogramCoordinateSystem":
        """Get coordinate system from echodata (supports both old and new builders)."""
        if hasattr(self.echodata, '_coord_system'):
            return self.echodata._coord_system
        return self.echodata

    def get_wci(self) -> np.ndarray:
        """Get water column image data for this ping."""
        return self.echodata.get_column(self.nr)

    def get_wci_layers(self) -> Dict[str, np.ndarray]:
        """Get water column data split by layers."""
        return self.echodata.get_wci_layers(self.nr)

    def get_extent_layers(self, axis_name: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
        """Get layer extents in specified coordinate system."""
        return self.echodata.get_extent_layers(self.nr, axis_name=axis_name)

    def get_limits_layers(self, axis_name: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
        """Get layer limits in specified coordinate system."""
        return self.echodata.get_limits_layers(self.nr, axis_name=axis_name)

    def get_ping_time(self) -> float:
        """Get ping timestamp."""
        return self._cs.feature_mapper.index_to_feature(self.nr)

    def get_datetime(self) -> dt.datetime:
        """Get ping datetime."""
        return dt.datetime.fromtimestamp(self.get_ping_time(), self._cs.time_zone)
