"""MapBuilder class for managing multiple map layers.

Provides a unified interface for loading, combining, and displaying
multiple geospatial data layers. Acts as a data provider with axis/resolution
control, similar to EchogramBuilder.

The MapBuilder controls:
- Layer management (add, remove, visibility, ordering)
- Axis/resolution settings for data retrieval
- Coordinate system handling

Rendering properties (colormap, opacity, blending) are handled by the viewer.
"""

from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .coordinate_system import MapCoordinateSystem, BoundingBox
from .backends import MapDataBackend, GeoTiffBackend


@dataclass
class MapLayer:
    """Configuration for a single map layer.
    
    Attributes:
        backend: The data backend for this layer.
        name: Display name for the layer.
        visible: Whether the layer is currently visible.
        z_order: Rendering order (higher = on top).
    """
    backend: MapDataBackend
    name: str
    visible: bool = True
    z_order: int = 0


class MapBuilder:
    """Builder for managing multiple map layers.
    
    MapBuilder provides:
    - Layer management (add, remove, reorder, visibility)
    - Axis/resolution control for data retrieval
    - Coordinate system handling
    
    Similar to EchogramBuilder, it acts as a data provider.
    Rendering properties are controlled by the viewer.
    
    Example usage:
        from themachinethatgoesping.pingprocessing.overview.map_builder import MapBuilder
        from themachinethatgoesping.pingprocessing.widgets import MapViewerPyQtGraph
        
        # Create builder and add layers
        builder = MapBuilder()
        builder.add_geotiff('map/BPNS_latlon.tiff', name="Bathymetry")
        
        # Set axis/resolution
        builder.set_axis(max_pixels=(1000, 1000))
        
        # Create viewer (handles colorscale, opacity, blending)
        viewer = MapViewerPyQtGraph(builder)
    """
    
    def __init__(self):
        """Initialize an empty MapBuilder."""
        self._layers: List[MapLayer] = []
        
        # Axis/resolution settings (similar to EchogramBuilder)
        self._max_pixels: Tuple[int, int] = (2000, 2000)  # (height, width)
        self._current_bounds: Optional[BoundingBox] = None
    
    # =========================================================================
    # Axis/resolution settings (like EchogramBuilder)
    # =========================================================================
    
    def set_axis_latlon(
        self,
        min_lat: float = np.nan,
        max_lat: float = np.nan,
        min_lon: float = np.nan,
        max_lon: float = np.nan,
        max_pixels: Tuple[int, int] = (2000, 2000),
    ) -> "MapBuilder":
        """Set axis extent in lat/lon coordinates.
        
        Similar to EchogramBuilder's set_x_axis_date_time pattern.
        Use np.nan for auto-detection from data bounds.
        
        Args:
            min_lat: Minimum latitude (nan = auto from data).
            max_lat: Maximum latitude (nan = auto from data).
            min_lon: Minimum longitude (nan = auto from data).
            max_lon: Maximum longitude (nan = auto from data).
            max_pixels: Maximum output size (height, width).
            
        Returns:
            Self for method chaining.
        """
        self._max_pixels = max_pixels
        
        # Get full bounds for auto values
        full_bounds = self.combined_bounds
        if full_bounds is None:
            self._current_bounds = None
            return self
        
        # Replace nan with full extent
        # In lat/lon: x=longitude, y=latitude
        xmin = min_lon if not np.isnan(min_lon) else full_bounds.xmin
        xmax = max_lon if not np.isnan(max_lon) else full_bounds.xmax
        ymin = min_lat if not np.isnan(min_lat) else full_bounds.ymin
        ymax = max_lat if not np.isnan(max_lat) else full_bounds.ymax
        
        self._current_bounds = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        return self
    
    def set_axis_utm(
        self,
        min_easting: float = np.nan,
        max_easting: float = np.nan,
        min_northing: float = np.nan,
        max_northing: float = np.nan,
        max_pixels: Tuple[int, int] = (2000, 2000),
    ) -> "MapBuilder":
        """Set axis extent in UTM/projected coordinates.
        
        Similar to EchogramBuilder's axis-setting pattern.
        Use np.nan for auto-detection from data bounds.
        
        Args:
            min_easting: Minimum easting (nan = auto from data).
            max_easting: Maximum easting (nan = auto from data).
            min_northing: Minimum northing (nan = auto from data).
            max_northing: Maximum northing (nan = auto from data).
            max_pixels: Maximum output size (height, width).
            
        Returns:
            Self for method chaining.
        """
        self._max_pixels = max_pixels
        
        # Get full bounds for auto values
        full_bounds = self.combined_bounds
        if full_bounds is None:
            self._current_bounds = None
            return self
        
        # Replace nan with full extent
        xmin = min_easting if not np.isnan(min_easting) else full_bounds.xmin
        xmax = max_easting if not np.isnan(max_easting) else full_bounds.xmax
        ymin = min_northing if not np.isnan(min_northing) else full_bounds.ymin
        ymax = max_northing if not np.isnan(max_northing) else full_bounds.ymax
        
        self._current_bounds = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        return self
    
    def set_axis(
        self,
        bounds: Optional[BoundingBox] = None,
        max_pixels: Optional[Tuple[int, int]] = None,
    ) -> "MapBuilder":
        """Set axis extent and resolution for data retrieval (legacy method).
        
        For more explicit control, use set_axis_latlon() or set_axis_utm().
        
        Args:
            bounds: Bounding box in world coordinates (None = full extent).
            max_pixels: Maximum output size (height, width) for downsampling.
            
        Returns:
            Self for method chaining.
        """
        if bounds is not None:
            self._current_bounds = bounds
        if max_pixels is not None:
            self._max_pixels = max_pixels
        return self
    
    def set_bounds(self, bounds: BoundingBox) -> "MapBuilder":
        """Set the current view bounds."""
        self._current_bounds = bounds
        return self
    
    def set_max_pixels(self, max_pixels: Tuple[int, int]) -> "MapBuilder":
        """Set maximum output resolution."""
        self._max_pixels = max_pixels
        return self
    
    def reset_bounds(self) -> "MapBuilder":
        """Reset bounds to full extent."""
        self._current_bounds = None
        return self
    
    @property
    def max_pixels(self) -> Tuple[int, int]:
        """Current max pixels setting."""
        return self._max_pixels
    
    @property
    def current_bounds(self) -> Optional[BoundingBox]:
        """Current view bounds (None = full extent)."""
        return self._current_bounds
    
    # =========================================================================
    # Layer management
    # =========================================================================
    
    def add_layer(
        self,
        backend: MapDataBackend,
        name: Optional[str] = None,
        visible: bool = True,
        z_order: Optional[int] = None,
    ) -> "MapBuilder":
        """Add a data layer.
        
        Args:
            backend: The data backend for this layer.
            name: Display name (default: backend feature name).
            visible: Whether layer is initially visible.
            z_order: Render order (default: next available).
            
        Returns:
            Self for method chaining.
        """
        if name is None:
            name = backend.feature_name.capitalize()
        
        if z_order is None:
            z_order = len(self._layers)
        
        layer = MapLayer(
            backend=backend,
            name=name,
            visible=visible,
            z_order=z_order,
        )
        
        self._layers.append(layer)
        self._layers.sort(key=lambda l: l.z_order)
        
        return self
    
    def add_geotiff(
        self,
        path: str,
        name: Optional[str] = None,
        band: int = 1,
        **kwargs,
    ) -> "MapBuilder":
        """Add a GeoTiff layer.
        
        Convenience method that creates a GeoTiffBackend and adds it.
        
        Args:
            path: Path to the GeoTiff file.
            name: Display name (default: inferred from file).
            band: Band number to read (1-indexed).
            **kwargs: Additional arguments for add_layer().
            
        Returns:
            Self for method chaining.
        """
        backend = GeoTiffBackend(path, band=band)
        
        if name is None:
            name = Path(path).stem
        
        return self.add_layer(backend, name=name, **kwargs)
    
    def remove_layer(self, name: str) -> "MapBuilder":
        """Remove a layer by name."""
        self._layers = [l for l in self._layers if l.name != name]
        return self
    
    def get_layer(self, name: str) -> Optional[MapLayer]:
        """Get a layer by name."""
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None
    
    def set_layer_visibility(self, name: str, visible: bool) -> "MapBuilder":
        """Set visibility of a layer."""
        layer = self.get_layer(name)
        if layer:
            layer.visible = visible
        return self
    
    def set_layer_order(self, name: str, z_order: int) -> "MapBuilder":
        """Set rendering order of a layer."""
        layer = self.get_layer(name)
        if layer:
            layer.z_order = z_order
            self._layers.sort(key=lambda l: l.z_order)
        return self
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def layers(self) -> List[MapLayer]:
        """List of all layers (sorted by z-order)."""
        return list(self._layers)
    
    @property
    def visible_layers(self) -> List[MapLayer]:
        """List of visible layers (sorted by z-order)."""
        return [l for l in self._layers if l.visible]
    
    @property
    def combined_bounds(self) -> Optional[BoundingBox]:
        """Get combined bounding box of all visible layers."""
        if not self._layers:
            return None
        
        bounds = None
        for layer in self.visible_layers:
            layer_bounds = layer.backend.bounds
            if bounds is None:
                bounds = layer_bounds
            else:
                bounds = bounds.union(layer_bounds)
        
        return bounds
    
    @property
    def primary_coordinate_system(self) -> Optional[MapCoordinateSystem]:
        """Get coordinate system from the first layer."""
        if not self._layers:
            return None
        return self._layers[0].backend.coordinate_system
    
    # =========================================================================
    # Data retrieval (like EchogramBuilder.get_echogram_image)
    # =========================================================================
    
    def get_layer_data(
        self,
        layer_name: str,
        bounds: Optional[BoundingBox] = None,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[np.ndarray, MapCoordinateSystem]]:
        """Get data for a single layer.
        
        Args:
            layer_name: Name of the layer.
            bounds: Bounding box (None = use current_bounds or full extent).
            max_size: Maximum output size (None = use max_pixels setting).
            
        Returns:
            Tuple of (data array, coordinate system) or None if layer not found.
        """
        layer = self.get_layer(layer_name)
        if layer is None:
            return None
        
        # Use current settings if not specified
        if bounds is None:
            bounds = self._current_bounds
        if max_size is None:
            max_size = self._max_pixels
        
        return layer.backend.get_data(bounds, max_size)
    
    def get_all_layer_data(
        self,
        bounds: Optional[BoundingBox] = None,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[MapLayer, np.ndarray, MapCoordinateSystem]]:
        """Get data for all visible layers.
        
        Args:
            bounds: Bounding box (None = use current_bounds or full extent).
            max_size: Maximum output size (None = use max_pixels setting).
            
        Returns:
            List of (layer, data array, coordinate system) tuples.
        """
        # Use current settings if not specified
        if bounds is None:
            bounds = self._current_bounds
        if max_size is None:
            max_size = self._max_pixels
        
        results = []
        for layer in self.visible_layers:
            data, cs = layer.backend.get_data(bounds, max_size)
            results.append((layer, data, cs))
        return results
    
    def get_value_at_position(
        self,
        lat: float,
        lon: float,
    ) -> Dict[str, Optional[float]]:
        """Get values from all layers at a lat/lon position."""
        results = {}
        for layer in self._layers:
            value = layer.backend.get_data_at_latlon(lat, lon)
            results[layer.name] = value
        return results
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def close(self) -> None:
        """Close all backends and release resources."""
        for layer in self._layers:
            layer.backend.close()
        self._layers.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __repr__(self) -> str:
        return f"MapBuilder(layers={len(self._layers)})"
