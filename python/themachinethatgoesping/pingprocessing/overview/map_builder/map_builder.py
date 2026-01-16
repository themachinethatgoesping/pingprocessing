"""MapBuilder class for managing multiple map layers.

Provides a unified interface for loading, combining, and displaying
multiple geospatial data layers with configurable overlay ordering.
"""

from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
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
        opacity: Layer opacity (0.0 = transparent, 1.0 = opaque).
        colormap: Colormap name for rendering.
        vmin: Minimum value for colormap scaling (None = auto).
        vmax: Maximum value for colormap scaling (None = auto).
        z_order: Rendering order (higher = on top).
    """
    backend: MapDataBackend
    name: str
    visible: bool = True
    opacity: float = 1.0
    colormap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    z_order: int = 0
    
    def __post_init__(self):
        """Set default colormap based on feature type."""
        if self.colormap is None:
            self.colormap = self._default_colormap()
        if self.vmin is None:
            self.vmin = self.backend.min_value
        if self.vmax is None:
            self.vmax = self.backend.max_value
    
    def _default_colormap(self) -> str:
        """Get default colormap based on feature name."""
        feature = self.backend.feature_name.lower()
        
        if 'bathymetry' in feature or 'depth' in feature:
            return 'viridis_r'  # Reversed: deep = dark blue
        elif 'backscatter' in feature:
            return 'gray'
        elif 'slope' in feature:
            return 'RdYlGn_r'
        elif 'aspect' in feature:
            return 'hsv'  # Cyclic colormap for aspect
        elif 'rugosity' in feature or 'roughness' in feature:
            return 'YlOrRd'
        
        return 'viridis'


class MapBuilder:
    """Builder for managing multiple map layers.
    
    MapBuilder provides:
    - Layer management (add, remove, reorder)
    - Unified coordinate system handling
    - Data retrieval for rendering
    
    Example usage:
        # Create builder and add layers
        builder = MapBuilder()
        builder.add_geotiff("bathymetry.tif", name="Bathymetry")
        builder.add_geotiff("backscatter.tif", name="Backscatter", opacity=0.5)
        
        # Create viewer
        viewer = MapViewerPyQtGraph(builder)
    """
    
    def __init__(self):
        """Initialize an empty MapBuilder."""
        self._layers: List[MapLayer] = []
        self._unified_crs: Optional[Any] = None
    
    # =========================================================================
    # Layer management
    # =========================================================================
    
    def add_layer(
        self,
        backend: MapDataBackend,
        name: Optional[str] = None,
        visible: bool = True,
        opacity: float = 1.0,
        colormap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        z_order: Optional[int] = None,
    ) -> "MapBuilder":
        """Add a data layer.
        
        Args:
            backend: The data backend for this layer.
            name: Display name (default: backend feature name).
            visible: Whether layer is initially visible.
            opacity: Layer opacity (0.0-1.0).
            colormap: Colormap name (default: auto from feature).
            vmin: Min value for colormap (default: from backend).
            vmax: Max value for colormap (default: from backend).
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
            opacity=opacity,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
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
        """Remove a layer by name.
        
        Args:
            name: Name of the layer to remove.
            
        Returns:
            Self for method chaining.
        """
        self._layers = [l for l in self._layers if l.name != name]
        return self
    
    def get_layer(self, name: str) -> Optional[MapLayer]:
        """Get a layer by name.
        
        Args:
            name: Name of the layer.
            
        Returns:
            MapLayer or None if not found.
        """
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None
    
    def set_layer_visibility(self, name: str, visible: bool) -> "MapBuilder":
        """Set visibility of a layer.
        
        Args:
            name: Layer name.
            visible: Whether layer should be visible.
            
        Returns:
            Self for method chaining.
        """
        layer = self.get_layer(name)
        if layer:
            layer.visible = visible
        return self
    
    def set_layer_opacity(self, name: str, opacity: float) -> "MapBuilder":
        """Set opacity of a layer.
        
        Args:
            name: Layer name.
            opacity: Opacity value (0.0-1.0).
            
        Returns:
            Self for method chaining.
        """
        layer = self.get_layer(name)
        if layer:
            layer.opacity = max(0.0, min(1.0, opacity))
        return self
    
    def set_layer_order(self, name: str, z_order: int) -> "MapBuilder":
        """Set rendering order of a layer.
        
        Args:
            name: Layer name.
            z_order: New z-order value.
            
        Returns:
            Self for method chaining.
        """
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
    # Data retrieval
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
            bounds: Bounding box (None = full extent).
            max_size: Maximum output size for downsampling.
            
        Returns:
            Tuple of (data array, coordinate system) or None if layer not found.
        """
        layer = self.get_layer(layer_name)
        if layer is None:
            return None
        
        return layer.backend.get_data(bounds, max_size)
    
    def get_all_layer_data(
        self,
        bounds: Optional[BoundingBox] = None,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[MapLayer, np.ndarray, MapCoordinateSystem]]:
        """Get data for all visible layers.
        
        Args:
            bounds: Bounding box (None = full extent).
            max_size: Maximum output size for downsampling.
            
        Returns:
            List of (layer, data array, coordinate system) tuples.
        """
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
        """Get values from all layers at a lat/lon position.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            
        Returns:
            Dictionary mapping layer names to values (or None if nodata).
        """
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
    
    # =========================================================================
    # String representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return f"MapBuilder(layers={len(self._layers)})"
