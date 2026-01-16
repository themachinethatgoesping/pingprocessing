"""MapBuilder module for loading and viewing geospatial data.

This module provides tools for loading, transforming, and visualizing
geospatial raster data such as bathymetry, backscatter, and other map layers.

Similar in design to the echograms module, it uses:
- Pluggable backends (GeoTiff, etc.) for data loading
- A coordinate system class for CRS transformations
- Affine transforms for pixel ↔ world coordinate mapping

Example:
    from themachinethatgoesping.pingprocessing.widgets import MapViewerPyQtGraph
    from themachinethatgoesping.pingprocessing.overview.map_builder import MapBuilder, TileBuilder
    
    # Data layers (numerical, with colorbar)
    builder = MapBuilder()
    builder.add_geotiff('map/BPNS_latlon.tiff')
    
    # Tile layers (pre-rendered images, no colorbar)
    tiles = TileBuilder()
    tiles.add_osm()  # or add_esri_worldimagery(), add_cartodb_positron(), etc.
    
    # Auto-displays in Jupyter
    viewer = MapViewerPyQtGraph(builder, tile_builder=tiles)
    
    # Connect to echogram viewer to show tracks
    viewer.connect_echogram_viewer(echogram_viewer)
"""

from .coordinate_system import MapCoordinateSystem, BoundingBox
from .map_builder import MapBuilder, MapLayer
from .backends import MapDataBackend, GeoTiffBackend
from .tile_builder import TileBuilder, TileSource, TILE_SOURCES, list_available_sources

__all__ = [
    "MapCoordinateSystem",
    "BoundingBox",
    "MapBuilder",
    "MapLayer",
    "MapDataBackend",
    "GeoTiffBackend",
    "TileBuilder",
    "TileSource",
    "TILE_SOURCES",
    "list_available_sources",
]
