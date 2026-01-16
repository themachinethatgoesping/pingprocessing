"""MapBuilder module for loading and viewing geospatial data.

This module provides tools for loading, transforming, and visualizing
geospatial raster data such as bathymetry, backscatter, and other map layers.

Similar in design to the echograms module, it uses:
- Pluggable backends (GeoTiff, etc.) for data loading
- A coordinate system class for CRS transformations
- Affine transforms for pixel ↔ world coordinate mapping

Example:
    from themachinethatgoesping.pingprocessing.widgets import MapViewerPyQtGraph
    from themachinethatgoesping.pingprocessing.overview.map_builder import MapBuilder
    
    builder = MapBuilder()
    builder.add_geotiff('map/BPNS_latlon.tiff')
    
    # Auto-displays in Jupyter (like EchogramViewer)
    viewer = MapViewerPyQtGraph(builder)
    
    # Connect to echogram viewer to show tracks
    viewer.connect_echogram_viewer(echogram_viewer)
"""

from .coordinate_system import MapCoordinateSystem, BoundingBox
from .map_builder import MapBuilder, MapLayer
from .backends import MapDataBackend, GeoTiffBackend

__all__ = [
    "MapCoordinateSystem",
    "BoundingBox",
    "MapBuilder",
    "MapLayer",
    "MapDataBackend",
    "GeoTiffBackend",
]
