"""MapDataBackend implementations for various geospatial data sources."""

from .base import MapDataBackend
from .geotiff_backend import GeoTiffBackend

__all__ = [
    "MapDataBackend",
    "GeoTiffBackend",
]
