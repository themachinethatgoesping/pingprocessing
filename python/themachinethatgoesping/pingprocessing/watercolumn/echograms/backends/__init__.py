"""Echogram data backends for different data sources."""

from .base import EchogramDataBackend
from .ping_backend import PingDataBackend
from .zarr_backend import ZarrDataBackend
from .mmap_backend import MmapDataBackend

# Keep old name as alias for backwards compatibility
MmapDataBackend = MmapDataBackend

__all__ = [
    "EchogramDataBackend",
    "PingDataBackend", 
    "ZarrDataBackend",
    "MmapDataBackend",
    "MmapDataBackend",  # alias for backwards compatibility
]
