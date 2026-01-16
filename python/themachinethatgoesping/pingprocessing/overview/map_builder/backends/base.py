"""Abstract base class for map data backends.

Defines the interface that all map data sources must implement,
including coordinate system access, data retrieval, and downsampling.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

import numpy as np

from ..coordinate_system import MapCoordinateSystem, BoundingBox


class MapDataBackend(ABC):
    """Abstract base class for map data sources.
    
    Backends handle loading and accessing geospatial raster data from
    various sources (GeoTiff, NetCDF, Zarr, etc.). They provide:
    
    - Coordinate system information (CRS, transform, bounds)
    - Feature/variable names for colormap selection
    - Data retrieval with optional downsampling
    - Memory-efficient access for large-than-memory datasets
    
    Each backend manages its own memory (caching, lazy loading, etc.)
    and must handle data retrieval efficiently for interactive visualization.
    """
    
    # =========================================================================
    # Required properties
    # =========================================================================
    
    @property
    @abstractmethod
    def coordinate_system(self) -> MapCoordinateSystem:
        """Get the coordinate system for this data.
        
        Returns:
            MapCoordinateSystem with CRS, transform, and bounds.
        """
        pass
    
    @property
    @abstractmethod
    def feature_name(self) -> str:
        """Get the feature/variable name (e.g., 'bathymetry', 'backscatter').
        
        Used for automatic colormap selection and display labeling.
        
        Returns:
            String identifier for the data type.
        """
        pass
    
    @property
    @abstractmethod
    def bounds(self) -> BoundingBox:
        """Get the bounding box in world coordinates.
        
        Returns:
            BoundingBox with (xmin, ymin, xmax, ymax) in native CRS.
        """
        pass
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Get the raster shape (height, width) in pixels.
        
        Returns:
            Tuple of (height, width).
        """
        pass
    
    @property
    @abstractmethod
    def nodata(self) -> Optional[float]:
        """Get the nodata value, or None if not defined.
        
        Returns:
            Nodata value used to mark invalid/missing pixels.
        """
        pass
    
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Get the data type of raster values.
        
        Returns:
            NumPy dtype of the data.
        """
        pass
    
    # =========================================================================
    # Optional properties (with default implementations)
    # =========================================================================
    
    @property
    def units(self) -> Optional[str]:
        """Get the units of the data values (e.g., 'm', 'dB').
        
        Returns:
            Unit string or None if not defined.
        """
        return None
    
    @property
    def min_value(self) -> Optional[float]:
        """Get the minimum data value (for colormap scaling).
        
        Returns:
            Minimum value or None if not known.
        """
        return None
    
    @property
    def max_value(self) -> Optional[float]:
        """Get the maximum data value (for colormap scaling).
        
        Returns:
            Maximum value or None if not known.
        """
        return None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get additional metadata as a dictionary.
        
        Returns:
            Dictionary of metadata (empty by default).
        """
        return {}
    
    # =========================================================================
    # Required methods
    # =========================================================================
    
    @abstractmethod
    def get_data(
        self,
        bounds: Optional[BoundingBox] = None,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, MapCoordinateSystem]:
        """Get raster data for a region with optional downsampling.
        
        This is the primary data retrieval method. Backends should:
        - Return only the requested region (or full extent if bounds=None)
        - Downsample if the region is larger than max_size
        - Handle memory efficiently (don't load entire file if not needed)
        - Replace nodata values with NaN for float output
        
        Args:
            bounds: Bounding box in world coordinates. If None, return full extent.
            max_size: Maximum output size as (height, width). If the requested
                     region is larger, downsample to fit. If None, return at
                     full resolution.
            
        Returns:
            Tuple of:
            - 2D NumPy array with data values (NaN for nodata)
            - MapCoordinateSystem for the returned data (may differ from original
              if subregion or downsampled)
        """
        pass
    
    @abstractmethod
    def get_value_at(
        self,
        x: float,
        y: float,
    ) -> Optional[float]:
        """Get the data value at a world coordinate.
        
        Args:
            x: X coordinate in world CRS.
            y: Y coordinate in world CRS.
            
        Returns:
            Data value at the point, or None if outside bounds or nodata.
        """
        pass
    
    # =========================================================================
    # Optional methods (with default implementations)
    # =========================================================================
    
    def get_data_at_latlon(
        self,
        lat: float,
        lon: float,
    ) -> Optional[float]:
        """Get the data value at a lat/lon coordinate.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            
        Returns:
            Data value at the point, or None if outside bounds or nodata.
        """
        x, y = self.coordinate_system.latlon_to_world(lat, lon)
        return self.get_value_at(x, y)
    
    def get_overview(
        self,
        max_size: Tuple[int, int] = (1000, 1000),
    ) -> Tuple[np.ndarray, MapCoordinateSystem]:
        """Get a downsampled overview of the entire dataset.
        
        Useful for initial display before zooming to regions of interest.
        
        Args:
            max_size: Maximum size as (height, width).
            
        Returns:
            Tuple of (data array, coordinate system).
        """
        return self.get_data(bounds=None, max_size=max_size)
    
    def close(self) -> None:
        """Release any resources held by the backend.
        
        Override this if your backend holds open file handles or connections.
        Default implementation does nothing.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the backend."""
        self.close()
        return False
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def compute_downsample_factor(
        self,
        bounds: BoundingBox,
        max_size: Tuple[int, int],
    ) -> int:
        """Compute the downsample factor needed to fit bounds within max_size.
        
        Args:
            bounds: Requested region bounds.
            max_size: Maximum output size (height, width).
            
        Returns:
            Downsample factor (1 = no downsampling, 2 = half resolution, etc.)
        """
        cs = self.coordinate_system
        
        # Get pixel bounds for the world extent
        col_min, row_min, col_max, row_max = cs.get_pixel_bounds_for_world_extent(bounds)
        
        # Compute region size in pixels
        region_width = col_max - col_min
        region_height = row_max - row_min
        
        if region_width <= 0 or region_height <= 0:
            return 1
        
        # Compute factor to fit within max_size
        factor_x = max(1, region_width // max_size[1])
        factor_y = max(1, region_height // max_size[0])
        
        return max(factor_x, factor_y)
    
    # =========================================================================
    # String representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"feature='{self.feature_name}', "
            f"shape={self.shape}, "
            f"bounds={self.bounds})"
        )
