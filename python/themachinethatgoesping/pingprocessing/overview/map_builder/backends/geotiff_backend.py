"""GeoTiff backend for loading raster data from GeoTiff files.

Uses rasterio for efficient windowed reading and built-in overview support.
"""

from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window, from_bounds
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from affine import Affine
except ImportError:
    Affine = None

from .base import MapDataBackend
from ..coordinate_system import MapCoordinateSystem, BoundingBox


class GeoTiffBackend(MapDataBackend):
    """Backend for loading data from GeoTiff files.
    
    Features:
    - Memory-efficient windowed reading (only loads requested regions)
    - Automatic use of internal overviews for downsampling
    - Support for multi-band files (reads first band by default)
    - Automatic nodata handling and NaN conversion
    
    Requires: rasterio (pip install rasterio)
    """
    
    def __init__(
        self,
        path: str,
        feature_name: Optional[str] = None,
        band: int = 1,
        preload_stats: bool = True,
    ):
        """Open a GeoTiff file.
        
        Args:
            path: Path to the GeoTiff file.
            feature_name: Name of the feature/variable. If None, infers from
                         filename or defaults to 'raster'.
            band: Band number to read (1-indexed). Default is 1.
            preload_stats: If True, compute min/max from overviews or sampling.
        """
        if not HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for GeoTiffBackend. "
                "Install with: pip install rasterio"
            )
        
        self._path = Path(path)
        self._band = band
        
        # Open the file
        self._ds = rasterio.open(str(path), 'r')
        
        # Infer feature name
        if feature_name is None:
            feature_name = self._infer_feature_name()
        self._feature_name = feature_name
        
        # Build coordinate system
        self._coordinate_system = MapCoordinateSystem(
            crs=self._ds.crs,
            transform=self._ds.transform,
            width=self._ds.width,
            height=self._ds.height,
        )
        
        # Cache nodata
        self._nodata = self._ds.nodata
        
        # Pre-compute stats if requested
        self._min_value: Optional[float] = None
        self._max_value: Optional[float] = None
        if preload_stats:
            self._compute_stats()
    
    def _infer_feature_name(self) -> str:
        """Infer feature name from filename or metadata."""
        name = self._path.stem.lower()
        
        # Common patterns
        if any(k in name for k in ['bath', 'depth', 'dem', 'dtm', 'elevation']):
            return 'bathymetry'
        elif any(k in name for k in ['backscatter', 'bsct', 'reflectivity', 'scatter']):
            return 'backscatter'
        elif any(k in name for k in ['slope']):
            return 'slope'
        elif any(k in name for k in ['aspect']):
            return 'aspect'
        elif any(k in name for k in ['rugosity', 'roughness']):
            return 'rugosity'
        
        return 'raster'
    
    def _compute_stats(self) -> None:
        """Compute min/max statistics from overviews or sampling."""
        # Try to read from file statistics
        if self._ds.tags(self._band):
            tags = self._ds.tags(self._band)
            if 'STATISTICS_MINIMUM' in tags and 'STATISTICS_MAXIMUM' in tags:
                self._min_value = float(tags['STATISTICS_MINIMUM'])
                self._max_value = float(tags['STATISTICS_MAXIMUM'])
                return
        
        # Use lowest overview level for fast stats
        if self._ds.overviews(self._band):
            # Read from smallest overview
            overview_idx = len(self._ds.overviews(self._band))
            factor = self._ds.overviews(self._band)[-1]  # Largest factor = smallest overview
            
            # Calculate output shape
            out_height = max(1, self._ds.height // factor)
            out_width = max(1, self._ds.width // factor)
            
            data = self._ds.read(
                self._band,
                out_shape=(out_height, out_width),
                resampling=Resampling.nearest,
            )
        else:
            # Sample the data (read at reduced resolution)
            max_samples = 1000
            factor = max(1, max(self._ds.height, self._ds.width) // max_samples)
            out_height = max(1, self._ds.height // factor)
            out_width = max(1, self._ds.width // factor)
            
            data = self._ds.read(
                self._band,
                out_shape=(out_height, out_width),
                resampling=Resampling.nearest,
            )
        
        # Mask nodata
        if self._nodata is not None:
            data = np.where(data == self._nodata, np.nan, data)
        
        self._min_value = float(np.nanmin(data))
        self._max_value = float(np.nanmax(data))
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def coordinate_system(self) -> MapCoordinateSystem:
        return self._coordinate_system
    
    @property
    def feature_name(self) -> str:
        return self._feature_name
    
    @property
    def bounds(self) -> BoundingBox:
        return self._coordinate_system.bounds
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self._ds.height, self._ds.width)
    
    @property
    def nodata(self) -> Optional[float]:
        return self._nodata
    
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._ds.dtypes[self._band - 1])
    
    @property
    def min_value(self) -> Optional[float]:
        return self._min_value
    
    @property
    def max_value(self) -> Optional[float]:
        return self._max_value
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            'path': str(self._path),
            'band': self._band,
            'crs': str(self._ds.crs),
            'driver': self._ds.driver,
            'has_overviews': len(self._ds.overviews(self._band)) > 0,
            'overview_factors': self._ds.overviews(self._band),
        }
    
    # =========================================================================
    # Data access
    # =========================================================================
    
    def get_data(
        self,
        bounds: Optional[BoundingBox] = None,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, MapCoordinateSystem]:
        """Get raster data for a region with optional downsampling.
        
        Uses rasterio's windowed reading for memory efficiency and
        built-in overview levels for fast downsampling.
        
        Args:
            bounds: Bounding box in world coordinates. If None, full extent.
            max_size: Maximum output size (height, width). Downsamples if needed.
            
        Returns:
            Tuple of (data array with NaN for nodata, coordinate system).
        """
        if bounds is None:
            # Full extent
            window = None
            col_off, row_off = 0, 0
            width, height = self._ds.width, self._ds.height
        else:
            # Calculate window from bounds
            col_min, row_min, col_max, row_max = \
                self._coordinate_system.get_pixel_bounds_for_world_extent(bounds)
            
            # Clip to valid range
            col_min = max(0, col_min)
            row_min = max(0, row_min)
            col_max = min(self._ds.width, col_max)
            row_max = min(self._ds.height, row_max)
            
            col_off = col_min
            row_off = row_min
            width = col_max - col_min
            height = row_max - row_min
            
            if width <= 0 or height <= 0:
                # Empty region
                return np.full((1, 1), np.nan, dtype=np.float32), self._coordinate_system
            
            window = Window(col_off, row_off, width, height)
        
        # Determine output size
        if max_size is not None:
            out_height = min(height, max_size[0])
            out_width = min(width, max_size[1])
            
            # If downsampling needed, find best overview level
            factor_x = width / out_width
            factor_y = height / out_height
            factor = max(factor_x, factor_y)
            
            if factor > 1:
                # Use overview if available
                overviews = self._ds.overviews(self._band)
                if overviews:
                    # Find best overview level
                    best_ovr = 1
                    for ovr in overviews:
                        if ovr <= factor:
                            best_ovr = ovr
                        else:
                            break
                
                # Calculate actual output shape
                out_height = max(1, int(height / factor))
                out_width = max(1, int(width / factor))
            
            out_shape = (out_height, out_width)
        else:
            out_shape = (height, width)
        
        # Read data
        data = self._ds.read(
            self._band,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.bilinear,
        )
        
        # Convert to float32 and handle nodata
        data = data.astype(np.float32)
        if self._nodata is not None:
            data = np.where(data == self._nodata, np.nan, data)
        
        # Create coordinate system for output
        # IMPORTANT: The transform maps pixel (col, row) -> world (x, y)
        # For GeoTiffs, transform.e is typically negative (y decreases as row increases)
        if window is not None or out_shape != (height, width):
            # Calculate scale factor from actual output size
            scale_x = width / out_shape[1]  # How many source pixels per output pixel
            scale_y = height / out_shape[0]
            
            # New pixel size in world coordinates
            dx = self._ds.transform.a * scale_x
            dy = self._ds.transform.e * scale_y  # Usually negative
            
            # Origin: world coordinate of top-left corner of the window
            # The transform gives us (x, y) for pixel (0, 0)
            # For a window starting at (col_off, row_off), we need the world coords there
            x_origin, y_origin = self._ds.transform * (col_off, row_off)
            
            new_transform = Affine(dx, 0, x_origin, 0, dy, y_origin)
            output_cs = MapCoordinateSystem(
                crs=self._ds.crs,
                transform=new_transform,
                width=out_shape[1],
                height=out_shape[0],
            )
        else:
            output_cs = self._coordinate_system
        
        return data, output_cs
    
    def get_value_at(self, x: float, y: float) -> Optional[float]:
        """Get the data value at a world coordinate.
        
        Args:
            x: X coordinate in world CRS.
            y: Y coordinate in world CRS.
            
        Returns:
            Data value or None if outside bounds or nodata.
        """
        # Check bounds
        if not self._coordinate_system.bounds.contains(x, y):
            return None
        
        # Convert to pixel
        col, row = self._coordinate_system.world_to_pixel(x, y)
        col, row = int(col), int(row)
        
        # Check pixel bounds
        if not (0 <= col < self._ds.width and 0 <= row < self._ds.height):
            return None
        
        # Read single pixel
        window = Window(col, row, 1, 1)
        value = self._ds.read(self._band, window=window)[0, 0]
        
        # Check nodata
        if self._nodata is not None and value == self._nodata:
            return None
        
        return float(value)
    
    def get_data_along_track(
        self,
        x: np.ndarray,
        y: np.ndarray,
        interpolation: str = 'nearest',
    ) -> np.ndarray:
        """Get data values along a track (e.g., ship navigation).
        
        Efficiently samples data at multiple points, useful for
        extracting values along a navigation track.
        
        Args:
            x: Array of X coordinates in world CRS.
            y: Array of Y coordinates in world CRS.
            interpolation: 'nearest' or 'bilinear'.
            
        Returns:
            Array of data values (NaN for nodata/out-of-bounds).
        """
        # Convert to pixel coordinates
        cols, rows = self._coordinate_system.world_to_pixel(x, y)
        
        # Initialize output
        values = np.full(len(x), np.nan, dtype=np.float32)
        
        # Find valid points (within bounds)
        valid = (
            (cols >= 0) & (cols < self._ds.width) &
            (rows >= 0) & (rows < self._ds.height)
        )
        
        if not np.any(valid):
            return values
        
        # For efficiency, read a bounding window and sample from it
        valid_cols = cols[valid]
        valid_rows = rows[valid]
        
        col_min = max(0, int(np.floor(valid_cols.min())))
        col_max = min(self._ds.width, int(np.ceil(valid_cols.max())) + 1)
        row_min = max(0, int(np.floor(valid_rows.min())))
        row_max = min(self._ds.height, int(np.ceil(valid_rows.max())) + 1)
        
        # Read the window
        window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
        data = self._ds.read(self._band, window=window).astype(np.float32)
        
        if self._nodata is not None:
            data = np.where(data == self._nodata, np.nan, data)
        
        # Sample values
        local_cols = (valid_cols - col_min).astype(int)
        local_rows = (valid_rows - row_min).astype(int)
        
        # Clip to valid indices
        local_cols = np.clip(local_cols, 0, data.shape[1] - 1)
        local_rows = np.clip(local_rows, 0, data.shape[0] - 1)
        
        values[valid] = data[local_rows, local_cols]
        
        return values
    
    def close(self) -> None:
        """Close the rasterio dataset."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None
    
    def __del__(self):
        """Ensure dataset is closed on garbage collection."""
        self.close()
    
    # =========================================================================
    # Factory methods
    # =========================================================================
    
    @classmethod
    def from_path(
        cls,
        path: str,
        feature_name: Optional[str] = None,
        band: int = 1,
    ) -> "GeoTiffBackend":
        """Create a GeoTiffBackend from a file path.
        
        This is the recommended way to create a GeoTiffBackend.
        
        Args:
            path: Path to the GeoTiff file.
            feature_name: Optional feature name override.
            band: Band number (1-indexed).
            
        Returns:
            GeoTiffBackend instance.
        """
        return cls(path, feature_name=feature_name, band=band)
