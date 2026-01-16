"""Coordinate system management for map data.

Provides CRS transformations, affine coordinate mapping, and extent handling
for geospatial data visualization.
"""

from typing import Tuple, Optional, Union, List
from dataclasses import dataclass

import numpy as np

try:
    from pyproj import CRS, Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    from affine import Affine
    HAS_AFFINE = True
except ImportError:
    # Fallback implementation if affine package not available
    HAS_AFFINE = False


@dataclass
class BoundingBox:
    """Bounding box in world coordinates.
    
    Attributes:
        xmin: Minimum x coordinate (west).
        ymin: Minimum y coordinate (south).
        xmax: Maximum x coordinate (east).
        ymax: Maximum y coordinate (north).
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    
    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.xmax - self.xmin
    
    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.ymax - self.ymin
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point (x, y) of the bounding box."""
        return (
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
        )
    
    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within the bounding box."""
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax
    
    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects another."""
        return not (
            self.xmax < other.xmin or
            self.xmin > other.xmax or
            self.ymax < other.ymin or
            self.ymin > other.ymax
        )
    
    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """Return the intersection of two bounding boxes, or None if no intersection."""
        if not self.intersects(other):
            return None
        return BoundingBox(
            xmin=max(self.xmin, other.xmin),
            ymin=max(self.ymin, other.ymin),
            xmax=min(self.xmax, other.xmax),
            ymax=min(self.ymax, other.ymax),
        )
    
    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Return the union (combined extent) of two bounding boxes."""
        return BoundingBox(
            xmin=min(self.xmin, other.xmin),
            ymin=min(self.ymin, other.ymin),
            xmax=max(self.xmax, other.xmax),
            ymax=max(self.ymax, other.ymax),
        )
    
    def expand(self, factor: float) -> "BoundingBox":
        """Return a new bounding box expanded by a factor around center."""
        cx, cy = self.center
        half_w = self.width / 2 * factor
        half_h = self.height / 2 * factor
        return BoundingBox(
            xmin=cx - half_w,
            ymin=cy - half_h,
            xmax=cx + half_w,
            ymax=cy + half_h,
        )
    
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (xmin, ymin, xmax, ymax) tuple."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)
    
    @classmethod
    def from_points(cls, x: np.ndarray, y: np.ndarray) -> "BoundingBox":
        """Create a bounding box from arrays of x and y coordinates."""
        return cls(
            xmin=float(np.nanmin(x)),
            ymin=float(np.nanmin(y)),
            xmax=float(np.nanmax(x)),
            ymax=float(np.nanmax(y)),
        )


class MapCoordinateSystem:
    """Coordinate system for map data with CRS transformations.
    
    Handles conversions between:
    - Pixel coordinates (row, col in raster)
    - World coordinates (in native CRS, e.g., UTM meters)
    - Geographic coordinates (latitude, longitude in WGS84)
    
    Attributes:
        crs: Coordinate Reference System (pyproj.CRS or EPSG code).
        transform: Affine transform from pixel to world coordinates.
        bounds: Bounding box in world coordinates.
    """
    
    def __init__(
        self,
        crs: Union[str, int, "CRS"],
        transform: "Affine",
        width: int,
        height: int,
    ):
        """Initialize coordinate system.
        
        Args:
            crs: Coordinate Reference System (EPSG code, WKT, or pyproj.CRS).
            transform: Affine transform from pixel to world coordinates.
                      Maps (col, row) to (x, y) in world coordinates.
            width: Raster width in pixels.
            height: Raster height in pixels.
        """
        if not HAS_PYPROJ:
            raise ImportError(
                "pyproj is required for MapCoordinateSystem. "
                "Install with: pip install pyproj"
            )
        
        # Normalize CRS
        if isinstance(crs, int):
            self._crs = CRS.from_epsg(crs)
        elif isinstance(crs, str):
            self._crs = CRS.from_user_input(crs)
        else:
            self._crs = crs
        
        self._transform = transform
        self._width = width
        self._height = height
        
        # Pre-compute inverse transform for world -> pixel
        self._inverse = ~transform
        
        # Transformer for CRS -> WGS84
        self._wgs84 = CRS.from_epsg(4326)
        self._to_wgs84: Optional[Transformer] = None
        self._from_wgs84: Optional[Transformer] = None
        
        # Compute bounds from pixel corners
        self._bounds = self._compute_bounds()
    
    def _compute_bounds(self) -> BoundingBox:
        """Compute world-coordinate bounds from raster extent."""
        # Four corners in pixel space
        corners_px = [
            (0, 0),                           # top-left
            (self._width, 0),                 # top-right
            (self._width, self._height),      # bottom-right
            (0, self._height),                # bottom-left
        ]
        
        # Transform to world coordinates
        xs, ys = [], []
        for col, row in corners_px:
            x, y = self._transform * (col, row)
            xs.append(x)
            ys.append(y)
        
        return BoundingBox(
            xmin=min(xs),
            ymin=min(ys),
            xmax=max(xs),
            ymax=max(ys),
        )
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def crs(self) -> "CRS":
        """Coordinate Reference System."""
        return self._crs
    
    @property
    def transform(self) -> "Affine":
        """Affine transform from pixel to world coordinates."""
        return self._transform
    
    @property
    def bounds(self) -> BoundingBox:
        """Bounding box in world coordinates."""
        return self._bounds
    
    @property
    def width(self) -> int:
        """Raster width in pixels."""
        return self._width
    
    @property
    def height(self) -> int:
        """Raster height in pixels."""
        return self._height
    
    @property
    def resolution(self) -> Tuple[float, float]:
        """Pixel resolution (dx, dy) in world coordinate units."""
        return (abs(self._transform.a), abs(self._transform.e))
    
    @property
    def is_geographic(self) -> bool:
        """True if CRS uses geographic (lat/lon) coordinates."""
        return self._crs.is_geographic
    
    @property
    def epsg(self) -> Optional[int]:
        """EPSG code of the CRS, or None if not available."""
        return self._crs.to_epsg()
    
    # =========================================================================
    # Coordinate transformations
    # =========================================================================
    
    def pixel_to_world(
        self,
        col: Union[float, np.ndarray],
        row: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert pixel coordinates to world coordinates.
        
        Args:
            col: Column index (x in pixel space).
            row: Row index (y in pixel space).
            
        Returns:
            Tuple of (x, y) in world coordinates.
        """
        if isinstance(col, np.ndarray):
            x = self._transform.a * col + self._transform.c
            y = self._transform.e * row + self._transform.f
            return x, y
        else:
            return self._transform * (col, row)
    
    def world_to_pixel(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert world coordinates to pixel coordinates.
        
        Args:
            x: X coordinate in world CRS.
            y: Y coordinate in world CRS.
            
        Returns:
            Tuple of (col, row) in pixel coordinates (float, may need rounding).
        """
        if isinstance(x, np.ndarray):
            col = (x - self._transform.c) / self._transform.a
            row = (y - self._transform.f) / self._transform.e
            return col, row
        else:
            return self._inverse * (x, y)
    
    def world_to_latlon(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert world coordinates to WGS84 lat/lon.
        
        Args:
            x: X coordinate in native CRS.
            y: Y coordinate in native CRS.
            
        Returns:
            Tuple of (latitude, longitude) in degrees.
        """
        if self._crs.is_geographic:
            # Already in geographic coordinates
            return y, x  # lat, lon
        
        if self._to_wgs84 is None:
            self._to_wgs84 = Transformer.from_crs(
                self._crs, self._wgs84, always_xy=True
            )
        
        lon, lat = self._to_wgs84.transform(x, y)
        return lat, lon
    
    def latlon_to_world(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert WGS84 lat/lon to world coordinates.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            
        Returns:
            Tuple of (x, y) in native CRS coordinates.
        """
        if self._crs.is_geographic:
            # Already in geographic coordinates
            return lon, lat  # x, y
        
        if self._from_wgs84 is None:
            self._from_wgs84 = Transformer.from_crs(
                self._wgs84, self._crs, always_xy=True
            )
        
        return self._from_wgs84.transform(lon, lat)
    
    def latlon_to_pixel(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert WGS84 lat/lon directly to pixel coordinates.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            
        Returns:
            Tuple of (col, row) in pixel coordinates.
        """
        x, y = self.latlon_to_world(lat, lon)
        return self.world_to_pixel(x, y)
    
    def pixel_to_latlon(
        self,
        col: Union[float, np.ndarray],
        row: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert pixel coordinates to WGS84 lat/lon.
        
        Args:
            col: Column index.
            row: Row index.
            
        Returns:
            Tuple of (latitude, longitude) in degrees.
        """
        x, y = self.pixel_to_world(col, row)
        return self.world_to_latlon(x, y)
    
    # =========================================================================
    # Region/extent operations
    # =========================================================================
    
    def get_pixel_bounds_for_world_extent(
        self,
        world_bounds: BoundingBox,
    ) -> Tuple[int, int, int, int]:
        """Get pixel bounds (col_min, row_min, col_max, row_max) for a world extent.
        
        Args:
            world_bounds: Bounding box in world coordinates.
            
        Returns:
            Tuple of (col_min, row_min, col_max, row_max) clipped to raster extent.
        """
        # Convert all corners (handle rotated/sheared transforms)
        corners = [
            (world_bounds.xmin, world_bounds.ymin),
            (world_bounds.xmax, world_bounds.ymin),
            (world_bounds.xmax, world_bounds.ymax),
            (world_bounds.xmin, world_bounds.ymax),
        ]
        
        cols, rows = [], []
        for wx, wy in corners:
            c, r = self.world_to_pixel(wx, wy)
            cols.append(c)
            rows.append(r)
        
        # Get bounding box in pixel space
        col_min = int(np.floor(min(cols)))
        col_max = int(np.ceil(max(cols)))
        row_min = int(np.floor(min(rows)))
        row_max = int(np.ceil(max(rows)))
        
        # Clip to raster extent
        col_min = max(0, col_min)
        row_min = max(0, row_min)
        col_max = min(self._width, col_max)
        row_max = min(self._height, row_max)
        
        return (col_min, row_min, col_max, row_max)
    
    def create_subregion(
        self,
        col_min: int,
        row_min: int,
        col_max: int,
        row_max: int,
    ) -> "MapCoordinateSystem":
        """Create a new coordinate system for a subregion.
        
        Args:
            col_min: Minimum column index.
            row_min: Minimum row index.
            col_max: Maximum column index.
            row_max: Maximum row index.
            
        Returns:
            New MapCoordinateSystem for the subregion.
        """
        # Compute new transform (shift origin to subregion top-left)
        new_transform = self._transform * Affine.translation(col_min, row_min)
        
        return MapCoordinateSystem(
            crs=self._crs,
            transform=new_transform,
            width=col_max - col_min,
            height=row_max - row_min,
        )
    
    def create_downsampled(self, factor: int) -> "MapCoordinateSystem":
        """Create a coordinate system for a downsampled version.
        
        Args:
            factor: Downsampling factor (e.g., 2 = half resolution).
            
        Returns:
            New MapCoordinateSystem with scaled transform.
        """
        # Scale transform to account for larger pixels
        new_transform = self._transform * Affine.scale(factor, factor)
        
        return MapCoordinateSystem(
            crs=self._crs,
            transform=new_transform,
            width=self._width // factor,
            height=self._height // factor,
        )
    
    # =========================================================================
    # Rotation for track-up display
    # =========================================================================
    
    def create_rotated_view(
        self,
        center_x: float,
        center_y: float,
        heading: float,
        view_width: int,
        view_height: int,
        resolution: float,
    ) -> Tuple["MapCoordinateSystem", BoundingBox]:
        """Create a rotated coordinate system for track-up display.
        
        Creates a new coordinate system rotated around a center point,
        useful for displaying maps in ship-relative (track-up) orientation.
        
        Args:
            center_x: X coordinate of rotation center (world coords).
            center_y: Y coordinate of rotation center (world coords).
            heading: Heading in degrees (0=north, clockwise).
            view_width: Output view width in pixels.
            view_height: Output view height in pixels.
            resolution: Desired pixel resolution in world units.
            
        Returns:
            Tuple of (rotated coordinate system, source bounds needed).
        """
        # Convert heading to radians (clockwise from north -> counterclockwise from east)
        angle_rad = np.radians(-heading + 90)
        
        # Create rotation transform centered at the center point
        # First translate center to origin, rotate, then translate to view center
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # View center in pixels
        view_cx = view_width / 2
        view_cy = view_height / 2
        
        # Affine transform: pixel -> world (rotated)
        # This maps view pixels to world coordinates through the rotation
        rotated_transform = Affine(
            resolution * cos_a, -resolution * sin_a, center_x - resolution * (cos_a * view_cx - sin_a * view_cy),
            resolution * sin_a, resolution * cos_a, center_y - resolution * (sin_a * view_cx + cos_a * view_cy),
        )
        
        rotated_cs = MapCoordinateSystem(
            crs=self._crs,
            transform=rotated_transform,
            width=view_width,
            height=view_height,
        )
        
        # Compute source bounds needed (corners of rotated view in world space)
        corners_px = [
            (0, 0), (view_width, 0),
            (view_width, view_height), (0, view_height),
        ]
        xs, ys = [], []
        for col, row in corners_px:
            x, y = rotated_transform * (col, row)
            xs.append(x)
            ys.append(y)
        
        source_bounds = BoundingBox(
            xmin=min(xs), ymin=min(ys),
            xmax=max(xs), ymax=max(ys),
        )
        
        return rotated_cs, source_bounds
    
    # =========================================================================
    # String representations
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"MapCoordinateSystem(crs={self.epsg or 'custom'}, "
            f"size=({self._width}×{self._height}), "
            f"resolution={self.resolution})"
        )
