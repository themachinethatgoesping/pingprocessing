"""TileBuilder for fetching and compositing web map tiles.

Provides a simple interface for loading XYZ tiles from web services
like OpenStreetMap, ESRI, Stamen, etc. with disk caching.

Tiles are pre-rendered RGB/RGBA images, unlike MapBuilder's numerical data.
This means no colormap/colorbar is needed - the images are displayed directly.

Example:
    from themachinethatgoesping.pingprocessing.overview.map_builder import TileBuilder
    from themachinethatgoesping.pingprocessing.widgets import MapViewerPyQtGraph
    
    # Create tile builder with OSM tiles
    tiles = TileBuilder()
    tiles.add_osm()
    
    # Or use other providers
    tiles.add_esri_worldimagery()
    tiles.add_cartodb_positron()
    
    # Use with MapViewer
    viewer = MapViewerPyQtGraph(tile_builder=tiles)
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import math
import io
import warnings
import os

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .coordinate_system import BoundingBox


# ============================================================================
# Common Tile Sources
# ============================================================================

@dataclass
class TileSource:
    """Configuration for a tile source.
    
    Attributes:
        name: Display name of the source.
        url_template: URL template with {z}, {x}, {y} placeholders.
        attribution: Attribution text (required by most providers).
        max_zoom: Maximum zoom level (typically 18-19).
        min_zoom: Minimum zoom level (typically 0).
        tile_size: Tile size in pixels (typically 256).
        headers: Optional HTTP headers for requests.
    """
    name: str
    url_template: str
    attribution: str = ""
    max_zoom: int = 19
    min_zoom: int = 0
    tile_size: int = 256
    headers: Dict[str, str] = field(default_factory=dict)
    visible: bool = True
    opacity: float = 1.0


# Pre-defined tile sources
TILE_SOURCES = {
    # OpenStreetMap
    "osm": TileSource(
        name="OpenStreetMap",
        url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap contributors",
        max_zoom=19,
        headers={"User-Agent": "themachinethatgoesping/1.0"},
    ),
    
    # ESRI
    "esri_worldimagery": TileSource(
        name="ESRI World Imagery",
        url_template="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="© Esri, Maxar, Earthstar Geographics",
        max_zoom=19,
    ),
    "esri_worldstreetmap": TileSource(
        name="ESRI World Street Map",
        url_template="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attribution="© Esri",
        max_zoom=19,
    ),
    "esri_ocean": TileSource(
        name="ESRI Ocean Basemap",
        url_template="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        attribution="© Esri, GEBCO, NOAA, National Geographic",
        max_zoom=13,
    ),
    "esri_natgeo": TileSource(
        name="ESRI National Geographic",
        url_template="https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        attribution="© Esri, National Geographic",
        max_zoom=16,
    ),
    
    # CartoDB
    "cartodb_positron": TileSource(
        name="CartoDB Positron",
        url_template="https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap, © CartoDB",
        max_zoom=19,
    ),
    "cartodb_darkmatter": TileSource(
        name="CartoDB Dark Matter",
        url_template="https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap, © CartoDB",
        max_zoom=19,
    ),
    "cartodb_voyager": TileSource(
        name="CartoDB Voyager",
        url_template="https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap, © CartoDB",
        max_zoom=19,
    ),
    
    # Stamen (now Stadia)
    "stadia_terrain": TileSource(
        name="Stadia Terrain",
        url_template="https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png",
        attribution="© Stamen Design, © OpenStreetMap",
        max_zoom=18,
    ),
    "stadia_toner": TileSource(
        name="Stadia Toner",
        url_template="https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}.png",
        attribution="© Stamen Design, © OpenStreetMap",
        max_zoom=18,
    ),
    "stadia_watercolor": TileSource(
        name="Stadia Watercolor",
        url_template="https://tiles.stadiamaps.com/tiles/stamen_watercolor/{z}/{x}/{y}.jpg",
        attribution="© Stamen Design, © OpenStreetMap",
        max_zoom=18,
    ),
    
    # OpenTopoMap
    "opentopomap": TileSource(
        name="OpenTopoMap",
        url_template="https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap, © SRTM, © OpenTopoMap",
        max_zoom=17,
    ),
    
    # EMODnet Bathymetry
    "emodnet_bathymetry": TileSource(
        name="EMODnet Bathymetry",
        url_template="https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{z}/{x}/{y}.png",
        attribution="© EMODnet Bathymetry",
        max_zoom=12,
    ),
    "emodnet_mean": TileSource(
        name="EMODnet Mean Depth",
        url_template="https://tiles.emodnet-bathymetry.eu/2020/mean_multicolour/web_mercator/{z}/{x}/{y}.png",
        attribution="© EMODnet Bathymetry",
        max_zoom=12,
    ),
}


def list_available_sources() -> List[str]:
    """List available pre-defined tile sources."""
    return list(TILE_SOURCES.keys())


# ============================================================================
# Tile Cache
# ============================================================================

class TileCache:
    """Simple disk-based tile cache."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize tile cache.
        
        Args:
            cache_dir: Directory for cached tiles. If None, uses ~/.cache/pingprocessing/tiles
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "pingprocessing" / "tiles"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._max_memory_tiles = 500  # Keep up to 500 tiles in memory
    
    def _tile_key(self, source_name: str, z: int, x: int, y: int) -> str:
        """Generate cache key for a tile."""
        return f"{source_name}_{z}_{x}_{y}"
    
    def _tile_path(self, source_name: str, z: int, x: int, y: int) -> Path:
        """Generate file path for cached tile."""
        # Use subdirectories to avoid too many files in one folder
        return self._cache_dir / source_name / str(z) / str(x) / f"{y}.png"
    
    def get(self, source_name: str, z: int, x: int, y: int) -> Optional[np.ndarray]:
        """Get tile from cache (memory or disk)."""
        key = self._tile_key(source_name, z, x, y)
        
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        path = self._tile_path(source_name, z, x, y)
        if path.exists():
            try:
                if HAS_PIL:
                    img = Image.open(path)
                    arr = np.array(img)
                    self._memory_cache[key] = arr
                    self._evict_memory_if_needed()
                    return arr
            except Exception:
                # Corrupted cache file, remove it
                path.unlink(missing_ok=True)
        
        return None
    
    def put(self, source_name: str, z: int, x: int, y: int, data: np.ndarray) -> None:
        """Store tile in cache (memory and disk)."""
        key = self._tile_key(source_name, z, x, y)
        
        # Store in memory
        self._memory_cache[key] = data
        self._evict_memory_if_needed()
        
        # Store on disk
        path = self._tile_path(source_name, z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if HAS_PIL:
            try:
                img = Image.fromarray(data)
                img.save(path, "PNG")
            except Exception as e:
                warnings.warn(f"Failed to cache tile: {e}")
    
    def _evict_memory_if_needed(self) -> None:
        """Evict oldest tiles from memory cache if too large."""
        while len(self._memory_cache) > self._max_memory_tiles:
            # Remove first item (oldest)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
    
    def clear_memory(self) -> None:
        """Clear memory cache."""
        self._memory_cache.clear()
    
    def clear_disk(self, source_name: Optional[str] = None) -> None:
        """Clear disk cache for a source or all sources."""
        import shutil
        if source_name:
            path = self._cache_dir / source_name
            if path.exists():
                shutil.rmtree(path)
        else:
            if self._cache_dir.exists():
                shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Coordinate Utilities
# ============================================================================

def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates at given zoom level.
    
    Uses Web Mercator (EPSG:3857) tile scheme.
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    # Clamp to valid range
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tile_to_latlon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon (top-left corner)."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def tile_bounds_latlon(x: int, y: int, zoom: int) -> BoundingBox:
    """Get bounding box of a tile in lat/lon."""
    lat_nw, lon_nw = tile_to_latlon(x, y, zoom)
    lat_se, lon_se = tile_to_latlon(x + 1, y + 1, zoom)
    return BoundingBox(
        xmin=lon_nw,  # West
        ymin=lat_se,  # South
        xmax=lon_se,  # East
        ymax=lat_nw,  # North
    )


def lat_to_mercator_y(lat: float) -> float:
    """Convert latitude to Web Mercator y coordinate (in degrees equivalent).
    
    Web Mercator uses the Spherical Mercator projection where:
    y = R * ln(tan(pi/4 + lat/2))
    
    We express this in "degree equivalents" so it can be compared with longitude.
    """
    lat_rad = math.radians(lat)
    return math.degrees(math.asinh(math.tan(lat_rad)))


def mercator_y_to_lat(y: float) -> float:
    """Convert Web Mercator y coordinate back to latitude."""
    return math.degrees(math.atan(math.sinh(math.radians(y))))


def reproject_mercator_to_latlon(
    image: np.ndarray,
    merc_bounds: BoundingBox,
    target_bounds: BoundingBox,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Reproject a Web Mercator image to linear lat/lon coordinates.
    
    Web Mercator tiles have pixels uniformly spaced in Mercator y-coordinate,
    not in latitude. This function resamples the image so pixels are uniformly
    spaced in latitude, allowing correct display with linear lat/lon axes.
    
    Args:
        image: Input RGBA image in Web Mercator projection.
        merc_bounds: Bounds in lat/lon of the Mercator image.
        target_bounds: Desired output bounds in lat/lon.
        target_size: Output size as (width, height).
        
    Returns:
        Reprojected RGBA image with pixels uniformly spaced in lat/lon.
    """
    if not HAS_PIL:
        return image
    
    from scipy import ndimage
    
    out_width, out_height = target_size
    in_height, in_width = image.shape[:2]
    
    # Input bounds in Mercator y-coordinates
    merc_y_min = lat_to_mercator_y(merc_bounds.ymin)  # south
    merc_y_max = lat_to_mercator_y(merc_bounds.ymax)  # north
    
    # Create output pixel lat/lon coordinates
    out_lats = np.linspace(target_bounds.ymax, target_bounds.ymin, out_height)  # north to south
    out_lons = np.linspace(target_bounds.xmin, target_bounds.xmax, out_width)
    
    # Convert output latitudes to Mercator y
    out_merc_y = np.array([lat_to_mercator_y(lat) for lat in out_lats])
    
    # Map Mercator y to input pixel row (north=row0, south=row[-1])
    # Input row 0 corresponds to merc_y_max (north)
    # Input row in_height-1 corresponds to merc_y_min (south)
    src_rows = (merc_y_max - out_merc_y) / (merc_y_max - merc_y_min) * (in_height - 1)
    
    # Map output lon to input pixel column
    # Input col 0 corresponds to xmin (west)
    # Input col in_width-1 corresponds to xmax (east)
    src_cols = (out_lons - merc_bounds.xmin) / (merc_bounds.xmax - merc_bounds.xmin) * (in_width - 1)
    
    # Create coordinate arrays for scipy.ndimage.map_coordinates
    row_coords, col_coords = np.meshgrid(src_rows, src_cols, indexing='ij')
    
    # Resample each channel
    output = np.zeros((out_height, out_width, image.shape[2]), dtype=image.dtype)
    for c in range(image.shape[2]):
        output[:, :, c] = ndimage.map_coordinates(
            image[:, :, c].astype(np.float64),
            [row_coords, col_coords],
            order=1,  # bilinear interpolation
            mode='constant',
            cval=0,
        ).astype(image.dtype)
    
    return output


def choose_zoom_level(bounds: BoundingBox, target_size: Tuple[int, int], tile_size: int = 256) -> int:
    """Choose appropriate zoom level for given bounds and target size."""
    # Calculate degrees per pixel for target
    target_width, target_height = target_size
    deg_per_pixel_x = bounds.width / target_width
    deg_per_pixel_y = bounds.height / target_height
    deg_per_pixel = min(deg_per_pixel_x, deg_per_pixel_y)
    
    # At zoom 0, the whole world (360°) fits in one tile (256px)
    # degrees_per_pixel at zoom z = 360 / (256 * 2^z)
    # So: 2^z = 360 / (256 * deg_per_pixel)
    # z = log2(360 / (256 * deg_per_pixel))
    
    if deg_per_pixel <= 0:
        return 0
    
    z = math.log2(360 / (tile_size * deg_per_pixel))
    # Round UP to get higher resolution, then add 1 for extra sharpness
    return max(0, min(19, int(math.ceil(z)) + 1))


# ============================================================================
# TileBuilder
# ============================================================================

class TileBuilder:
    """Builder for fetching and compositing web map tiles.
    
    TileBuilder handles:
    - Multiple tile sources (OSM, ESRI, etc.)
    - Tile fetching with disk caching
    - Compositing tiles for arbitrary view bounds
    
    Unlike MapBuilder's numerical data, tiles are pre-rendered RGBA images.
    No colormap/colorbar is needed.
    
    Example:
        tiles = TileBuilder()
        tiles.add_osm()  # Add OpenStreetMap tiles
        
        # Get composited image for a region
        image = tiles.get_image(bounds, target_size=(800, 600))
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize TileBuilder.
        
        Args:
            cache_dir: Directory for tile cache. Default: ~/.cache/pingprocessing/tiles
        """
        if not HAS_REQUESTS:
            warnings.warn("requests package not installed. Tile fetching will not work.")
        if not HAS_PIL:
            warnings.warn("Pillow package not installed. Tile loading will not work.")
        
        self._sources: Dict[str, TileSource] = {}
        self._cache = TileCache(cache_dir)
        self._session: Optional[requests.Session] = None
        
        # Axis/resolution settings (like MapBuilder/EchogramBuilder)
        self._max_pixels: Tuple[int, int] = (2000, 2000)  # (height, width)
        self._current_bounds: Optional[BoundingBox] = None
    
    # =========================================================================
    # Axis/resolution settings (like MapBuilder/EchogramBuilder)
    # =========================================================================
    
    def set_axis_latlon(
        self,
        min_lat: float = np.nan,
        max_lat: float = np.nan,
        min_lon: float = np.nan,
        max_lon: float = np.nan,
        max_pixels: Optional[Tuple[int, int]] = None,
    ) -> "TileBuilder":
        """Set axis extent in lat/lon coordinates.
        
        Similar to MapBuilder/EchogramBuilder pattern.
        Use np.nan for auto-detection (full world).
        
        Args:
            min_lat: Minimum latitude (-85.05 = Web Mercator limit).
            max_lat: Maximum latitude (85.05 = Web Mercator limit).
            min_lon: Minimum longitude (-180).
            max_lon: Maximum longitude (180).
            max_pixels: Maximum output size (height, width).
            
        Returns:
            Self for method chaining.
        """
        if max_pixels is not None:
            self._max_pixels = max_pixels
        
        # Web Mercator limits
        xmin = min_lon if not np.isnan(min_lon) else -180.0
        xmax = max_lon if not np.isnan(max_lon) else 180.0
        ymin = min_lat if not np.isnan(min_lat) else -85.05
        ymax = max_lat if not np.isnan(max_lat) else 85.05
        
        # Clamp to Web Mercator limits
        ymin = max(-85.05, min(85.05, ymin))
        ymax = max(-85.05, min(85.05, ymax))
        
        self._current_bounds = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        return self
    
    def set_bounds(self, bounds: BoundingBox) -> "TileBuilder":
        """Set the current view bounds (in lat/lon).
        
        Args:
            bounds: BoundingBox with lon as x, lat as y.
            
        Returns:
            Self for method chaining.
        """
        # Clamp to Web Mercator limits
        self._current_bounds = BoundingBox(
            xmin=max(-180.0, min(180.0, bounds.xmin)),
            ymin=max(-85.05, min(85.05, bounds.ymin)),
            xmax=max(-180.0, min(180.0, bounds.xmax)),
            ymax=max(-85.05, min(85.05, bounds.ymax)),
        )
        return self
    
    def set_max_pixels(self, max_pixels: Tuple[int, int]) -> "TileBuilder":
        """Set maximum output resolution (height, width).
        
        Args:
            max_pixels: Maximum size as (height, width) tuple.
            
        Returns:
            Self for method chaining.
        """
        self._max_pixels = max_pixels
        return self
    
    def reset_bounds(self) -> "TileBuilder":
        """Reset bounds to full Web Mercator extent."""
        self._current_bounds = None
        return self
    
    @property
    def max_pixels(self) -> Tuple[int, int]:
        """Current max pixels setting (height, width)."""
        return self._max_pixels
    
    @property
    def current_bounds(self) -> Optional[BoundingBox]:
        """Current view bounds in lat/lon (None = not set)."""
        return self._current_bounds
    
    def _get_session(self) -> "requests.Session":
        """Get or create requests session."""
        if self._session is None and HAS_REQUESTS:
            self._session = requests.Session()
        return self._session
    
    # =========================================================================
    # Add tile sources
    # =========================================================================
    
    def add_source(self, source: TileSource) -> "TileBuilder":
        """Add a tile source.
        
        Args:
            source: TileSource configuration.
        """
        self._sources[source.name] = source
        return self
    
    def add_xyz(
        self,
        name: str,
        url_template: str,
        attribution: str = "",
        max_zoom: int = 19,
        **kwargs
    ) -> "TileBuilder":
        """Add a custom XYZ tile source.
        
        Args:
            name: Display name.
            url_template: URL with {z}, {x}, {y} placeholders.
            attribution: Attribution text.
            max_zoom: Maximum zoom level.
        """
        source = TileSource(
            name=name,
            url_template=url_template,
            attribution=attribution,
            max_zoom=max_zoom,
            **kwargs
        )
        return self.add_source(source)
    
    def add_preset(self, preset_name: str) -> "TileBuilder":
        """Add a pre-defined tile source.
        
        Args:
            preset_name: Name from TILE_SOURCES (e.g., 'osm', 'esri_worldimagery').
        """
        if preset_name not in TILE_SOURCES:
            available = ", ".join(TILE_SOURCES.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        source = TILE_SOURCES[preset_name]
        # Create a copy so we don't modify the global
        # Store under preset_name for consistent lookup
        self._sources[preset_name] = TileSource(
            name=source.name,
            url_template=source.url_template,
            attribution=source.attribution,
            max_zoom=source.max_zoom,
            min_zoom=source.min_zoom,
            tile_size=source.tile_size,
            headers=dict(source.headers),
        )
        return self
    
    # Convenience methods for common sources
    def add_osm(self) -> "TileBuilder":
        """Add OpenStreetMap tiles."""
        return self.add_preset("osm")
    
    def add_esri_worldimagery(self) -> "TileBuilder":
        """Add ESRI World Imagery (satellite)."""
        return self.add_preset("esri_worldimagery")
    
    def add_esri_ocean(self) -> "TileBuilder":
        """Add ESRI Ocean Basemap."""
        return self.add_preset("esri_ocean")
    
    def add_esri_natgeo(self) -> "TileBuilder":
        """Add ESRI National Geographic."""
        return self.add_preset("esri_natgeo")
    
    def add_cartodb_positron(self) -> "TileBuilder":
        """Add CartoDB Positron (light theme)."""
        return self.add_preset("cartodb_positron")
    
    def add_cartodb_darkmatter(self) -> "TileBuilder":
        """Add CartoDB Dark Matter (dark theme)."""
        return self.add_preset("cartodb_darkmatter")
    
    def add_opentopomap(self) -> "TileBuilder":
        """Add OpenTopoMap (topographic)."""
        return self.add_preset("opentopomap")
    
    # =========================================================================
    # Source management
    # =========================================================================
    
    @property
    def sources(self) -> List[TileSource]:
        """List of all tile sources."""
        return list(self._sources.values())
    
    @property
    def source_names(self) -> List[str]:
        """List of source names."""
        return list(self._sources.keys())
    
    @property
    def visible_sources(self) -> List[TileSource]:
        """List of visible tile sources."""
        return [s for s in self._sources.values() if s.visible]
    
    def set_source_visible(self, name: str, visible: bool) -> "TileBuilder":
        """Set visibility of a source."""
        if name in self._sources:
            self._sources[name].visible = visible
        return self
    
    def set_source_opacity(self, name: str, opacity: float) -> "TileBuilder":
        """Set opacity of a source (0.0 - 1.0)."""
        if name in self._sources:
            self._sources[name].opacity = max(0.0, min(1.0, opacity))
        return self
    
    def remove_source(self, name: str) -> "TileBuilder":
        """Remove a tile source."""
        if name in self._sources:
            del self._sources[name]
        return self
    
    def clear_sources(self) -> "TileBuilder":
        """Remove all tile sources."""
        self._sources.clear()
        return self
    
    def set_source(self, preset_name: str) -> "TileBuilder":
        """Set a single tile source (clears others and adds this one).
        
        Convenience method for simple single-source usage.
        
        Args:
            preset_name: Name from TILE_SOURCES (e.g., 'osm', 'esri_worldimagery').
        """
        self.clear_sources()
        return self.add_preset(preset_name)
    
    # =========================================================================
    # Tile fetching
    # =========================================================================
    
    def _fetch_tile(self, source: TileSource, z: int, x: int, y: int) -> Optional[np.ndarray]:
        """Fetch a single tile, using cache if available."""
        # Check cache first
        cached = self._cache.get(source.name, z, x, y)
        if cached is not None:
            return cached
        
        if not HAS_REQUESTS or not HAS_PIL:
            return None
        
        # Build URL
        url = source.url_template.format(z=z, x=x, y=y)
        
        try:
            session = self._get_session()
            headers = {"User-Agent": "themachinethatgoesping/1.0"}
            headers.update(source.headers)
            
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Load image
            img = Image.open(io.BytesIO(response.content))
            
            # Convert to RGBA
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            
            arr = np.array(img)
            
            # Cache it
            self._cache.put(source.name, z, x, y, arr)
            
            return arr
            
        except Exception as e:
            warnings.warn(f"Failed to fetch tile {url}: {e}")
            return None
    
    def _get_tiles_for_bounds(
        self,
        source: TileSource,
        bounds: BoundingBox,
        zoom: int
    ) -> Tuple[np.ndarray, BoundingBox]:
        """Get all tiles covering bounds and composite them.
        
        Returns:
            Tuple of (composited image array, actual bounds covered).
        """
        tile_size = source.tile_size
        
        # Clamp bounds to valid lat/lon range for Web Mercator
        # Web Mercator doesn't work beyond ~85.06 degrees latitude
        clamped_bounds = BoundingBox(
            xmin=max(-180.0, min(180.0, bounds.xmin)),
            ymin=max(-85.05, min(85.05, bounds.ymin)),
            xmax=max(-180.0, min(180.0, bounds.xmax)),
            ymax=max(-85.05, min(85.05, bounds.ymax)),
        )
        
        # Get tile range
        x_min, y_max = latlon_to_tile(clamped_bounds.ymax, clamped_bounds.xmin, zoom)  # NW corner
        x_max, y_min = latlon_to_tile(clamped_bounds.ymin, clamped_bounds.xmax, zoom)  # SE corner
        
        # Ensure valid range
        n_tiles = 2 ** zoom
        x_min = max(0, min(n_tiles - 1, x_min))
        x_max = max(0, min(n_tiles - 1, x_max))
        y_min = max(0, min(n_tiles - 1, y_min))
        y_max = max(0, min(n_tiles - 1, y_max))
        
        # Ensure x_max >= x_min and y_max >= y_min
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        
        # Calculate output size
        n_tiles_x = x_max - x_min + 1
        n_tiles_y = y_max - y_min + 1
        
        # Sanity check
        if n_tiles_x <= 0 or n_tiles_y <= 0:
            # Return empty transparent image
            empty = np.zeros((256, 256, 4), dtype=np.uint8)
            return empty, clamped_bounds
        
        # Limit maximum tiles to prevent memory issues
        max_tiles = 100
        if n_tiles_x * n_tiles_y > max_tiles:
            # Reduce zoom level
            return self._get_tiles_for_bounds(source, bounds, max(0, zoom - 1))
        
        # Create output array
        output_width = n_tiles_x * tile_size
        output_height = n_tiles_y * tile_size
        output = np.zeros((output_height, output_width, 4), dtype=np.uint8)
        
        # Fetch and place tiles
        for ty in range(y_min, y_max + 1):
            for tx in range(x_min, x_max + 1):
                tile_data = self._fetch_tile(source, zoom, tx, ty)
                
                if tile_data is not None:
                    # Calculate position in output
                    px = (tx - x_min) * tile_size
                    py = (ty - y_min) * tile_size
                    
                    # Handle tile size mismatch
                    th, tw = tile_data.shape[:2]
                    h = min(th, output_height - py)
                    w = min(tw, output_width - px)
                    
                    output[py:py+h, px:px+w] = tile_data[:h, :w]
        
        # Calculate actual bounds covered
        nw_lat, nw_lon = tile_to_latlon(x_min, y_min, zoom)
        se_lat, se_lon = tile_to_latlon(x_max + 1, y_max + 1, zoom)
        actual_bounds = BoundingBox(
            xmin=nw_lon,
            ymin=se_lat,
            xmax=se_lon,
            ymax=nw_lat,
        )
        
        return output, actual_bounds
    
    # =========================================================================
    # Main interface
    # =========================================================================
    
    def get_image(
        self,
        bounds: BoundingBox,
        target_size: Tuple[int, int] = (800, 600),
        source_name: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], BoundingBox]:
        """Get composited tile image for given bounds.
        
        Args:
            bounds: Bounding box in lat/lon (WGS84).
            target_size: Desired output size (width, height) in pixels.
            source_name: Specific source to use, or None for first visible.
        
        Returns:
            Tuple of (RGBA image array, actual bounds covered).
            Returns (None, bounds) if no tiles could be loaded.
        """
        # Get source
        if source_name:
            if source_name not in self._sources:
                return None, bounds
            source = self._sources[source_name]
        else:
            visible = self.visible_sources
            if not visible:
                return None, bounds
            source = visible[0]
        
        # Choose zoom level
        zoom = choose_zoom_level(bounds, target_size, source.tile_size)
        zoom = max(source.min_zoom, min(source.max_zoom, zoom))
        
        # Get tiles
        image, actual_bounds = self._get_tiles_for_bounds(source, bounds, zoom)
        
        # Crop to requested bounds if needed
        if image is not None and HAS_PIL:
            image = self._crop_to_bounds(image, actual_bounds, bounds)
        
        # Resize to target size
        if image is not None and HAS_PIL:
            img = Image.fromarray(image)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            image = np.array(img)
        
        return image, bounds
    
    def get_image_with_bounds(
        self,
        bounds: BoundingBox,
        target_size: Tuple[int, int] = (800, 600),
        source_name: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], BoundingBox]:
        """Get composited tile image reprojected to linear lat/lon coordinates.
        
        Web Mercator tiles have non-linear latitude spacing. This method 
        reprojects the tiles to have uniform lat/lon pixel spacing, allowing
        correct display with simple setRect positioning.
        
        Args:
            bounds: Bounding box in lat/lon (WGS84) - the requested area.
            target_size: Desired output size (width, height) in pixels.
            source_name: Specific source to use, or None for first visible.
        
        Returns:
            Tuple of (RGBA image array, bounds).
            The image is reprojected to linear lat/lon matching the returned bounds.
            Returns (None, bounds) if no tiles could be loaded.
        """
        # Get source
        if source_name:
            if source_name not in self._sources:
                return None, bounds
            source = self._sources[source_name]
        else:
            visible = self.visible_sources
            if not visible:
                return None, bounds
            source = visible[0]
        
        # Choose zoom level
        zoom = choose_zoom_level(bounds, target_size, source.tile_size)
        zoom = max(source.min_zoom, min(source.max_zoom, zoom))
        
        # Get tiles - actual_bounds is the Mercator tile bounds in lat/lon
        image, merc_bounds = self._get_tiles_for_bounds(source, bounds, zoom)
        
        if image is None:
            return None, bounds
        
        # Reproject from Web Mercator to linear lat/lon
        # This corrects the non-linear latitude distortion
        try:
            output = reproject_mercator_to_latlon(
                image=image,
                merc_bounds=merc_bounds,
                target_bounds=bounds,  # Use requested bounds for output
                target_size=target_size,
            )
            # Return the requested bounds since the image now matches them
            return output, bounds
        except Exception as e:
            # Fallback: just resize without reprojection (will have distortion)
            warnings.warn(f"Reprojection failed, using simple resize: {e}")
            if HAS_PIL:
                img = Image.fromarray(image)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                output = np.array(img)
            else:
                output = image
            return output, merc_bounds

    def build_image(
        self,
        source_name: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
        """Build a tile image for the current axis settings.
        
        This is the main interface following the EchogramBuilder pattern.
        Call set_axis_latlon() or set_bounds() + set_max_pixels() first.
        
        The returned extent tuple is (xmin, xmax, ymin, ymax) for use with
        PyQtGraph's ImageItem.setRect() (like EchogramBuilder).
        
        Args:
            source_name: Specific source to use, or None for first visible.
            
        Returns:
            Tuple of (RGBA image array, extent).
            extent is (lon_min, lon_max, lat_min, lat_max) in lat/lon coordinates.
            Returns (None, None) if no bounds set or no tiles could be loaded.
            
        Example:
            tiles = TileBuilder()
            tiles.add_osm()
            
            # Set view region
            tiles.set_axis_latlon(min_lat=51.0, max_lat=52.0,
                                  min_lon=2.5, max_lon=3.5,
                                  max_pixels=(800, 600))
            
            # Build image
            image, extent = tiles.build_image()
            # extent = (lon_min, lon_max, lat_min, lat_max)
        """
        if self._current_bounds is None:
            return None, None
        
        bounds = self._current_bounds
        target_size = (self._max_pixels[1], self._max_pixels[0])  # (width, height)
        
        # Use get_image_with_bounds for precise bounds
        image, actual_bounds = self.get_image_with_bounds(
            bounds=bounds,
            target_size=target_size,
            source_name=source_name,
        )
        
        if image is None:
            return None, None
        
        # Return extent as (xmin, xmax, ymin, ymax) like EchogramBuilder
        extent = (
            actual_bounds.xmin,  # lon_min
            actual_bounds.xmax,  # lon_max
            actual_bounds.ymin,  # lat_min
            actual_bounds.ymax,  # lat_max
        )
        
        return image, extent
