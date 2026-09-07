"""Fast web-map tiles for the map viewer.

Thin wrapper around two maintained packages:

* **contextily** – parallel tile download (``n_connections``), on-disk caching and
  Web-Mercator → lat/lon reprojection (``warp_tiles``).
* **xyzservices** – maintained catalogue of XYZ tile providers (OSM, Esri, CartoDB, …).

The builder returns RGBA images already reprojected to linear lat/lon (WGS84) so they
line up with the map viewer's lat/lon axes.

Sources cover street/imagery maps, hybrids (imagery + labels/seamarks, alpha-composited),
ocean/bathymetry (GEBCO, EMODnet, Esri Ocean), near-real-time satellite and science layers
(NASA GIBS VIIRS/MODIS true colour, sea-surface temperature, chlorophyll, night lights) and
cloud-free Sentinel-2. See :data:`TILE_SOURCES` for the full list.

Example::

    from themachinethatgoesping.pingprocessing.overview.map_builder import TileBuilder
    tiles = TileBuilder()
    tiles.set_source("imagery_seamarks")  # or "esri_hybrid", "gebco",
                                          # "nasa_viirs_truecolor" (near-real-time), "nasa_sst", ...
    image, bounds = tiles.get_image_with_bounds(bbox, target_size=(1200, 900))
"""
from __future__ import annotations

import math
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .coordinate_system import BoundingBox

try:
    import contextily as _cx
    import xyzservices
    import xyzservices.providers as _xyz
    HAS_TILES = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TILES = False

# Web-Mercator is only defined up to this latitude.
_MERCATOR_LAT_LIMIT = 85.0511287798
# Concurrent tile downloads. contextily defaults to 1 (slow); QGIS-like speed needs several.
# Note: OpenStreetMap's usage policy asks for <=2 parallel connections.
DEFAULT_N_CONNECTIONS = 16


# ============================================================================
# Provider catalogue (xyzservices) + a few custom sources
# ============================================================================

# friendly name -> dotted path in xyzservices.providers
_PRESET_PATHS = {
    # base / street maps
    "osm": "OpenStreetMap.Mapnik",
    "cartodb_positron": "CartoDB.Positron",
    "cartodb_darkmatter": "CartoDB.DarkMatter",
    "cartodb_voyager": "CartoDB.Voyager",
    "opentopomap": "OpenTopoMap",
    "esri_worldstreetmap": "Esri.WorldStreetMap",
    "esri_worldgraycanvas": "Esri.WorldGrayCanvas",
    "esri_natgeo": "Esri.NatGeoWorldMap",
    # imagery
    "esri_worldimagery": "Esri.WorldImagery",
    # terrain / physical
    "esri_worldshadedrelief": "Esri.WorldShadedRelief",
    "esri_worldphysical": "Esri.WorldPhysical",
    # ocean
    "esri_oceanbasemap": "Esri.OceanBasemap",
    # NASA GIBS science layers (near-real-time; time=default -> latest available)
    "nasa_viirs_truecolor": "NASAGIBS.ViirsTrueColorCR",
    "nasa_modis_truecolor": "NASAGIBS.ModisTerraTrueColorCR",
    "nasa_chlorophyll": "NASAGIBS.ModisTerraChlorophyll",
    "nasa_earth_at_night": "NASAGIBS.ViirsEarthAtNight2012",
    "nasa_bluemarble": "NASAGIBS.BlueMarble",
}

# GIBS presets whose {time} should resolve to the latest available imagery.
_PRESET_TIME = {
    "nasa_viirs_truecolor": "default",
    "nasa_modis_truecolor": "default",
    "nasa_chlorophyll": "default",
}

# custom single providers not in xyzservices (built as xyzservices.TileProvider)
_CUSTOM_SPECS = {
    "gebco": dict(
        name="GEBCO",
        url="https://tiles.arcgis.com/tiles/C8EMgrsFcRFL6LrL/arcgis/rest/services/"
        "GEBCO_basemap_NCEI/MapServer/tile/{z}/{y}/{x}",
        attribution="GEBCO Compilation Group / NCEI",
        max_zoom=9,
    ),
    "emodnet_bathymetry": dict(
        name="EMODnet.Bathymetry",
        url="https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{z}/{x}/{y}.png",
        attribution="© EMODnet Bathymetry",
        max_zoom=12,
    ),
    "nasa_sst": dict(  # GHRSST L4 MUR sea-surface temperature (latest)
        name="NASA.SeaSurfaceTemperature",
        url="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "GHRSST_L4_MUR_Sea_Surface_Temperature/default/{time}/"
        "GoogleMapsCompatible_Level7/{z}/{y}/{x}.png",
        time="default",
        attribution="NASA GIBS / GHRSST MUR",
        max_zoom=7,
    ),
    "eox_sentinel2": dict(  # cloud-free Sentinel-2 mosaic
        name="EOX.Sentinel2Cloudless",
        url="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/"
        "GoogleMapsCompatible/{z}/{y}/{x}.jpg",
        attribution="Sentinel-2 cloudless (EOX, contains modified Copernicus data)",
        max_zoom=16,
    ),
    "google_satellite": dict(
        name="Google.Satellite",
        url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attribution="© Google",
        max_zoom=20,
    ),
    "google_hybrid": dict(
        name="Google.Hybrid",
        url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attribution="© Google",
        max_zoom=20,
    ),
}

# transparent overlays used only to build hybrids (not offered standalone)
_OVERLAY_SPECS = {
    "esri_boundaries": dict(
        name="Esri.BoundariesPlaces",
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
        "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attribution="Esri",
        max_zoom=19,
    ),
    "esri_ocean_reference": dict(
        name="Esri.OceanReference",
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/"
        "World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}",
        attribution="Esri",
        max_zoom=16,
    ),
    "openseamap": dict(
        name="OpenSeaMap.Seamark",
        url="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        attribution="© OpenSeaMap contributors",
        max_zoom=18,
    ),
}

# transparent label overlays from xyzservices (offered in the GUI overlay dropdown)
_OVERLAY_PRESET_PATHS = {
    "cartodb_labels": "CartoDB.PositronOnlyLabels",
    "cartodb_darklabels": "CartoDB.DarkMatterOnlyLabels",
}

# hybrids = base + transparent overlay(s), alpha-composited (keys into the pools above)
_COMPOSITE_SPECS = {
    "esri_hybrid": ["esri_worldimagery", "esri_boundaries"],
    "esri_ocean_labeled": ["esri_oceanbasemap", "esri_ocean_reference"],
    "imagery_seamarks": ["esri_worldimagery", "openseamap"],
    "sentinel2_seamarks": ["eox_sentinel2", "openseamap"],
}


def _resolve_provider(path: str):
    """Resolve a dotted xyzservices provider path (e.g. ``Esri.WorldImagery``)."""
    obj = _xyz
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _build_catalog():
    """Build the source + overlay catalogues (token-free only).

    Returns ``(catalog, overlays)``. Catalog values are an :class:`xyzservices.TileProvider`
    (single-layer source) or a list of providers (base first) for composited hybrids. Overlays are
    transparent single providers offered as the second (overlay) layer.
    """
    catalog: Dict[str, object] = {}
    overlays: Dict[str, object] = {}
    if not HAS_TILES:
        return catalog, overlays

    pool: Dict[str, object] = {}  # singles + overlays, referenced by the composites
    for name, path in _PRESET_PATHS.items():
        try:
            provider = _resolve_provider(path)
            if provider.requires_token():  # needs an API key -> skip
                continue
            if name in _PRESET_TIME:
                provider = provider(time=_PRESET_TIME[name])
            catalog[name] = pool[name] = provider
        except Exception:
            continue
    for name, spec in _CUSTOM_SPECS.items():
        try:
            catalog[name] = pool[name] = xyzservices.TileProvider(**spec)
        except Exception:
            continue
    for name, spec in _OVERLAY_SPECS.items():
        try:
            pool[name] = overlays[name] = xyzservices.TileProvider(**spec)
        except Exception:
            continue
    for name, path in _OVERLAY_PRESET_PATHS.items():
        try:
            provider = _resolve_provider(path)
            if provider.requires_token():
                continue
            pool[name] = overlays[name] = provider
        except Exception:
            continue
    for name, layer_keys in _COMPOSITE_SPECS.items():
        layers = [pool.get(key) for key in layer_keys]
        if all(layers):
            catalog[name] = list(layers)
    return catalog, overlays


#: Available tile sources: friendly name -> :class:`xyzservices.TileProvider` (or list for hybrids).
#: Overlay sources: transparent layers offered as the GUI's second (overlay) layer.
TILE_SOURCES: Dict[str, object]
OVERLAY_SOURCES: Dict[str, object]
TILE_SOURCES, OVERLAY_SOURCES = _build_catalog()


def list_available_sources() -> List[str]:
    """List the names of the available pre-defined (base) tile sources."""
    return list(TILE_SOURCES.keys())


def list_overlay_sources() -> List[str]:
    """List the names of the available transparent overlay sources."""
    return list(OVERLAY_SOURCES.keys())


@dataclass
class TileSource:
    """Lightweight description of a custom XYZ tile source.

    Kept for backwards compatibility; :meth:`TileBuilder.add_source` also accepts an
    :class:`xyzservices.TileProvider` directly.
    """

    name: str
    url_template: str
    attribution: str = ""
    max_zoom: int = 19
    min_zoom: int = 0
    tile_size: int = 256
    headers: Dict[str, str] = field(default_factory=dict)

    def to_provider(self):
        """Convert to an :class:`xyzservices.TileProvider`."""
        return xyzservices.TileProvider(
            name=self.name,
            url=self.url_template,
            attribution=self.attribution,
            max_zoom=self.max_zoom,
            min_zoom=self.min_zoom,
            html_attribution=self.attribution,
        )


# ============================================================================
# Helpers
# ============================================================================

def _to_rgba(image: np.ndarray) -> np.ndarray:
    """Return the image as a contiguous uint8 RGBA array (adds an opaque alpha channel)."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        image = np.dstack([image, alpha])
    return np.ascontiguousarray(image, dtype=np.uint8)


def _alpha_over(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    """Alpha-composite RGBA *over* on top of RGBA *base* (same shape) -> uint8 RGBA."""
    b = base.astype(np.float32) / 255.0
    o = over.astype(np.float32) / 255.0
    oa = o[..., 3:4]
    ba = b[..., 3:4]
    out_a = oa + ba * (1.0 - oa)
    out_rgb = (o[..., :3] * oa + b[..., :3] * ba * (1.0 - oa)) / np.clip(out_a, 1e-6, None)
    return (np.concatenate([out_rgb, out_a], axis=-1) * 255.0).astype(np.uint8)


# ============================================================================
# TileBuilder
# ============================================================================

class TileBuilder:
    """Fetch and composite web-map tiles for a lat/lon view.

    Sources are :class:`xyzservices.TileProvider` objects (see :data:`TILE_SOURCES` for the
    presets). Tiles are downloaded in parallel and cached on disk by contextily; the returned
    images are reprojected to linear lat/lon (WGS84).
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        n_connections: int = DEFAULT_N_CONNECTIONS,
        max_pixels: Tuple[int, int] = (2000, 2000),
    ) -> None:
        if not HAS_TILES:
            warnings.warn(
                "TileBuilder needs 'contextily' and 'xyzservices' "
                "(mamba/pip install contextily xyzservices); tiles are disabled."
            )

        self._sources: Dict[str, object] = {}       # name -> TileProvider
        self._visible: Dict[str, bool] = {}          # name -> visible
        self._opacity: Dict[str, float] = {}         # name -> 0..1
        self._n_connections = int(n_connections)
        self._max_pixels = tuple(max_pixels)
        self._current_bounds: Optional[BoundingBox] = None
        self._time: Optional[str] = None  # None = latest ('default') for time-dependent layers

        self._cache_dir = Path(cache_dir) if cache_dir else (
            Path.home() / ".cache" / "pingprocessing" / "tiles"
        )
        if HAS_TILES:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            _cx.set_cache_dir(str(self._cache_dir))

    # ------------------------------------------------------------------ sources
    @property
    def source_names(self) -> List[str]:
        return list(self._sources)

    @property
    def sources(self) -> List[object]:
        return list(self._sources.values())

    @property
    def visible_sources(self) -> List[object]:
        return [self._sources[n] for n in self._sources if self._visible.get(n)]

    @property
    def active_source_name(self) -> Optional[str]:
        """Name of the first visible source, or ``None`` if no source is active."""
        for name in self._sources:
            if self._visible.get(name):
                return name
        return None

    def add_source(self, source, name: Optional[str] = None, visible: bool = True) -> "TileBuilder":
        """Add a tile source.

        *source* may be an :class:`xyzservices.TileProvider`, a :class:`TileSource`, or a list of
        providers (base first) that are alpha-composited into a hybrid.
        """
        if isinstance(source, TileSource):
            name = name or source.name
            source = source.to_provider()
        if isinstance(source, (list, tuple)):
            source = [p.to_provider() if isinstance(p, TileSource) else p for p in source]
            name = name or "hybrid"
        elif name is None:
            name = source.get("name", "tiles") if hasattr(source, "get") else "tiles"
        self._sources[name] = source
        self._visible[name] = visible
        self._opacity.setdefault(name, 1.0)
        return self

    def add_preset(self, name: str) -> "TileBuilder":
        """Add a pre-defined source (or overlay) from the catalogue by name."""
        provider = TILE_SOURCES.get(name)
        if provider is None:
            provider = OVERLAY_SOURCES.get(name)
        if provider is None:
            raise ValueError(
                f"Unknown tile source '{name}'. Available: {list_available_sources()}"
            )
        return self.add_source(provider, name=name)

    def add_xyz(
        self, name: str, url_template: str, attribution: str = "", max_zoom: int = 19, **kwargs
    ) -> "TileBuilder":
        """Add a custom XYZ source from a ``{z}/{x}/{y}`` URL template."""
        provider = xyzservices.TileProvider(
            name=name, url=url_template, attribution=attribution, max_zoom=max_zoom, **kwargs
        )
        return self.add_source(provider, name=name)

    def set_source(self, name: str) -> "TileBuilder":
        """Use a single preset source (clears any others)."""
        self.clear_sources()
        return self.add_preset(name)

    # convenience presets
    def add_osm(self) -> "TileBuilder":
        return self.add_preset("osm")

    def add_esri_worldimagery(self) -> "TileBuilder":
        return self.add_preset("esri_worldimagery")

    def add_esri_ocean(self) -> "TileBuilder":
        return self.add_preset("esri_oceanbasemap")

    def add_cartodb_positron(self) -> "TileBuilder":
        return self.add_preset("cartodb_positron")

    def add_cartodb_darkmatter(self) -> "TileBuilder":
        return self.add_preset("cartodb_darkmatter")

    def add_opentopomap(self) -> "TileBuilder":
        return self.add_preset("opentopomap")

    def set_source_visible(self, name: str, visible: bool) -> "TileBuilder":
        if name in self._sources:
            self._visible[name] = visible
        return self

    def set_source_opacity(self, name: str, opacity: float) -> "TileBuilder":
        if name in self._sources:
            self._opacity[name] = float(min(1.0, max(0.0, opacity)))
        return self

    def remove_source(self, name: str) -> "TileBuilder":
        self._sources.pop(name, None)
        self._visible.pop(name, None)
        self._opacity.pop(name, None)
        return self

    def clear_sources(self) -> "TileBuilder":
        self._sources.clear()
        self._visible.clear()
        self._opacity.clear()
        return self

    def set_layers(self, names: List[Optional[str]]) -> "TileBuilder":
        """Set the ordered composite stack of active sources (base first).

        Each name is a key in :data:`TILE_SOURCES`; ``None`` / ``"None"`` entries are ignored.
        Replaces any previously active sources (used for the GUI base + overlay selection).
        """
        self.clear_sources()
        for name in names:
            if name and name != "None":
                self.add_preset(name)
        return self

    @property
    def active_layer_names(self) -> List[str]:
        """Names of the active (visible) sources, base first."""
        return [n for n in self._sources if self._visible.get(n)]

    def is_time_dependent(self, name: Optional[str] = None) -> bool:
        """Whether *name* (or any active source, if None) depends on an acquisition date."""
        names = [name] if name else self.active_layer_names
        for entry in names:
            source = self._sources.get(entry)
            if source is None:
                source = TILE_SOURCES.get(entry) or OVERLAY_SOURCES.get(entry)
            for provider in (source if isinstance(source, (list, tuple)) else [source]):
                if provider is not None and "{time}" in str(provider.get("url", "")):
                    return True
        return False

    # -------------------------------------------------------------- axis / view
    def set_axis_latlon(
        self,
        min_lat: float = np.nan,
        max_lat: float = np.nan,
        min_lon: float = np.nan,
        max_lon: float = np.nan,
        max_pixels: Optional[Tuple[int, int]] = None,
    ) -> "TileBuilder":
        """Set the view extent in lat/lon (``np.nan`` = full extent)."""
        if max_pixels is not None:
            self._max_pixels = tuple(max_pixels)
        xmin = -180.0 if np.isnan(min_lon) else min_lon
        xmax = 180.0 if np.isnan(max_lon) else max_lon
        ymin = -_MERCATOR_LAT_LIMIT if np.isnan(min_lat) else max(-_MERCATOR_LAT_LIMIT, min_lat)
        ymax = _MERCATOR_LAT_LIMIT if np.isnan(max_lat) else min(_MERCATOR_LAT_LIMIT, max_lat)
        self._current_bounds = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        return self

    def set_bounds(self, bounds: BoundingBox) -> "TileBuilder":
        """Set the current view bounds (lon as x, lat as y)."""
        self._current_bounds = BoundingBox(
            xmin=max(-180.0, min(180.0, bounds.xmin)),
            ymin=max(-_MERCATOR_LAT_LIMIT, min(_MERCATOR_LAT_LIMIT, bounds.ymin)),
            xmax=max(-180.0, min(180.0, bounds.xmax)),
            ymax=max(-_MERCATOR_LAT_LIMIT, min(_MERCATOR_LAT_LIMIT, bounds.ymax)),
        )
        return self

    def set_max_pixels(self, max_pixels: Tuple[int, int]) -> "TileBuilder":
        self._max_pixels = tuple(max_pixels)
        return self

    def reset_bounds(self) -> "TileBuilder":
        self._current_bounds = None
        return self

    @property
    def max_pixels(self) -> Tuple[int, int]:
        return self._max_pixels

    @property
    def current_bounds(self) -> Optional[BoundingBox]:
        return self._current_bounds

    # ------------------------------------------------------------------- time
    def set_time(self, time) -> "TileBuilder":
        """Set the acquisition date for time-dependent layers.

        *time* may be a ``datetime`` / ``date``, an ISO ``"YYYY-MM-DD"`` string, or ``None`` for the
        latest available imagery. Non-time-dependent layers ignore it.
        """
        if time is None:
            self._time = None
        elif hasattr(time, "strftime"):
            self._time = time.strftime("%Y-%m-%d")
        else:
            self._time = str(time)
        return self

    def get_time(self) -> Optional[str]:
        return self._time

    @property
    def time(self) -> Optional[str]:
        return self._time

    def _with_time(self, provider):
        """Return *provider* with its ``{time}`` bound to the requested date (or 'default')."""
        if provider is None or "{time}" not in str(provider.get("url", "")):
            return provider
        return provider(time=self._time or "default")

    # --------------------------------------------------------------- rendering
    def _select_source(self, source_name: Optional[str]):
        """Return the provider(s) to render: a named source, or the active visible stack."""
        if source_name:
            if source_name not in self._sources and (
                source_name in TILE_SOURCES or source_name in OVERLAY_SOURCES
            ):
                self.add_preset(source_name)
            return self._sources.get(source_name)
        # no explicit source -> composite all visible sources (base first), flattening presets
        layers: List[object] = []
        for src in self.visible_sources:
            layers.extend(src if isinstance(src, (list, tuple)) else [src])
        if not layers:
            return None
        return layers[0] if len(layers) == 1 else layers

    def _source_name_for(self, provider) -> Optional[str]:
        for name, prov in self._sources.items():
            if prov is provider:
                return name
        return None

    def _choose_zoom(self, west: float, east: float, target_width: int, providers) -> int:
        """Pick the smallest zoom whose mosaic is at least the target pixel width.

        For a composited source (several providers) the zoom is clamped to the range valid for
        every layer.
        """
        layers = providers if isinstance(providers, (list, tuple)) else [providers]
        span = max(1e-9, east - west)
        zoom = int(math.ceil(math.log2(360.0 * max(1, target_width) / (256.0 * span))))
        zmin = max(int(p.get("min_zoom", 0) or 0) for p in layers)
        zmax = min(int(p.get("max_zoom", 19) or 19) for p in layers)
        return max(zmin, min(zmax, zoom))

    def _fetch(
        self, bounds: BoundingBox, target_size: Tuple[int, int], source
    ) -> Tuple[Optional[np.ndarray], BoundingBox]:
        """Download + composite + reproject tiles for *bounds* (RGBA, lat/lon).

        *source* is a single provider or a list of providers (base first) to alpha-composite.
        """
        if not HAS_TILES or source is None:
            return None, bounds

        west = max(-180.0, bounds.xmin)
        east = min(180.0, bounds.xmax)
        south = max(-_MERCATOR_LAT_LIMIT, bounds.ymin)
        north = min(_MERCATOR_LAT_LIMIT, bounds.ymax)
        if east <= west or north <= south:
            return None, bounds

        layers = source if isinstance(source, (list, tuple)) else [source]
        zoom = self._choose_zoom(west, east, target_size[0], layers)

        image = None
        extent = None
        for index, provider in enumerate(layers):
            provider = self._with_time(provider)
            try:
                # contextily fetches all tiles in parallel and returns a Web-Mercator mosaic
                tile_img, tile_ext = _cx.bounds2img(
                    west, south, east, north,
                    zoom=zoom, source=provider, ll=True, n_connections=self._n_connections,
                )
            except Exception as error:
                if index == 0:  # the base layer must succeed
                    warnings.warn(f"Tile loading failed: {error}")
                    return None, bounds
                continue  # a missing overlay is not fatal
            tile_img = _to_rgba(tile_img)
            if image is None:
                image, extent = tile_img, tile_ext
            else:
                image = _alpha_over(image, tile_img)

        if image is None:
            return None, bounds

        try:
            # reproject the composited mosaic to linear lat/lon so it matches the viewer axes
            image, extent = _cx.warp_tiles(image, extent, t_crs="EPSG:4326")
        except Exception as error:
            warnings.warn(f"Tile reprojection failed: {error}")
            return None, bounds

        image = _to_rgba(image)

        opacity = self._opacity.get(self._source_name_for(source), 1.0)
        if opacity < 1.0:
            image = image.copy()
            image[..., 3] = (image[..., 3] * opacity).astype(np.uint8)

        # contextily extent = (left, right, bottom, top) = (lon_min, lon_max, lat_min, lat_max)
        actual = BoundingBox(xmin=extent[0], ymin=extent[2], xmax=extent[1], ymax=extent[3])
        return image, actual

    def get_image_with_bounds(
        self,
        bounds: BoundingBox,
        target_size: Tuple[int, int] = (800, 600),
        source_name: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], BoundingBox]:
        """Return ``(rgba_image, actual_bounds)`` reprojected to linear lat/lon."""
        return self._fetch(bounds, target_size, self._select_source(source_name))

    def get_image(
        self,
        bounds: BoundingBox,
        target_size: Tuple[int, int] = (800, 600),
        source_name: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], BoundingBox]:
        """Return ``(rgba_image, bounds)`` (alias of :meth:`get_image_with_bounds`)."""
        image, actual = self.get_image_with_bounds(bounds, target_size, source_name)
        return image, (actual if image is not None else bounds)

    def build_image(
        self, source_name: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
        """Build a tile image for the current axis settings (EchogramBuilder-style).

        Returns ``(rgba_image, extent)`` with ``extent = (lon_min, lon_max, lat_min, lat_max)``,
        or ``(None, None)`` if no bounds are set or no tiles could be loaded.
        """
        if self._current_bounds is None:
            return None, None
        target_size = (self._max_pixels[1], self._max_pixels[0])  # (width, height)
        image, actual = self.get_image_with_bounds(self._current_bounds, target_size, source_name)
        if image is None:
            return None, None
        return image, (actual.xmin, actual.xmax, actual.ymin, actual.ymax)

    # ------------------------------------------------------------------- export
    def export_geotiff(
        self,
        path,
        bounds: BoundingBox,
        target_size: Tuple[int, int] = (4000, 4000),
        source_name: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch *bounds* at *target_size* and save a georeferenced GeoTIFF (EPSG:4326).

        Returns the written path, or ``None`` if no tiles could be loaded.
        """
        image, actual = self.get_image_with_bounds(bounds, target_size, source_name)
        if image is None:
            return None
        return self.save_geotiff(path, image, actual)

    @staticmethod
    def save_geotiff(path, image: np.ndarray, bounds: BoundingBox) -> str:
        """Write an RGBA lat/lon *image* (row 0 = north) to a georeferenced GeoTIFF (EPSG:4326)."""
        import rasterio
        from rasterio.transform import from_bounds

        image = _to_rgba(image)
        height, width = image.shape[:2]
        transform = from_bounds(
            bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax, width, height
        )
        path = str(path)
        with rasterio.open(
            path, "w", driver="GTiff", height=height, width=width, count=4,
            dtype="uint8", crs="EPSG:4326", transform=transform, photometric="RGB",
        ) as dst:
            for band in range(4):
                dst.write(image[:, :, band], band + 1)
        return path

    # ------------------------------------------------------------------- cache
    def clear_cache(self) -> None:
        """Delete the on-disk tile cache."""
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
