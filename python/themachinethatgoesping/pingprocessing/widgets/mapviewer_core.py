"""Core map viewer logic independent of the UI toolkit.

Manages pyqtgraph scene-graph items (PlotItem, ImageItem, ScatterPlotItem, …)
and reads / writes control state through a :class:`ControlPanel` abstraction.
Adapters (``mapviewer_jupyter``, ``mapviewer_qt``) create the concrete
controls and wire up display / async loading.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from .control_spec import ControlPanel, MAP_COLORMAPS
from . import pyqtgraph_helpers as pgh

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000.0  # Earth radius in metres
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _format_distance(metres: float, unit: str = "m") -> str:
    """Format *metres* into the requested display *unit*."""
    if unit == "nm":
        value = metres / 1852.0
        if value < 0.01:
            return f"{value:.4f} nm"
        if value < 1:
            return f"{value:.2f} nm"
        return f"{value:.1f} nm"
    if unit == "km":
        value = metres / 1000.0
        if value < 0.01:
            return f"{metres:.0f} m"
        if value < 10:
            return f"{value:.2f} km"
        return f"{value:.1f} km"
    # metres
    if metres < 1:
        return f"{metres:.2f} m"
    if metres < 1000:
        return f"{metres:.0f} m"
    return f"{metres / 1000:.2f} km"


# "Nice" distances for the scale bar (metres)
_NICE_DISTANCES = [
    1, 2, 5, 10, 20, 50, 100, 200, 500,
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 500_000, 1_000_000, 2_000_000,
]

def _get_colormap_lut(name: str, n_colors: int = 256) -> np.ndarray:
    """Get a colormap LUT (Look-Up Table) for pyqtgraph."""
    if not HAS_MATPLOTLIB:
        gray = np.linspace(0, 255, n_colors, dtype=np.uint8)
        return np.stack([gray, gray, gray, np.full(n_colors, 255, dtype=np.uint8)], axis=1)
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(0, 1, n_colors))
    return (colors * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerRenderSettings:
    """Viewer-side rendering settings for a layer."""
    colormap: str = "viridis"
    opacity: float = 1.0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    blend_mode: str = "alpha"


@dataclass
class TrackInfo:
    """Track display information."""
    name: str
    latitudes: np.ndarray
    longitudes: np.ndarray
    color: str
    line_width: float = 2.0
    is_active: bool = False
    visible: bool = True
    slot_idx: Optional[int] = None


@dataclass
class OverviewTrackInfo:
    """Track backed by a PingOverview with zoom-adaptive downsampling."""
    name: str
    overview: Any
    color: str
    max_points: int = 50_000
    line_width: float = 2.0
    is_active: bool = False
    visible: bool = True
    slot_idx: Optional[int] = None
    show_points: bool = False
    point_size: float = 5.0
    point_symbol: str = 'o'
    point_outline: bool = True
    latitudes: Optional[np.ndarray] = None
    longitudes: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# MapCore
# ---------------------------------------------------------------------------

class MapCore:
    """Backend-agnostic map viewer core.

    Parameters
    ----------
    builder : any, optional
        MapBuilder with data layers to display.
    tile_builder : any, optional
        TileBuilder for background tiles.
    panel : ControlPanel
        Named :class:`ControlHandle` objects for reading / writing UI state.
    graphics : pg.GraphicsLayoutWidget
        The pyqtgraph widget (jupyter-rfb or native).
    max_render_size : tuple[int, int]
        Maximum size for rendered layers.
    """

    TRACK_COLORS = [
        "#FF0000", "#00FF00", "#0000FF", "#FF00FF",
        "#00FFFF", "#FFFF00", "#FF8000", "#8000FF",
    ]

    def __init__(
        self,
        builder: Any = None,
        tile_builder: Any = None,
        panel: Optional[ControlPanel] = None,
        graphics: Any = None,
        max_render_size: Tuple[int, int] = (2000, 2000),
    ) -> None:
        self._builder = builder
        self._tile_builder = tile_builder
        self._panel = panel
        self.graphics = graphics
        self._max_render_size = max_render_size

        # State
        self._current_bounds = None
        self._layer_images: Dict[str, pg.ImageItem] = {}
        self._coordinate_system = None

        # Viewer-controlled rendering settings per layer
        self._layer_render_settings: Dict[str, LayerRenderSettings] = {}

        # Tile background layer
        self._tile_image: Optional[pg.ImageItem] = None
        self._tile_visible: bool = True if tile_builder else False

        # Track overlays
        self._tracks: Dict[str, TrackInfo] = {}
        self._overview_tracks: Dict[str, OverviewTrackInfo] = {}
        self._track_plots: List[Any] = []
        self._active_track_name: Optional[str] = None

        # User marker overlays
        self._user_markers: Dict[str, Any] = {}

        # Ping position marker
        self._ping_marker: Optional[pg.ScatterPlotItem] = None
        self._current_ping_latlon: Optional[Tuple[float, float]] = None

        # Connected viewers
        self._echogram_viewer = None
        self._wci_viewer = None

        # Callbacks
        self._click_callbacks: List[Callable] = []
        self._view_change_callbacks: List[Callable] = []

        # Colorbar state
        self._colorbar_item: Optional[pg.ColorBarItem] = None
        self._active_colorbar_layer: Optional[str] = None
        self._layer_colorbar_levels: Dict[str, Tuple[float, float]] = {}

        # Tile cache key (for avoiding redundant loads)
        self._tile_cache_key: Optional[Tuple] = None

        # Scale bar
        self._scale_bar_visible: bool = True
        self._scale_bar_line: Optional[pg.PlotCurveItem] = None
        self._scale_bar_text: Optional[pg.TextItem] = None
        self._scale_bar_bg: Optional[pg.QtWidgets.QGraphicsRectItem] = None

        # Measurement tool
        self._measure_active: bool = False
        self._measure_unit: str = "m"  # "m", "km", "nm"
        self._measure_points: List[Tuple[float, float]] = []  # (lon, lat) pairs
        self._measure_plots: List[Any] = []
        self._measure_markers: Optional[pg.ScatterPlotItem] = None
        self._measure_labels: List[pg.TextItem] = []
        self._measure_total_label: Optional[pg.TextItem] = None
        self._ctx_measure_action: Optional[Any] = None
        self._ctx_unit_actions: Dict[str, Any] = {}

        # Adapter callbacks (set by the adapter)
        self._schedule_update: Optional[Callable] = None
        self._request_draw: Optional[Callable] = None
        self._trigger_tile_load: Optional[Callable] = None
        self._report_error: Optional[Callable] = None
        self._update_track_legend: Optional[Callable] = None

        # Ignore range changes flag (used during programmatic pan/zoom)
        self._ignore_range_changes = False
        # Timestamp until which deferred sigRangeChanged should be ignored
        self._ignore_range_until: float = 0.0

        # Throttle ping update
        self._last_ping_update_time: float = 0.0

        # Throttle WCI ping-change callback
        self._wci_ping_dirty: bool = False
        self._wci_ping_timer: Optional[QtCore.QTimer] = None
        self._last_wci_ping_time: float = 0.0

        # Throttle track re-rendering
        self._last_track_update_time: float = 0.0
        self._track_update_pending: bool = False

        # Initialize default render settings
        self._init_layer_render_settings()

        # Build pyqtgraph scene items
        self._build_scene()

    # =====================================================================
    # Initialisation
    # =====================================================================

    def _init_layer_render_settings(self) -> None:
        """Initialize default render settings for all layers from builder."""
        if self._builder is None:
            return
        for layer in self._builder.layers:
            if layer.name not in self._layer_render_settings:
                self._layer_render_settings[layer.name] = LayerRenderSettings()

    def _build_scene(self) -> None:
        """Create pyqtgraph plot and items."""
        pg.setConfigOptions(imageAxisOrder='row-major')

        # Create plot for map display
        self._plot = self.graphics.addPlot(row=0, col=0)
        self._plot.setAspectLocked(True)
        vb = self._plot.getViewBox()
        vb.setBackgroundColor("w")
        # Disable autoRange so that adding/updating ImageItems does not
        # cause the ViewBox to progressively expand the view.
        vb.disableAutoRange()
        self._plot.setLabel('bottom', 'Longitude')
        self._plot.setLabel('left', 'Latitude')

        # Coordinate label
        self._coord_label = pg.TextItem("", anchor=(0, 1))
        self._coord_label.setPos(10, 10)
        self._plot.addItem(self._coord_label)

        # Connect mouse move for coordinate display
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_click)
        self._plot.sigRangeChanged.connect(self._on_view_changed)

        # Build scale bar overlay
        self._build_scale_bar()

        # Add measurement entries to the ViewBox right-click context menu
        self._build_measure_context_menu()

    # =====================================================================
    # Colorbar
    # =====================================================================

    def create_colorbar(self) -> None:
        """Create an interactive colorbar for the selected layer."""
        if self._colorbar_item is None:
            self._colorbar_item = pg.ColorBarItem(
                interactive=True,
                orientation='vertical',
                colorMap=pg.colormap.get('viridis'),
                width=15,
            )
            self.graphics.addItem(self._colorbar_item, row=0, col=1)

            if self._layer_images:
                first_image = list(self._layer_images.values())[0]
                self._colorbar_item.setImageItem(first_image)

            if hasattr(self._colorbar_item, 'sigLevelsChanged'):
                self._colorbar_item.sigLevelsChanged.connect(
                    lambda cb=self._colorbar_item: self._on_colorbar_levels_changed(cb)
                )

        self._update_colorbar()

    def _update_colorbar(self) -> None:
        """Update the colorbar for the currently selected layer."""
        if self._colorbar_item is None:
            return

        layer_name = self._active_colorbar_layer
        if layer_name is None:
            self._colorbar_item.hide()
            return

        self._colorbar_item.show()
        if layer_name in self._layer_images:
            self._colorbar_item.setImageItem(self._layer_images[layer_name])

        settings = self._layer_render_settings.get(layer_name, LayerRenderSettings())

        try:
            cmap = pg.colormap.get(settings.colormap, source='matplotlib')
            self._colorbar_item.setColorMap(cmap)
        except Exception as e:
            warnings.warn(f"Could not set colorbar colormap: {e}")

        if layer_name in self._layer_colorbar_levels:
            vmin, vmax = self._layer_colorbar_levels[layer_name]
            self._colorbar_item.setLevels((vmin, vmax))
        else:
            vmin = settings.vmin
            vmax = settings.vmax
            if vmin is None or vmax is None:
                if self._builder is not None:
                    result = self._builder.get_layer_data(layer_name, max_size=(100, 100))
                    if result is not None:
                        data, _ = result
                        if vmin is None:
                            vmin = float(np.nanmin(data))
                        if vmax is None:
                            vmax = float(np.nanmax(data))
            if vmin is not None and vmax is not None:
                self._colorbar_item.setLevels((vmin, vmax))
                self._layer_colorbar_levels[layer_name] = (vmin, vmax)

    def _on_colorbar_levels_changed(self, colorbar) -> None:
        """Handle colorbar level change from user interaction."""
        layer_name = self._active_colorbar_layer
        if layer_name is None:
            return
        vmin, vmax = colorbar.levels()
        self._layer_colorbar_levels[layer_name] = (vmin, vmax)

    def _get_layer_levels(self, layer_name: str, data: np.ndarray) -> Tuple[float, float]:
        """Get rendering levels for a layer."""
        if (layer_name == self._active_colorbar_layer and
                self._colorbar_item is not None):
            try:
                return self._colorbar_item.levels()
            except Exception:
                pass

        if layer_name in self._layer_colorbar_levels:
            return self._layer_colorbar_levels[layer_name]

        settings = self._layer_render_settings.get(layer_name, LayerRenderSettings())
        vmin = settings.vmin if settings.vmin is not None else float(np.nanmin(data))
        vmax = settings.vmax if settings.vmax is not None else float(np.nanmax(data))
        return (vmin, vmax)

    def on_colorbar_layer_change(self, layer_name: Optional[str]) -> None:
        """Handle colorbar layer selection change."""
        old_layer = self._active_colorbar_layer
        if old_layer and self._colorbar_item is not None:
            try:
                vmin, vmax = self._colorbar_item.levels()
                self._layer_colorbar_levels[old_layer] = (vmin, vmax)
            except Exception:
                pass

        self._active_colorbar_layer = layer_name
        self._update_colorbar()

    # =====================================================================
    # Viewer-side rendering settings (API)
    # =====================================================================

    def set_layer_colormap(self, layer_name: str, colormap: str) -> None:
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].colormap = colormap
        if self._builder is not None:
            layer = self._builder.get_layer(layer_name)
            if layer:
                self._render_layer(layer)

    def set_layer_opacity(self, layer_name: str, opacity: float) -> None:
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].opacity = opacity
        if self._builder is not None:
            layer = self._builder.get_layer(layer_name)
            if layer:
                self._render_layer(layer)

    def set_layer_range(self, layer_name: str, vmin: float, vmax: float) -> None:
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].vmin = vmin
        self._layer_render_settings[layer_name].vmax = vmax
        if self._builder is not None:
            layer = self._builder.get_layer(layer_name)
            if layer:
                self._render_layer(layer)

    def set_layer_blend_mode(self, layer_name: str, blend_mode: str) -> None:
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].blend_mode = blend_mode
        if self._builder is not None:
            layer = self._builder.get_layer(layer_name)
            if layer:
                self._render_layer(layer)

    # =====================================================================
    # Tile background control
    # =====================================================================

    @property
    def tile_visible(self) -> bool:
        return self._tile_visible

    @tile_visible.setter
    def tile_visible(self, visible: bool) -> None:
        self._tile_visible = visible
        if self._tile_image is not None:
            self._tile_image.setVisible(visible)
        if visible and self._tile_builder is not None:
            self.render_tiles()
            self._do_request_draw()

    @property
    def tile_builder(self):
        return self._tile_builder

    @tile_builder.setter
    def tile_builder(self, builder) -> None:
        self._tile_builder = builder
        if builder is not None:
            self._tile_visible = True
            self.render_tiles()

    def change_tile_source(self, source_name: str) -> None:
        """Change the tile source by name."""
        if self._tile_builder is None:
            return

        if source_name == 'None':
            self._tile_visible = False
            if self._tile_image is not None:
                self._tile_image.setVisible(False)
            return

        try:
            if source_name not in self._tile_builder.source_names:
                self._tile_builder.add_preset(source_name)
            for name in self._tile_builder.source_names:
                self._tile_builder.set_source_visible(name, name == source_name)
            self._tile_cache_key = None
            self._tile_visible = True
            self.render_tiles()
            self._do_request_draw()
        except Exception as e:
            warnings.warn(f"Failed to change tile source to {source_name}: {e}")

    def list_tile_sources(self) -> List[str]:
        from ..overview.map_builder.tile_builder import TILE_SOURCES
        return list(TILE_SOURCES.keys())

    # =====================================================================
    # Rendering
    # =====================================================================

    def update_view(self) -> None:
        """Update the displayed layers based on current view bounds."""
        if self._current_bounds is None:
            if self._builder is not None:
                self._current_bounds = self._builder.combined_bounds

        if self._current_bounds is None:
            return

        if self._tile_builder is not None and self._tile_visible:
            self.render_tiles()

        if self._builder is not None:
            for layer in self._builder.visible_layers:
                self._render_layer(layer)

        self._update_tracks(force=True)
        self._update_ping_marker()

    def render_tiles(self) -> None:
        """Trigger background tile loading via the adapter callback."""
        if self._tile_builder is None or self._current_bounds is None:
            return

        bounds = self._current_bounds
        if hasattr(bounds, 'xmin'):
            bbox = bounds
        else:
            from ..overview.map_builder.coordinate_system import BoundingBox
            bbox = BoundingBox(
                xmin=bounds[0], ymin=bounds[1],
                xmax=bounds[2], ymax=bounds[3],
            )

        vb = self._plot.getViewBox()
        try:
            screen_rect = vb.screenGeometry()
            pixel_width = screen_rect.width()
            pixel_height = screen_rect.height()
        except Exception:
            pixel_width = 800
            pixel_height = 600

        pixel_width = max(512, min(pixel_width, self._max_render_size[0]))
        pixel_height = max(512, min(pixel_height, self._max_render_size[1]))

        cache_key = (
            round(bbox.xmin, 6), round(bbox.ymin, 6),
            round(bbox.xmax, 6), round(bbox.ymax, 6),
            pixel_width // 32, pixel_height // 32,
        )
        if self._tile_cache_key == cache_key:
            return

        # Delegate actual async loading to the adapter
        if self._trigger_tile_load is not None:
            self._trigger_tile_load(bbox, (pixel_width, pixel_height), cache_key)

    def apply_tile_result(self, result: Dict[str, Any]) -> None:
        """Apply a tile loading result (called from adapter after async load)."""
        tile_image = result['image']
        actual_bounds = result['bounds']
        cache_key = result['cache_key']

        if tile_image is None:
            return

        self._tile_cache_key = cache_key

        self._ignore_range_changes = True
        try:
            if self._tile_image is None:
                self._tile_image = pg.ImageItem(axisOrder="row-major")
                self._plot.addItem(self._tile_image)
                self._tile_image.setZValue(-100)

            flipped = tile_image[::-1]
            self._tile_image.setImage(flipped, autoLevels=False)

            x0 = actual_bounds.xmin
            x1 = actual_bounds.xmax
            y0 = actual_bounds.ymin
            y1 = actual_bounds.ymax

            rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
            self._tile_image.setRect(rect)
            self._tile_image.setVisible(self._tile_visible)

            self._do_request_draw()
        finally:
            self._ignore_range_changes = False
            self._ignore_range_until = time.time() + 0.3

    def _render_layer(self, layer) -> None:
        """Render a single data layer."""
        try:
            result = self._builder.get_layer_data(
                layer.name,
                bounds=self._current_bounds,
                max_size=self._max_render_size,
            )
            if result is None:
                return
            data, cs = result
        except Exception as e:
            warnings.warn(f"Failed to load layer {layer.name}: {e}")
            return

        self._render_layer_from_data(layer, data, cs)

    def _render_layer_from_data(self, layer, data: np.ndarray, cs) -> None:
        """Render a layer from pre-loaded data."""
        self._coordinate_system = cs

        if layer.name not in self._layer_images:
            img = pg.ImageItem(axisOrder="row-major")
            self._plot.addItem(img)
            self._layer_images[layer.name] = img

        img = self._layer_images[layer.name]
        settings = self._layer_render_settings.get(layer.name, LayerRenderSettings())

        data_for_display = data.copy()

        if hasattr(cs, 'transform') and cs.transform.e < 0:
            data_for_display = np.flipud(data_for_display)

        img.setImage(data_for_display, autoLevels=False)

        try:
            cmap = pg.colormap.get(settings.colormap, source='matplotlib')
            if hasattr(img, 'setColorMap'):
                img.setColorMap(cmap)
            else:
                lut = cmap.getLookupTable(256)
                img.setLookupTable(lut)
        except Exception:
            lut = _get_colormap_lut(settings.colormap)
            img.setLookupTable(lut)

        vmin, vmax = self._get_layer_levels(layer.name, data)
        img.setLevels((vmin, vmax))
        img.setOpacity(settings.opacity)

        bounds = cs.bounds
        img.setRect(QtCore.QRectF(
            bounds.xmin, bounds.ymin,
            bounds.width, bounds.height,
        ))
        img.setZValue(layer.z_order)
        img.setVisible(layer.visible)

    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """Set visibility for a data layer."""
        if self._builder is not None:
            self._builder.set_layer_visibility(layer_name, visible)
        if layer_name in self._layer_images:
            self._layer_images[layer_name].setVisible(visible)

    def set_layer_opacity_image(self, layer_name: str, opacity: float) -> None:
        """Set opacity and re-render a layer."""
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].opacity = opacity
        if self._builder is not None:
            layer = self._builder.get_layer(layer_name)
            if layer:
                self._render_layer(layer)

    def set_layer_colormap_image(self, layer_name: str, colormap: str) -> None:
        """Set colormap and re-render a layer."""
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].colormap = colormap
        if layer_name == self._active_colorbar_layer:
            self._update_colorbar()
        if self._builder is not None:
            layer = self._builder.get_layer(layer_name)
            if layer:
                self._render_layer(layer)

    # =====================================================================
    # Add layers after construction
    # =====================================================================

    def add_geotiff(self, path: str, name: Optional[str] = None, band: int = 1, **kwargs) -> None:
        from ..overview.map_builder import MapBuilder

        if self._builder is None:
            self._builder = MapBuilder()

        self._builder.add_geotiff(path, name=name, band=band, **kwargs)
        added_layer = self._builder.layers[-1]
        if added_layer.name not in self._layer_render_settings:
            self._layer_render_settings[added_layer.name] = LayerRenderSettings()

        self._current_bounds = None
        self.update_view()

    def add_layer(self, backend: Any, name: Optional[str] = None,
                  visible: bool = True, z_order: Optional[int] = None) -> None:
        from ..overview.map_builder import MapBuilder

        if self._builder is None:
            self._builder = MapBuilder()

        self._builder.add_layer(backend, name=name, visible=visible, z_order=z_order)
        added_layer = self._builder.layers[-1]
        if added_layer.name not in self._layer_render_settings:
            self._layer_render_settings[added_layer.name] = LayerRenderSettings()

        self._current_bounds = None
        self.update_view()

    @property
    def layer_names(self) -> List[str]:
        """Get names of all data layers."""
        if self._builder is None:
            return []
        return [l.name for l in self._builder.layers]

    # =====================================================================
    # Tracks
    # =====================================================================

    def _update_tracks(self, force: bool = False) -> None:
        """Update track overlays.

        Throttled to at most once every 200 ms unless *force* is True
        to avoid tearing down / recreating all PlotDataItems on every
        ping callback when connected viewers trigger frequent updates.
        """
        now = time.time()
        if not force and now - self._last_track_update_time < 0.2:
            self._track_update_pending = True
            return
        self._last_track_update_time = now
        self._track_update_pending = False

        for plot in self._track_plots:
            self._plot.removeItem(plot)
        self._track_plots.clear()

        self._refresh_overview_tracks()

        for name, track_info in self._tracks.items():
            self._render_track_info(track_info)

        for name, ov_info in self._overview_tracks.items():
            if not ov_info.visible or ov_info.latitudes is None or len(ov_info.latitudes) == 0:
                continue
            self._render_overview_track(ov_info)

    def _render_track_info(self, track_info: TrackInfo) -> None:
        if not track_info.visible:
            return

        x = track_info.longitudes
        y = track_info.latitudes

        darker_color = self._darken_color(track_info.color, 0.5)
        pen_full = pg.mkPen(color=darker_color, width=track_info.line_width * 0.5)
        plot_full = self._plot.plot(x, y, pen=pen_full)
        self._track_plots.append(plot_full)

        visible_range = self._get_slot_visible_ping_range(track_info.slot_idx)

        if visible_range is not None:
            self._render_visible_range(x, y, visible_range, track_info)
        elif track_info.is_active:
            line_width = track_info.line_width * 2
            pen = pg.mkPen(color=track_info.color, width=line_width)
            plot = self._plot.plot(x, y, pen=pen)
            self._track_plots.append(plot)

    def _render_overview_track(self, ov_info: OverviewTrackInfo) -> None:
        x = ov_info.longitudes
        y = ov_info.latitudes

        pen = pg.mkPen(color=ov_info.color, width=ov_info.line_width)
        plot_line = self._plot.plot(x, y, pen=pen)
        self._track_plots.append(plot_line)

        if ov_info.show_points and len(x) > 0:
            pt_pen = pg.mkPen('w', width=1) if ov_info.point_outline else pg.mkPen(None)
            scatter = pg.ScatterPlotItem(
                x, y,
                size=ov_info.point_size,
                brush=pg.mkBrush(ov_info.color),
                pen=pt_pen,
                symbol=ov_info.point_symbol,
            )
            self._plot.addItem(scatter)
            self._track_plots.append(scatter)

        visible_range = self._get_slot_visible_ping_range(ov_info.slot_idx)
        if visible_range is not None and ov_info.indices is not None:
            start_ping, end_ping = visible_range
            mask = (ov_info.indices >= start_ping) & (ov_info.indices <= end_ping)
            if np.any(mask):
                x_vis = x[mask]
                y_vis = y[mask]
                pen_vis = pg.mkPen(color=ov_info.color, width=ov_info.line_width * 2)
                plot_vis = self._plot.plot(x_vis, y_vis, pen=pen_vis)
                self._track_plots.append(plot_vis)
                self._add_track_markers(x_vis, y_vis, ov_info.color)
        elif ov_info.is_active:
            pen_active = pg.mkPen(color=ov_info.color, width=ov_info.line_width * 2)
            plot_active = self._plot.plot(x, y, pen=pen_active)
            self._track_plots.append(plot_active)

    def _render_visible_range(self, x, y, visible_range, track_info: TrackInfo) -> None:
        start_idx, end_idx = visible_range
        start_idx = max(0, int(start_idx))
        end_idx = min(len(x), int(end_idx) + 1)

        if start_idx >= end_idx:
            return

        x_visible = x[start_idx:end_idx]
        y_visible = y[start_idx:end_idx]

        line_width = track_info.line_width * 2
        pen = pg.mkPen(color=track_info.color, width=line_width)
        plot_visible = self._plot.plot(x_visible, y_visible, pen=pen)
        self._track_plots.append(plot_visible)

        self._add_track_markers(x_visible, y_visible, track_info.color)

    def _add_track_markers(self, x, y, color) -> None:
        if len(x) == 0:
            return

        n_points = len(x)
        marker_interval = max(1, n_points // 10)
        marker_indices = list(range(0, n_points, marker_interval))
        if 0 not in marker_indices:
            marker_indices.insert(0, 0)
        if n_points - 1 not in marker_indices:
            marker_indices.append(n_points - 1)

        marker_x = [x[i] for i in marker_indices]
        marker_y = [y[i] for i in marker_indices]

        markers = pg.ScatterPlotItem(
            marker_x, marker_y,
            size=8, brush=pg.mkBrush(color),
            pen=pg.mkPen('w', width=1.5), symbol='o'
        )
        self._plot.addItem(markers)
        self._track_plots.append(markers)

        marker_start = pg.ScatterPlotItem(
            [x[0]], [y[0]],
            size=14, brush=pg.mkBrush(color),
            pen=pg.mkPen('w', width=2), symbol='t'
        )
        self._plot.addItem(marker_start)
        self._track_plots.append(marker_start)

        marker_end = pg.ScatterPlotItem(
            [x[-1]], [y[-1]],
            size=12, brush=pg.mkBrush(color),
            pen=pg.mkPen('w', width=2), symbol='s'
        )
        self._plot.addItem(marker_end)
        self._track_plots.append(marker_end)

    def _get_slot_visible_ping_range(self, slot_idx: Optional[int]) -> Optional[Tuple[int, int]]:
        if slot_idx is None or self._echogram_viewer is None:
            return None
        if not hasattr(self._echogram_viewer, 'slots'):
            return None
        slots = self._echogram_viewer.slots
        if slot_idx >= len(slots):
            return None
        slot = slots[slot_idx]
        if not slot.is_visible or slot.plot_item is None:
            return None
        try:
            vb = slot.plot_item.getViewBox()
            view_range = vb.viewRange()
            x_range = view_range[0]
            start_ping = int(np.floor(x_range[0]))
            end_ping = int(np.ceil(x_range[1]))
            return (start_ping, end_ping)
        except Exception:
            return None

    def _darken_color(self, color: str, factor: float = 0.5) -> str:
        if color.startswith('#'):
            color = color[1:]
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    def _refresh_overview_tracks(self) -> None:
        if not self._overview_tracks:
            return

        bounds = self._current_bounds
        if bounds is not None:
            min_lat, max_lat = bounds.ymin, bounds.ymax
            min_lon, max_lon = bounds.xmin, bounds.xmax
        else:
            min_lat = max_lat = min_lon = max_lon = None

        for info in self._overview_tracks.values():
            if not info.visible:
                continue
            try:
                lats, lons, idx = info.overview.get_track_data(
                    min_lat=min_lat, max_lat=max_lat,
                    min_lon=min_lon, max_lon=max_lon,
                    max_points=info.max_points,
                )
                info.latitudes = lats
                info.longitudes = lons
                info.indices = idx
            except Exception as e:
                warnings.warn(f"Failed to refresh overview track '{info.name}': {e}")

    # =====================================================================
    # Track management API
    # =====================================================================

    def add_track(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        name: str = "Track",
        color: Optional[str] = None,
        line_width: float = 2.0,
        is_active: bool = False,
        slot_idx: Optional[int] = None,
    ) -> None:
        if color is None:
            color = self.TRACK_COLORS[len(self._tracks) % len(self.TRACK_COLORS)]

        self._tracks[name] = TrackInfo(
            name=name,
            latitudes=np.asarray(latitudes),
            longitudes=np.asarray(longitudes),
            color=color,
            line_width=line_width,
            is_active=is_active,
            slot_idx=slot_idx,
        )

        if is_active:
            self._active_track_name = name

        self._update_tracks()
        if self._update_track_legend:
            self._update_track_legend()

    def add_overview_track(
        self,
        overview,
        name: str = "Overview",
        color: Optional[str] = None,
        line_width: float = 2.0,
        max_points: int = 50_000,
        is_active: bool = False,
        slot_idx: Optional[int] = None,
        show_points: bool = False,
        point_size: float = 5.0,
        point_symbol: str = 'o',
        point_outline: bool = True,
    ) -> None:
        n_all = len(self._tracks) + len(self._overview_tracks)
        if color is None:
            color = self.TRACK_COLORS[n_all % len(self.TRACK_COLORS)]

        info = OverviewTrackInfo(
            name=name, overview=overview, color=color,
            max_points=max_points, line_width=line_width,
            is_active=is_active, slot_idx=slot_idx,
            show_points=show_points, point_size=point_size,
            point_symbol=point_symbol, point_outline=point_outline,
        )
        self._overview_tracks[name] = info

        if is_active:
            self._active_track_name = name

        self._refresh_overview_tracks()
        self._update_tracks()
        if self._update_track_legend:
            self._update_track_legend()

    def set_active_track(self, name: str) -> None:
        for track_name, track_info in self._tracks.items():
            track_info.is_active = (track_name == name)
        for track_name, track_info in self._overview_tracks.items():
            track_info.is_active = (track_name == name)
        self._active_track_name = name
        self._update_tracks()

    def clear_tracks(self) -> None:
        self._tracks.clear()
        self._overview_tracks.clear()
        for plot in self._track_plots:
            self._plot.removeItem(plot)
        self._track_plots.clear()
        if self._update_track_legend:
            self._update_track_legend()

    def set_track_color(self, name: str, color: str) -> None:
        if name in self._tracks:
            self._tracks[name].color = color
        if name in self._overview_tracks:
            self._overview_tracks[name].color = color
        self._update_tracks()
        if self._update_track_legend:
            self._update_track_legend()

    def set_track_visibility(self, track_name: str, visible: bool) -> None:
        if track_name in self._tracks:
            self._tracks[track_name].visible = visible
        if track_name in self._overview_tracks:
            self._overview_tracks[track_name].visible = visible
        self._update_tracks()

    # =====================================================================
    # User marker overlays
    # =====================================================================

    def add_markers(
        self,
        latitudes, longitudes,
        name: str = "markers",
        labels: Optional[List[str]] = None,
        color: str = "white",
        edge_color: str = "black",
        size: float = 10,
        symbol: str = "o",
        edge_width: float = 1.5,
        z_value: float = 100,
        label_color: str = "black",
        label_size: str = "9pt",
    ) -> pg.ScatterPlotItem:
        self.remove_markers(name)

        lons = np.asarray(longitudes)
        lats = np.asarray(latitudes)

        scatter = pg.ScatterPlotItem(
            lons, lats,
            size=size,
            brush=pg.mkBrush(color),
            pen=pg.mkPen(edge_color, width=edge_width),
            symbol=symbol,
        )
        scatter.setZValue(z_value)
        self._plot.addItem(scatter)
        self._user_markers[name] = scatter

        if labels is not None:
            text_items = []
            for lon, lat, lbl in zip(lons, lats, labels):
                txt = pg.TextItem(
                    lbl, color=label_color,
                    fill=pg.mkBrush(255, 255, 255, 200),
                    border=pg.mkPen(0, 0, 0, 150),
                )
                txt.setFont(pg.QtGui.QFont(
                    "sans-serif",
                    int(label_size.replace("pt", "")),
                ))
                txt.setPos(lon, lat)
                txt.setZValue(z_value + 1)
                self._plot.addItem(txt)
                text_items.append(txt)
            self._user_markers[f"{name}__labels"] = text_items

        return scatter

    def add_markers_tuples(self, positions, name: str = "markers", **kwargs) -> pg.ScatterPlotItem:
        pts = list(positions)
        lats = np.array([p[0] for p in pts])
        lons = np.array([p[1] for p in pts])
        return self.add_markers(lats, lons, name=name, **kwargs)

    def remove_markers(self, name: str) -> None:
        if name in self._user_markers:
            item = self._user_markers.pop(name)
            if isinstance(item, list):
                for sub in item:
                    self._plot.removeItem(sub)
            else:
                self._plot.removeItem(item)
        label_key = f"{name}__labels"
        if label_key in self._user_markers:
            for txt in self._user_markers.pop(label_key):
                self._plot.removeItem(txt)

    def clear_markers(self) -> None:
        for item in self._user_markers.values():
            if isinstance(item, list):
                for sub in item:
                    self._plot.removeItem(sub)
            else:
                self._plot.removeItem(item)
        self._user_markers.clear()

    # =====================================================================
    # Ping position
    # =====================================================================

    def _update_ping_marker(self) -> None:
        if self._current_ping_latlon is None:
            if self._ping_marker is not None:
                self._ping_marker.hide()
            return

        lat, lon = self._current_ping_latlon
        x, y = lon, lat

        if self._ping_marker is None:
            self._ping_marker = pg.ScatterPlotItem(
                [x], [y], size=20,
                brush=pg.mkBrush('#FF00FF'),
                pen=pg.mkPen('#000000', width=2),
                symbol='o',
            )
            self._plot.addItem(self._ping_marker)
        else:
            self._ping_marker.setData([x], [y])
            self._ping_marker.show()

    def update_ping_position(self, lat: float, lon: float) -> None:
        """Update the current ping position marker."""
        self._current_ping_latlon = (lat, lon)

        now = time.time()
        if now - self._last_ping_update_time < 0.05:
            return
        self._last_ping_update_time = now

        self._update_ping_marker()

        if self._panel and "auto_center_wci" in self._panel:
            if self._panel["auto_center_wci"].value:
                self.pan_to_position_if_near_edge(lat, lon, edge_fraction=0.2)

    # =====================================================================
    # Navigation
    # =====================================================================

    def zoom_to_fit(self) -> None:
        bounds = None
        if self._builder is not None:
            bounds = self._builder.combined_bounds

        if bounds is None and (self._tracks or self._overview_tracks):
            self.zoom_to_track()
            return

        if bounds:
            self._ignore_range_changes = True
            rect = QtCore.QRectF(
                bounds.xmin, bounds.ymin,
                bounds.width, bounds.height,
            )
            self._plot.getViewBox().setRange(rect, padding=0.05)
            self._ignore_range_changes = False
            # Grace period: aspect-locked ViewBox may emit deferred
            # sigRangeChanged events after the range is set.
            self._ignore_range_until = time.time() + 0.5
            self._current_bounds = bounds
            self.update_view()

    def zoom_to_track(self) -> None:
        if not self._tracks and not self._overview_tracks:
            return

        from ..overview.map_builder.coordinate_system import BoundingBox

        bboxes = []
        for track_info in self._tracks.values():
            if len(track_info.latitudes) == 0:
                continue
            bboxes.append(BoundingBox(
                xmin=float(np.nanmin(track_info.longitudes)),
                ymin=float(np.nanmin(track_info.latitudes)),
                xmax=float(np.nanmax(track_info.longitudes)),
                ymax=float(np.nanmax(track_info.latitudes)),
            ))
        for ov_info in self._overview_tracks.values():
            try:
                ov_info.overview._ensure_latlon_arrays()
                la = ov_info.overview._lat_arr
                lo = ov_info.overview._lon_arr
                if len(la) == 0:
                    continue
                bboxes.append(BoundingBox(
                    xmin=float(np.nanmin(lo)),
                    ymin=float(np.nanmin(la)),
                    xmax=float(np.nanmax(lo)),
                    ymax=float(np.nanmax(la)),
                ))
            except Exception:
                pass

        if not bboxes:
            return

        bounds = bboxes[0]
        for bb in bboxes[1:]:
            bounds = bounds.union(bb)
        bounds = bounds.expand(1.1)

        self._ignore_range_changes = True
        rect = QtCore.QRectF(
            bounds.xmin, bounds.ymin,
            bounds.width, bounds.height,
        )
        self._plot.getViewBox().setRange(rect, padding=0.02)
        self._ignore_range_changes = False
        # Grace period: aspect-locked ViewBox may emit deferred
        # sigRangeChanged events after the range is set.
        self._ignore_range_until = time.time() + 0.5
        self._current_bounds = bounds
        if self._schedule_update:
            self._schedule_update()

    def zoom_to_position(self, lat: float, lon: float, radius_deg: float = 0.01) -> None:
        from ..overview.map_builder.coordinate_system import BoundingBox

        bounds = BoundingBox(
            xmin=lon - radius_deg, ymin=lat - radius_deg,
            xmax=lon + radius_deg, ymax=lat + radius_deg,
        )

        rect = QtCore.QRectF(
            bounds.xmin, bounds.ymin, bounds.width, bounds.height
        )
        self._ignore_range_changes = True
        self._plot.getViewBox().setRange(rect, padding=0)
        self._ignore_range_changes = False
        self._ignore_range_until = time.time() + 0.15
        self._current_bounds = bounds
        if self._schedule_update:
            self._schedule_update()

    def pan_to_wci_position(self) -> None:
        if self._current_ping_latlon is None:
            warnings.warn("No WCI position available")
            return
        lat, lon = self._current_ping_latlon
        self.pan_to_position(lat, lon)

    def pan_to_position(self, lat: float, lon: float) -> None:
        view_range = self._plot.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]
        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]

        # Use a single setRange call so aspect-locked ViewBox doesn't
        # expand one axis between two separate setXRange / setYRange
        # calls, which was causing progressive zoom-out.
        rect = QtCore.QRectF(
            lon - width / 2, lat - height / 2, width, height
        )
        self._ignore_range_changes = True
        self._plot.getViewBox().setRange(rect, padding=0)
        self._ignore_range_changes = False
        # Grace period: Qt may deliver sigRangeChanged asynchronously
        # after _ignore_range_changes is already False.
        self._ignore_range_until = time.time() + 0.15
        # Don't call update_view() synchronously — existing items pan with
        # the viewbox automatically.  Let the adapter's debounce timer
        # handle the expensive tile/layer refresh.
        if self._schedule_update:
            self._schedule_update()

    def is_position_near_edge(self, lat: float, lon: float, edge_fraction: float = 0.2) -> bool:
        view_range = self._plot.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]

        x_margin = (x_range[1] - x_range[0]) * edge_fraction
        y_margin = (y_range[1] - y_range[0]) * edge_fraction

        inner_xmin = x_range[0] + x_margin
        inner_xmax = x_range[1] - x_margin
        inner_ymin = y_range[0] + y_margin
        inner_ymax = y_range[1] - y_margin

        return not (inner_xmin <= lon <= inner_xmax and inner_ymin <= lat <= inner_ymax)

    def pan_to_position_if_near_edge(self, lat: float, lon: float, edge_fraction: float = 0.2) -> None:
        if self.is_position_near_edge(lat, lon, edge_fraction):
            self.pan_to_position(lat, lon)

    # =====================================================================
    # Mouse interaction
    # =====================================================================

    def _on_mouse_move(self, pos) -> None:
        try:
            mouse_point = self._plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            lon, lat = x, y

            coord_text = f"Lat: {lat:.6f}\u00b0, Lon: {lon:.6f}\u00b0"
            self._coord_label.setText(coord_text)

            if self._panel and "lbl_coords" in self._panel:
                self._panel["lbl_coords"].value = coord_text
        except Exception:
            pass

    def _on_mouse_click(self, ev) -> None:
        """Handle mouse clicks — left-click places measurement points."""
        try:
            btn = ev.button()
        except Exception:
            return

        if btn == QtCore.Qt.MouseButton.LeftButton and self._measure_active:
            pos = ev.scenePos()
            mouse_point = self._plot.vb.mapSceneToView(pos)
            lon, lat = mouse_point.x(), mouse_point.y()
            self._measure_points.append((lon, lat))
            self._update_measurement_overlay()
            ev.accept()
            return

        if btn == QtCore.Qt.MouseButton.LeftButton:
            try:
                pos = ev.scenePos()
                mouse_point = self._plot.vb.mapSceneToView(pos)
                lon, lat = mouse_point.x(), mouse_point.y()
                for cb in self._click_callbacks:
                    cb(lat, lon)
            except Exception:
                pass

    # =====================================================================
    # Scale bar
    # =====================================================================

    def _build_scale_bar(self) -> None:
        """Create the scale bar graphics items."""
        self._scale_bar_line = pg.PlotCurveItem(
            pen=pg.mkPen(color="k", width=3),
        )
        self._scale_bar_line.setZValue(1000)
        self._plot.addItem(self._scale_bar_line)

        self._scale_bar_text = pg.TextItem(
            "", anchor=(0.5, 1), color="k",
        )
        self._scale_bar_text.setZValue(1001)
        self._plot.addItem(self._scale_bar_text)

        # End ticks stored as separate items so they render cleanly
        self._scale_bar_tick_left = pg.PlotCurveItem(
            pen=pg.mkPen(color="k", width=2),
        )
        self._scale_bar_tick_left.setZValue(1000)
        self._plot.addItem(self._scale_bar_tick_left)
        self._scale_bar_tick_right = pg.PlotCurveItem(
            pen=pg.mkPen(color="k", width=2),
        )
        self._scale_bar_tick_right.setZValue(1000)
        self._plot.addItem(self._scale_bar_tick_right)

        self._set_scale_bar_items_visible(self._scale_bar_visible)

    def _set_scale_bar_items_visible(self, visible: bool) -> None:
        for item in (self._scale_bar_line, self._scale_bar_text,
                     self._scale_bar_tick_left, self._scale_bar_tick_right):
            if item is not None:
                item.setVisible(visible)

    def set_scale_bar_visible(self, visible: bool) -> None:
        self._scale_bar_visible = visible
        self._set_scale_bar_items_visible(visible)
        if visible:
            self._update_scale_bar()

    def _update_scale_bar(self) -> None:
        """Recompute and reposition the scale bar for the current view."""
        if not self._scale_bar_visible or self._scale_bar_line is None:
            return

        vb = self._plot.getViewBox()
        view_range = vb.viewRange()
        lon_min, lon_max = view_range[0]
        lat_min, lat_max = view_range[1]
        if lon_max <= lon_min or lat_max <= lat_min:
            return

        center_lat = (lat_min + lat_max) / 2.0
        view_width_m = _haversine_distance(
            center_lat, lon_min, center_lat, lon_max,
        )

        # Pick a nice bar length ≈ 15–25 % of the view width
        target_m = view_width_m * 0.2
        bar_m = _NICE_DISTANCES[0]
        for d in _NICE_DISTANCES:
            if d <= target_m:
                bar_m = d
            else:
                break

        # Convert bar_m back to degrees of longitude at center_lat
        cos_lat = math.cos(math.radians(center_lat))
        if cos_lat < 1e-8:
            return
        bar_deg = bar_m / (111_320.0 * cos_lat)

        # Position: bottom-left corner with some margin
        margin_x = (lon_max - lon_min) * 0.05
        margin_y = (lat_max - lat_min) * 0.06
        x0 = lon_min + margin_x
        x1 = x0 + bar_deg
        y0 = lat_min + margin_y

        # Tick height in data coords (small fraction of view)
        tick_h = (lat_max - lat_min) * 0.012

        self._scale_bar_line.setData([x0, x1], [y0, y0])
        self._scale_bar_tick_left.setData([x0, x0], [y0 - tick_h, y0 + tick_h])
        self._scale_bar_tick_right.setData([x1, x1], [y0 - tick_h, y0 + tick_h])

        label = _format_distance(bar_m, "m")
        self._scale_bar_text.setText(label)
        self._scale_bar_text.setPos((x0 + x1) / 2, y0 + tick_h * 1.5)

    # =====================================================================
    # Context menu (right-click) – measurement entries
    # =====================================================================

    def _build_measure_context_menu(self) -> None:
        """Add measurement tool actions to the ViewBox right-click menu."""
        try:
            from pyqtgraph.Qt import QtWidgets as _QtW
            # QAction lives in QtGui (Qt6) or QtWidgets (Qt5)
            try:
                from pyqtgraph.Qt.QtGui import QAction, QActionGroup
            except ImportError:
                from pyqtgraph.Qt.QtWidgets import QAction, QActionGroup
        except Exception:
            return

        vb = self._plot.getViewBox()
        menu = vb.menu
        if menu is None:
            return

        menu.addSeparator()

        # ── Toggle measure ──
        self._ctx_measure_action = QAction("Distance measurement", menu)
        self._ctx_measure_action.setCheckable(True)
        self._ctx_measure_action.setChecked(self._measure_active)
        self._ctx_measure_action.toggled.connect(self.set_measure_active)
        menu.addAction(self._ctx_measure_action)

        # ── Unit sub-menu ──
        unit_menu = menu.addMenu("Measure unit")
        unit_group = QActionGroup(unit_menu)
        unit_group.setExclusive(True)
        for label, key in [("Metres", "m"), ("Kilometres", "km"),
                           ("Nautical miles", "nm")]:
            act = QAction(label, unit_menu)
            act.setCheckable(True)
            act.setChecked(key == self._measure_unit)
            act.triggered.connect(
                lambda _checked, k=key: self.set_measure_unit(k))
            unit_group.addAction(act)
            unit_menu.addAction(act)
        self._ctx_unit_actions = {a.text(): a for a in unit_group.actions()}

        # ── Undo / Clear ──
        undo_action = QAction("Undo last point", menu)
        undo_action.triggered.connect(lambda: self._measure_undo())
        menu.addAction(undo_action)

        clear_action = QAction("Clear measurement", menu)
        clear_action.triggered.connect(lambda: self.clear_measurement())
        menu.addAction(clear_action)

    # =====================================================================
    # Measurement tool
    # =====================================================================

    def set_measure_active(self, active: bool) -> None:
        """Enable or disable the measurement tool (left-click to place)."""
        self._measure_active = active
        # Sync the context-menu checkable action
        if hasattr(self, '_ctx_measure_action') and self._ctx_measure_action is not None:
            if self._ctx_measure_action.isChecked() != active:
                self._ctx_measure_action.setChecked(active)
        # Sync the control-panel checkbox
        if self._panel and "measure_tool" in self._panel:
            if self._panel["measure_tool"].value != active:
                self._panel["measure_tool"].value = active
        if not active:
            self.clear_measurement()

    def set_measure_unit(self, unit: str) -> None:
        """Set the measurement display unit ('m', 'km', or 'nm')."""
        if unit not in ("m", "km", "nm"):
            return
        self._measure_unit = unit
        # Sync context-menu radio buttons
        _label_map = {"m": "Metres", "km": "Kilometres", "nm": "Nautical miles"}
        if hasattr(self, '_ctx_unit_actions'):
            lbl = _label_map.get(unit)
            if lbl and lbl in self._ctx_unit_actions:
                act = self._ctx_unit_actions[lbl]
                if not act.isChecked():
                    act.setChecked(True)
        # Sync control-panel dropdown
        if self._panel and "measure_unit" in self._panel:
            if self._panel["measure_unit"].value != unit:
                self._panel["measure_unit"].value = unit
        if self._measure_points:
            self._update_measurement_overlay()

    def clear_measurement(self) -> None:
        """Remove all measurement points and overlays."""
        self._measure_points.clear()
        self._remove_measurement_overlay()

    def _measure_undo(self) -> None:
        """Remove the last measurement point."""
        if self._measure_points:
            self._measure_points.pop()
            if self._measure_points:
                self._update_measurement_overlay()
            else:
                self._remove_measurement_overlay()

    def _remove_measurement_overlay(self) -> None:
        """Remove all measurement graphics items from the plot."""
        for item in self._measure_plots:
            self._plot.removeItem(item)
        self._measure_plots.clear()
        if self._measure_markers is not None:
            self._plot.removeItem(self._measure_markers)
            self._measure_markers = None
        for lbl in self._measure_labels:
            self._plot.removeItem(lbl)
        self._measure_labels.clear()
        if self._measure_total_label is not None:
            self._plot.removeItem(self._measure_total_label)
            self._measure_total_label = None
        if self._panel and "lbl_measure" in self._panel:
            self._panel["lbl_measure"].value = ""

    def _update_measurement_overlay(self) -> None:
        """Redraw measurement lines, markers, and distance labels."""
        self._remove_measurement_overlay()
        pts = self._measure_points
        if not pts:
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Markers at each point
        self._measure_markers = pg.ScatterPlotItem(
            x=xs, y=ys, size=10,
            pen=pg.mkPen("r", width=2),
            brush=pg.mkBrush(255, 0, 0, 120),
            symbol="o",
        )
        self._measure_markers.setZValue(2000)
        self._plot.addItem(self._measure_markers)

        total_m = 0.0
        for i in range(1, len(pts)):
            lon1, lat1 = pts[i - 1]
            lon2, lat2 = pts[i]
            seg_m = _haversine_distance(lat1, lon1, lat2, lon2)
            total_m += seg_m

            # Segment line
            line = pg.PlotCurveItem(
                [lon1, lon2], [lat1, lat2],
                pen=pg.mkPen(color="r", width=2, style=QtCore.Qt.PenStyle.DashLine),
            )
            line.setZValue(1999)
            self._plot.addItem(line)
            self._measure_plots.append(line)

            # Segment label at midpoint
            mx = (lon1 + lon2) / 2
            my = (lat1 + lat2) / 2
            lbl = pg.TextItem(
                _format_distance(seg_m, self._measure_unit),
                anchor=(0.5, 1), color="r",
            )
            lbl.setZValue(2001)
            lbl.setPos(mx, my)
            self._plot.addItem(lbl)
            self._measure_labels.append(lbl)

        # Total distance label at last point
        if len(pts) > 1:
            total_text = f"Total: {_format_distance(total_m, self._measure_unit)}"
            self._measure_total_label = pg.TextItem(
                total_text, anchor=(0, 0), color=(180, 0, 0),
            )
            self._measure_total_label.setZValue(2001)
            self._measure_total_label.setPos(xs[-1], ys[-1])
            self._plot.addItem(self._measure_total_label)

        # Update info panel
        if self._panel and "lbl_measure" in self._panel:
            if len(pts) == 1:
                self._panel["lbl_measure"].value = "Click to add next point"
            else:
                self._panel["lbl_measure"].value = (
                    f"Total: {_format_distance(total_m, self._measure_unit)} "
                    f"({len(pts)} points)"
                )

    def _on_view_changed(self) -> None:
        if self._ignore_range_changes:
            return
        # Ignore deferred signals that arrive after a programmatic pan/zoom
        if time.time() < self._ignore_range_until:
            return

        vb = self._plot.vb
        view_range = vb.viewRange()

        from ..overview.map_builder.coordinate_system import BoundingBox

        self._current_bounds = BoundingBox(
            xmin=view_range[0][0], xmax=view_range[0][1],
            ymin=view_range[1][0], ymax=view_range[1][1],
        )

        self._update_scale_bar()

        for callback in self._view_change_callbacks:
            try:
                callback(self._current_bounds)
            except Exception as e:
                warnings.warn(f"View change callback error: {e}")

    # =====================================================================
    # Viewer integration
    # =====================================================================

    def connect_echogram_viewer(self, echogram_viewer) -> None:
        self._echogram_viewer = echogram_viewer
        self._load_tracks_from_echogram_viewer()

    def connect_wci_viewer(self, wci_viewer) -> None:
        self._wci_viewer = wci_viewer
        self._load_tracks_from_wci_viewer()
        if hasattr(wci_viewer, 'register_ping_change_callback'):
            wci_viewer.register_ping_change_callback(self._on_wci_ping_change)
        self._on_wci_ping_change()

    def _on_wci_ping_change(self) -> None:
        if self._wci_viewer is None:
            return

        now = time.time()
        elapsed = now - self._last_wci_ping_time
        if elapsed < 0.1:
            # Throttled – mark dirty and ensure a trailing timer fires
            self._wci_ping_dirty = True
            if self._wci_ping_timer is None:
                self._wci_ping_timer = QtCore.QTimer()
                self._wci_ping_timer.setSingleShot(True)
                self._wci_ping_timer.timeout.connect(self._flush_wci_ping_change)
            if not self._wci_ping_timer.isActive():
                remaining_ms = int((0.1 - elapsed) * 1000) + 1
                self._wci_ping_timer.start(remaining_ms)
            return

        self._wci_ping_dirty = False
        self._last_wci_ping_time = now
        self._do_wci_ping_update()

    def _flush_wci_ping_change(self) -> None:
        """Trailing-edge callback: render the last skipped ping position."""
        if not self._wci_ping_dirty:
            return
        self._wci_ping_dirty = False
        self._last_wci_ping_time = time.time()
        self._do_wci_ping_update()

    def _do_wci_ping_update(self) -> None:
        """Resolve geolocation from the WCI viewer and update the marker."""
        if self._wci_viewer is None:
            return
        slots = getattr(self._wci_viewer, 'slots', [])
        if not slots:
            return
        for slot in slots:
            if slot.is_visible and slot.channel_key is not None:
                ping = slot.get_ping()
                if ping is not None:
                    try:
                        if hasattr(ping, 'get_geolocation'):
                            geo = ping.get_geolocation()
                            if hasattr(geo, 'latitude') and hasattr(geo, 'longitude'):
                                self.update_ping_position(geo.latitude, geo.longitude)
                                return
                    except Exception:
                        pass

    @staticmethod
    def _ev_attr(viewer, name):
        """Resolve an attribute on the echogram viewer or its .core."""
        if hasattr(viewer, name):
            return getattr(viewer, name)
        core = getattr(viewer, 'core', None)
        if core is not None and hasattr(core, name):
            return getattr(core, name)
        raise AttributeError(f"Echogram viewer has no attribute '{name}'")

    def _load_tracks_from_echogram_viewer(self) -> None:
        if self._echogram_viewer is None:
            return
        self.clear_tracks()
        if not hasattr(self._echogram_viewer, 'slots'):
            return

        echograms = self._ev_attr(self._echogram_viewer, 'echograms')
        grid_rows = self._ev_attr(self._echogram_viewer, 'grid_rows')
        grid_cols = self._ev_attr(self._echogram_viewer, 'grid_cols')
        n_visible = grid_rows * grid_cols

        for slot_idx, slot in enumerate(self._echogram_viewer.slots[:n_visible]):
            if not slot.is_visible or slot.echogram_key is None:
                continue
            echogram = echograms.get(slot.echogram_key)
            if echogram is None:
                continue
            if hasattr(echogram, 'get_track') and hasattr(echogram, 'has_track') and echogram.has_track:
                track_data = echogram.get_track()
                if track_data is not None:
                    lats, lons = track_data
                    color = self.TRACK_COLORS[slot_idx % len(self.TRACK_COLORS)]
                    is_active = (slot_idx == 0)
                    self.add_track(
                        latitudes=lats, longitudes=lons,
                        name=str(slot.echogram_key), color=color,
                        is_active=is_active, slot_idx=slot_idx,
                    )
        self._update_tracks()

    def _load_tracks_from_wci_viewer(self) -> None:
        if self._wci_viewer is None:
            return
        channels = getattr(self._wci_viewer, 'channels', {})
        for i, (name, pings) in enumerate(channels.items()):
            try:
                if pings and len(pings) > 0:
                    lats = []
                    lons = []
                    for ping in pings:
                        if hasattr(ping, 'get_geolocation'):
                            geo = ping.get_geolocation()
                            if hasattr(geo, 'latitude') and hasattr(geo, 'longitude'):
                                lats.append(geo.latitude)
                                lons.append(geo.longitude)
                    if lats:
                        color = self.TRACK_COLORS[i % len(self.TRACK_COLORS)]
                        self.add_track(
                            latitudes=np.array(lats), longitudes=np.array(lons),
                            name=str(name), color=color,
                        )
            except Exception as e:
                warnings.warn(f"Failed to extract track from WCI channel {name}: {e}")
        self._update_tracks()

    def refresh_tracks(self) -> None:
        visibility_state = {name: track.visible for name, track in self._tracks.items()}
        if self._echogram_viewer is not None:
            self._load_tracks_from_echogram_viewer()
        if self._wci_viewer is not None:
            self._load_tracks_from_wci_viewer()
        for name, was_visible in visibility_state.items():
            if name in self._tracks:
                self._tracks[name].visible = was_visible
        self._update_tracks()

    # =====================================================================
    # Callbacks
    # =====================================================================

    def register_click_callback(self, callback: Callable[[float, float], None]) -> None:
        self._click_callbacks.append(callback)

    def register_view_change_callback(self, callback: Callable) -> None:
        self._view_change_callbacks.append(callback)

    # =====================================================================
    # High-res update helpers (called from adapters)
    # =====================================================================

    def build_high_res_sync(self, cancel_flag) -> Optional[Dict[str, Any]]:
        """Load all visible layer data synchronously (for background thread)."""
        if self._builder is None:
            return None
        if self._current_bounds is None:
            return None

        results = {}
        for layer in self._builder.visible_layers:
            if cancel_flag.is_set():
                return None
            try:
                result = self._builder.get_layer_data(
                    layer.name,
                    bounds=self._current_bounds,
                    max_size=self._max_render_size,
                )
                if result is not None:
                    results[layer.name] = {
                        'data': result[0],
                        'cs': result[1],
                        'layer': layer,
                    }
            except Exception as e:
                warnings.warn(f"Failed to load layer {layer.name}: {e}")
        return results

    def apply_high_res_results(self, results: Dict[str, Any]) -> None:
        """Apply preloaded layer data from background thread."""
        self._ignore_range_changes = True
        try:
            for layer_name, layer_data in results.items():
                self._render_layer_from_data(
                    layer_data['layer'],
                    layer_data['data'],
                    layer_data['cs'],
                )
            self._update_tracks(force=True)
            self._update_ping_marker()
            self._do_request_draw()
        finally:
            self._ignore_range_changes = False
            # Grace period so deferred sigRangeChanged from setRect /
            # aspect-locked ViewBox adjustments don't retrigger the
            # high-res update cycle.
            self._ignore_range_until = time.time() + 0.3

    # =====================================================================
    # Wire observers
    # =====================================================================

    def wire_observers(self) -> None:
        """Wire control panel observers to core actions."""
        if self._panel is None:
            return

        p = self._panel

        if "btn_zoom_fit" in p:
            p["btn_zoom_fit"].on_click(lambda _=None: self.zoom_to_fit())
        if "btn_zoom_track" in p:
            p["btn_zoom_track"].on_click(lambda _=None: self.zoom_to_track())
        if "btn_zoom_wci" in p:
            p["btn_zoom_wci"].on_click(lambda _=None: self.pan_to_wci_position())
        if "btn_refresh_tracks" in p:
            p["btn_refresh_tracks"].on_click(lambda _=None: self.refresh_tracks())
        if "auto_update" in p:
            p["auto_update"].on_change(self._on_auto_update_toggle)
        if "colorbar_layer" in p:
            p["colorbar_layer"].on_change(self._on_colorbar_layer_panel_change)
        if "scale_bar" in p:
            p["scale_bar"].on_change(lambda v: self.set_scale_bar_visible(v))
        if "measure_tool" in p:
            p["measure_tool"].on_change(lambda v: self.set_measure_active(v))
        if "measure_unit" in p:
            p["measure_unit"].on_change(lambda v: self.set_measure_unit(v))
        if "btn_measure_clear" in p:
            p["btn_measure_clear"].on_click(lambda _=None: self.clear_measurement())
        if "btn_measure_undo" in p:
            p["btn_measure_undo"].on_click(lambda _=None: self._measure_undo())

    def _on_auto_update_toggle(self, value) -> None:
        # Adapter will handle actual enable/disable
        pass

    def _on_colorbar_layer_panel_change(self, value) -> None:
        self.on_colorbar_layer_change(value)

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _do_request_draw(self) -> None:
        """Request a draw from the adapter."""
        if self._request_draw:
            self._request_draw()
        elif hasattr(self.graphics, 'request_draw'):
            self.graphics.request_draw()

    @property
    def plot(self):
        """Access the main PlotItem."""
        return self._plot
