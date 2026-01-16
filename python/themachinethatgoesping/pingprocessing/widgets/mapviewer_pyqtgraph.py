"""PyQtGraph-based map viewer widget for Jupyter notebooks.

Provides interactive visualization of map layers with pan/zoom,
track overlays, and integration with echogram/WCI viewers.

The viewer handles:
- Colorscale, opacity, blending per layer
- Auto-update with debouncing on pan/zoom
- Track overlays with direct lat/lon coordinates
- Ping position markers from connected viewers

Example:
    from themachinethatgoesping.pingprocessing.widgets import MapViewerPyQtGraph
    from themachinethatgoesping.pingprocessing.overview.map_builder import MapBuilder
    
    builder = MapBuilder()
    builder.add_geotiff('map/BPNS_latlon.tiff')
    
    # Auto-displays in Jupyter (like EchogramViewer)
    viewer = MapViewerPyQtGraph(builder)
    
    # Connect to echogram viewer - tracks are loaded automatically
    viewer.connect_echogram_viewer(echogram_viewer)
    
    # Or connect to WCI viewer
    viewer.connect_wci_viewer(wci_viewer)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
import warnings
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ipywidgets
from IPython.display import display

import pyqtgraph as pg
from pyqtgraph.jupyter import GraphicsLayoutWidget
from pyqtgraph.Qt import QtCore

from . import pyqtgraph_helpers as pgh

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _get_colormap_lut(name: str, n_colors: int = 256) -> np.ndarray:
    """Get a colormap LUT (Look-Up Table) for pyqtgraph.
    
    Args:
        name: Matplotlib colormap name.
        n_colors: Number of colors in the LUT.
        
    Returns:
        RGBA array of shape (n_colors, 4) with values 0-255.
    """
    if not HAS_MATPLOTLIB:
        # Fallback to grayscale
        gray = np.linspace(0, 255, n_colors, dtype=np.uint8)
        return np.stack([gray, gray, gray, np.full(n_colors, 255, dtype=np.uint8)], axis=1)
    
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(0, 1, n_colors))
    return (colors * 255).astype(np.uint8)


@dataclass
class LayerRenderSettings:
    """Viewer-side rendering settings for a layer.
    
    These settings are controlled by the viewer, not the builder.
    """
    colormap: str = "viridis"
    opacity: float = 1.0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    blend_mode: str = "alpha"  # "alpha", "additive", "overlay"


@dataclass
class TrackInfo:
    """Track display information."""
    name: str
    latitudes: np.ndarray
    longitudes: np.ndarray
    color: str
    line_width: float = 2.0
    is_active: bool = False  # Whether this is the currently selected channel
    visible: bool = True  # Whether to display this track
    slot_idx: Optional[int] = None  # Index of the echogram viewer slot (for visible range)


class MapViewerPyQtGraph:
    """PyQtGraph-based map viewer for geospatial data.
    
    Features:
    - Interactive pan/zoom with mouse
    - Layer management with visibility/opacity/colorscale controls (viewer-side)
    - Auto-update with debouncing on pan/zoom (like EchogramViewer)
    - Track overlays showing navigation paths from echograms
    - Current ping position marker (larger points from WCI viewer)
    - Integration with EchogramViewerMultiChannel and WCIViewerMultiChannel
    - Coordinate display (lat/lon)
    
    The MapBuilder provides data; the viewer controls all rendering properties.
    
    Example:
        from themachinethatgoesping.pingprocessing.widgets import MapViewerPyQtGraph
        from themachinethatgoesping.pingprocessing.overview.map_builder import MapBuilder
        
        builder = MapBuilder()
        builder.add_geotiff('map/BPNS_latlon.tiff')
        
        # Auto-displays in Jupyter (like EchogramViewer)
        viewer = MapViewerPyQtGraph(builder)
        
        # Control rendering from viewer
        viewer.set_layer_colormap("BPNS_latlon", "terrain")
        viewer.set_layer_opacity("BPNS_latlon", 0.8)
        
        # Connect to echogram viewer - tracks are loaded automatically
        viewer.connect_echogram_viewer(echogram_viewer)
    """
    
    # Default track colors for different channels
    TRACK_COLORS = [
        "#FF0000",  # Red
        "#00FF00",  # Green  
        "#0000FF",  # Blue
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#FFFF00",  # Yellow
        "#FF8000",  # Orange
        "#8000FF",  # Purple
    ]
    
    def __init__(
        self,
        builder: Any,  # MapBuilder
        width: int = 800,
        height: int = 600,
        show_controls: bool = True,
        max_render_size: Tuple[int, int] = (2000, 2000),
        auto_update: bool = True,
        auto_update_delay_ms: int = 300,
        show: bool = True,
    ):
        """Initialize the map viewer.
        
        Args:
            builder: MapBuilder with layers to display.
            width: Widget width in pixels.
            height: Widget height in pixels.
            show_controls: Whether to show layer control widgets.
            max_render_size: Maximum size for rendered layers (for performance).
            auto_update: Whether to auto-update on pan/zoom.
            auto_update_delay_ms: Delay before auto-update (debounce).
            show: Whether to display immediately. Default True.
        """
        # Ensure Qt application exists
        pgh.ensure_qapp()
        
        self._builder = builder
        self._width = width
        self._height = height
        self._show_controls = show_controls
        self._max_render_size = max_render_size
        
        # Auto-update settings (like EchogramViewer)
        self._auto_update_enabled = auto_update
        self._auto_update_delay_ms = auto_update_delay_ms
        self._debounce_task: Optional[asyncio.Task] = None
        self._last_view_range: Optional[Tuple] = None
        self._last_range_change_time: float = 0.0
        self._startup_complete = False
        self._is_loading = False
        self._ignore_range_changes = False
        self._view_changed_during_load = False
        self._cancel_flag = threading.Event()
        self._loading_future: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        # State
        self._current_bounds = None
        self._layer_images: Dict[str, pg.ImageItem] = {}
        self._coordinate_system = None
        
        # Viewer-controlled rendering settings per layer
        self._layer_render_settings: Dict[str, LayerRenderSettings] = {}
        
        # Track overlays
        self._tracks: Dict[str, TrackInfo] = {}
        self._track_plots: List[Any] = []
        self._active_track_name: Optional[str] = None
        
        # Ping position marker
        self._ping_marker: Optional[pg.ScatterPlotItem] = None
        self._current_ping_latlon: Optional[Tuple[float, float]] = None
        
        # Connected viewers
        self._echogram_viewer = None
        self._wci_viewer = None
        
        # Callbacks
        self._click_callbacks: List[Callable] = []
        self._view_change_callbacks: List[Callable] = []
        
        # Output for errors
        self.output = ipywidgets.Output()
        
        # Initialize default render settings for existing layers
        self._init_layer_render_settings()
        
        # Build UI
        self._build_ui()
        
        # Initial render
        self._update_view()
        
        # Mark startup complete for auto-update
        self._startup_complete = True
        
        # Auto-display like EchogramViewer
        if show:
            self.show()
    
    def _init_layer_render_settings(self):
        """Initialize default render settings for all layers from builder."""
        for layer in self._builder.layers:
            if layer.name not in self._layer_render_settings:
                self._layer_render_settings[layer.name] = LayerRenderSettings()
    
    def _build_ui(self):
        """Build the PyQtGraph and ipywidgets UI."""
        # Create PyQtGraph widget using pyqtgraph.jupyter
        pg.setConfigOptions(imageAxisOrder='row-major')
        
        self.graphics = GraphicsLayoutWidget(
            css_width=f"{self._width}px",
            css_height=f"{self._height}px"
        )
        pgh.apply_widget_layout(self.graphics, self._width, self._height)
        
        # Set background color
        if hasattr(self.graphics, "gfxView"):
            self.graphics.gfxView.setBackground("w")
        
        # Create plot for map display - DO NOT invert Y for lat/lon maps
        self._plot = self.graphics.addPlot(row=0, col=0)
        self._plot.setAspectLocked(True)
        # For geographic coords (lat/lon): Y (lat) increases northward (up), so don't invertY
        # The image data will be flipped if needed based on the transform
        self._plot.getViewBox().setBackgroundColor("w")
        
        # Set axis labels
        self._plot.setLabel('bottom', 'Longitude')
        self._plot.setLabel('left', 'Latitude')
        
        # Add coordinate label
        self._coord_label = pg.TextItem("", anchor=(0, 1))
        self._coord_label.setPos(10, 10)
        self._plot.addItem(self._coord_label)
        
        # Connect signals
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self._plot.sigRangeChanged.connect(self._on_view_changed)
        
        # Set up auto-update hook (like EchogramViewer)
        self._setup_auto_update_hook()
        
        # Build control widgets if requested
        if self._show_controls:
            self._build_controls()
        else:
            self._controls = None
    
    def _setup_auto_update_hook(self):
        """Set up hook for auto-update on pan/zoom (like EchogramViewer)."""
        self._original_request_draw = self.graphics.request_draw
        viewer = self
        
        def patched_request_draw():
            viewer._original_request_draw()
            if not viewer._startup_complete or not viewer._auto_update_enabled:
                return
            if viewer._ignore_range_changes or viewer._is_loading:
                return
            # Check for view range changes
            vb = viewer._plot.getViewBox()
            current_range = vb.viewRange()
            if viewer._last_view_range is not None:
                old_x, old_y = viewer._last_view_range
                new_x, new_y = current_range
                if not (np.allclose(old_x, new_x, rtol=1e-6) and np.allclose(old_y, new_y, rtol=1e-6)):
                    viewer._last_view_range = current_range
                    viewer._last_range_change_time = time.time()
                    viewer._schedule_debounced_update()
            else:
                viewer._last_view_range = current_range
        
        self.graphics.request_draw = patched_request_draw
    
    def _build_controls(self):
        """Build ipywidgets controls for layer management."""
        # Layer visibility checkboxes
        self._layer_checkboxes: Dict[str, ipywidgets.Checkbox] = {}
        self._layer_sliders: Dict[str, ipywidgets.FloatSlider] = {}
        self._layer_colormap_dropdowns: Dict[str, ipywidgets.Dropdown] = {}
        
        # Available colormaps
        colormaps = ['viridis', 'terrain', 'gray', 'plasma', 'inferno', 'magma', 
                     'cividis', 'coolwarm', 'RdBu', 'Blues', 'Greens', 'ocean']
        
        layer_widgets = []
        layer_names = []
        for layer in self._builder.layers:
            settings = self._layer_render_settings.get(layer.name, LayerRenderSettings())
            layer_names.append(layer.name)
            
            # Visibility checkbox
            cb = ipywidgets.Checkbox(
                value=layer.visible,
                description=layer.name,
                indent=False,
                layout=ipywidgets.Layout(width='auto'),
            )
            cb.observe(
                lambda change, name=layer.name: self._on_visibility_change(name, change['new']),
                names='value',
            )
            self._layer_checkboxes[layer.name] = cb
            
            # Opacity slider (viewer-controlled)
            slider = ipywidgets.FloatSlider(
                value=settings.opacity,
                min=0.0,
                max=1.0,
                step=0.1,
                description='',
                continuous_update=True,
                readout=False,
                layout=ipywidgets.Layout(width='100px'),
            )
            slider.observe(
                lambda change, name=layer.name: self._on_opacity_change(name, change['new']),
                names='value',
            )
            self._layer_sliders[layer.name] = slider
            
            # Colormap dropdown (viewer-controlled)
            cmap_dropdown = ipywidgets.Dropdown(
                options=colormaps,
                value=settings.colormap,
                layout=ipywidgets.Layout(width='100px'),
            )
            cmap_dropdown.observe(
                lambda change, name=layer.name: self._on_colormap_change(name, change['new']),
                names='value',
            )
            self._layer_colormap_dropdowns[layer.name] = cmap_dropdown
            
            layer_widgets.append(ipywidgets.HBox([cb, slider, cmap_dropdown]))
        
        # Colorbar selection dropdown
        self._colorbar_layer_dropdown = ipywidgets.Dropdown(
            options=['None'] + layer_names,
            value=layer_names[0] if layer_names else 'None',
            description='Colorbar:',
            layout=ipywidgets.Layout(width='200px'),
        )
        self._colorbar_layer_dropdown.observe(
            lambda change: self._on_colorbar_layer_change(change['new']),
            names='value',
        )
        self._active_colorbar_layer = layer_names[0] if layer_names else None
        
        # Navigation buttons
        self._btn_zoom_fit = ipywidgets.Button(
            description="Fit All",
            layout=ipywidgets.Layout(width='70px'),
        )
        self._btn_zoom_fit.on_click(lambda _: self.zoom_to_fit())
        
        self._btn_zoom_track = ipywidgets.Button(
            description="Fit Track",
            layout=ipywidgets.Layout(width='70px'),
        )
        self._btn_zoom_track.on_click(lambda _: self.zoom_to_track())
        
        self._btn_zoom_wci = ipywidgets.Button(
            description="Go to WCI",
            layout=ipywidgets.Layout(width='80px'),
        )
        self._btn_zoom_wci.on_click(lambda _: self.pan_to_wci_position())
        
        self._btn_refresh_tracks = ipywidgets.Button(
            description="Refresh",
            layout=ipywidgets.Layout(width='70px'),
        )
        self._btn_refresh_tracks.on_click(lambda _: self.refresh_tracks())
        
        # Auto-update checkbox
        self.w_auto_update = ipywidgets.Checkbox(
            value=self._auto_update_enabled,
            description="Auto-update map",
            indent=False,
        )
        self.w_auto_update.observe(self._on_auto_update_toggle, names='value')
        
        # Auto-center on WCI position checkbox
        self.w_auto_center_wci = ipywidgets.Checkbox(
            value=False,
            description="Follow WCI position",
            indent=False,
        )
        self._auto_center_wci = False
        self.w_auto_center_wci.observe(
            lambda change: setattr(self, '_auto_center_wci', change['new']),
            names='value',
        )
        
        # Coordinate display
        self._lbl_coords = ipywidgets.Label(value="Lat: --, Lon: --")
        
        # Assemble controls
        layers_box = ipywidgets.VBox(layer_widgets) if layer_widgets else ipywidgets.VBox([])
        nav_box = ipywidgets.HBox([self._btn_zoom_fit, self._btn_zoom_track, self._btn_zoom_wci, self._btn_refresh_tracks])
        
        self._controls = ipywidgets.VBox([
            ipywidgets.HTML("<b>Layers</b>"),
            layers_box,
            self._colorbar_layer_dropdown,
            ipywidgets.HTML("<b>Navigation</b>"),
            nav_box,
            ipywidgets.HBox([self.w_auto_update, self.w_auto_center_wci]),
            self._lbl_coords,
        ])
        
        # Track legend container (will be populated with checkboxes when tracks are added)
        self._track_legend_label = ipywidgets.HTML("<b>Tracks:</b>")
        self._track_checkboxes: Dict[str, ipywidgets.Checkbox] = {}
        self._track_legend = ipywidgets.VBox([])
    
    # =========================================================================
    # Display
    # =========================================================================
    
    def show(self) -> None:
        """Display the viewer widget."""
        # Create colorbar (pyqtgraph ColorBarItem)
        self._create_colorbar()
        
        widgets = [ipywidgets.HBox([self.graphics])]
        
        if self._controls is not None:
            widgets.append(self._controls)
        
        # Add track legend
        if hasattr(self, '_track_legend'):
            widgets.append(self._track_legend)
        
        widgets.append(self.output)
        
        self.layout = ipywidgets.VBox(widgets)
        display(self.layout)
        
        # Start at fit-all zoom level
        self.zoom_to_fit()
    
    def _create_colorbar(self):
        """Create a pyqtgraph colorbar for the selected layer."""
        if not hasattr(self, '_colorbar_item') or self._colorbar_item is None:
            # Create INTERACTIVE colorbar item (like echogramviewer)
            self._colorbar_item = pg.ColorBarItem(
                interactive=True,  # Allow user to drag color range
                orientation='vertical',
                colorMap=pg.colormap.get('viridis'),
                width=15,
            )
            # Add colorbar to the layout (right of the plot)
            self.graphics.addItem(self._colorbar_item, row=0, col=1)
            
            # Link to first image if available
            if self._layer_images:
                first_image = list(self._layer_images.values())[0]
                self._colorbar_item.setImageItem(first_image)
            
            # Connect level change signal to store user-set levels
            if hasattr(self._colorbar_item, 'sigLevelsChanged'):
                self._colorbar_item.sigLevelsChanged.connect(
                    lambda cb=self._colorbar_item: self._on_colorbar_levels_changed(cb)
                )
        
        # Update colorbar for current layer
        self._update_colorbar()
    
    def _update_colorbar(self):
        """Update the colorbar for the currently selected layer."""
        if not hasattr(self, '_colorbar_item') or self._colorbar_item is None:
            return
        
        layer_name = getattr(self, '_active_colorbar_layer', None)
        if layer_name is None or layer_name == 'None':
            # Hide colorbar if no layer selected
            self._colorbar_item.hide()
            return
        
        # Show and link to the selected layer's image
        self._colorbar_item.show()
        if layer_name in self._layer_images:
            img_item = self._layer_images[layer_name]
            self._colorbar_item.setImageItem(img_item)
        
        settings = self._layer_render_settings.get(layer_name, LayerRenderSettings())
        
        # Update colormap (always update when colormap changes)
        try:
            cmap = pg.colormap.get(settings.colormap, source='matplotlib')
            self._colorbar_item.setColorMap(cmap)
        except Exception as e:
            warnings.warn(f"Could not set colorbar colormap: {e}")
        
        # Initialize layer levels storage if needed
        if not hasattr(self, '_layer_colorbar_levels'):
            self._layer_colorbar_levels = {}
        
        # Get stored levels for this layer or compute initial values
        if layer_name in self._layer_colorbar_levels:
            # Restore user-set levels
            vmin, vmax = self._layer_colorbar_levels[layer_name]
            self._colorbar_item.setLevels((vmin, vmax))
        else:
            # Compute initial levels from settings or data
            vmin = settings.vmin
            vmax = settings.vmax
            if vmin is None or vmax is None:
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
    
    def _on_colorbar_levels_changed(self, colorbar):
        """Handle colorbar level change from user interaction."""
        layer_name = getattr(self, '_active_colorbar_layer', None)
        if layer_name is None or layer_name == 'None':
            return
        
        # Store the user-set levels for this layer
        vmin, vmax = colorbar.levels()
        if not hasattr(self, '_layer_colorbar_levels'):
            self._layer_colorbar_levels = {}
        self._layer_colorbar_levels[layer_name] = (vmin, vmax)
    
    def _get_layer_levels(self, layer_name: str, data: np.ndarray) -> Tuple[float, float]:
        """Get rendering levels for a layer (from colorbar or data).
        
        Args:
            layer_name: Name of the layer.
            data: Data array to compute default levels from.
            
        Returns:
            (vmin, vmax) tuple for rendering.
        """
        # If this is the active colorbar layer, use colorbar levels directly
        if (layer_name == getattr(self, '_active_colorbar_layer', None) and
            hasattr(self, '_colorbar_item') and self._colorbar_item is not None):
            try:
                return self._colorbar_item.levels()
            except Exception:
                pass
        
        # Check stored levels for non-active layers
        if hasattr(self, '_layer_colorbar_levels') and layer_name in self._layer_colorbar_levels:
            return self._layer_colorbar_levels[layer_name]
        
        # Compute from data as fallback
        settings = self._layer_render_settings.get(layer_name, LayerRenderSettings())
        vmin = settings.vmin if settings.vmin is not None else float(np.nanmin(data))
        vmax = settings.vmax if settings.vmax is not None else float(np.nanmax(data))
        return (vmin, vmax)
    
    def _on_colorbar_layer_change(self, layer_name: str):
        """Handle colorbar layer selection change."""
        # Save current colorbar levels before switching
        old_layer = getattr(self, '_active_colorbar_layer', None)
        if old_layer and old_layer != 'None' and hasattr(self, '_colorbar_item') and self._colorbar_item is not None:
            try:
                vmin, vmax = self._colorbar_item.levels()
                if not hasattr(self, '_layer_colorbar_levels'):
                    self._layer_colorbar_levels = {}
                self._layer_colorbar_levels[old_layer] = (vmin, vmax)
            except Exception:
                pass
        
        self._active_colorbar_layer = layer_name if layer_name != 'None' else None
        self._update_colorbar()
    
    # =========================================================================
    # Viewer-side rendering settings
    # =========================================================================
    
    def set_layer_colormap(self, layer_name: str, colormap: str) -> "MapViewerPyQtGraph":
        """Set colormap for a layer (viewer-controlled)."""
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].colormap = colormap
        
        # Update dropdown if exists
        if layer_name in self._layer_colormap_dropdowns:
            self._layer_colormap_dropdowns[layer_name].value = colormap
        
        # Re-render
        layer = self._builder.get_layer(layer_name)
        if layer:
            self._render_layer(layer)
        return self
    
    def set_layer_opacity(self, layer_name: str, opacity: float) -> "MapViewerPyQtGraph":
        """Set opacity for a layer (viewer-controlled)."""
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].opacity = opacity
        
        # Update slider if exists
        if layer_name in self._layer_sliders:
            self._layer_sliders[layer_name].value = opacity
        
        # Re-render
        layer = self._builder.get_layer(layer_name)
        if layer:
            self._render_layer(layer)
        return self
    
    def set_layer_range(self, layer_name: str, vmin: float, vmax: float) -> "MapViewerPyQtGraph":
        """Set value range for a layer (viewer-controlled)."""
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].vmin = vmin
        self._layer_render_settings[layer_name].vmax = vmax
        
        # Re-render
        layer = self._builder.get_layer(layer_name)
        if layer:
            self._render_layer(layer)
        return self
    
    def set_layer_blend_mode(self, layer_name: str, blend_mode: str) -> "MapViewerPyQtGraph":
        """Set blend mode for a layer (viewer-controlled).
        
        Args:
            layer_name: Layer name.
            blend_mode: One of "alpha", "additive", "overlay".
        """
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].blend_mode = blend_mode
        
        # Re-render
        layer = self._builder.get_layer(layer_name)
        if layer:
            self._render_layer(layer)
        return self
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    def _update_view(self):
        """Update the displayed layers based on current view bounds."""
        if self._current_bounds is None:
            # Use combined bounds of all layers
            self._current_bounds = self._builder.combined_bounds
        
        if self._current_bounds is None:
            return
        
        # Get data for each visible layer
        for layer in self._builder.visible_layers:
            self._render_layer(layer)
        
        # Update track overlays
        self._update_tracks()
        
        # Update ping marker
        self._update_ping_marker()
    
    def _render_layer(self, layer):
        """Render a single layer using viewer-controlled settings."""
        try:
            # Get data from builder
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
        
        self._coordinate_system = cs
        
        # Create or update image item
        if layer.name not in self._layer_images:
            img = pg.ImageItem(axisOrder="row-major")
            self._plot.addItem(img)
            self._layer_images[layer.name] = img
        
        img = self._layer_images[layer.name]
        
        # Get viewer-controlled render settings
        settings = self._layer_render_settings.get(layer.name, LayerRenderSettings())
        
        # Handle NaN -> replace with a value outside normal range for masking
        # We'll use the alpha channel for transparency
        data_for_display = data.copy()
        nan_mask = np.isnan(data_for_display)
        
        # Set raw data to image (like echogramviewer does)
        # PyQtGraph will apply colormap and levels interactively
        img.setImage(data_for_display, autoLevels=False)
        
        # Apply colormap using PyQtGraph's native colormap system
        try:
            cmap = pg.colormap.get(settings.colormap, source='matplotlib')
            if hasattr(img, 'setColorMap'):
                img.setColorMap(cmap)
            else:
                lut = cmap.getLookupTable(256)
                img.setLookupTable(lut)
        except Exception as e:
            # Fallback to LUT approach
            lut = _get_colormap_lut(settings.colormap)
            img.setLookupTable(lut)
        
        # Get and apply levels (from user colorbar or computed from data)
        vmin, vmax = self._get_layer_levels(layer.name, data)
        img.setLevels((vmin, vmax))
        
        # Apply opacity
        img.setOpacity(settings.opacity)
        
        # Set transform (position in world coordinates)
        bounds = cs.bounds
        
        # Check if the transform has negative dy (typical for north-up geotiffs)
        if hasattr(cs, 'transform') and cs.transform.e < 0:
            # Standard GeoTiff: row 0 is north (top), row n is south (bottom)
            # Flip the data for display
            data_for_display = np.flipud(data_for_display)
            img.setImage(data_for_display, autoLevels=False)
            # Re-apply colormap after setImage
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
            img.setLevels((vmin, vmax))
        
        # setRect takes (x, y, width, height) where (x,y) is bottom-left corner
        img.setRect(QtCore.QRectF(
            bounds.xmin, bounds.ymin,
            bounds.width, bounds.height,
        ))
        
        # Set z-order
        img.setZValue(layer.z_order)
        
        # Set visibility
        img.setVisible(layer.visible)
    
    def _render_layer_from_data(self, layer, data: np.ndarray, cs):
        """Render a layer from pre-loaded data (for threaded loading)."""
        self._coordinate_system = cs
        
        # Create or update image item
        if layer.name not in self._layer_images:
            img = pg.ImageItem(axisOrder="row-major")
            self._plot.addItem(img)
            self._layer_images[layer.name] = img
        
        img = self._layer_images[layer.name]
        
        # Get viewer-controlled render settings
        settings = self._layer_render_settings.get(layer.name, LayerRenderSettings())
        
        # Handle data for display
        data_for_display = data.copy()
        
        # Check if we need to flip for geotiff orientation
        if hasattr(cs, 'transform') and cs.transform.e < 0:
            data_for_display = np.flipud(data_for_display)
        
        # Set raw data to image (like echogramviewer does)
        img.setImage(data_for_display, autoLevels=False)
        
        # Apply colormap using PyQtGraph's native colormap system
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
        
        # Get and apply levels (from user colorbar or computed from data)
        vmin, vmax = self._get_layer_levels(layer.name, data)
        img.setLevels((vmin, vmax))
        
        # Apply opacity
        img.setOpacity(settings.opacity)
        
        # Set position in world coordinates
        bounds = cs.bounds
        img.setRect(QtCore.QRectF(
            bounds.xmin, bounds.ymin,
            bounds.width, bounds.height,
        ))
        
        # Set z-order and visibility
        img.setZValue(layer.z_order)
        img.setVisible(layer.visible)

    def _update_tracks(self):
        """Update track overlays - show full track as darker, visible region as brighter."""
        # Clear existing track plots
        for plot in self._track_plots:
            self._plot.removeItem(plot)
        self._track_plots.clear()
        
        # Add tracks - use lat/lon directly as x/y (assuming lat/lon coord system)
        for name, track_info in self._tracks.items():
            # Skip hidden tracks
            if not track_info.visible:
                continue
            
            # Use longitude as X, latitude as Y (standard lat/lon convention)
            x = track_info.longitudes
            y = track_info.latitudes
            
            # First, draw full track with darker/thinner line (background)
            darker_color = self._darken_color(track_info.color, 0.5)
            pen_full = pg.mkPen(color=darker_color, width=track_info.line_width * 0.5)
            plot_full = self._plot.plot(x, y, pen=pen_full)
            self._track_plots.append(plot_full)
            
            # Try to get visible ping range from echogram slot
            visible_range = self._get_slot_visible_ping_range(track_info.slot_idx)
            
            if visible_range is not None:
                # Draw visible portion with brighter color and thicker line
                start_idx, end_idx = visible_range
                # Clamp to valid indices
                start_idx = max(0, int(start_idx))
                end_idx = min(len(x), int(end_idx) + 1)
                
                if start_idx < end_idx:
                    x_visible = x[start_idx:end_idx]
                    y_visible = y[start_idx:end_idx]
                    
                    line_width = track_info.line_width * 2
                    pen = pg.mkPen(color=track_info.color, width=line_width)
                    plot_visible = self._plot.plot(x_visible, y_visible, pen=pen)
                    self._track_plots.append(plot_visible)
                    
                    # Add markers along the visible portion of the track
                    if len(x_visible) > 0:
                        # Calculate marker interval - roughly 10 markers along visible track
                        n_points = len(x_visible)
                        marker_interval = max(1, n_points // 10)
                        marker_indices = list(range(0, n_points, marker_interval))
                        # Always include start and end
                        if 0 not in marker_indices:
                            marker_indices.insert(0, 0)
                        if n_points - 1 not in marker_indices:
                            marker_indices.append(n_points - 1)
                        
                        marker_x = [x_visible[i] for i in marker_indices]
                        marker_y = [y_visible[i] for i in marker_indices]
                        
                        # Add circle markers along the visible track
                        markers = pg.ScatterPlotItem(
                            marker_x, marker_y,
                            size=8, brush=pg.mkBrush(track_info.color),
                            pen=pg.mkPen('w', width=1.5), symbol='o'
                        )
                        self._plot.addItem(markers)
                        self._track_plots.append(markers)
                        
                        # Add larger triangle at start
                        marker_start = pg.ScatterPlotItem(
                            [x_visible[0]], [y_visible[0]],
                            size=14, brush=pg.mkBrush(track_info.color),
                            pen=pg.mkPen('w', width=2), symbol='t'
                        )
                        self._plot.addItem(marker_start)
                        self._track_plots.append(marker_start)
                        
                        # Add larger square at end
                        marker_end = pg.ScatterPlotItem(
                            [x_visible[-1]], [y_visible[-1]],
                            size=12, brush=pg.mkBrush(track_info.color),
                            pen=pg.mkPen('w', width=2), symbol='s'
                        )
                        self._plot.addItem(marker_end)
                        self._track_plots.append(marker_end)
            elif track_info.is_active:
                # Fallback: Use thicker line for active track if no visible range
                line_width = track_info.line_width * 2
                pen = pg.mkPen(color=track_info.color, width=line_width)
                plot = self._plot.plot(x, y, pen=pen)
                self._track_plots.append(plot)
    
    def _get_slot_visible_ping_range(self, slot_idx: Optional[int]) -> Optional[Tuple[int, int]]:
        """Get the visible ping index range for an echogram slot.
        
        Returns:
            Tuple of (start_ping, end_ping) or None if not available.
        """
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
        
        # Get the x-axis range from the slot's plot
        try:
            vb = slot.plot_item.getViewBox()
            view_range = vb.viewRange()
            x_range = view_range[0]  # [xmin, xmax] - these are ping indices
            
            # Convert to integer ping indices
            start_ping = int(np.floor(x_range[0]))
            end_ping = int(np.ceil(x_range[1]))
            
            return (start_ping, end_ping)
        except Exception:
            return None
    
    def _darken_color(self, color: str, factor: float = 0.5) -> str:
        """Darken a hex color by a factor."""
        if color.startswith('#'):
            color = color[1:]
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _update_ping_marker(self):
        """Update the current ping position marker - larger point for WCI visibility."""
        if self._ping_marker is not None:
            self._plot.removeItem(self._ping_marker)
            self._ping_marker = None
        
        if self._current_ping_latlon is None:
            return
        
        lat, lon = self._current_ping_latlon
        
        # Use longitude as X, latitude as Y (assuming lat/lon coord system)
        x, y = lon, lat
        
        # Create larger ping marker (size=20 for better visibility)
        self._ping_marker = pg.ScatterPlotItem(
            [x], [y],
            size=20,  # Larger size for better visibility
            brush=pg.mkBrush('#FF00FF'),
            pen=pg.mkPen('#000000', width=2),  # Black outline
            symbol='o',
        )
        self._plot.addItem(self._ping_marker)
    
    # =========================================================================
    # User interaction
    # =========================================================================
    
    def _on_mouse_move(self, pos):
        """Handle mouse move for coordinate display."""
        try:
            # Convert to scene coordinates
            mouse_point = self._plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            # In lat/lon coordinate system: x=lon, y=lat
            lon, lat = x, y
            
            # Update label
            coord_text = f"Lat: {lat:.6f}°, Lon: {lon:.6f}°"
            self._coord_label.setText(coord_text)
            
            if self._controls and hasattr(self, '_lbl_coords'):
                self._lbl_coords.value = coord_text
                
        except Exception:
            pass
    
    def _on_view_changed(self):
        """Handle view range change (pan/zoom)."""
        if self._ignore_range_changes:
            return
        
        # Get new view bounds
        vb = self._plot.vb
        view_range = vb.viewRange()
        
        # Import BoundingBox here to avoid circular imports
        from ..overview.map_builder.coordinate_system import BoundingBox
        
        self._current_bounds = BoundingBox(
            xmin=view_range[0][0],
            xmax=view_range[0][1],
            ymin=view_range[1][0],
            ymax=view_range[1][1],
        )
        
        # Notify callbacks
        for callback in self._view_change_callbacks:
            try:
                callback(self._current_bounds)
            except Exception as e:
                warnings.warn(f"View change callback error: {e}")
    
    def _on_visibility_change(self, layer_name: str, visible: bool):
        """Handle layer visibility toggle."""
        self._builder.set_layer_visibility(layer_name, visible)
        
        if layer_name in self._layer_images:
            self._layer_images[layer_name].setVisible(visible)
    
    def _on_opacity_change(self, layer_name: str, opacity: float):
        """Handle layer opacity change (viewer-controlled)."""
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].opacity = opacity
        
        # Re-render the layer with new opacity
        layer = self._builder.get_layer(layer_name)
        if layer:
            self._render_layer(layer)
    
    def _on_colormap_change(self, layer_name: str, colormap: str):
        """Handle layer colormap change (viewer-controlled)."""
        if layer_name not in self._layer_render_settings:
            self._layer_render_settings[layer_name] = LayerRenderSettings()
        self._layer_render_settings[layer_name].colormap = colormap
        
        # Update colorbar if this is the active layer
        if layer_name == self._active_colorbar_layer:
            self._update_colorbar()
        
        # Re-render the layer with new colormap
        layer = self._builder.get_layer(layer_name)
        if layer:
            self._render_layer(layer)
    
    def _on_auto_update_toggle(self, change):
        """Handle auto-update checkbox toggle."""
        self._auto_update_enabled = change['new']
        if not self._auto_update_enabled and self._debounce_task is not None:
            self._debounce_task.cancel()
    
    # =========================================================================
    # Auto-update (like EchogramViewer)
    # =========================================================================
    
    def _schedule_debounced_update(self) -> None:
        """Schedule a debounced high-res update."""
        if self._is_loading:
            self._view_changed_during_load = True
            return
        
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
        
        async def debounced():
            try:
                await asyncio.sleep(self._auto_update_delay_ms / 1000.0)
                elapsed = time.time() - self._last_range_change_time
                if elapsed >= (self._auto_update_delay_ms / 1000.0) - 0.01:
                    if not self._is_loading:
                        self._trigger_high_res_update()
            except asyncio.CancelledError:
                pass
        
        try:
            loop = asyncio.get_running_loop()
            self._debounce_task = loop.create_task(debounced())
        except RuntimeError:
            self._trigger_high_res_update()
    
    def _cancel_pending_load(self) -> None:
        """Cancel pending background load."""
        self._cancel_flag.set()
        if self._loading_future is not None:
            try:
                self._loading_future.cancel()
            except Exception:
                pass
            self._loading_future = None
        self._is_loading = False
    
    def _trigger_high_res_update(self) -> None:
        """Trigger high-resolution update for all visible layers (threaded)."""
        self._cancel_pending_load()
        
        # Capture current view state
        if self._current_bounds is None:
            self._current_bounds = self._builder.combined_bounds
        if self._current_bounds is None:
            return
        
        self._is_loading = True
        self._view_changed_during_load = False
        self._cancel_flag.clear()
        
        viewer = self
        current_bounds = self._current_bounds
        visible_layers = list(self._builder.visible_layers)
        
        def load_layer_data():
            """Load layer data in background thread."""
            results = {}
            for layer in visible_layers:
                if viewer._cancel_flag.is_set():
                    return None
                try:
                    result = viewer._builder.get_layer_data(
                        layer.name,
                        bounds=current_bounds,
                        max_size=viewer._max_render_size,
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
        
        def apply_results(results):
            viewer._is_loading = False
            if results is None:
                if viewer._view_changed_during_load:
                    viewer._view_changed_during_load = False
                    viewer._schedule_debounced_update()
                return
            
            for layer_name, layer_data in results.items():
                viewer._render_layer_from_data(
                    layer_data['layer'],
                    layer_data['data'],
                    layer_data['cs']
                )
            
            # Update track overlays  
            viewer._update_tracks()
            viewer._update_ping_marker()
            
            # Request redraw
            if hasattr(viewer.graphics, 'request_draw'):
                viewer.graphics.request_draw()
            
            if viewer._view_changed_during_load:
                viewer._view_changed_during_load = False
                viewer._schedule_debounced_update()
        
        async def run_async():
            try:
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(viewer._executor, load_layer_data)
                apply_results(results)
            except Exception as e:
                viewer._is_loading = False
                warnings.warn(f"Map update error: {e}")
        
        try:
            loop = asyncio.get_running_loop()
            self._loading_future = loop.create_task(run_async())
        except RuntimeError:
            # No event loop, run synchronously
            results = load_layer_data()
            apply_results(results)
    
    # =========================================================================
    # Navigation
    # =========================================================================
    
    def zoom_to_fit(self):
        """Zoom to fit all visible layers."""
        bounds = self._builder.combined_bounds
        if bounds:
            self._ignore_range_changes = True
            self._plot.setXRange(bounds.xmin, bounds.xmax, padding=0.05)
            self._plot.setYRange(bounds.ymin, bounds.ymax, padding=0.05)
            self._ignore_range_changes = False
            self._current_bounds = bounds
            self._update_view()
    
    def zoom_to_track(self):
        """Zoom to fit all navigation tracks."""
        if not self._tracks:
            return
        
        # Import BoundingBox here to avoid circular imports
        from ..overview.map_builder.coordinate_system import BoundingBox
        
        # Combine all tracks - use lon as X, lat as Y
        all_x = []
        all_y = []
        for track_info in self._tracks.values():
            all_x.extend(track_info.longitudes)
            all_y.extend(track_info.latitudes)
        
        if not all_x:
            return
        
        bounds = BoundingBox.from_points(np.array(all_x), np.array(all_y)).expand(1.1)
        
        self._ignore_range_changes = True
        self._plot.setXRange(bounds.xmin, bounds.xmax, padding=0)
        self._plot.setYRange(bounds.ymin, bounds.ymax, padding=0)
        self._ignore_range_changes = False
        self._current_bounds = bounds
        self._update_view()
    
    def zoom_to_position(self, lat: float, lon: float, radius_deg: float = 0.01):
        """Zoom to center on a lat/lon position.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            radius_deg: View radius in degrees.
        """
        from ..overview.map_builder.coordinate_system import BoundingBox
        
        # In lat/lon system: x=lon, y=lat
        bounds = BoundingBox(
            xmin=lon - radius_deg,
            ymin=lat - radius_deg,
            xmax=lon + radius_deg,
            ymax=lat + radius_deg,
        )
        
        self._ignore_range_changes = True
        self._plot.setXRange(bounds.xmin, bounds.xmax, padding=0)
        self._plot.setYRange(bounds.ymin, bounds.ymax, padding=0)
        self._ignore_range_changes = False
        self._current_bounds = bounds
        self._update_view()
    
    def pan_to_wci_position(self):
        """Pan to center on the current WCI ping position without changing zoom."""
        if self._current_ping_latlon is None:
            warnings.warn("No WCI position available")
            return
        
        lat, lon = self._current_ping_latlon
        self.pan_to_position(lat, lon)
    
    def pan_to_position(self, lat: float, lon: float):
        """Pan to center on a position without changing zoom level.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
        """
        # Get current view range
        view_range = self._plot.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]
        
        # Calculate current view size
        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]
        
        # Set new range centered on position
        self._ignore_range_changes = True
        self._plot.setXRange(lon - width/2, lon + width/2, padding=0)
        self._plot.setYRange(lat - height/2, lat + height/2, padding=0)
        self._ignore_range_changes = False
        
        self._update_view()
    
    def is_position_near_edge(self, lat: float, lon: float, edge_fraction: float = 0.2) -> bool:
        """Check if a position is near the edge of the current view.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            edge_fraction: Fraction of view to consider as 'edge' (0.2 = 20%).
            
        Returns:
            True if position is in the outer edge_fraction of the view.
        """
        view_range = self._plot.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]
        
        # Calculate inner bounds (non-edge area)
        x_margin = (x_range[1] - x_range[0]) * edge_fraction
        y_margin = (y_range[1] - y_range[0]) * edge_fraction
        
        inner_xmin = x_range[0] + x_margin
        inner_xmax = x_range[1] - x_margin
        inner_ymin = y_range[0] + y_margin
        inner_ymax = y_range[1] - y_margin
        
        # Check if position is outside inner bounds (i.e., in edge area)
        return not (inner_xmin <= lon <= inner_xmax and inner_ymin <= lat <= inner_ymax)
    
    def pan_to_position_if_near_edge(self, lat: float, lon: float, edge_fraction: float = 0.2):
        """Pan to center on position only if it's near the edge of the view.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            edge_fraction: Fraction of view to consider as 'edge' (0.2 = 20%).
        """
        if self.is_position_near_edge(lat, lon, edge_fraction):
            self.pan_to_position(lat, lon)
    
    # =========================================================================
    # Track management
    # =========================================================================
    
    def add_track(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        name: str = "Track",
        color: Optional[str] = None,
        line_width: float = 2.0,
        is_active: bool = False,
        slot_idx: Optional[int] = None,
    ):
        """Add a navigation track overlay.
        
        Args:
            latitudes: Array of latitudes in degrees.
            longitudes: Array of longitudes in degrees.
            name: Track name (used as key).
            color: Track color. If None, auto-assigned.
            line_width: Line width.
            is_active: Whether this is the active/selected track.
            slot_idx: Echogram viewer slot index (for visible range highlighting).
        """
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
        self._update_track_legend()
    
    def set_active_track(self, name: str):
        """Set which track is the active/highlighted one.
        
        Args:
            name: Name of the track to make active.
        """
        for track_name, track_info in self._tracks.items():
            track_info.is_active = (track_name == name)
        
        self._active_track_name = name
        self._update_tracks()
    
    def clear_tracks(self):
        """Remove all tracks."""
        self._tracks.clear()
        for plot in self._track_plots:
            self._plot.removeItem(plot)
        self._track_plots.clear()
        self._update_track_legend()
    
    def _update_track_legend(self):
        """Update the track legend with interactive checkboxes."""
        if not hasattr(self, '_track_legend'):
            return
        
        if not self._tracks:
            self._track_legend.children = []
            return
        
        # Build checkbox widgets for each track
        checkbox_widgets = [self._track_legend_label]
        self._track_checkboxes.clear()
        
        for name, track in self._tracks.items():
            # Create styled checkbox with track color indicator
            checkbox = ipywidgets.Checkbox(
                value=track.visible,
                description='',
                indent=False,
                layout=ipywidgets.Layout(width='20px'),
            )
            # Create color indicator and name label
            color_label = ipywidgets.HTML(
                f'<span style="color:{track.color}; font-weight:bold;">●</span> {name}'
            )
            
            # Create handler for this track (capture name in closure)
            def make_handler(track_name):
                def handler(change):
                    self._on_track_visibility_change(track_name, change['new'])
                return handler
            
            checkbox.observe(make_handler(name), names='value')
            self._track_checkboxes[name] = checkbox
            
            # Combine checkbox and label
            row = ipywidgets.HBox([checkbox, color_label])
            checkbox_widgets.append(row)
        
        self._track_legend.children = checkbox_widgets
    
    def _on_track_visibility_change(self, track_name: str, visible: bool):
        """Handle track visibility checkbox change."""
        if track_name in self._tracks:
            self._tracks[track_name].visible = visible
            self._update_tracks()
    
    # =========================================================================
    # Ping position
    # =========================================================================
    
    def update_ping_position(self, lat: float, lon: float):
        """Update the current ping position marker.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
        """
        self._current_ping_latlon = (lat, lon)
        self._update_ping_marker()
        
        # Auto-center if enabled (only pan if position is near edge)
        if getattr(self, '_auto_center_wci', False):
            self.pan_to_position_if_near_edge(lat, lon, edge_fraction=0.2)
    
    # =========================================================================
    # Echogram viewer integration
    # =========================================================================
    
    def connect_echogram_viewer(self, echogram_viewer):
        """Connect to an EchogramViewerMultiChannel to show tracks and ping positions.
        
        This will:
        - Add tracks for each visible channel (from echogram builders with get_track())
        - Sync track visibility with echogramviewer slot visibility
        - Highlight the track for the currently active slot
        - Update ping position when the ping changes
        
        Args:
            echogram_viewer: EchogramViewerMultiChannel instance.
        """
        self._echogram_viewer = echogram_viewer
        
        # Load tracks from visible echograms
        self._load_tracks_from_echogram_viewer()
    
    def _get_visible_echogram_names(self) -> set:
        """Get names of echograms currently visible in the echogram viewer."""
        if self._echogram_viewer is None:
            return set()
        
        visible_names = set()
        # Check which echograms are shown in visible slots
        if hasattr(self._echogram_viewer, 'slots'):
            n_visible = self._echogram_viewer.grid_rows * self._echogram_viewer.grid_cols
            for i, slot in enumerate(self._echogram_viewer.slots[:n_visible]):
                if slot.is_visible and slot.echogram_key is not None:
                    visible_names.add(str(slot.echogram_key))
        return visible_names
    
    def _load_tracks_from_echogram_viewer(self):
        """Load tracks from visible slots in the connected viewer.
        
        Only loads tracks for echograms that are currently displayed in a visible slot.
        When grid is 1x1, only 1 track will be shown.
        """
        if self._echogram_viewer is None:
            return
        
        self.clear_tracks()
        
        # Iterate directly over visible slots
        if not hasattr(self._echogram_viewer, 'slots'):
            return
        
        echograms = self._echogram_viewer.echograms
        n_visible = self._echogram_viewer.grid_rows * self._echogram_viewer.grid_cols
        
        for slot_idx, slot in enumerate(self._echogram_viewer.slots[:n_visible]):
            if not slot.is_visible or slot.echogram_key is None:
                continue
            
            echogram = echograms.get(slot.echogram_key)
            if echogram is None:
                continue
            
            # Check if echogram has track data
            if hasattr(echogram, 'get_track') and hasattr(echogram, 'has_track') and echogram.has_track:
                track_data = echogram.get_track()
                if track_data is not None:
                    lats, lons = track_data
                    # Use slot index for color to match slot ordering
                    color = self.TRACK_COLORS[slot_idx % len(self.TRACK_COLORS)]
                    
                    # Slot 0 is the active/primary slot
                    is_active = (slot_idx == 0)
                    
                    self.add_track(
                        latitudes=lats,
                        longitudes=lons,
                        name=str(slot.echogram_key),
                        color=color,
                        is_active=is_active,
                        slot_idx=slot_idx,  # Store slot index for visible range query
                    )
        
        self._update_tracks()
    
    def _is_echogram_active(self, echogram_name: str) -> bool:
        """Check if an echogram is in the currently active slot."""
        if self._echogram_viewer is None:
            return False
        
        # Check slot 0 (or whatever is considered "primary")
        if hasattr(self._echogram_viewer, 'slots') and self._echogram_viewer.slots:
            active_slot = self._echogram_viewer.slots[0]
            return active_slot.echogram_key == echogram_name
        
        return False
    
    # =========================================================================
    # WCI viewer integration
    # =========================================================================
    
    def connect_wci_viewer(self, wci_viewer):
        """Connect to a WCIViewerMultiChannel to show tracks and ping positions.
        
        This will:
        - Add tracks for each channel (if echogram data is available)
        - Update ping position when the ping changes
        
        Args:
            wci_viewer: WCIViewerMultiChannel instance.
        """
        self._wci_viewer = wci_viewer
        
        # Load tracks from WCI channels if they have echogram/navigation data
        self._load_tracks_from_wci_viewer()
        
        # Register ping change callback to update position marker
        if hasattr(wci_viewer, 'register_ping_change_callback'):
            wci_viewer.register_ping_change_callback(self._on_wci_ping_change)
        
        # Update position now to show current ping
        self._on_wci_ping_change()
    
    def _on_wci_ping_change(self):
        """Handle WCI ping change - update ping position marker."""
        if self._wci_viewer is None:
            return
        
        # Get the current ping from the reference slot (slot 0 or active)
        slots = getattr(self._wci_viewer, 'slots', [])
        if not slots:
            return
        
        # Find first visible slot with data
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
                        pass  # Skip if geolocation not available
    
    def _load_tracks_from_wci_viewer(self):
        """Load tracks from WCI viewer channels."""
        if self._wci_viewer is None:
            return
        
        # WCI viewer channels are ping sequences, need to extract navigation
        # This will depend on the actual structure of WCIViewerMultiChannel
        channels = getattr(self._wci_viewer, 'channels', {})
        
        for i, (name, pings) in enumerate(channels.items()):
            try:
                # Try to extract lat/lon from pings
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
                            latitudes=np.array(lats),
                            longitudes=np.array(lons),
                            name=str(name),
                            color=color,
                        )
            except Exception as e:
                warnings.warn(f"Failed to extract track from WCI channel {name}: {e}")
        
        self._update_tracks()
    
    def refresh_tracks(self):
        """Refresh tracks from connected viewers, preserving visibility state."""
        # Save current visibility state
        visibility_state = {name: track.visible for name, track in self._tracks.items()}
        
        if self._echogram_viewer is not None:
            self._load_tracks_from_echogram_viewer()
        if self._wci_viewer is not None:
            self._load_tracks_from_wci_viewer()
        
        # Restore visibility state for tracks that still exist
        for name, was_visible in visibility_state.items():
            if name in self._tracks:
                self._tracks[name].visible = was_visible
                # Update checkbox if it exists
                if name in self._track_checkboxes:
                    self._track_checkboxes[name].value = was_visible
        
        self._update_tracks()
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    def register_click_callback(self, callback: Callable[[float, float], None]):
        """Register a callback for map clicks.
        
        Callback receives (lat, lon) of clicked position.
        
        Args:
            callback: Function to call on click.
        """
        self._click_callbacks.append(callback)
    
    def register_view_change_callback(self, callback: Callable):
        """Register a callback for view changes.
        
        Callback receives new view bounds.
        
        Args:
            callback: Function to call on view change.
        """
        self._view_change_callbacks.append(callback)
