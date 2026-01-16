"""PyQtGraph-based map viewer widget for Jupyter notebooks.

Provides interactive visualization of map layers with pan/zoom,
track overlays, and integration with echogram/WCI viewers.

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
from dataclasses import dataclass
import warnings

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
class TrackInfo:
    """Track display information."""
    name: str
    latitudes: np.ndarray
    longitudes: np.ndarray
    color: str
    line_width: float = 2.0
    is_active: bool = False  # Whether this is the currently selected channel


class MapViewerPyQtGraph:
    """PyQtGraph-based map viewer for geospatial data.
    
    Features:
    - Interactive pan/zoom with mouse
    - Layer management with visibility/opacity controls
    - Track overlays showing navigation paths from echograms
    - Current ping position marker
    - Integration with EchogramViewerMultiChannel and WCIViewerMultiChannel
    - Coordinate display (lat/lon)
    
    Example:
        from themachinethatgoesping.pingprocessing.widgets import MapViewerPyQtGraph
        from themachinethatgoesping.pingprocessing.overview.map_builder import MapBuilder
        
        builder = MapBuilder()
        builder.add_geotiff('map/BPNS_latlon.tiff')
        
        # Auto-displays in Jupyter (like EchogramViewer)
        viewer = MapViewerPyQtGraph(builder)
        
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
        show: bool = True,
    ):
        """Initialize the map viewer.
        
        Args:
            builder: MapBuilder with layers to display.
            width: Widget width in pixels.
            height: Widget height in pixels.
            show_controls: Whether to show layer control widgets.
            max_render_size: Maximum size for rendered layers (for performance).
            show: Whether to display immediately. Default True.
        """
        # Ensure Qt application exists
        pgh.ensure_qapp()
        
        self._builder = builder
        self._width = width
        self._height = height
        self._show_controls = show_controls
        self._max_render_size = max_render_size
        
        # State
        self._current_bounds = None
        self._layer_images: Dict[str, pg.ImageItem] = {}
        self._coordinate_system = None
        
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
        
        # Build UI
        self._build_ui()
        
        # Initial render
        self._update_view()
        
        # Auto-display like EchogramViewer
        if show:
            self.show()
    
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
        
        # Create plot for map display
        self._plot = self.graphics.addPlot(row=0, col=0)
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)  # Y increases downward in image coords
        self._plot.getViewBox().setBackgroundColor("w")
        
        # Add coordinate label
        self._coord_label = pg.TextItem("", anchor=(0, 1))
        self._coord_label.setPos(10, 10)
        self._plot.addItem(self._coord_label)
        
        # Connect signals
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self._plot.sigRangeChanged.connect(self._on_view_changed)
        
        # Build control widgets if requested
        if self._show_controls:
            self._build_controls()
        else:
            self._controls = None
    
    def _build_controls(self):
        """Build ipywidgets controls for layer management."""
        # Layer visibility checkboxes
        self._layer_checkboxes: Dict[str, ipywidgets.Checkbox] = {}
        self._layer_sliders: Dict[str, ipywidgets.FloatSlider] = {}
        
        layer_widgets = []
        for layer in self._builder.layers:
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
            
            # Opacity slider
            slider = ipywidgets.FloatSlider(
                value=layer.opacity,
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
            
            layer_widgets.append(ipywidgets.HBox([cb, slider]))
        
        # Navigation buttons
        self._btn_zoom_fit = ipywidgets.Button(
            description="Fit All",
            layout=ipywidgets.Layout(width='80px'),
        )
        self._btn_zoom_fit.on_click(lambda _: self.zoom_to_fit())
        
        self._btn_zoom_track = ipywidgets.Button(
            description="Fit Track",
            layout=ipywidgets.Layout(width='80px'),
        )
        self._btn_zoom_track.on_click(lambda _: self.zoom_to_track())
        
        self._btn_refresh_tracks = ipywidgets.Button(
            description="Refresh Tracks",
            layout=ipywidgets.Layout(width='100px'),
        )
        self._btn_refresh_tracks.on_click(lambda _: self.refresh_tracks())
        
        # Coordinate display
        self._lbl_coords = ipywidgets.Label(value="Lat: --, Lon: --")
        
        # Assemble controls
        layers_box = ipywidgets.VBox(layer_widgets) if layer_widgets else ipywidgets.VBox([])
        nav_box = ipywidgets.HBox([self._btn_zoom_fit, self._btn_zoom_track, self._btn_refresh_tracks])
        
        self._controls = ipywidgets.VBox([
            ipywidgets.HTML("<b>Layers</b>"),
            layers_box,
            ipywidgets.HTML("<b>Navigation</b>"),
            nav_box,
            self._lbl_coords,
        ])
    
    # =========================================================================
    # Display
    # =========================================================================
    
    def show(self) -> None:
        """Display the viewer widget."""
        widgets = [ipywidgets.HBox([self.graphics])]
        
        if self._controls is not None:
            widgets.append(self._controls)
        
        widgets.append(self.output)
        
        self.layout = ipywidgets.VBox(widgets)
        display(self.layout)
    
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
        """Render a single layer."""
        try:
            # Get data
            data, cs = layer.backend.get_data(
                bounds=self._current_bounds,
                max_size=self._max_render_size,
            )
        except Exception as e:
            warnings.warn(f"Failed to load layer {layer.name}: {e}")
            return
        
        self._coordinate_system = cs
        
        # Create or update image item
        if layer.name not in self._layer_images:
            img = pg.ImageItem()
            self._plot.addItem(img)
            self._layer_images[layer.name] = img
        
        img = self._layer_images[layer.name]
        
        # Apply colormap
        lut = _get_colormap_lut(layer.colormap or 'viridis')
        
        # Normalize data to 0-255 for LUT
        vmin = layer.vmin if layer.vmin is not None else np.nanmin(data)
        vmax = layer.vmax if layer.vmax is not None else np.nanmax(data)
        
        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0, 1)
            indices = (normalized * 255).astype(np.uint8)
        else:
            indices = np.zeros_like(data, dtype=np.uint8)
        
        # Handle NaN (transparent)
        rgba = lut[indices]
        rgba[np.isnan(data)] = [0, 0, 0, 0]
        
        # Apply opacity
        if layer.opacity < 1.0:
            rgba[:, :, 3] = (rgba[:, :, 3] * layer.opacity).astype(np.uint8)
        
        # Set image
        img.setImage(rgba)
        
        # Set transform (position in world coordinates)
        bounds = cs.bounds
        img.setRect(QtCore.QRectF(
            bounds.xmin, bounds.ymin,
            bounds.width, bounds.height,
        ))
        
        # Set z-order
        img.setZValue(layer.z_order)
        
        # Set visibility
        img.setVisible(layer.visible)
    
    def _update_tracks(self):
        """Update track overlays from connected viewers."""
        if self._coordinate_system is None:
            return
        
        # Clear existing track plots
        for plot in self._track_plots:
            self._plot.removeItem(plot)
        self._track_plots.clear()
        
        # Add tracks
        for name, track_info in self._tracks.items():
            # Convert to world coordinates
            x, y = self._coordinate_system.latlon_to_world(
                track_info.latitudes, track_info.longitudes
            )
            
            # Use thicker line and brighter color for active track
            line_width = track_info.line_width * 2 if track_info.is_active else track_info.line_width
            
            pen = pg.mkPen(color=track_info.color, width=line_width)
            plot = self._plot.plot(x, y, pen=pen)
            self._track_plots.append(plot)
    
    def _update_ping_marker(self):
        """Update the current ping position marker."""
        if self._ping_marker is not None:
            self._plot.removeItem(self._ping_marker)
            self._ping_marker = None
        
        if self._current_ping_latlon is None or self._coordinate_system is None:
            return
        
        lat, lon = self._current_ping_latlon
        x, y = self._coordinate_system.latlon_to_world(lat, lon)
        
        self._ping_marker = pg.ScatterPlotItem(
            [x], [y],
            size=15,
            brush=pg.mkBrush('#FF00FF'),
            symbol='o',
        )
        self._plot.addItem(self._ping_marker)
    
    # =========================================================================
    # User interaction
    # =========================================================================
    
    def _on_mouse_move(self, pos):
        """Handle mouse move for coordinate display."""
        if self._coordinate_system is None:
            return
        
        try:
            # Convert to scene coordinates
            mouse_point = self._plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            # Convert to lat/lon
            lat, lon = self._coordinate_system.world_to_latlon(x, y)
            
            # Update label
            coord_text = f"Lat: {lat:.6f}°, Lon: {lon:.6f}°"
            self._coord_label.setText(coord_text)
            
            if self._controls and hasattr(self, '_lbl_coords'):
                self._lbl_coords.value = coord_text
                
        except Exception:
            pass
    
    def _on_view_changed(self):
        """Handle view range change (pan/zoom)."""
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
        """Handle layer opacity change."""
        self._builder.set_layer_opacity(layer_name, opacity)
        
        # Re-render the layer with new opacity
        layer = self._builder.get_layer(layer_name)
        if layer:
            self._render_layer(layer)
    
    # =========================================================================
    # Navigation
    # =========================================================================
    
    def zoom_to_fit(self):
        """Zoom to fit all visible layers."""
        bounds = self._builder.combined_bounds
        if bounds:
            self._plot.setXRange(bounds.xmin, bounds.xmax, padding=0.05)
            self._plot.setYRange(bounds.ymin, bounds.ymax, padding=0.05)
            self._current_bounds = bounds
            self._update_view()
    
    def zoom_to_track(self):
        """Zoom to fit all navigation tracks."""
        if not self._tracks:
            return
        
        # Import BoundingBox here to avoid circular imports
        from ..overview.map_builder.coordinate_system import BoundingBox
        
        # Combine all tracks
        all_lats = []
        all_lons = []
        for track_info in self._tracks.values():
            all_lats.extend(track_info.latitudes)
            all_lons.extend(track_info.longitudes)
        
        if not all_lats or self._coordinate_system is None:
            return
        
        # Convert to world coordinates
        x, y = self._coordinate_system.latlon_to_world(
            np.array(all_lats), np.array(all_lons)
        )
        
        bounds = BoundingBox.from_points(x, y).expand(1.1)
        
        self._plot.setXRange(bounds.xmin, bounds.xmax, padding=0)
        self._plot.setYRange(bounds.ymin, bounds.ymax, padding=0)
        self._current_bounds = bounds
        self._update_view()
    
    def zoom_to_position(self, lat: float, lon: float, radius_m: float = 1000):
        """Zoom to center on a lat/lon position.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            radius_m: View radius in meters.
        """
        from ..overview.map_builder.coordinate_system import BoundingBox
        
        if self._coordinate_system is None:
            return
        
        x, y = self._coordinate_system.latlon_to_world(lat, lon)
        
        bounds = BoundingBox(
            xmin=x - radius_m,
            ymin=y - radius_m,
            xmax=x + radius_m,
            ymax=y + radius_m,
        )
        
        self._plot.setXRange(bounds.xmin, bounds.xmax, padding=0)
        self._plot.setYRange(bounds.ymin, bounds.ymax, padding=0)
        self._current_bounds = bounds
        self._update_view()
    
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
    ):
        """Add a navigation track overlay.
        
        Args:
            latitudes: Array of latitudes in degrees.
            longitudes: Array of longitudes in degrees.
            name: Track name (used as key).
            color: Track color. If None, auto-assigned.
            line_width: Line width.
            is_active: Whether this is the active/selected track.
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
        )
        
        if is_active:
            self._active_track_name = name
        
        self._update_tracks()
    
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
    
    # =========================================================================
    # Echogram viewer integration
    # =========================================================================
    
    def connect_echogram_viewer(self, echogram_viewer):
        """Connect to an EchogramViewerMultiChannel to show tracks and ping positions.
        
        This will:
        - Add tracks for each visible channel (from echogram builders with get_track())
        - Highlight the track for the currently active slot
        - Update ping position when the ping changes
        
        Args:
            echogram_viewer: EchogramViewerMultiChannel instance.
        """
        self._echogram_viewer = echogram_viewer
        
        # Load tracks from visible echograms
        self._load_tracks_from_echogram_viewer()
    
    def _load_tracks_from_echogram_viewer(self):
        """Load tracks from all echograms in the connected viewer."""
        if self._echogram_viewer is None:
            return
        
        self.clear_tracks()
        
        # Get all echograms from the viewer
        echograms = self._echogram_viewer.echograms
        
        for i, (name, echogram) in enumerate(echograms.items()):
            # Check if echogram has track data
            if hasattr(echogram, 'get_track') and hasattr(echogram, 'has_track') and echogram.has_track:
                track_data = echogram.get_track()
                if track_data is not None:
                    lats, lons = track_data
                    color = self.TRACK_COLORS[i % len(self.TRACK_COLORS)]
                    
                    # Check if this is the active slot
                    is_active = self._is_echogram_active(name)
                    
                    self.add_track(
                        latitudes=lats,
                        longitudes=lons,
                        name=str(name),
                        color=color,
                        is_active=is_active,
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
        """Refresh tracks from connected viewers."""
        if self._echogram_viewer is not None:
            self._load_tracks_from_echogram_viewer()
        if self._wci_viewer is not None:
            self._load_tracks_from_wci_viewer()
    
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
