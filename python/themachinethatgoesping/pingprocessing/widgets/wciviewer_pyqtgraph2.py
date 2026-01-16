"""PyQtGraph-based Multi-Channel Water Column Image (WCI) viewer.

Features:
- Grid layout selector (1, 2, 2x2, 3x2, 4x2)
- Per-slot ping number selection with time synchronization
- Global controls except per-slot ping selection and color levels
- Time difference display for synchronized pings
- Per-slot interactive colorbars with global override
- Smooth view transitions when switching slots
"""
from __future__ import annotations

import time as time_module
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ipywidgets
import numpy as np
import pyqtgraph as pg
from IPython.display import display
from pyqtgraph.jupyter import GraphicsLayoutWidget
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from themachinethatgoesping import echosounders
import themachinethatgoesping.pingprocessing.watercolumn.image as mi
from themachinethatgoesping.pingprocessing.widgets import TqdmWidget

from . import pyqtgraph_helpers as pgh

WCI_VALUE_CHOICES = [
    "sv/av/pv/rv",
    "sv/av/pv",
    "sv/av",
    "sp/ap/pp/rp",
    "sp/ap/pp",
    "sp/ap",
    "power/amp",
    "av",
    "ap",
    "amp",
    "sv",
    "sp",
    "pv",
    "pp",
    "rv",
    "rp",
    "power",
    "sv_vs_av",
    "sp_vs_ap"
]


class WCISlot:
    """Manages a single WCI display slot with per-slot ping and color levels."""
    
    def __init__(self, slot_idx: int, parent: 'WCIViewerMultiChannel'):
        self.slot_idx = slot_idx
        self.parent = parent
        self.channel_key: Optional[str] = None  # Key into parent.channels dict
        self.is_visible = False
        
        # Per-slot ping index
        self.ping_index: int = 0
        
        # Stored ping index (the last explicitly set ping, not synced)
        self.stored_ping_index: int = 0
        
        # Per-slot color levels (stored when switching between slots)
        self.color_levels: Optional[Tuple[float, float]] = None
        
        # Time offset from reference timestamp (seconds)
        self.time_offset: float = 0.0
        
        # Crosshair items
        self.crosshair_v: Optional[pg.InfiniteLine] = None
        self.crosshair_h: Optional[pg.InfiniteLine] = None
        
        # Cached WCI image and extent
        self.wci_image: Optional[np.ndarray] = None
        self.wci_extent: Optional[Tuple[float, float, float, float]] = None
        
        # Image cache by ping index for fast switching
        self._image_cache: Dict[int, Dict[str, Any]] = {}
        
        # PyQtGraph items (set by parent when creating plots)
        self.plot_item: Optional[pg.PlotItem] = None
        self.image_item: Optional[pg.ImageItem] = None
        self.colorbar: Optional[pg.ColorBarItem] = None
        self.time_offset_text: Optional[pg.TextItem] = None
        
        # ImageBuilder for this slot (one per channel)
        self.imagebuilder: Optional[mi.ImageBuilder] = None
    
    def set_visible(self, visible: bool):
        """Set visibility."""
        self.is_visible = visible
    
    def assign_channel(self, channel_key: Optional[str]):
        """Assign a channel (ping source) to this slot."""
        if channel_key != self.channel_key:
            # Cache current image if we have it
            if self.channel_key is not None and self.wci_image is not None:
                self._image_cache[self.ping_index] = {
                    'wci_image': self.wci_image,
                    'wci_extent': self.wci_extent,
                }
            
            self.channel_key = channel_key
            self.wci_image = None
            self.wci_extent = None
            self._image_cache.clear()
            
            # Create new imagebuilder for this channel
            if channel_key is not None:
                pings = self.parent.channels.get(channel_key)
                if pings is not None and len(pings) > 0:
                    self.imagebuilder = mi.ImageBuilder(
                        pings,
                        horizontal_pixels=self.parent.args_imagebuilder["horizontal_pixels"],
                        progress=self.parent.progress,
                    )
                else:
                    self.imagebuilder = None
            else:
                self.imagebuilder = None
    
    def get_pings(self) -> Optional[Sequence[Any]]:
        """Get the pings assigned to this slot."""
        if self.channel_key is None:
            return None
        return self.parent.channels.get(self.channel_key)
    
    def get_ping(self, index: Optional[int] = None) -> Optional[Any]:
        """Get a specific ping from this slot's channel."""
        pings = self.get_pings()
        if pings is None or len(pings) == 0:
            return None
        idx = index if index is not None else self.ping_index
        if 0 <= idx < len(pings):
            ping = pings[idx]
            if isinstance(ping, dict):
                return next(iter(ping.values()))
            return ping
        return None
    
    def get_timestamp(self, index: Optional[int] = None) -> Optional[float]:
        """Get timestamp of a ping (unix time)."""
        ping = self.get_ping(index)
        if ping is None:
            return None
        try:
            return ping.get_timestamp()
        except Exception:
            return None
    
    def find_closest_ping_index(self, target_timestamp: float) -> int:
        """Find the ping index closest to the target timestamp."""
        pings = self.get_pings()
        if pings is None or len(pings) == 0:
            return 0
        
        best_idx = 0
        best_diff = float('inf')
        
        for i in range(len(pings)):
            ts = self.get_timestamp(i)
            if ts is not None:
                diff = abs(ts - target_timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
        
        return best_idx


class WCIViewerMultiChannel:
    """Multi-channel Water Column Image viewer with time synchronization.
    
    Features:
    - Multiple WCI channels displayed in a grid layout
    - Per-slot ping number selection
    - Time synchronization across channels
    - Per-slot color scales with global override
    - Smooth view transitions
    """
    
    GRID_LAYOUTS = [
        (1, 1, "1"),
        (1, 2, "1×2"),
        (2, 1, "2×1"),
        (2, 2, "2×2"),
        (3, 2, "3×2"),
        (4, 2, "4×2"),
    ]
    
    def __init__(
        self,
        channels: Union[Dict[str, Sequence[Any]], Sequence[Sequence[Any]]],
        name: str = "Multi-Channel WCI",
        names: Optional[Sequence[Optional[str]]] = None,
        horizontal_pixels: int = 1024,
        progress: Optional[Any] = None,
        show: bool = True,
        cmap: str = "YlGnBu_r",
        widget_height_px: int = 600,
        widget_width_px: int = 1000,
        initial_grid: Tuple[int, int] = (2, 2),
        time_sync_enabled: bool = True,
        time_warning_threshold: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the multi-channel WCI viewer.
        
        Parameters
        ----------
        channels : dict or sequence
            Either a dict mapping channel names to ping sequences,
            or a sequence of ping sequences.
        name : str
            Viewer name/title.
        names : sequence, optional
            Names for each channel if channels is a sequence.
        horizontal_pixels : int
            Horizontal resolution for WCI images.
        progress : widget, optional
            Progress widget for loading feedback.
        show : bool
            If True, display immediately.
        cmap : str
            Colormap name.
        widget_height_px : int
            Widget height in pixels.
        widget_width_px : int
            Widget width in pixels.
        initial_grid : tuple
            Initial grid layout (rows, cols).
        time_sync_enabled : bool
            If True, synchronize ping times across channels.
        time_warning_threshold : float
            Time difference (seconds) above which to show red warning text.
        **kwargs
            Additional arguments passed to image builder.
        """
        pgh.ensure_qapp()
        pg.setConfigOptions(imageAxisOrder="row-major")
        
        # Image builder arguments
        self.args_imagebuilder: Dict[str, Any] = {
            "horizontal_pixels": horizontal_pixels,
            "linear_mean": True,
            "hmin": None,
            "hmax": None,
            "vmin": None,
            "vmax": None,
            "wci_value": "sv/av/pv/rv",
            "wci_render": "linear",
            "ping_sample_selector": echosounders.pingtools.PingSampleSelector(),
            "apply_pss_to_bottom": False,
            "mp_cores": 1,
        }
        
        # Plot arguments
        self.args_plot: Dict[str, Any] = {
            "vmin": kwargs.pop("vmin", -90),
            "vmax": kwargs.pop("vmax", -25),
        }
        
        # Apply kwargs to imagebuilder args
        for key in list(kwargs.keys()):
            if key in self.args_imagebuilder:
                self.args_imagebuilder[key] = kwargs.pop(key)
        self.args_imagebuilder.update(kwargs)
        
        # Convert input to dict format
        if isinstance(channels, dict):
            self.channels: Dict[str, Sequence[Any]] = dict(channels)
            self.channel_names = list(channels.keys())
        elif hasattr(channels, '__iter__') and not isinstance(channels, (str, bytes)):
            channels_list = list(channels)
            # Check if this is a single ping list or a list of ping lists
            if len(channels_list) > 0:
                first_item = channels_list[0]
                # If first item looks like a ping (has get_timestamp), treat as single channel
                if hasattr(first_item, 'get_timestamp') or hasattr(first_item, 'file_data'):
                    # Single ping list - wrap in dict with "default" key
                    self.channel_names = ["default"]
                    self.channels = {"default": channels_list}
                else:
                    # List of ping lists
                    if names is not None:
                        self.channel_names = [str(n) if n else f"Channel {i}" for i, n in enumerate(names)]
                    else:
                        self.channel_names = [f"Channel {i}" for i in range(len(channels_list))]
                    self.channels = {name: ch for name, ch in zip(self.channel_names, channels_list)}
            else:
                self.channel_names = []
                self.channels = {}
        else:
            # Single item - wrap in list and dict
            self.channel_names = ["default"]
            self.channels = {"default": [channels]}
        
        self.name = name
        self.cmap_name = cmap
        self._colormap = pgh.resolve_colormap(cmap)
        
        # Progress widget
        self.progress = progress or TqdmWidget()
        self.display_progress = progress is None
        
        # Time synchronization
        self.time_sync_enabled = time_sync_enabled
        self.time_warning_threshold = time_warning_threshold
        self._reference_slot_idx: int = 0  # Slot that drives time sync
        self._syncing = False  # Flag to prevent recursive sync
        self._reference_timestamp: Optional[float] = None  # Global reference time
        
        # Crosshair state
        self._crosshair_enabled = True
        self._crosshair_position: Optional[Tuple[float, float]] = None
        
        # Ping change callbacks for connected viewers (e.g., echogram viewer)
        self._ping_change_callbacks: List[Any] = []

        # Grid layout state
        n_channels = len(self.channel_names)
        if initial_grid == (2, 2):
            if n_channels == 1:
                self.grid_rows, self.grid_cols = (1, 1)
            elif n_channels == 2:
                self.grid_rows, self.grid_cols = (1, 2)
            elif n_channels <= 4:
                self.grid_rows, self.grid_cols = (2, 2)
            elif n_channels <= 6:
                self.grid_rows, self.grid_cols = (3, 2)
            else:
                self.grid_rows, self.grid_cols = (4, 2)
        else:
            self.grid_rows, self.grid_cols = initial_grid
        self.max_slots = 8
        
        # Widget dimensions
        self.widget_height_px = widget_height_px
        self.widget_width_px = widget_width_px
        
        # Create slots
        self.slots: List[WCISlot] = []
        for i in range(self.max_slots):
            slot = WCISlot(i, self)
            self.slots.append(slot)
        
        # Assign initial channels to slots
        for i, ch_name in enumerate(self.channel_names[:self.max_slots]):
            self.slots[i].assign_channel(ch_name)
        
        # Output widget for errors
        self.output = ipywidgets.Output()
        self.hover_label = ipywidgets.HTML(value="&nbsp;")
        
        # View state
        self._ignore_range_changes = False
        self._first_draw = True
        
        # Build UI
        self._build_ui()
        self._make_graphics_widget()
        self._update_grid_layout()
        self._assemble_layout()
        
        if show:
            display(self.layout)
        
        # Initial data load for visible slots
        self._update_all_visible()
        
        # Initialize reference timestamp from first slot
        if self.slots[0].channel_key is not None:
            ts = self.slots[0].get_timestamp()
            if ts is not None:
                self._reference_timestamp = ts
                self._update_ref_time_display()
    
    def _build_ui(self) -> None:
        """Build the ipywidgets UI components."""
        # Layout selector
        layout_options = [(label, (r, c)) for r, c, label in self.GRID_LAYOUTS]
        self.w_layout = ipywidgets.Dropdown(
            description="Grid:",
            options=layout_options,
            value=(self.grid_rows, self.grid_cols),
            layout=ipywidgets.Layout(width='120px'),
        )
        self.w_layout.observe(self._on_layout_change, names='value')
        
        # Channel selectors for each slot
        channel_options = [(name, name) for name in self.channel_names]
        channel_options.insert(0, ("(none)", None))
        
        self.slot_selectors: List[ipywidgets.Dropdown] = []
        for i in range(self.max_slots):
            selector = ipywidgets.Dropdown(
                description=f"Ch {i+1}:",
                options=channel_options,
                value=self.slots[i].channel_key,
                layout=ipywidgets.Layout(width='180px'),
            )
            selector.observe(lambda change, idx=i: self._on_slot_change(idx, change), names='value')
            self.slot_selectors.append(selector)
        
        # Per-slot ping sliders
        self.ping_sliders: List[ipywidgets.IntSlider] = []
        for i in range(self.max_slots):
            pings = self.slots[i].get_pings()
            max_ping = max(0, len(pings) - 1) if pings else 0
            slider = ipywidgets.IntSlider(
                description=f"Ping:",
                min=0,
                max=max_ping,
                value=0,
                layout=ipywidgets.Layout(width='250px'),
            )
            slider.observe(lambda change, idx=i: self._on_ping_change(idx, change), names='value')
            self.ping_sliders.append(slider)
        
        # Tab buttons for quick single-view access
        self.tab_buttons: List[ipywidgets.Button] = []
        for ch_name in self.channel_names:
            name_str = str(ch_name)
            btn = ipywidgets.Button(
                description=name_str[:15],
                tooltip=f"Show {name_str} full-size",
                layout=ipywidgets.Layout(width='auto', min_width='60px'),
            )
            btn.on_click(lambda _, n=ch_name: self._show_single(n))
            self.tab_buttons.append(btn)
        
        # Global color scale sliders
        self.w_vmin = ipywidgets.FloatSlider(
            description="vmin", min=-150, max=100, step=0.5,
            value=self.args_plot["vmin"],
            layout=ipywidgets.Layout(width='220px'),
        )
        self.w_vmax = ipywidgets.FloatSlider(
            description="vmax", min=-150, max=100, step=0.5,
            value=self.args_plot["vmax"],
            layout=ipywidgets.Layout(width='220px'),
        )
        self.w_vmin.observe(self._on_global_color_change, names='value')
        self.w_vmax.observe(self._on_global_color_change, names='value')
        
        # Time sync controls
        self.w_time_sync = ipywidgets.Checkbox(
            value=self.time_sync_enabled,
            description="Sync time",
            indent=False,
        )
        self.w_time_sync.observe(lambda c: setattr(self, 'time_sync_enabled', c['new']), names='value')
        
        self.w_time_warning = ipywidgets.FloatText(
            value=self.time_warning_threshold,
            description="Warn Δt (s):",
            layout=ipywidgets.Layout(width='130px'),
        )
        self.w_time_warning.observe(
            lambda c: setattr(self, 'time_warning_threshold', c['new']),
            names='value'
        )
        
        # Crosshair checkbox
        self.w_crosshair = ipywidgets.Checkbox(
            value=self._crosshair_enabled,
            description="Crosshair",
            indent=False,
        )
        self.w_crosshair.observe(lambda c: setattr(self, '_crosshair_enabled', c['new']), names='value')
        
        # Reference time display (read-only)
        self.w_ref_time = ipywidgets.Text(
            value="",
            description="Ref time:",
            disabled=True,
            layout=ipywidgets.Layout(width='220px'),
        )
        
        # Image builder controls (global)
        self.w_stack = ipywidgets.IntText(value=1, description="stack", layout=ipywidgets.Layout(width='140px'))
        self.w_stack_step = ipywidgets.IntText(value=1, description="step", layout=ipywidgets.Layout(width='140px'))
        self.w_mp_cores = ipywidgets.IntText(
            value=self.args_imagebuilder["mp_cores"],
            description="cores",
            layout=ipywidgets.Layout(width='140px')
        )
        
        self.w_stack_linear = ipywidgets.Checkbox(
            description="linear stack",
            value=self.args_imagebuilder["linear_mean"],
        )
        self.w_wci_value = ipywidgets.Dropdown(
            description="value",
            options=WCI_VALUE_CHOICES,
            value=self.args_imagebuilder["wci_value"],
            layout=ipywidgets.Layout(width='180px'),
        )
        self.w_wci_render = ipywidgets.Dropdown(
            description="render",
            options=["linear", "beamsample"],
            value=self.args_imagebuilder["wci_render"],
            layout=ipywidgets.Layout(width='150px'),
        )
        self.w_horizontal_pixels = ipywidgets.IntSlider(
            description="h_pixels",
            min=2,
            max=2048,
            step=1,
            value=self.args_imagebuilder["horizontal_pixels"],
            layout=ipywidgets.Layout(width='200px'),
        )
        
        # Observe global controls for rebuild
        for widget in [self.w_stack, self.w_stack_step, self.w_mp_cores,
                       self.w_stack_linear, self.w_wci_value, self.w_wci_render,
                       self.w_horizontal_pixels]:
            widget.observe(self._on_global_param_change, names="value")
        
        # Fix/unfix view buttons
        self.w_fix_xy = ipywidgets.Button(description="Fix view", layout=ipywidgets.Layout(width='80px'))
        self.w_unfix_xy = ipywidgets.Button(description="Unfix", layout=ipywidgets.Layout(width='70px'))
        self.w_fix_xy.on_click(self._fix_xy)
        self.w_unfix_xy.on_click(self._unfix_xy)
        
        # Timing display
        self.w_proctime = ipywidgets.Text(description="time", disabled=True, layout=ipywidgets.Layout(width='280px'))
        self.w_procrate = ipywidgets.Text(description="rate", disabled=True, layout=ipywidgets.Layout(width='280px'))
        
        # === PLAYBACK CONTROLS ===
        # Ping step control
        self.w_ping_step = ipywidgets.IntText(
            value=1,
            description="ping step",
            layout=ipywidgets.Layout(width='140px'),
        )
        self.w_ping_step.observe(self._on_ping_step_change, names='value')
        
        # Step buttons
        self.w_step_prev = ipywidgets.Button(
            description="◀ Prev",
            tooltip="Step to previous ping (by step amount)",
            layout=ipywidgets.Layout(width='80px'),
        )
        self.w_step_next = ipywidgets.Button(
            description="Next ▶",
            tooltip="Step to next ping (by step amount)",
            layout=ipywidgets.Layout(width='80px'),
        )
        self.w_step_prev.on_click(self._step_prev)
        self.w_step_next.on_click(self._step_next)
        
        # Autoplay controls
        self._autoplay_active = False
        self._autoplay_timer = None
        self._autoplay_last_time = None
        
        # Real fps display
        self.w_real_fps = ipywidgets.Label(
            value="real: --",
            layout=ipywidgets.Layout(width='100px'),
        )
        
        self.w_play_button = ipywidgets.Button(
            description="▶ Play",
            tooltip="Start/stop autoplay",
            layout=ipywidgets.Layout(width='80px'),
        )
        self.w_play_button.on_click(self._toggle_autoplay)
        
        self.w_play_fps = ipywidgets.FloatText(
            value=2.0,
            description="fps",
            layout=ipywidgets.Layout(width='160px'),
        )
        
        self.w_use_ping_time = ipywidgets.Checkbox(
            value=False,
            description="ping time",
            tooltip="Use actual ping timestamps for timing (fps becomes speed multiplier)",
            indent=False,
        )
        
        # === VIDEO EXPORT CONTROLS ===
        self.w_video_frames = ipywidgets.IntText(
            value=100,
            description="frames",
            tooltip="Number of frames to export (0 = all pings)",
            layout=ipywidgets.Layout(width='140px'),
        )
        self.w_video_fps = ipywidgets.FloatText(
            value=10.0,
            description="video fps",
            tooltip="Frame rate of output video",
            layout=ipywidgets.Layout(width='140px'),
        )
        self.w_video_format = ipywidgets.Dropdown(
            description="format",
            options=["mp4", "gif", "webm", "avi"],
            value="mp4",
            layout=ipywidgets.Layout(width='130px'),
        )
        self.w_video_filename = ipywidgets.Text(
            value="wci_video",
            description="filename",
            layout=ipywidgets.Layout(width='200px'),
        )
        self.w_export_video = ipywidgets.Button(
            description="Export Video",
            tooltip="Export video from current position",
            layout=ipywidgets.Layout(width='120px'),
        )
        self.w_export_video.on_click(self._export_video)
        self.w_video_status = ipywidgets.Label(value="")
        
        self.w_video_ping_time = ipywidgets.Checkbox(
            value=False,
            description="ping time",
            tooltip="Use ping timestamps for video timing (fps becomes speed multiplier)",
            indent=False,
        )
        
        self.w_video_live = ipywidgets.Checkbox(
            value=True,
            description="live",
            tooltip="Show live preview during capture (slower but shows progress)",
            indent=False,
        )
    
    def _make_graphics_widget(self) -> None:
        """Create the PyQtGraph graphics widget."""
        self.graphics = GraphicsLayoutWidget(
            css_width=f"{self.widget_width_px}px",
            css_height=f"{self.widget_height_px}px"
        )
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, self.widget_height_px)
        if hasattr(self.graphics, "gfxView"):
            self.graphics.gfxView.setBackground("w")
    
    def _update_grid_layout(self) -> None:
        """Update the graphics widget to reflect current grid layout."""
        self.graphics.clear()
        
        n_visible = self.grid_rows * self.grid_cols
        
        for i, slot in enumerate(self.slots):
            slot.set_visible(i < n_visible)
        
        master_plot = None
        for i in range(n_visible):
            row = i // self.grid_cols
            col = i % self.grid_cols
            slot = self.slots[i]
            
            plot: pg.PlotItem = self.graphics.addPlot(row=row, col=col * 2)
            slot.plot_item = plot
            
            title = str(slot.channel_key) if slot.channel_key else f"Slot {i+1}"
            plot.setTitle(title)
            plot.setLabel("left", "Depth (m)" if col == 0 else "")
            plot.setLabel("bottom", "Horizontal distance" if row == self.grid_rows - 1 else "")
            plot.getViewBox().invertY(True)
            plot.getViewBox().setBackgroundColor("w")
            plot.getViewBox().setAspectLocked(True, ratio=1)
            
            # Create image item
            image_item = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image_item)
            slot.image_item = image_item
            
            # Create colorbar
            try:
                colorbar = pg.ColorBarItem(
                    label="(dB)",
                    values=(self.args_plot["vmin"], self.args_plot["vmax"]),
                    interactive=True,
                )
                colorbar.setImageItem(image_item, insert_in=plot)
                if hasattr(colorbar, "setColorMap"):
                    colorbar.setColorMap(self._colormap)
                slot.colorbar = colorbar
                slot.color_levels = (self.args_plot["vmin"], self.args_plot["vmax"])
                
                # Connect colorbar changes
                if hasattr(colorbar, 'sigLevelsChanged'):
                    colorbar.sigLevelsChanged.connect(
                        lambda cb=colorbar, s=slot: self._on_colorbar_levels_changed(s, cb)
                    )
            except AttributeError:
                slot.colorbar = None
            
            # Create time offset text item
            time_text = pg.TextItem(
                text="",
                color=(0, 0, 0),
                anchor=(0, 0),
            )
            time_text.setPos(0, 0)
            time_text.hide()
            plot.addItem(time_text)
            slot.time_offset_text = time_text
            
            # Create crosshairs
            pen_cross = pg.mkPen(color='r', width=1, style=QtCore.Qt.PenStyle.DashLine)
            slot.crosshair_v = pg.InfiniteLine(angle=90, pen=pen_cross)
            slot.crosshair_h = pg.InfiniteLine(angle=0, pen=pen_cross)
            slot.crosshair_v.hide()
            slot.crosshair_h.hide()
            plot.addItem(slot.crosshair_v)
            plot.addItem(slot.crosshair_h)
            
            # Link axes
            if master_plot is None:
                master_plot = plot
            else:
                plot.setXLink(master_plot)
                plot.setYLink(master_plot)
        
        self._connect_scene_events()
    
    def _connect_scene_events(self) -> None:
        """Connect mouse events."""
        gfx_view = getattr(self.graphics, "gfxView", None)
        scene = gfx_view.scene() if gfx_view is not None else None
        if scene is None:
            return
        
        # Disconnect existing
        if hasattr(self, '_scene_move_connection') and self._scene_move_connection:
            try:
                scene.sigMouseMoved.disconnect(self._handle_scene_move)
            except (TypeError, RuntimeError):
                pass
        
        self._scene_move_connection = scene.sigMouseMoved.connect(self._handle_scene_move)
    
    def _assemble_layout(self) -> None:
        """Assemble the final widget layout with tabbed interface."""
        n_visible = self.grid_rows * self.grid_cols
        
        # Combined slot selectors and ping sliders for visible slots only
        slot_controls = []
        for i in range(n_visible):
            slot_box = ipywidgets.HBox([
                self.slot_selectors[i],
                self.ping_sliders[i],
            ])
            slot_controls.append(slot_box)
        self.slot_selector_box = ipywidgets.VBox(slot_controls)
        
        # Tab buttons row (grid selector + channel quick buttons)
        tab_box = ipywidgets.HBox([self.w_layout] + self.tab_buttons)
        
        # === TABBED SETTINGS (right side) ===
        # Tab 1: Rendering settings (color, sync, crosshair, warn)
        tab_render = ipywidgets.VBox([
            ipywidgets.HBox([self.w_vmin, self.w_vmax]),
            ipywidgets.HBox([self.w_wci_value, self.w_wci_render]),
            ipywidgets.HBox([self.w_horizontal_pixels]),
            ipywidgets.HBox([self.w_time_sync, self.w_crosshair, self.w_time_warning]),
        ])
        
        # Tab 2: Stacking settings  
        tab_stack = ipywidgets.HBox([
            self.w_stack, self.w_stack_step, self.w_mp_cores, self.w_stack_linear,
        ])
        
        # Tab 3: Timing info
        tab_timing = ipywidgets.HBox([self.w_proctime, self.w_procrate])
        
        # Tab 4: Playback controls
        tab_playback = ipywidgets.VBox([
            ipywidgets.HBox([self.w_ping_step, self.w_step_prev, self.w_step_next]),
            ipywidgets.HBox([self.w_play_button, self.w_play_fps, self.w_use_ping_time, self.w_real_fps]),
        ])
        
        # Tab 5: Video export
        tab_video = ipywidgets.VBox([
            ipywidgets.HBox([self.w_video_frames, self.w_video_fps, self.w_video_format]),
            ipywidgets.HBox([self.w_video_filename, self.w_video_ping_time, self.w_video_live, self.w_export_video]),
            self.w_video_status,
        ])
        
        # Create Tab widget for settings
        self.settings_tabs = ipywidgets.Tab(
            children=[tab_render, tab_stack, tab_timing, tab_playback, tab_video],
        )
        self.settings_tabs.set_title(0, 'Render')
        self.settings_tabs.set_title(1, 'Stack')
        self.settings_tabs.set_title(2, 'Timing')
        self.settings_tabs.set_title(3, 'Playback')
        self.settings_tabs.set_title(4, 'Video')
        
        # === MAIN CONTROLS ROW: slots/ref time on left, tabs on right ===
        main_left = ipywidgets.VBox([
            self.slot_selector_box,
            ipywidgets.HBox([self.w_ref_time, self.w_fix_xy, self.w_unfix_xy]),
        ])
        
        main_controls_row = ipywidgets.HBox([
            main_left,
            self.settings_tabs,
        ])
        
        # Progress
        if self.display_progress:
            progress_box = ipywidgets.HBox([self.progress])
        else:
            progress_box = ipywidgets.HBox([])
        
        self.layout = ipywidgets.VBox([
            ipywidgets.HBox([self.graphics]),
            progress_box,
            tab_box,
            main_controls_row,
            self.hover_label,
            self.output,
        ])
    
    def _on_layout_change(self, change: Dict[str, Any]) -> None:
        """Handle grid layout change."""
        new_rows, new_cols = change['new']
        if (new_rows, new_cols) == (self.grid_rows, self.grid_cols):
            return
        
        # Capture view range
        current_range = self._capture_current_view_range()
        
        self.grid_rows, self.grid_cols = new_rows, new_cols
        self._update_grid_layout()
        
        # Dynamically update slot selector box to show only visible slots
        self._update_slot_selector_visibility()
        
        # Restore view
        if current_range is not None:
            self._restore_view_range(current_range)
        
        # Update all visible slots
        self._update_all_visible()
    
    def _update_slot_selector_visibility(self) -> None:
        """Update slot selector box to show only slots for current grid."""
        n_visible = self.grid_rows * self.grid_cols
        slot_controls = []
        for i in range(n_visible):
            slot_box = ipywidgets.HBox([
                self.slot_selectors[i],
                self.ping_sliders[i],
            ])
            slot_controls.append(slot_box)
        self.slot_selector_box.children = slot_controls
    
    def _on_slot_change(self, slot_idx: int, change: Dict[str, Any]) -> None:
        """Handle slot channel assignment change.
        
        Note: Changing channel does NOT update the reference timestamp.
        Only explicit ping changes update the reference.
        """
        new_key = change['new']
        slot = self.slots[slot_idx]
        
        current_range = self._capture_current_view_range()
        self._ignore_range_changes = True
        
        try:
            slot.assign_channel(new_key)
            
            # Update ping slider max
            pings = slot.get_pings()
            max_ping = max(0, len(pings) - 1) if pings else 0
            self.ping_sliders[slot_idx].max = max_ping
            
            # Sync to reference timestamp when switching channels
            # NOTE: Changing channel does NOT update reference timestamp
            if self.time_sync_enabled and pings and len(pings) > 0:
                ref_timestamp = self._reference_timestamp
                
                if ref_timestamp is not None:
                    # Find closest ping to reference time
                    closest_idx = slot.find_closest_ping_index(ref_timestamp)
                    slot.ping_index = closest_idx
                    
                    # Block syncing to avoid recursive updates
                    self._syncing = True
                    try:
                        self.ping_sliders[slot_idx].value = closest_idx
                    finally:
                        self._syncing = False
                    
                    # Calculate time offset from reference (show as delta in image)
                    closest_ts = slot.get_timestamp(closest_idx)
                    if closest_ts is not None:
                        slot.time_offset = closest_ts - ref_timestamp
                    else:
                        slot.time_offset = 0.0
                else:
                    # No reference yet - just use ping 0
                    self.ping_sliders[slot_idx].value = min(slot.ping_index, max_ping)
                    slot.time_offset = 0.0
            else:
                self.ping_sliders[slot_idx].value = min(slot.ping_index, max_ping)
            
            if slot.is_visible:
                self._update_slot(slot)
                self._update_time_offset_text(slot)
                self._process_qt_events()
        finally:
            self._ignore_range_changes = False
        
        if current_range is not None:
            self._restore_view_range(current_range)
        
        self._request_remote_draw()
    
    def _on_ping_change(self, slot_idx: int, change: Dict[str, Any]) -> None:
        """Handle per-slot ping number change.
        
        This is an EXPLICIT ping change by the user, so we update
        the global reference timestamp.
        """
        if self._syncing:
            return
        
        new_ping = change['new']
        slot = self.slots[slot_idx]
        slot.ping_index = new_ping
        slot.stored_ping_index = new_ping  # Store as explicit ping selection
        
        # Update the global reference timestamp (explicit ping change)
        new_timestamp = slot.get_timestamp()
        if new_timestamp is not None:
            self._reference_timestamp = new_timestamp
            self._update_ref_time_display()
        
        # Update this slot
        if slot.is_visible:
            self._update_slot(slot)
        
        # Time synchronization - sync other slots to this reference
        if self.time_sync_enabled:
            self._sync_other_slots(slot_idx)
        
        # Notify registered callbacks (for connected viewers)
        for callback in self._ping_change_callbacks:
            try:
                callback()
            except Exception:
                pass  # Don't let callback errors block ping updates
        
        self._request_remote_draw()
    
    def _sync_other_slots(self, reference_slot_idx: int) -> None:
        """Synchronize other slots to the global reference timestamp."""
        ref_timestamp = self._reference_timestamp
        
        if ref_timestamp is None:
            return
        
        self._syncing = True
        self._reference_slot_idx = reference_slot_idx
        
        try:
            n_visible = self.grid_rows * self.grid_cols
            for i in range(n_visible):
                slot = self.slots[i]
                if slot.channel_key is None:
                    continue
                
                if i == reference_slot_idx:
                    # Reference slot has zero offset
                    slot.time_offset = 0.0
                    self._update_time_offset_text(slot)
                    continue
                
                # Find closest ping to reference timestamp
                closest_idx = slot.find_closest_ping_index(ref_timestamp)
                closest_ts = slot.get_timestamp(closest_idx)
                
                if closest_ts is not None:
                    slot.time_offset = closest_ts - ref_timestamp
                else:
                    slot.time_offset = 0.0
                
                # Update slider without triggering recursion
                slot.ping_index = closest_idx
                self.ping_sliders[i].value = closest_idx
                
                # Update display
                if slot.is_visible:
                    self._update_slot(slot)
                
                # Update time offset text
                self._update_time_offset_text(slot)
        finally:
            self._syncing = False
        
        self._request_remote_draw()
    
    def _update_ref_time_display(self) -> None:
        """Update the reference time display widget."""
        if self._reference_timestamp is not None:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(self._reference_timestamp, tz=timezone.utc)
            self.w_ref_time.value = dt.strftime("%H:%M:%S.%f")[:-3]
        else:
            self.w_ref_time.value = ""
    
    def _update_time_offset_text(self, slot: WCISlot) -> None:
        """Update the time offset text display for a slot.
        
        Always shows delta if > 0, even with single slot (grid 1x1).
        """
        if slot.time_offset_text is None:
            return
        
        # No time sync - hide text
        if not self.time_sync_enabled:
            slot.time_offset_text.hide()
            return
        
        offset = slot.time_offset
        abs_offset = abs(offset)
        
        # Don't show for very small offsets
        if abs_offset < 0.01:
            slot.time_offset_text.hide()
            return
        
        # Format text (compact) - always show if > 0
        if abs_offset >= self.time_warning_threshold:
            # Red/orange for large offsets - warning
            text = f"\u0394{offset:+.1f}s"
            slot.time_offset_text.setColor((200, 60, 60, 220))  # Semi-transparent red
        else:
            # Gray for small offsets - subtle
            text = f"\u0394{offset:+.2f}s"
            slot.time_offset_text.setColor((80, 80, 80, 180))  # Semi-transparent gray
        
        slot.time_offset_text.setText(text)
        
        # Position in top-right of plot (less intrusive)
        if slot.wci_extent is not None:
            x0, x1, y0, y1 = slot.wci_extent
            # Place near top-right corner with small offset
            margin_x = (x1 - x0) * 0.02
            slot.time_offset_text.setPos(x1 - margin_x, min(y0, y1))
            slot.time_offset_text.setAnchor((1, 0))  # Anchor to top-right
        
        slot.time_offset_text.show()
    
    def _on_global_color_change(self, change: Dict[str, Any]) -> None:
        """Handle global color scale change - override all slot levels."""
        vmin = float(self.w_vmin.value)
        vmax = float(self.w_vmax.value)
        
        self.args_plot["vmin"] = vmin
        self.args_plot["vmax"] = vmax
        
        # Update all visible slots
        n_visible = self.grid_rows * self.grid_cols
        for i in range(n_visible):
            slot = self.slots[i]
            slot.color_levels = (vmin, vmax)
            
            if slot.image_item is not None:
                slot.image_item.setLevels((vmin, vmax))
            
            if slot.colorbar is not None:
                try:
                    slot.colorbar.blockSignals(True)
                    slot.colorbar.setLevels((vmin, vmax))
                finally:
                    slot.colorbar.blockSignals(False)
        
        self._request_remote_draw()
    
    def _on_colorbar_levels_changed(self, slot: WCISlot, colorbar: pg.ColorBarItem) -> None:
        """Handle per-slot colorbar level changes."""
        levels = colorbar.levels()
        slot.color_levels = levels
        
        if slot.image_item is not None:
            slot.image_item.setLevels(levels)
    
    def _on_global_param_change(self, change: Dict[str, Any]) -> None:
        """Handle changes to global image builder parameters."""
        self._sync_builder_args()
        self._update_all_visible()
    
    def _sync_builder_args(self) -> None:
        """Sync widget values to image builder args."""
        self.args_imagebuilder["linear_mean"] = self.w_stack_linear.value
        self.args_imagebuilder["wci_value"] = self.w_wci_value.value
        self.args_imagebuilder["wci_render"] = self.w_wci_render.value
        self.args_imagebuilder["horizontal_pixels"] = self.w_horizontal_pixels.value
        self.args_imagebuilder["mp_cores"] = self.w_mp_cores.value
        
        # Update all slot imagebuilders
        for slot in self.slots:
            if slot.imagebuilder is not None:
                slot.imagebuilder.update_args(**self.args_imagebuilder)
    
    def _fix_xy(self, _event: Any = None) -> None:
        """Fix the view to current range."""
        master = self._get_master_plot()
        if master is None:
            return
        
        view = master.getViewBox()
        (xmin, xmax), (ymin, ymax) = view.viewRange()
        xmin, xmax = sorted((float(xmin), float(xmax)))
        ymin, ymax = sorted((float(ymin), float(ymax)))
        
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            return
        
        self.args_imagebuilder["hmin"] = xmin
        self.args_imagebuilder["hmax"] = xmax
        self.args_imagebuilder["vmin"] = ymin
        self.args_imagebuilder["vmax"] = ymax
        
        self._update_all_visible()
    
    def _unfix_xy(self, _event: Any = None) -> None:
        """Unfix the view."""
        for key in ("hmin", "hmax", "vmin", "vmax"):
            self.args_imagebuilder[key] = None
        self._update_all_visible()
    
    # === PLAYBACK METHODS ===
    
    def _step_prev(self, _event: Any = None) -> None:
        """Step backward by ping step amount (applies to first visible slot)."""
        step = max(1, self.w_ping_step.value)
        slot = self.slots[0]
        pings = slot.get_pings()
        if not pings:
            return
        
        new_idx = max(0, self.ping_sliders[0].value - step)
        self.ping_sliders[0].value = new_idx
    
    def _step_next(self, _event: Any = None) -> None:
        """Step forward by ping step amount (applies to first visible slot)."""
        step = max(1, self.w_ping_step.value)
        slot = self.slots[0]
        pings = slot.get_pings()
        if not pings:
            return
        
        max_idx = len(pings) - 1
        new_idx = min(max_idx, self.ping_sliders[0].value + step)
        self.ping_sliders[0].value = new_idx
    
    def _toggle_autoplay(self, _event: Any = None) -> None:
        """Toggle autoplay on/off."""
        if self._autoplay_active:
            self._stop_autoplay()
        else:
            self._start_autoplay()
    
    def _start_autoplay(self) -> None:
        """Start the autoplay timer using asyncio for thread-safe updates."""
        import asyncio
        
        self._autoplay_active = True
        self.w_play_button.description = "Stop"
        self._autoplay_last_time = None
        
        async def play_loop():
            while self._autoplay_active:
                t0 = time_module.time()
                
                # Read settings dynamically each iteration
                use_ping_time = self.w_use_ping_time.value
                speed_mult = max(0.1, self.w_play_fps.value)  # fps or speed multiplier
                step = max(1, self.w_ping_step.value)
                
                slot = self.slots[0]
                pings = slot.get_pings()
                interval = 1.0 / speed_mult  # default interval
                
                if pings:
                    max_idx = len(pings) - 1
                    current_idx = self.ping_sliders[0].value
                    new_idx = current_idx + step
                    
                    if new_idx > max_idx:
                        # Loop back to start
                        new_idx = 0
                    
                    # Calculate interval based on mode
                    if use_ping_time:
                        # Use actual ping timestamp difference
                        current_ts = slot.get_timestamp(current_idx)
                        next_ts = slot.get_timestamp(new_idx)
                        if current_ts is not None and next_ts is not None:
                            ping_dt = abs(next_ts - current_ts)
                            interval = ping_dt / speed_mult  # speed_mult is speedup factor
                    
                    # Update slider value (triggers redraw via observer)
                    self.ping_sliders[0].value = new_idx
                
                # Calculate real fps
                if self._autoplay_last_time is not None:
                    real_interval = t0 - self._autoplay_last_time
                    if real_interval > 0:
                        real_fps = 1.0 / real_interval
                        self.w_real_fps.value = f"real: {real_fps:.1f}"
                self._autoplay_last_time = t0
                
                await asyncio.sleep(max(0.01, interval))
        
        # Get or create event loop and schedule the coroutine
        try:
            loop = asyncio.get_running_loop()
            self._autoplay_task = loop.create_task(play_loop())
        except RuntimeError:
            # No running loop - create one using ensure_future
            self._autoplay_task = asyncio.ensure_future(play_loop())
    
    def _stop_autoplay(self) -> None:
        """Stop the autoplay timer."""
        self._autoplay_active = False
        self.w_play_button.description = "Play"
        self.w_real_fps.value = "real: --"
        if hasattr(self, '_autoplay_task') and self._autoplay_task:
            self._autoplay_task.cancel()
            self._autoplay_task = None
    
    def _on_ping_step_change(self, change: Dict[str, Any]) -> None:
        """Update all ping sliders' step property when ping step changes."""
        new_step = max(1, change['new'])
        for slider in self.ping_sliders:
            slider.step = new_step
    
    def _export_video(self, _event: Any = None) -> None:
        """Export video from current position.
        
        Captures frames by stepping through pings and saves as video file.
        Requires imageio (and imageio-ffmpeg for mp4/avi/webm).
        """
        try:
            import imageio.v3 as iio
        except ImportError:
            try:
                import imageio as iio
            except ImportError:
                self.w_video_status.value = "Error: pip install imageio imageio-ffmpeg"
                return
        
        slot = self.slots[0]
        pings = slot.get_pings()
        if not pings:
            self.w_video_status.value = "Error: No pings available"
            return
        
        # Get parameters
        num_frames = self.w_video_frames.value
        if num_frames <= 0:
            num_frames = len(pings)
        
        video_fps = max(1, self.w_video_fps.value)  # fps or speed multiplier
        step = max(1, self.w_ping_step.value)
        fmt = self.w_video_format.value
        filename = self.w_video_filename.value.strip()
        use_ping_time = self.w_video_ping_time.value
        
        if not filename:
            filename = "wci_video"
        
        # Ensure proper extension
        if not filename.endswith(f".{fmt}"):
            filename = f"{filename}.{fmt}"
        
        start_idx = self.ping_sliders[0].value
        max_idx = len(pings) - 1
        
        # Limit frames to avoid hanging
        actual_frames = min(num_frames, (max_idx - start_idx) // step + 1)
        self.w_video_status.value = f"Exporting {actual_frames} frames..."
        
        try:
            # Use pyqtgraph's Qt bindings
            from pyqtgraph.Qt import QtGui
            
            # Get view reference
            if not hasattr(self.graphics, 'gfxView'):
                self.w_video_status.value = "Error: No graphics view available"
                return
            view = self.graphics.gfxView
            
            # Capture frames and durations
            frames = []
            durations = []  # Duration for each frame (in seconds)
            current_idx = start_idx
            prev_timestamp = None
            
            # Option: show live preview during capture
            show_live = getattr(self, 'w_video_live', None)
            show_live = show_live.value if show_live else False
            
            # Disable syncing for faster capture unless showing live
            old_syncing = self._syncing
            if not show_live:
                self._syncing = True
            
            t_start = time_module.time()
            
            for i in range(actual_frames):
                t0 = time_module.time()
                
                # Get timestamp for duration calculation
                if use_ping_time:
                    current_ts = slot.get_timestamp(current_idx)
                    if prev_timestamp is not None and current_ts is not None:
                        dt = abs(current_ts - prev_timestamp) / video_fps
                        durations.append(max(0.01, dt))
                    prev_timestamp = current_ts
                
                # Update ALL visible slots, not just the first one
                for slot_i, s in enumerate(self.slots):
                    if s.is_visible and s.get_pings():
                        s.ping_index = current_idx
                        self._update_slot(s, fast_mode=not show_live)
                        if slot_i < len(self.ping_sliders):
                            self.ping_sliders[slot_i].value = current_idx
                
                # Process Qt events so the display updates
                self._process_qt_events()
                
                # Capture the viewport directly using grab() - this captures exactly what's visible
                try:
                    pixmap = view.grab()
                    image = pixmap.toImage()
                    
                    # Convert QImage to numpy array (PySide6 compatible)
                    width = image.width()
                    height = image.height()
                    ptr = image.bits()
                    if hasattr(image, 'sizeInBytes'):
                        nbytes = image.sizeInBytes()
                    else:
                        nbytes = image.byteCount()
                    if hasattr(ptr, 'tobytes'):
                        arr = np.frombuffer(ptr.tobytes(), dtype=np.uint8)
                    else:
                        ptr.setsize(nbytes)
                        arr = np.array(ptr, dtype=np.uint8)
                    arr = arr.reshape(height, width, 4)
                    # Convert BGRA to RGB
                    frame = arr[:, :, [2, 1, 0]].copy()
                    frames.append(frame)
                except Exception as e:
                    self._syncing = old_syncing
                    self.w_video_status.value = f"Frame capture error: {e}"
                    return
                
                # Step to next ping
                current_idx += step
                if current_idx > max_idx:
                    break
                
                # Show progress with timing info
                t1 = time_module.time()
                fps_current = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
                elapsed = t1 - t_start
                remaining = (actual_frames - i - 1) * (elapsed / (i + 1)) if i > 0 else 0
                self.w_video_status.value = f"Frame {i+1}/{actual_frames} ({fps_current:.1f} fps, ~{remaining:.0f}s left)"
            
            # Restore syncing
            self._syncing = old_syncing
            
            if not frames:
                self.w_video_status.value = "Error: No frames captured"
                return
            
            # Write video
            self.w_video_status.value = f"Writing {filename} ({len(frames)} frames)..."
            self._process_qt_events()
            
            if fmt == "gif":
                # Use Pillow for GIF - much faster than imageio
                try:
                    from PIL import Image
                    pil_frames = [Image.fromarray(f) for f in frames]
                    
                    if use_ping_time and durations:
                        # Convert durations to milliseconds
                        duration_ms = [int(d * 1000) for d in durations]
                        # Pad durations if needed
                        while len(duration_ms) < len(pil_frames):
                            duration_ms.append(int(1000 / video_fps))
                    else:
                        duration_ms = int(1000 / video_fps)
                    
                    pil_frames[0].save(
                        filename,
                        save_all=True,
                        append_images=pil_frames[1:],
                        duration=duration_ms,
                        loop=0
                    )
                except ImportError:
                    # Fall back to imageio legacy API
                    import imageio
                    if use_ping_time and durations:
                        imageio.mimsave(filename, frames, duration=durations)
                    else:
                        imageio.mimsave(filename, frames, duration=1.0/video_fps)
            else:
                # For video formats (mp4, avi, webm), use imageio with ffmpeg plugin
                try:
                    import imageio_ffmpeg
                    import imageio
                    if use_ping_time and durations:
                        avg_duration = sum(durations) / len(durations) if durations else 1.0/video_fps
                        avg_fps = 1.0 / avg_duration if avg_duration > 0 else video_fps
                    else:
                        avg_fps = video_fps
                    
                    # Use imageio with ffmpeg writer explicitly
                    writer = imageio.get_writer(filename, fps=avg_fps, codec='libx264', quality=8)
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()
                except ImportError:
                    self.w_video_status.value = "Error: pip install imageio-ffmpeg for mp4/avi/webm"
                    return
            
            self.w_video_status.value = f"Saved: {filename} ({len(frames)} frames)"
            
            # Restore original position
            self.ping_sliders[0].value = start_idx
            
        except Exception as e:
            self.w_video_status.value = f"Export error: {e}"
    
    def _show_single(self, channel_name: str) -> None:
        """Show a single channel full-size."""
        current_range = self._capture_current_view_range()
        self._ignore_range_changes = True
        
        try:
            need_grid_change = (self.grid_rows, self.grid_cols) != (1, 1)
            
            if need_grid_change:
                self.w_layout.value = (1, 1)
            
            if self.slots[0].channel_key != channel_name:
                self.slots[0].assign_channel(channel_name)
                self.slot_selectors[0].value = channel_name
                
                # Update ping slider
                pings = self.slots[0].get_pings()
                max_ping = max(0, len(pings) - 1) if pings else 0
                self.ping_sliders[0].max = max_ping
            
            if not need_grid_change:
                self._update_slot(self.slots[0])
        finally:
            self._ignore_range_changes = False
        
        if current_range is not None:
            self._restore_view_range(current_range)
        
        self._request_remote_draw()
    
    def _update_all_visible(self) -> None:
        """Update all visible slots."""
        t0 = time_module.time()
        self._sync_builder_args()
        
        n_visible = self.grid_rows * self.grid_cols
        for i in range(n_visible):
            slot = self.slots[i]
            if slot.is_visible and slot.channel_key is not None:
                self._update_slot(slot)
        
        t1 = time_module.time()
        self._process_qt_events()
        self._request_remote_draw()
        t2 = time_module.time()
        
        self._update_timing_fields(t0, t1, t2)
    
    def _update_slot(self, slot: WCISlot, fast_mode: bool = False) -> None:
        """Update a single slot's WCI image.
        
        Args:
            slot: The slot to update
            fast_mode: If True, skip UI updates for faster rendering (used in video export)
        """
        if slot.imagebuilder is None or slot.plot_item is None:
            return
        
        if fast_mode:
            # Fast path: just build image and update display, skip progress/title updates
            try:
                slot.imagebuilder.update_args(**self.args_imagebuilder)
                slot.wci_image, slot.wci_extent = slot.imagebuilder.build(
                    index=slot.ping_index,
                    stack=self.w_stack.value,
                    stack_step=self.w_stack_step.value,
                )
            except Exception:
                return
            self._update_slot_image(slot)
            return
        
        with self.output:
            self.output.clear_output()
            
            # Update title
            title = str(slot.channel_key) if slot.channel_key else f"Slot {slot.slot_idx + 1}"
            slot.plot_item.setTitle(title)
            
            # Build image
            try:
                self.progress.set_description(f"Building {slot.channel_key}...")
                slot.imagebuilder.update_args(**self.args_imagebuilder)
                slot.wci_image, slot.wci_extent = slot.imagebuilder.build(
                    index=slot.ping_index,
                    stack=self.w_stack.value,
                    stack_step=self.w_stack_step.value,
                )
            except Exception as e:
                self.progress.set_description(f"Error: {e}")
                return
            
            self.progress.set_description("Idle")
            
            # Update image display
            self._update_slot_image(slot)
            
            # Update time offset text position
            self._update_time_offset_text(slot)
    
    def _update_slot_image(self, slot: WCISlot) -> None:
        """Update the image display for a slot."""
        if slot.image_item is None or slot.wci_image is None or slot.wci_extent is None:
            if slot.image_item is not None:
                slot.image_item.hide()
            return
        
        array = slot.wci_image.transpose()
        if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
            slot.image_item.hide()
            return
        
        slot.image_item.setImage(array, autoLevels=False)
        
        x0, x1, y0, y1 = slot.wci_extent
        vb = slot.plot_item.getViewBox()
        if vb.yInverted():
            y0, y1 = y1, y0
        
        width = x1 - x0
        height = y1 - y0
        if width == 0 or height == 0:
            slot.image_item.hide()
            return
        
        rect = QtCore.QRectF(x0, y0, width, height)
        slot.image_item.setRect(rect)
        
        # Set colormap
        if hasattr(slot.image_item, "setColorMap"):
            slot.image_item.setColorMap(self._colormap)
        else:
            lut = self._colormap.getLookupTable(256)
            slot.image_item.setLookupTable(lut)
        
        # Set color levels (per-slot)
        if slot.color_levels is not None:
            vmin, vmax = slot.color_levels
        else:
            vmin = float(self.w_vmin.value)
            vmax = float(self.w_vmax.value)
        
        slot.image_item.setLevels((vmin, vmax))
        
        # Update view on first draw
        if self._first_draw or self.args_imagebuilder["hmin"] is None:
            vb.setXRange(x0, x1, padding=0)
        if self._first_draw or self.args_imagebuilder["vmin"] is None:
            vb.setYRange(min(y0, y1), max(y0, y1), padding=0)
        self._first_draw = False
        
        slot.image_item.show()
    
    def _handle_scene_move(self, pos: QtCore.QPointF) -> None:
        """Handle mouse movement for hover display and crosshairs."""
        found_slot = None
        data_pos = None
        
        for slot in self.slots:
            if not slot.is_visible or slot.plot_item is None:
                continue
            
            vb = slot.plot_item.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                point = vb.mapSceneToView(pos)
                found_slot = slot
                data_pos = (point.x(), point.y())
                value = self._sample_value(slot, point.x(), point.y())
                label = (
                    f"<b>Slot {slot.slot_idx + 1}</b> | "
                    f"<b>x</b>: {point.x():0.2f} | <b>y</b>: {point.y():0.2f} | "
                    f"<b>value</b>: {value:0.2f}" if value is not None else "--"
                )
                self.hover_label.value = label
                break
        
        # Update crosshairs
        if self._crosshair_enabled and found_slot is not None and data_pos is not None:
            self._crosshair_position = data_pos
            self._update_crosshairs()
        else:
            # Hide crosshairs when mouse leaves plot area or crosshairs disabled
            self._hide_crosshairs()
        
        if found_slot is None:
            self.hover_label.value = "&nbsp;"
    
    def _update_crosshairs(self) -> None:
        """Update crosshair positions on all visible slots."""
        if self._crosshair_position is None:
            self._hide_crosshairs()
            return
        
        x, y = self._crosshair_position
        n_visible = self.grid_rows * self.grid_cols
        for i in range(n_visible):
            slot = self.slots[i]
            if slot.crosshair_v is not None:
                slot.crosshair_v.setPos(x)
                slot.crosshair_v.show()
            if slot.crosshair_h is not None:
                slot.crosshair_h.setPos(y)
                slot.crosshair_h.show()
    
    def _hide_crosshairs(self) -> None:
        """Hide all crosshairs."""
        for slot in self.slots:
            if slot.crosshair_v is not None:
                slot.crosshair_v.hide()
            if slot.crosshair_h is not None:
                slot.crosshair_h.hide()
    
    def _sample_value(self, slot: WCISlot, x_coord: float, y_coord: float) -> Optional[float]:
        """Sample value at coordinates in a slot."""
        if slot.wci_image is None or slot.wci_extent is None:
            return None
        
        x0, x1, y0, y1 = slot.wci_extent
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0 or dy == 0:
            return None
        
        col = (x_coord - x0) / dx * (slot.wci_image.shape[0] - 1)
        row = (y_coord - y0) / dy * (slot.wci_image.shape[1] - 1)
        
        if 0 <= col < slot.wci_image.shape[0] and 0 <= row < slot.wci_image.shape[1]:
            return float(slot.wci_image[int(col), int(row)])
        return None
    
    def _get_master_plot(self) -> Optional[pg.PlotItem]:
        """Get the master plot (first visible slot with a plot)."""
        for slot in self.slots:
            if slot.is_visible and slot.plot_item is not None:
                return slot.plot_item
        return None
    
    def _capture_current_view_range(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Capture current view range from master plot."""
        master = self._get_master_plot()
        if master is None:
            return None
        vb = master.getViewBox()
        return vb.viewRange()
    
    def _restore_view_range(self, view_range: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        """Restore view range to master plot."""
        master = self._get_master_plot()
        if master is None:
            return
        vb = master.getViewBox()
        (xmin, xmax), (ymin, ymax) = view_range
        vb.setXRange(xmin, xmax, padding=0)
        vb.setYRange(ymin, ymax, padding=0)
    
    def _update_timing_fields(self, t0: float, t1: float, t2: float) -> None:
        """Update timing display fields."""
        build_time = t1 - t0
        draw_time = t2 - t1
        total_time = t2 - t0
        self.w_proctime.value = f"{build_time:0.3f} / {draw_time:0.3f} / [{total_time:0.3f}] s"
        r1 = 1 / build_time if build_time > 0 else 0
        r2 = 1 / draw_time if draw_time > 0 else 0
        r3 = 1 / total_time if total_time > 0 else 0
        self.w_procrate.value = f"r1: {r1:0.1f} / r2: {r2:0.1f} / r3: [{r3:0.1f}] Hz"
    
    @staticmethod
    def _process_qt_events() -> None:
        """Process pending Qt events."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
    
    def _request_remote_draw(self) -> None:
        """Request a remote draw update."""
        request_draw = getattr(self.graphics, "request_draw", None)
        if callable(request_draw):
            request_draw()
    
    def process_events(self) -> None:
        """Process pending Qt events for responsiveness."""
        self._process_qt_events()
    
    def redraw(self, force: bool = True) -> None:
        """Force a redraw of the widget."""
        self._process_qt_events()
        if force:
            self._force_send_frame()
        else:
            self._request_remote_draw()
    
    def _force_send_frame(self) -> None:
        """Force send frame to browser."""
        gfx_view = getattr(self.graphics, "gfxView", None)
        if gfx_view is None:
            return
        
        # Grab the rendered image
        img = gfx_view.grab()
        if img.isNull():
            return
        
        # Convert to bytes
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        img.save(buffer, "PNG")
        buffer.close()
        
        # Send via widget's data stream if available
        if hasattr(self.graphics, "_send_frame"):
            self.graphics._send_frame(buffer.data().data())
        else:
            self._request_remote_draw()
    
    def set_widget_height(self, height_px: int) -> None:
        """Set the widget height."""
        self.widget_height_px = height_px
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, height_px)
    
    # =========================================================================
    # Ping change callbacks (for connected viewers)
    # =========================================================================
    
    def register_ping_change_callback(self, callback: Any) -> None:
        """Register a callback to be called when ping changes.
        
        The callback should be a callable with no arguments.
        Useful for connecting echogram viewers to update their pinglines.
        
        Args:
            callback: A callable to be invoked on ping change.
        """
        if callback not in self._ping_change_callbacks:
            self._ping_change_callbacks.append(callback)
    
    def unregister_ping_change_callback(self, callback: Any) -> None:
        """Unregister a previously registered ping change callback.
        
        Args:
            callback: The callback to remove.
        """
        if callback in self._ping_change_callbacks:
            self._ping_change_callbacks.remove(callback)
    
    # =========================================================================
    # Compatibility properties for connect_pingviewer
    # =========================================================================
    
    @property
    def w_index(self) -> ipywidgets.IntSlider:
        """Return ping slider for first visible slot (compatibility with single-channel viewer)."""
        for i, slot in enumerate(self.slots):
            if slot.is_visible:
                return self.ping_sliders[i]
        return self.ping_sliders[0]
    
    @property
    def imagebuilder(self) -> Optional[mi.ImageBuilder]:
        """Return imagebuilder for first visible slot (compatibility with single-channel viewer)."""
        for slot in self.slots:
            if slot.is_visible and slot.imagebuilder is not None:
                return slot.imagebuilder
        return None
