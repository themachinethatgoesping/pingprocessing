"""Enhanced PyQtGraph-based multi-echogram viewer with grid layout and lazy updates.

Features:
- Grid layout selector (1, 2, 2x2, 3x2, 4x2)
- Per-slot dropdown to select which echogram/frequency to display
- Visibility-based updates (inactive echograms don't update until shown)
- Synchronized crosshair for target investigation across frequencies
- Tab-based quick access for single echogram view
- Lazy loading pattern for performance
- Interactive parameter editing (add, move, delete points)
"""
from __future__ import annotations

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import ipywidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.jupyter import GraphicsLayoutWidget
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

import themachinethatgoesping as theping
from . import pyqtgraph_helpers as pgh


def _get_axis_names(echogram):
    """Get x_axis_name and y_axis_name from echogram (old or new builder)."""
    if hasattr(echogram, 'coord_system'):
        return echogram.coord_system.x_axis_name, echogram.coord_system.y_axis_name
    return echogram.x_axis_name, echogram.y_axis_name


class DraggableScatterPlotItem(pg.ScatterPlotItem):
    """ScatterPlotItem that supports dragging individual points."""
    
    # Signal emitted when a point is dragged: (point_index, new_x, new_y)
    sigPointDragged = QtCore.Signal(int, float, float)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dragging_point_idx: Optional[int] = None
        self._drag_start_pos: Optional[QtCore.QPointF] = None
    
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        """Handle mouse press - start drag if on a point."""
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        
        # Find clicked point
        pos = ev.pos()
        pts = self.pointsAt(pos)
        if len(pts) > 0:
            self._dragging_point_idx = pts[0].index()
            self._drag_start_pos = pos
            ev.accept()
        else:
            ev.ignore()
    
    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        """Handle mouse move - drag point if dragging."""
        if self._dragging_point_idx is None:
            ev.ignore()
            return
        
        # Map to view coordinates
        vb = self.getViewBox()
        if vb is None:
            ev.ignore()
            return
        
        scene_pos = ev.scenePos()
        view_pos = vb.mapSceneToView(scene_pos)
        
        # Emit signal with new position (y only changes, x stays same for order)
        self.sigPointDragged.emit(self._dragging_point_idx, view_pos.x(), view_pos.y())
        ev.accept()
    
    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        """Handle mouse release - end drag."""
        if self._dragging_point_idx is not None:
            self._dragging_point_idx = None
            self._drag_start_pos = None
            ev.accept()
        else:
            ev.ignore()


class SafePolyLineROI(pg.PolyLineROI):
    """PolyLineROI subclass that disables right-click context menu on handles.
    
    The default PolyLineROI handles can crash when right-clicked due to
    internal state issues. This subclass intercepts and ignores right-click
    events to prevent crashes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override hoverEvent and context menu on all handles
        self._disable_handle_context_menus()
    
    def _disable_handle_context_menus(self):
        """Disable context menus on all handles to prevent crashes."""
        for handle in self.handles:
            handle_item = handle.get('item')
            if handle_item is not None:
                # Override the mouseClickEvent to ignore right clicks
                original_click = handle_item.mouseClickEvent
                def safe_click(ev, orig=original_click):
                    if ev.button() == QtCore.Qt.MouseButton.RightButton:
                        ev.accept()  # Accept but do nothing
                        return
                    orig(ev)
                handle_item.mouseClickEvent = safe_click
    
    def addHandle(self, *args, **kwargs):
        """Override to disable context menu on newly added handles."""
        result = super().addHandle(*args, **kwargs)
        self._disable_handle_context_menus()
        return result
    
    def setPoints(self, points, closed=None):
        """Override to disable context menu on handles after setting points."""
        result = super().setPoints(points, closed)
        self._disable_handle_context_menus()
        return result


class EchogramSlot:
    """Manages a single echogram display slot with lazy loading."""
    
    def __init__(self, slot_idx: int, parent: 'EchogramViewerMultiChannel'):
        self.slot_idx = slot_idx
        self.parent = parent
        self.echogram_key: Optional[str] = None  # Key into parent.echograms dict
        self.is_visible = False
        self.needs_update = False  # Dirty flag
        
        # Per-layer color scales (stored when switching between layers)
        self.background_levels: Optional[Tuple[float, float]] = None
        self.layer_levels: Optional[Tuple[float, float]] = None
        
        # Active colorbar layer: 'background' or 'layer'
        self.active_colorbar_layer: str = 'background'
        
        # Image data cache - keyed by echogram_key to support re-assignment
        self._image_cache: Dict[str, Dict[str, Any]] = {}
        
        # Image data (current)
        self.background_image: Optional[np.ndarray] = None
        self.background_extent: Optional[Tuple[float, float, float, float]] = None
        self.high_res_image: Optional[np.ndarray] = None
        self.high_res_extent: Optional[Tuple[float, float, float, float]] = None
        self.layer_image: Optional[np.ndarray] = None
        self.layer_extent: Optional[Tuple[float, float, float, float]] = None
        
        # PyQtGraph items (set by parent when creating plots)
        self.plot_item: Optional[pg.PlotItem] = None
        self.image_layers: Dict[str, pg.ImageItem] = {}
        self.colorbar: Optional[pg.ColorBarItem] = None
        self.layer_colorbar: Optional[pg.ColorBarItem] = None
        self.crosshair_v: Optional[pg.InfiniteLine] = None
        self.crosshair_h: Optional[pg.InfiniteLine] = None
        self.pingline: Optional[pg.InfiniteLine] = None
        
        # Station time markers (dict of station_name -> {'start_line', 'end_line', 'region', 'label'})
        self.station_markers: Dict[str, Dict[str, Any]] = {}
    
    def mark_dirty(self):
        """Mark that data needs refresh when shown."""
        self.needs_update = True
    
    def set_visible(self, visible: bool):
        """Set visibility and trigger update if needed."""
        was_visible = self.is_visible
        self.is_visible = visible
        if visible and not was_visible and self.needs_update:
            # Will be handled by parent's refresh cycle
            pass
    
    def assign_echogram(self, echogram_key: Optional[str]):
        """Assign an echogram to this slot."""
        if echogram_key != self.echogram_key:
            # Cache current images if we have them (including high-res)
            if self.echogram_key is not None and self.background_image is not None:
                self._image_cache[self.echogram_key] = {
                    'background_image': self.background_image,
                    'background_extent': self.background_extent,
                    'layer_image': self.layer_image,
                    'layer_extent': self.layer_extent,
                    'high_res_image': self.high_res_image,
                    'high_res_extent': self.high_res_extent,
                }
            
            self.echogram_key = echogram_key
            
            # Check slot's local cache first
            if echogram_key is not None and echogram_key in self._image_cache:
                cached = self._image_cache[echogram_key]
                self.background_image = cached.get('background_image')
                self.background_extent = cached.get('background_extent')
                self.layer_image = cached.get('layer_image')
                self.layer_extent = cached.get('layer_extent')
                self.high_res_image = cached.get('high_res_image')
                self.high_res_extent = cached.get('high_res_extent')
            # Then check parent's global cache
            elif (echogram_key is not None
                  and hasattr(self.parent, '_global_image_cache')
                  and echogram_key in self.parent._global_image_cache):
                cached = self.parent._global_image_cache[echogram_key]
                self.background_image = cached.get('background_image')
                self.background_extent = cached.get('background_extent')
                self.layer_image = cached.get('layer_image')
                self.layer_extent = cached.get('layer_extent')
                self.high_res_image = cached.get('high_res_image')
                self.high_res_extent = cached.get('high_res_extent')
            else:
                self.background_image = None
                self.background_extent = None
                self.layer_image = None
                self.layer_extent = None
                self.high_res_image = None
                self.high_res_extent = None
            
            self.needs_update = True
    
    def get_echogram(self) -> Optional[Any]:
        """Get the echogram assigned to this slot."""
        if self.echogram_key is None:
            return None
        return self.parent.echograms.get(self.echogram_key)
    
    def clear_high_res(self):
        """Clear high-res data (keeps background)."""
        self.high_res_image = None
        self.high_res_extent = None


class EchogramViewerMultiChannel:
    """Enhanced multi-echogram viewer with grid layout and lazy updates."""
    
    # Available grid layouts: (rows, cols, label)
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
        echogramdata: Union[Dict[str, Any], Sequence[Any]],
        name: str = "Multi-Echogram Viewer",
        names: Optional[Sequence[Optional[str]]] = None,
        progress: Optional[Any] = None,
        show: bool = True,
        voffsets: Optional[Dict[str, float]] = None,
        cmap: str = "Greys_r",
        cmap_layer: str = "YlGnBu_r",
        fps: int = 25,
        widget_height_px: int = 600,
        widget_width_px: int = 1000,
        auto_update: bool = True,
        auto_update_delay_ms: int = 300,
        initial_grid: Tuple[int, int] = (2, 2),
        **kwargs: Any,
    ) -> None:
        pg.setConfigOptions(imageAxisOrder="row-major")
        pgh.ensure_qapp()
        
        self.args_plot: Dict[str, Any] = {
            "vmin": kwargs.pop("vmin", -100),
            "vmax": kwargs.pop("vmax", -25),
        }
        self.args_plot.update(kwargs)
        self.args_plot_layer = dict(self.args_plot)
        self.cmap_name = cmap
        self.cmap_layer_name = cmap_layer
        self._colormap = pgh.resolve_colormap(cmap)
        self._colormap_layer = pgh.resolve_colormap(cmap_layer)
        
        # Convert input to dict format
        if isinstance(echogramdata, dict):
            self.echograms: Dict[str, Any] = dict(echogramdata)
            self.echogram_names = list(echogramdata.keys())
        elif hasattr(echogramdata, '__iter__') and not isinstance(echogramdata, (str, bytes)):
            echogramdata_list = list(echogramdata)
            # Check if this is a single echogram or a list of echograms
            if len(echogramdata_list) > 0:
                first_item = echogramdata_list[0]
                # If first item looks like an echogram (has layers, build_image), treat as single
                if hasattr(first_item, 'layers') or hasattr(first_item, 'build_image'):
                    # Single echogram - wrap in dict with "default" key
                    self.echogram_names = ["default"]
                    self.echograms = {"default": first_item}
                else:
                    # List of echograms
                    if names is not None:
                        self.echogram_names = [n if n else f"Echogram {i}" for i, n in enumerate(names)]
                    else:
                        self.echogram_names = [f"Echogram {i}" for i in range(len(echogramdata_list))]
                    self.echograms = {name: eg for name, eg in zip(self.echogram_names, echogramdata_list)}
            else:
                self.echogram_names = []
                self.echograms = {}
        else:
            # Single echogram object - wrap in dict with "default" key
            self.echogram_names = ["default"]
            self.echograms = {"default": echogramdata}
        
        # Vertical offsets per echogram
        self.voffsets: Dict[str, float] = {}
        if voffsets is not None:
            self.voffsets = dict(voffsets)
        for name in self.echogram_names:
            if name not in self.voffsets:
                self.voffsets[name] = 0.0
        
        # Determine axis names from first echogram
        if self.echograms:
            first_eg = next(iter(self.echograms.values()))
            self.x_axis_name, self.y_axis_name = _get_axis_names(first_eg)
        else:
            self.x_axis_name = "Ping number"
            self.y_axis_name = "Depth (m)"
        self._x_axis_is_datetime = self.x_axis_name == "Date time"
        
        # Progress widget
        self.progress = progress or theping.pingprocessing.widgets.TqdmWidget()
        self.display_progress = progress is None
        
        # Grid layout state - adapt to number of echograms if not specified
        n_echograms = len(self.echogram_names)
        if initial_grid == (2, 2):  # Default value, auto-adapt
            if n_echograms == 1:
                self.grid_rows, self.grid_cols = (1, 1)
            elif n_echograms == 2:
                self.grid_rows, self.grid_cols = (1, 2)
            elif n_echograms <= 4:
                self.grid_rows, self.grid_cols = (2, 2)
            elif n_echograms <= 6:
                self.grid_rows, self.grid_cols = (3, 2)
            else:
                self.grid_rows, self.grid_cols = (4, 2)
        else:
            self.grid_rows, self.grid_cols = initial_grid
        self.max_slots = 8  # Maximum number of slots
        
        # Global image cache for loaded echograms (shared across all slots)
        # Must be created BEFORE slots so assign_echogram can check it
        self._global_image_cache: Dict[Any, Dict[str, Any]] = {}
        
        # Create slots
        self.slots: List[EchogramSlot] = []
        for i in range(self.max_slots):
            slot = EchogramSlot(i, self)
            self.slots.append(slot)
        
        # Assign initial echograms to slots
        for i, name in enumerate(self.echogram_names[:self.max_slots]):
            self.slots[i].assign_echogram(name)
        
        # Widget dimensions
        self.widget_height_px = widget_height_px
        self.widget_width_px = widget_width_px
        
        # Auto-update state
        self._auto_update_enabled = auto_update
        self._auto_update_delay_ms = auto_update_delay_ms
        self._ignore_range_changes = False
        self._last_range_change_time: float = 0.0
        self._debounce_task: Optional[asyncio.Task] = None
        self._startup_complete = False
        self._last_view_range = None
        
        # Background loading state
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="echogram_loader")
        self._cancel_flag = threading.Event()
        self._loading_future: Optional[asyncio.Task] = None
        self._is_loading = False
        self._is_shutting_down = False
        self._view_changed_during_load = False
        
        # Crosshair sync state
        self._crosshair_enabled = True
        self._crosshair_position: Optional[Tuple[float, float]] = None
        
        # Pingviewer connection
        self.pingviewer = None
        
        # Station time markers storage (for recreation after layout changes)
        self._station_data: Optional[Dict[str, Any]] = None
        
        # Parameter editor state
        self._param_edit_state = {
            'active_param': None,  # Currently selected parameter name
            'editing_data': None,  # Working copy of (x_coords, y_coords) in view coordinates
            'native_data': None,   # Original (y_reference, vec_y_val) in native coordinates
            'has_unsaved_changes': False,
            'selected_point_idx': None,  # Index of selected point for delete
            'synced_params': set(),  # Set of parameter names that should sync across echograms
            'roi_items': {},  # Dict of slot_idx -> PolyLineROI (native PyQtGraph draggable line)
            'line_items': {},  # Dict of slot_idx -> PlotCurveItem (for full-resolution display)
            'display_indices': None,  # Mapping from display indices to actual data indices (for downsampling)
            '_updating_roi': False,  # Flag to prevent feedback loops during ROI updates
        }
        
        # Output widget for errors/debug
        self.output = ipywidgets.Output()
        
        # Build UI
        self._build_ui()
        self._make_graphics_widget()
        self._update_grid_layout()
        
        if show:
            self.show()
        
        # Load initial background images
        self._load_all_backgrounds()
        self._startup_complete = True
    
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
        
        # Slot selectors (dropdowns to choose which echogram in each slot)
        echogram_options = [(name, name) for name in self.echogram_names]
        echogram_options.insert(0, ("(none)", None))
        
        self.slot_selectors: List[ipywidgets.Dropdown] = []
        for i in range(self.max_slots):
            selector = ipywidgets.Dropdown(
                description=f"Slot {i+1}:",
                options=echogram_options,
                value=self.slots[i].echogram_key,
                layout=ipywidgets.Layout(width='200px'),
            )
            selector.observe(lambda change, idx=i: self._on_slot_change(idx, change), names='value')
            self.slot_selectors.append(selector)
        
        # Tab buttons for quick single-view access
        self.tab_buttons: List[ipywidgets.Button] = []
        for name in self.echogram_names:
            name_str = str(name)
            btn = ipywidgets.Button(
                description=name_str[:15],  # Truncate long names
                tooltip=f"Show {name_str} full-size",
                layout=ipywidgets.Layout(width='auto', min_width='60px'),
            )
            btn.on_click(lambda _, n=name: self._show_single(n))
            self.tab_buttons.append(btn)
        
        # Global color scale sliders (sets all colorbars when changed)
        self.w_vmin = ipywidgets.FloatSlider(
            description="vmin (all)", min=-150, max=100, step=5,
            value=self.args_plot["vmin"],
            layout=ipywidgets.Layout(width='250px'),
        )
        self.w_vmax = ipywidgets.FloatSlider(
            description="vmax (all)", min=-150, max=100, step=5,
            value=self.args_plot["vmax"],
            layout=ipywidgets.Layout(width='250px'),
        )
        self.w_vmin.observe(self._on_global_color_change, names='value')
        self.w_vmax.observe(self._on_global_color_change, names='value')
        
        # Layer selector for colorbar display (applies to all slots)
        self.w_colorbar_layer = ipywidgets.Dropdown(
            description="Colorbar:",
            options=[
                ("Background", "background"),
                ("Layer", "layer"),
            ],
            value="background",
            layout=ipywidgets.Layout(width='180px'),
        )
        self.w_colorbar_layer.observe(self._on_colorbar_layer_change, names='value')
        
        # Auto-update checkbox
        self.w_auto_update = ipywidgets.Checkbox(
            value=self._auto_update_enabled,
            description="Auto-update",
            indent=False,
        )
        self.w_auto_update.observe(self._on_auto_update_toggle, names='value')
        
        # Crosshair sync checkbox
        self.w_crosshair = ipywidgets.Checkbox(
            value=self._crosshair_enabled,
            description="Sync crosshair",
            indent=False,
        )
        self.w_crosshair.observe(lambda c: setattr(self, '_crosshair_enabled', c['new']), names='value')
        
        # Action buttons
        self.btn_update = ipywidgets.Button(description="Update", tooltip="Force update visible echograms")
        self.btn_update.on_click(self._on_update_click)
        
        self.btn_reset = ipywidgets.Button(description="Reset View", tooltip="Reset to full extent")
        self.btn_reset.on_click(self._on_reset_click)
        
        # Auto-follow pingline checkbox (visible when pingviewer connected)
        self.w_auto_follow = ipywidgets.Checkbox(
            value=False,
            description="Follow ping",
            tooltip="Automatically keep pingline in view (smooth scroll when near edge)",
            indent=False,
            layout=ipywidgets.Layout(width='110px'),
        )
        
        self.btn_goto_pingline = ipywidgets.Button(
            description="→ Ping",
            tooltip="Jump to current ping line position (center view on pingline)",
            layout=ipywidgets.Layout(width='70px'),
        )
        self.btn_goto_pingline.on_click(lambda _: self._goto_pingline())
        
        # Navigation buttons
        self._nav_fraction = 0.25
        self.btn_nav_left = ipywidgets.Button(description='◀', layout=ipywidgets.Layout(width='35px'))
        self.btn_nav_right = ipywidgets.Button(description='▶', layout=ipywidgets.Layout(width='35px'))
        self.btn_nav_up = ipywidgets.Button(description='▲', layout=ipywidgets.Layout(width='35px'))
        self.btn_nav_down = ipywidgets.Button(description='▼', layout=ipywidgets.Layout(width='35px'))
        self.btn_nav_left.on_click(lambda _: self.pan_view('left'))
        self.btn_nav_right.on_click(lambda _: self.pan_view('right'))
        self.btn_nav_up.on_click(lambda _: self.pan_view('up'))
        self.btn_nav_down.on_click(lambda _: self.pan_view('down'))
        
        # Hover label
        self.hover_label = ipywidgets.HTML(value="&nbsp;")
        
        # =====================================================================
        # Parameter Editor UI
        # =====================================================================
        self._build_param_editor_ui()
    
    def _build_param_editor_ui(self) -> None:
        """Build UI components for interactive parameter editing."""
        # Master echogram selector
        self.w_param_master = ipywidgets.Dropdown(
            description="Master:",
            options=[],
            value=None,
            layout=ipywidgets.Layout(width='150px'),
        )
        self.w_param_master.observe(self._on_param_master_change, names='value')
        
        # Parameter selector dropdown
        self.w_param_select = ipywidgets.Dropdown(
            description="Param:",
            options=[("(none)", None)],
            value=None,
            layout=ipywidgets.Layout(width='150px'),
        )
        self.w_param_select.observe(self._on_param_select_change, names='value')
        
        # Button to refresh parameter list
        self.btn_refresh_params = ipywidgets.Button(
            description="↻",
            tooltip="Refresh master and parameter lists",
            layout=ipywidgets.Layout(width='35px'),
        )
        self.btn_refresh_params.on_click(lambda _: self._refresh_param_master_list())
        
        # New empty parameter button
        self.btn_new_param = ipywidgets.Button(
            description="New",
            tooltip="Create a new empty parameter",
            layout=ipywidgets.Layout(width='50px'),
        )
        self.btn_new_param.on_click(self._on_new_param_click)
        
        # Copy parameter controls
        self.w_new_param_name = ipywidgets.Text(
            placeholder="Name",
            layout=ipywidgets.Layout(width='100px'),
        )
        self.btn_copy_param = ipywidgets.Button(
            description="Copy",
            tooltip="Copy selected parameter with new name",
            layout=ipywidgets.Layout(width='50px'),
        )
        self.btn_copy_param.on_click(self._on_copy_param_click)
        
        # Copy to all button (propagate from master to all echograms)
        self.btn_copy_to_all = ipywidgets.Button(
            description="→All",
            tooltip="Copy this parameter from master to all other echograms",
            layout=ipywidgets.Layout(width='50px'),
        )
        self.btn_copy_to_all.on_click(self._on_copy_to_all_click)
        
        # Sync checkbox
        self.w_param_sync = ipywidgets.Checkbox(
            value=False,
            description="Sync",
            tooltip="Sync edits across all echograms (only when param exists in all)",
            indent=False,
            disabled=True,  # Disabled until param is in all echograms
            layout=ipywidgets.Layout(width='70px'),
        )
        self.w_param_sync.observe(self._on_param_sync_change, names='value')
        
        # Apply/Discard buttons
        self.btn_apply_param = ipywidgets.Button(
            description="Apply",
            tooltip="Save changes to echogram(s)",
            button_style='success',
            layout=ipywidgets.Layout(width='60px'),
        )
        self.btn_apply_param.on_click(self._on_apply_param_click)
        
        self.btn_discard_param = ipywidgets.Button(
            description="Discard",
            tooltip="Discard unsaved changes",
            button_style='warning',
            layout=ipywidgets.Layout(width='60px'),
        )
        self.btn_discard_param.on_click(self._on_discard_param_click)
        
        # Add point button (alternative to 'a' key)
        self.btn_add_point = ipywidgets.Button(
            description="+Point",
            tooltip="Add a point at the current crosshair position",
            layout=ipywidgets.Layout(width='60px'),
        )
        self.btn_add_point.on_click(lambda _: self._add_point_at_cursor())
        
        # Delete point button (alternative to Delete key)
        self.btn_del_point = ipywidgets.Button(
            description="-Point",
            tooltip="Delete the selected point",
            layout=ipywidgets.Layout(width='60px'),
        )
        self.btn_del_point.on_click(lambda _: self._delete_selected_point())
        
        # Status label
        self.w_param_status = ipywidgets.HTML(value="")
        
        # Help text (updated for PolyLineROI interaction + keyboard shortcuts)
        self.w_param_help = ipywidgets.HTML(
            value="<small>Drag handles to move | <b>Click plot, then A</b>=add point | <b>Del/Backspace</b>=delete nearest point | Buttons: +Point/-Point</small>",
            layout=ipywidgets.Layout(width='auto'),
        )
        
        # Initially refresh master and parameter list
        self._refresh_param_master_list()
    
    def _refresh_param_master_list(self) -> None:
        """Refresh the master echogram dropdown."""
        # Build options from echogram keys
        options = []
        for key in self.echograms.keys():
            options.append((str(key), key))
        
        if not options:
            self.w_param_master.options = []
            self.w_param_master.value = None
            self._refresh_param_list()
            return
        
        old_value = self.w_param_master.value
        self.w_param_master.options = options
        
        # Preserve selection if still valid
        valid_keys = [k for _, k in options]
        if old_value in valid_keys:
            self.w_param_master.value = old_value
        else:
            self.w_param_master.value = options[0][1] if options else None
        
        # Refresh param list based on selected master
        self._refresh_param_list()
    
    def _on_param_master_change(self, change: Dict[str, Any]) -> None:
        """Handle master echogram selection change."""
        # Refresh parameter list for the new master
        self._refresh_param_list()
        
        # Clear current selection if it doesn't exist in new master
        self._param_edit_state['active_param'] = None
        self._param_edit_state['editing_data'] = None
        self._param_edit_state['native_data'] = None
        self._param_edit_state['has_unsaved_changes'] = False
        self._param_edit_state['selected_point_idx'] = None
        self._clear_param_visualization()
        self.w_param_status.value = ""
    
    def _get_master_echogram(self) -> Optional[Any]:
        """Get the currently selected master echogram."""
        master_key = self.w_param_master.value
        if master_key is None:
            return None
        return self.echograms.get(master_key)
    
    def _refresh_param_list(self) -> None:
        """Refresh the parameter dropdown with parameters from the master echogram."""
        master_eg = self._get_master_echogram()
        
        if master_eg is None or not hasattr(master_eg, '_coord_system'):
            self.w_param_select.options = [("(none)", None)]
            self.w_param_select.value = None
            self.w_param_sync.disabled = True
            return
        
        cs = master_eg._coord_system
        if not hasattr(cs, 'param'):
            self.w_param_select.options = [("(none)", None)]
            self.w_param_select.value = None
            self.w_param_sync.disabled = True
            return
        
        params = set(cs.param.keys())
        
        # Build options list
        options = [("(none)", None)]
        for name in sorted(params):
            options.append((name, name))
        
        # Preserve selection if still valid
        old_value = self.w_param_select.value
        self.w_param_select.options = options
        if old_value in params:
            self.w_param_select.value = old_value
        else:
            self.w_param_select.value = None
        
        # Update sync checkbox state based on current param
        self._update_sync_checkbox_state()
    
    def _update_sync_checkbox_state(self) -> None:
        """Update sync checkbox enabled state based on whether param exists in all echograms."""
        param_name = self.w_param_select.value
        if param_name is None:
            self.w_param_sync.disabled = True
            self.w_param_sync.value = False
            return
        
        # Check if param exists in all echograms
        all_have_param = True
        for eg in self.echograms.values():
            if not hasattr(eg, '_coord_system') or not hasattr(eg._coord_system, 'param'):
                all_have_param = False
                break
            if param_name not in eg._coord_system.param:
                all_have_param = False
                break
        
        self.w_param_sync.disabled = not all_have_param
        if not all_have_param:
            self.w_param_sync.value = False
    
    def _on_param_select_change(self, change: Dict[str, Any]) -> None:
        """Handle parameter selection change."""
        new_param = change['new']
        
        # Check for unsaved changes
        if self._param_edit_state['has_unsaved_changes']:
            # Show warning but allow change (user can discard or apply first)
            self.w_param_status.value = "<span style='color:orange'>⚠ Unsaved changes exist</span>"
        
        # Update state
        self._param_edit_state['active_param'] = new_param
        self._param_edit_state['selected_point_idx'] = None
        
        # Update sync checkbox state
        self._update_sync_checkbox_state()
        
        # Load parameter data
        if new_param is not None:
            self._load_param_for_editing(new_param)
        else:
            self._param_edit_state['editing_data'] = None
            self._param_edit_state['native_data'] = None
            self._param_edit_state['has_unsaved_changes'] = False
            self.w_param_status.value = ""
        
        # Update visualization
        self._update_param_visualization()
    
    def _load_param_for_editing(self, param_name: str) -> None:
        """Load parameter data for editing from the selected master echogram.
        
        Handles both sparse format (control points) and dense format (per-ping values).
        Sparse format is preferred for editing as it shows exact control points.
        """
        master_eg = self._get_master_echogram()
        
        if master_eg is None:
            self.w_param_status.value = "<span style='color:red'>No master echogram selected</span>"
            return
        
        if not hasattr(master_eg, '_coord_system'):
            self.w_param_status.value = "<span style='color:red'>Echogram has no coordinate system</span>"
            return
        
        cs = master_eg._coord_system
        if param_name not in cs.param:
            self.w_param_status.value = f"<span style='color:red'>Parameter '{param_name}' not found</span>"
            return
        
        y_reference, param_data = cs.param[param_name]
        
        # Check if sparse format: (y_reference, (sparse_x_ping_time, sparse_y_native))
        is_sparse = isinstance(param_data, tuple) and len(param_data) == 2
        
        try:
            if is_sparse:
                # Sparse format - load the exact control points
                sparse_x_ping_time, sparse_y_native = param_data
                sparse_x_ping_time = np.asarray(sparse_x_ping_time, dtype=np.float64)
                sparse_y_native = np.asarray(sparse_y_native, dtype=np.float64)
                
                # Store native data
                self._param_edit_state['native_data'] = (y_reference, (sparse_x_ping_time.copy(), sparse_y_native.copy()))
                
                # Convert sparse x (ping_time in unix seconds) to view x coordinates
                # View x depends on x_axis_name:
                # - "Ping time": unix timestamp (seconds)
                # - "Date time": matplotlib day number
                # - "Ping index": ping index
                all_ping_times = np.array(cs.ping_times)
                
                if cs.x_axis_name == "Date time":
                    # Convert ping_times to matplotlib day numbers for view
                    all_view_x = all_ping_times / 86400.0  # unix seconds to days
                elif cs.x_axis_name == "Ping time":
                    all_view_x = all_ping_times  # already in seconds
                else:
                    # Ping index - need to interpolate
                    all_view_x = np.arange(len(all_ping_times), dtype=np.float64)
                
                # Sort for interpolation
                sort_idx = np.argsort(all_ping_times)
                sorted_ping_times = all_ping_times[sort_idx]
                sorted_view_x = all_view_x[sort_idx]
                
                # Convert ping_times to view x
                x_view = np.interp(sparse_x_ping_time, sorted_ping_times, sorted_view_x)
                
                # Convert sparse y (native) to view y coordinates
                y_view = self._convert_native_to_view_y(cs, sparse_y_native, y_reference)
                
                self._param_edit_state['editing_data'] = (x_view, y_view)
                self._param_edit_state['has_unsaved_changes'] = False
                self.w_param_status.value = f"<span style='color:green'>Loaded '{param_name}' ({len(x_view)} control points)</span>"
            else:
                # Dense format - load downsampled for editing
                self._param_edit_state['native_data'] = (y_reference, np.array(param_data).copy())
                
                x_coords, y_coords = cs.get_ping_param(param_name, use_x_coordinates=False)
                x_coords = np.array([self._extent_value_to_float(x) for x in x_coords])
                y_coords = np.array(y_coords, dtype=np.float64)
                
                # Filter out NaN values
                valid_mask = np.isfinite(y_coords)
                x_valid = x_coords[valid_mask]
                y_valid = y_coords[valid_mask]
                
                # Downsample if too many points
                MAX_EDIT_POINTS = 500
                if len(x_valid) > MAX_EDIT_POINTS:
                    step = len(x_valid) // MAX_EDIT_POINTS
                    x_valid = x_valid[::step]
                    y_valid = y_valid[::step]
                    self.w_param_status.value = f"<span style='color:green'>Loaded '{param_name}' ({len(x_valid)} points, downsampled from dense)</span>"
                else:
                    n_valid = len(x_valid)
                    if n_valid == 0:
                        self.w_param_status.value = f"<span style='color:green'>Loaded '{param_name}' (empty - use 'a' to add)</span>"
                    else:
                        self.w_param_status.value = f"<span style='color:green'>Loaded '{param_name}' ({n_valid} dense points)</span>"
                
                self._param_edit_state['editing_data'] = (x_valid, y_valid)
                self._param_edit_state['has_unsaved_changes'] = False
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.w_param_status.value = f"<span style='color:red'>Error: {e}</span>"
            self._param_edit_state['editing_data'] = None
    
    def _on_copy_param_click(self, _: Any) -> None:
        """Handle copy parameter button click - copy to master echogram only."""
        source_param = self.w_param_select.value
        new_name = self.w_new_param_name.value.strip()
        
        if source_param is None:
            self.w_param_status.value = "<span style='color:red'>Select a parameter to copy</span>"
            return
        
        if not new_name:
            self.w_param_status.value = "<span style='color:red'>Enter a name for the copy</span>"
            return
        
        master_eg = self._get_master_echogram()
        if master_eg is None:
            self.w_param_status.value = "<span style='color:red'>No master echogram selected</span>"
            return
        
        if not hasattr(master_eg, '_coord_system'):
            self.w_param_status.value = "<span style='color:red'>Master has no coordinate system</span>"
            return
        
        cs = master_eg._coord_system
        if source_param not in cs.param:
            self.w_param_status.value = f"<span style='color:red'>Parameter '{source_param}' not found in master</span>"
            return
        
        # Check if new name already exists
        if new_name in cs.param:
            self.w_param_status.value = f"<span style='color:red'>Parameter '{new_name}' already exists</span>"
            return
        
        # Copy the parameter within master
        y_ref, vec_y_val = cs.param[source_param]
        cs.param[new_name] = (y_ref, np.array(vec_y_val).copy())
        
        self.w_param_status.value = f"<span style='color:green'>Copied '{source_param}' to '{new_name}'</span>"
        self._refresh_param_list()
        # Select the new parameter
        self.w_param_select.value = new_name
        self.w_new_param_name.value = ""
    
    def _on_new_param_click(self, _: Any) -> None:
        """Create a new empty parameter in the master echogram."""
        new_name = self.w_new_param_name.value.strip()
        
        if not new_name:
            self.w_param_status.value = "<span style='color:red'>Enter a name for the new parameter</span>"
            return
        
        master_eg = self._get_master_echogram()
        if master_eg is None:
            self.w_param_status.value = "<span style='color:red'>No master echogram selected</span>"
            return
        
        if not hasattr(master_eg, '_coord_system'):
            self.w_param_status.value = "<span style='color:red'>Master has no coordinate system</span>"
            return
        
        cs = master_eg._coord_system
        
        # Check if already exists
        if new_name in cs.param:
            self.w_param_status.value = f"<span style='color:red'>Parameter '{new_name}' already exists</span>"
            return
        
        # Create empty parameter with NaN values (one per ping)
        n_pings = cs.n_pings
        empty_values = np.full(n_pings, np.nan, dtype=np.float64)
        
        # Use current y axis name as y_reference (e.g., 'Depth (m)', 'Range (m)', etc.)
        y_reference = cs.y_axis_name if cs.y_axis_name else "Y indice"
        cs.param[new_name] = (y_reference, empty_values)
        
        self.w_param_status.value = f"<span style='color:green'>Created empty '{new_name}'</span>"
        self._refresh_param_list()
        self.w_param_select.value = new_name
        self.w_new_param_name.value = ""
    
    def _on_copy_to_all_click(self, _: Any) -> None:
        """Copy the current parameter from master to all other echograms."""
        param_name = self.w_param_select.value
        if param_name is None:
            self.w_param_status.value = "<span style='color:red'>Select a parameter first</span>"
            return
        
        master_eg = self._get_master_echogram()
        if master_eg is None:
            self.w_param_status.value = "<span style='color:red'>No master echogram selected</span>"
            return
        
        if not hasattr(master_eg, '_coord_system'):
            self.w_param_status.value = "<span style='color:red'>Master has no coordinate system</span>"
            return
        
        master_cs = master_eg._coord_system
        if param_name not in master_cs.param:
            self.w_param_status.value = f"<span style='color:red'>Parameter '{param_name}' not in master</span>"
            return
        
        y_ref, vec_y_val = master_cs.param[param_name]
        master_n_pings = len(vec_y_val)
        
        # Pre-convert to numpy array once
        vec_y_val_np = np.asarray(vec_y_val, dtype=np.float64)
        
        # Copy to all other echograms
        copied_count = 0
        for key, eg in self.echograms.items():
            if eg is master_eg:
                continue
            if not hasattr(eg, '_coord_system'):
                continue
            
            cs = eg._coord_system
            n_pings = cs.n_pings
            
            # Create appropriate array
            if n_pings != master_n_pings:
                new_values = np.full(n_pings, np.nan, dtype=np.float64)
            else:
                new_values = vec_y_val_np.copy()
            
            cs.param[param_name] = (y_ref, new_values)
            copied_count += 1
        
        # Update UI
        if copied_count > 0:
            self.w_param_status.value = f"<span style='color:green'>Copied to {copied_count} echogram(s)</span>"
            # Enable sync checkbox since param now exists in all
            self.w_param_sync.disabled = False
        else:
            self.w_param_status.value = "<span style='color:orange'>No other echograms to copy to</span>"
    
    def _on_param_sync_change(self, change: Dict[str, Any]) -> None:
        """Handle sync checkbox change."""
        param_name = self._param_edit_state['active_param']
        if param_name is None:
            return
        
        if change['new']:
            self._param_edit_state['synced_params'].add(param_name)
        else:
            self._param_edit_state['synced_params'].discard(param_name)
    
    def _on_apply_param_click(self, _: Any) -> None:
        """Apply edited parameter back to echogram(s).
        
        Stores sparse control points directly in cs.param as:
        (y_reference, (sparse_x_in_ping_time, sparse_y_in_native))
        """
        param_name = self._param_edit_state['active_param']
        if param_name is None:
            return
        
        editing_data = self._param_edit_state['editing_data']
        native_data = self._param_edit_state['native_data']
        
        if editing_data is None or native_data is None:
            self.w_param_status.value = "<span style='color:red'>No data to apply</span>"
            return
        
        x_coords, y_coords = editing_data
        y_reference, _ = native_data
        
        # Determine which echograms to update
        if self.w_param_sync.value and param_name in self._param_edit_state['synced_params']:
            echograms_to_update = list(self.echograms.values())
        else:
            # Just the master echogram
            master_eg = self._get_master_echogram()
            echograms_to_update = [master_eg] if master_eg is not None else []
        
        # Convert from view coordinates back to native and save as sparse
        updated_count = 0
        for eg in echograms_to_update:
            if not hasattr(eg, '_coord_system'):
                continue
            cs = eg._coord_system
            
            try:
                # Convert view y_coords back to native y_reference coordinates
                native_y = self._convert_view_to_native_y(cs, y_coords, y_reference)
                
                # Convert view x_coords to ping_times (unix seconds)
                # View x depends on x_axis_name:
                # - "Ping time": unix timestamp (seconds)
                # - "Date time": matplotlib day number (days since 1970)
                # - "Ping index": ping index
                all_ping_times = np.array(cs.ping_times)
                
                if cs.x_axis_name == "Date time":
                    # View x is in matplotlib day numbers, convert to view format
                    all_view_x = all_ping_times / 86400.0  # unix seconds to days
                elif cs.x_axis_name == "Ping time":
                    all_view_x = all_ping_times  # already in seconds
                else:
                    # Ping index
                    all_view_x = np.arange(len(all_ping_times), dtype=np.float64)
                
                # Sort for proper interpolation
                sort_idx = np.argsort(all_view_x)
                sorted_view_x = all_view_x[sort_idx]
                sorted_ping_times = all_ping_times[sort_idx]
                
                # Convert x_coords (view) to ping_times
                x_as_ping_times = np.interp(x_coords, sorted_view_x, sorted_ping_times)
                
                # Sort by x for storage
                sort_order = np.argsort(x_as_ping_times)
                x_as_ping_times = x_as_ping_times[sort_order]
                native_y = native_y[sort_order]
                
                # Deduplicate x values (average y for duplicates) - required by LinearInterpolator
                unique_x, indices = np.unique(x_as_ping_times, return_inverse=True)
                if len(unique_x) < len(x_as_ping_times):
                    # Average y values for duplicate x
                    sums = np.bincount(indices, weights=native_y)
                    counts = np.bincount(indices)
                    unique_y = sums / counts
                    x_as_ping_times = unique_x
                    native_y = unique_y
                
                # Store as sparse format: (y_reference, (x_ping_times, y_native))
                cs.param[param_name] = (y_reference, (x_as_ping_times.copy(), native_y.copy()))
                updated_count += 1
            except Exception as e:
                with self.output:
                    print(f"Error updating {eg}: {e}")
                import traceback
                traceback.print_exc()
        
        if updated_count > 0:
            self._param_edit_state['has_unsaved_changes'] = False
            self._param_edit_state['native_data'] = (y_reference, (x_as_ping_times.copy(), native_y.copy()))
            self.w_param_status.value = f"<span style='color:green'>Applied {len(x_coords)} points to {updated_count} echogram(s)</span>"
        else:
            self.w_param_status.value = "<span style='color:red'>Failed to apply changes</span>"
    
    def _on_discard_param_click(self, _: Any) -> None:
        """Discard unsaved changes and reload from echogram."""
        param_name = self._param_edit_state['active_param']
        if param_name is not None:
            self._load_param_for_editing(param_name)
            self._update_param_visualization()
            self.w_param_status.value = "<span style='color:blue'>Changes discarded</span>"
    
    def _convert_view_to_native_y(self, cs, view_y: np.ndarray, y_reference: str) -> np.ndarray:
        """Convert y values from current view coordinates to native parameter coordinates.
        
        Args:
            cs: EchogramCoordinateSystem
            view_y: Y values in current view coordinates
            y_reference: Target native coordinate type
            
        Returns:
            Y values in native coordinate system
        """
        # Current y coordinates are in cs.y_axis_name
        # We need to convert to y_reference
        
        # First convert view_y to sample indices
        # view_y = a_y + b_y * sample_idx, so sample_idx = (view_y - a_y) / b_y
        if cs._affine_sample_to_y is None:
            return view_y  # No conversion available
        
        a_y, b_y = cs._affine_sample_to_y
        # Use mean coefficients for now (could be more sophisticated)
        a_y_mean = np.nanmean(a_y)
        b_y_mean = np.nanmean(b_y)
        
        if b_y_mean == 0:
            sample_indices = view_y
        else:
            sample_indices = (view_y - a_y_mean) / b_y_mean
        
        # Now convert sample indices to native coordinate
        match y_reference:
            case "Y indice":
                return sample_indices
            case "Sample number":
                if not cs.has_sample_nrs:
                    return sample_indices
                a, b = cs._affine_sample_to_sample_nr
                a_mean, b_mean = np.nanmean(a), np.nanmean(b)
                return a_mean + b_mean * sample_indices
            case "Depth (m)":
                if not cs.has_depths:
                    return sample_indices
                a, b = cs._affine_sample_to_depth
                a_mean, b_mean = np.nanmean(a), np.nanmean(b)
                return a_mean + b_mean * sample_indices
            case "Range (m)":
                if not cs.has_ranges:
                    return sample_indices
                a, b = cs._affine_sample_to_range
                a_mean, b_mean = np.nanmean(a), np.nanmean(b)
                return a_mean + b_mean * sample_indices
            case _:
                return view_y
    
    def _convert_native_to_view_y(self, cs, native_y: np.ndarray, y_reference: str) -> np.ndarray:
        """Convert y values from native parameter coordinates to current view coordinates.
        
        Args:
            cs: EchogramCoordinateSystem
            native_y: Y values in native coordinate system (y_reference)
            y_reference: Source native coordinate type
            
        Returns:
            Y values in current view coordinates
        """
        # First convert native_y to sample indices
        match y_reference:
            case "Y indice":
                sample_indices = native_y
            case "Sample number":
                if not cs.has_sample_nrs:
                    sample_indices = native_y
                else:
                    a, b = cs._affine_sample_to_sample_nr
                    a_mean, b_mean = np.nanmean(a), np.nanmean(b)
                    if b_mean != 0:
                        sample_indices = (native_y - a_mean) / b_mean
                    else:
                        sample_indices = native_y
            case "Depth (m)":
                if not cs.has_depths:
                    sample_indices = native_y
                else:
                    a, b = cs._affine_sample_to_depth
                    a_mean, b_mean = np.nanmean(a), np.nanmean(b)
                    if b_mean != 0:
                        sample_indices = (native_y - a_mean) / b_mean
                    else:
                        sample_indices = native_y
            case "Range (m)":
                if not cs.has_ranges:
                    sample_indices = native_y
                else:
                    a, b = cs._affine_sample_to_range
                    a_mean, b_mean = np.nanmean(a), np.nanmean(b)
                    if b_mean != 0:
                        sample_indices = (native_y - a_mean) / b_mean
                    else:
                        sample_indices = native_y
            case _:
                sample_indices = native_y
        
        # Now convert sample indices to current view coordinates
        if cs._affine_sample_to_y is None:
            return sample_indices
        
        a_y, b_y = cs._affine_sample_to_y
        a_y_mean = np.nanmean(a_y)
        b_y_mean = np.nanmean(b_y)
        
        return a_y_mean + b_y_mean * sample_indices
    
    def _update_param_visualization(self) -> None:
        """Update the parameter visualization on all visible plots using PolyLineROI."""
        # Remove old visualization items
        self._clear_param_visualization()
        
        param_name = self._param_edit_state['active_param']
        editing_data = self._param_edit_state['editing_data']
        
        if param_name is None:
            return
        
        # Handle None or empty editing_data
        if editing_data is None:
            x_coords = np.array([], dtype=np.float64)
            y_coords = np.array([], dtype=np.float64)
        else:
            x_coords, y_coords = editing_data
        
        n_points = len(x_coords)
        
        # For large datasets, use downsampling for ROI handles
        # Show all points as a line but only create draggable handles for subset
        MAX_ROI_HANDLES = 500  # PolyLineROI can get slow with too many handles
        if n_points > MAX_ROI_HANDLES:
            # Downsample for ROI handles (keep every nth point)
            step = max(1, n_points // MAX_ROI_HANDLES)
            display_indices = np.arange(0, n_points, step)
            x_display = x_coords[display_indices]
            y_display = y_coords[display_indices]
            is_downsampled = True
        else:
            x_display = x_coords
            y_display = y_coords
            display_indices = np.arange(n_points) if n_points > 0 else np.array([], dtype=int)
            is_downsampled = False
        
        # Store mapping for handle->data index conversion
        self._param_edit_state['display_indices'] = display_indices
        
        # Create visualization on each visible slot
        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            
            # If downsampled, show full line separately for accuracy
            if is_downsampled and n_points > 0:
                pen = pg.mkPen(color='#FF6600', width=2, style=QtCore.Qt.PenStyle.DashLine)
                line_item = pg.PlotCurveItem(x_coords, y_coords, pen=pen)
                slot.plot_item.addItem(line_item)
                self._param_edit_state['line_items'][slot.slot_idx] = line_item
            
            # Create PolyLineROI for interactive editing
            # Use SafePolyLineROI to prevent crashes from right-click on handles
            if len(x_display) > 0:
                # Convert to list of [x, y] positions for PolyLineROI
                positions = [[float(x_display[i]), float(y_display[i])] for i in range(len(x_display))]
                
                roi = SafePolyLineROI(
                    positions,
                    closed=False,
                    pen=pg.mkPen('#FF6600', width=2),
                    hoverPen=pg.mkPen('#FFAA00', width=3),
                    handlePen=pg.mkPen('#FF6600', width=1),
                    handleHoverPen=pg.mkPen('#FF0000', width=2),
                    movable=False,  # Don't allow moving entire ROI, only handles
                    removable=False,
                )
                
                # Connect signal for when handles are dragged
                roi.sigRegionChanged.connect(lambda r=roi, s=slot: self._on_roi_changed(r, s))
                roi.sigRegionChangeFinished.connect(lambda r=roi, s=slot: self._on_roi_change_finished(r, s))
                
                slot.plot_item.addItem(roi)
                self._param_edit_state['roi_items'][slot.slot_idx] = roi
            
            elif n_points == 0:
                # No points - show message (handled via status label)
                pass
        
        if is_downsampled:
            self.w_param_status.value = f"<span style='color:orange'>Showing {len(x_display)}/{n_points} handles (downsampled)</span>"
        
        self._request_remote_draw()
    
    def _on_roi_changed(self, roi: pg.PolyLineROI, slot: EchogramSlot) -> None:
        """Handle ROI being dragged - update data in real-time.
        
        During drag we update X and Y coordinates from the active ROI.
        We don't sync other ROIs during drag to avoid feedback loops.
        Sorting happens only when drag finishes.
        """
        if self._param_edit_state.get('_updating_roi'):
            return  # Prevent feedback loops
        
        # Mark that we're dragging
        self._param_edit_state['_is_dragging_roi'] = True
        self._param_edit_state['_active_drag_slot'] = slot.slot_idx
        self._param_edit_state['has_unsaved_changes'] = True
    
    def _on_roi_change_finished(self, roi: pg.PolyLineROI, slot: EchogramSlot) -> None:
        """Handle ROI drag finished - read final positions, resort if needed, rebuild.
        
        Only rebuilds the visualization if points crossed each other on X axis.
        """
        if not self._param_edit_state.get('_is_dragging_roi'):
            return
        
        self._param_edit_state['_is_dragging_roi'] = False
        self._param_edit_state['_active_drag_slot'] = None
        
        editing_data = self._param_edit_state['editing_data']
        if editing_data is None:
            return
        
        try:
            # Get current positions from ROI handles
            handle_positions = roi.getLocalHandlePositions()
            if len(handle_positions) == 0:
                return
            
            x_coords, y_coords = editing_data
            display_indices = self._param_edit_state.get('display_indices')
            
            # Read final positions from the ROI
            for i, (name, pos) in enumerate(handle_positions):
                if display_indices is not None and i < len(display_indices):
                    actual_idx = display_indices[i]
                else:
                    actual_idx = i
                
                if 0 <= actual_idx < len(x_coords):
                    x_coords[actual_idx] = pos.x()
                    y_coords[actual_idx] = pos.y()
            
            # Check if points are out of order (need resorting)
            needs_resort = False
            for i in range(len(x_coords) - 1):
                if x_coords[i] > x_coords[i + 1]:
                    needs_resort = True
                    break
            
            if needs_resort:
                # Sort by X coordinate
                sort_indices = np.argsort(x_coords)
                x_coords = x_coords[sort_indices].copy()
                y_coords = y_coords[sort_indices].copy()
                self._param_edit_state['editing_data'] = (x_coords, y_coords)
                
                # Rebuild visualization with sorted data
                self._clear_param_visualization()
                self._update_param_visualization()
                self.w_param_status.value = "<span style='color:orange'>Points reordered (unsaved)</span>"
            else:
                # Just update the data, no need to rebuild ROI
                self._param_edit_state['editing_data'] = (x_coords, y_coords)
                
                # Update line items if downsampled
                for slot_idx, line_item in list(self._param_edit_state['line_items'].items()):
                    try:
                        line_item.setData(x_coords, y_coords)
                    except Exception:
                        pass
                
                # Sync other ROIs
                self._param_edit_state['_updating_roi'] = True
                try:
                    for other_slot_idx, other_roi in list(self._param_edit_state['roi_items'].items()):
                        if other_slot_idx != slot.slot_idx:
                            try:
                                if display_indices is not None:
                                    x_disp = x_coords[display_indices]
                                    y_disp = y_coords[display_indices]
                                else:
                                    x_disp = x_coords
                                    y_disp = y_coords
                                positions = [[float(x_disp[j]), float(y_disp[j])] for j in range(len(x_disp))]
                                other_roi.setPoints(positions)
                            except Exception:
                                pass
                finally:
                    self._param_edit_state['_updating_roi'] = False
                
                self.w_param_status.value = "<span style='color:orange'>Modified (unsaved)</span>"
            
        except Exception as e:
            with self.output:
                print(f"ROI finish error: {e}")
            self.w_param_status.value = f"<span style='color:red'>Error: {e}</span>"
    
    def _clear_param_visualization(self) -> None:
        """Remove parameter visualization items from all plots."""
        # Remove ROI items
        for slot_idx, roi in list(self._param_edit_state.get('roi_items', {}).items()):
            slot = self.slots[slot_idx] if slot_idx < len(self.slots) else None
            if slot and slot.plot_item:
                try:
                    slot.plot_item.removeItem(roi)
                except Exception:
                    pass
        self._param_edit_state['roi_items'] = {}
        
        # Remove line items
        for slot_idx, line in list(self._param_edit_state['line_items'].items()):
            slot = self.slots[slot_idx] if slot_idx < len(self.slots) else None
            if slot and slot.plot_item:
                try:
                    slot.plot_item.removeItem(line)
                except Exception:
                    pass
        self._param_edit_state['line_items'].clear()
    

    
    def _delete_selected_point(self) -> bool:
        """Delete the point closest to the cursor position on the X-axis.
        
        The deletion finds the point with the smallest X-axis distance to the
        current crosshair position, ensuring the user deletes the intended point.
        """
        # Prevent deletion during active drag to avoid crashes
        if self._param_edit_state.get('_is_dragging_roi'):
            self.w_param_status.value = "<span style='color:red'>Cannot delete while dragging</span>"
            return False
        
        # Prevent rapid-fire deletions
        if self._param_edit_state.get('_deletion_in_progress'):
            return False
        
        try:
            self._param_edit_state['_deletion_in_progress'] = True
            
            editing_data = self._param_edit_state['editing_data']
            
            if editing_data is None:
                self.w_param_status.value = "<span style='color:red'>No parameter data to delete from</span>"
                return False
            
            x_coords, y_coords = editing_data
            
            if len(x_coords) == 0:
                self.w_param_status.value = "<span style='color:red'>No points to delete</span>"
                return False
            
            # Find point closest to cursor on X-axis
            cursor_pos = getattr(self, '_last_crosshair_position', None)
            if cursor_pos is not None:
                cursor_x = cursor_pos[0]
                # Find index of point with minimum X distance to cursor
                x_distances = np.abs(x_coords - cursor_x)
                idx = int(np.argmin(x_distances))
            else:
                # Fallback to last point if no cursor position
                idx = len(x_coords) - 1
            
            new_x = np.delete(x_coords, idx)
            new_y = np.delete(y_coords, idx)
            
            # Update state before visualization to ensure consistency
            self._param_edit_state['editing_data'] = (new_x, new_y)
            self._param_edit_state['selected_point_idx'] = None
            self._param_edit_state['has_unsaved_changes'] = True
            
            # Clear existing visualization first
            self._clear_param_visualization()
            
            # Update visualization with new data
            self._update_param_visualization()
            self.w_param_status.value = "<span style='color:orange'>Point deleted (unsaved)</span>"
            return True
            
        except Exception as e:
            with self.output:
                print(f"Delete point error: {e}")
            self.w_param_status.value = f"<span style='color:red'>Delete error: {e}</span>"
            return False
        finally:
            self._param_edit_state['_deletion_in_progress'] = False
    
    def _add_point_at_cursor(self) -> bool:
        """Add a new point at the current cursor position."""
        # Use last known position (persists after mouse leaves plot)
        cursor_pos = getattr(self, '_last_crosshair_position', None)
        if cursor_pos is None:
            self.w_param_status.value = "<span style='color:red'>Move cursor over plot first</span>"
            return False
        
        # Check if we have an active parameter
        if self._param_edit_state['active_param'] is None:
            self.w_param_status.value = "<span style='color:red'>Select a parameter first</span>"
            return False
        
        editing_data = self._param_edit_state['editing_data']
        cursor_x, cursor_y = cursor_pos
        
        # Handle case where editing_data is None or empty
        if editing_data is None:
            # Initialize with empty arrays
            x_coords = np.array([], dtype=np.float64)
            y_coords = np.array([], dtype=np.float64)
        else:
            x_coords, y_coords = editing_data
        
        # Find insertion point to maintain sorted order by x
        insert_idx = np.searchsorted(x_coords, cursor_x)
        
        # Insert the new point
        new_x = np.insert(x_coords, insert_idx, cursor_x)
        new_y = np.insert(y_coords, insert_idx, cursor_y)
        
        self._param_edit_state['editing_data'] = (new_x, new_y)
        self._param_edit_state['selected_point_idx'] = insert_idx
        self._param_edit_state['has_unsaved_changes'] = True
        
        self._update_param_visualization()
        self.w_param_status.value = "<span style='color:orange'>Point added (unsaved)</span>"
        return True
    
    def _build_param_editor_row(self) -> ipywidgets.VBox:
        """Build the parameter editor row widget."""
        # First row: master selection and parameter selection
        row1 = ipywidgets.HBox([
            self.w_param_master,
            self.w_param_select,
            self.btn_refresh_params,
            self.w_new_param_name,
            self.btn_new_param,
            self.btn_copy_param,
            self.btn_copy_to_all,
        ])
        # Second row: point editing actions
        row2 = ipywidgets.HBox([
            self.btn_add_point,
            self.btn_del_point,
            self.w_param_sync,
            self.btn_apply_param,
            self.btn_discard_param,
            self.w_param_status,
            self.w_param_help,
        ])
        return ipywidgets.VBox([row1, row2])

    def _make_graphics_widget(self) -> None:
        """Create the PyQtGraph graphics widget."""
        self.graphics = GraphicsLayoutWidget(
            css_width=f"{self.widget_width_px}px",
            css_height=f"{self.widget_height_px}px"
        )
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, self.widget_height_px)
        if hasattr(self.graphics, "gfxView"):
            self.graphics.gfxView.setBackground("w")
        
        # Hook into jupyter_rfb's handle_event to capture keyboard events
        # since pyqtgraph.jupyter doesn't forward key events to Qt by default
        self._setup_rfb_event_handler()
        
        # Set up auto-update hook
        self._original_request_draw = self.graphics.request_draw
        viewer = self
        
        def patched_request_draw():
            viewer._original_request_draw()
            if not viewer._startup_complete or not viewer._auto_update_enabled:
                return
            if viewer._ignore_range_changes or viewer._is_loading:
                return
            # Check for view range changes
            if viewer._get_master_plot():
                vb = viewer._get_master_plot().getViewBox()
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
    
    def _update_grid_layout(self) -> None:
        """Update the graphics widget to reflect current grid layout."""
        # Clear existing plots
        self.graphics.clear()
        
        # Determine which slots are visible
        n_visible = self.grid_rows * self.grid_cols
        
        for i, slot in enumerate(self.slots):
            slot.set_visible(i < n_visible)
        
        # Create plot items for visible slots
        master_plot = None
        for i in range(n_visible):
            row = i // self.grid_cols
            col = i % self.grid_cols
            slot = self.slots[i]
            
            # Create axis items
            axis_items = None
            if self._x_axis_is_datetime:
                axis_items = {"bottom": pgh.MatplotlibDateAxis(self._mpl_num_to_datetime, orientation="bottom")}
            
            plot: pg.PlotItem = self.graphics.addPlot(row=row, col=col * 2, axisItems=axis_items)
            slot.plot_item = plot
            
            # Configure plot
            title = str(slot.echogram_key) if slot.echogram_key is not None else f"Slot {i+1}"
            plot.setTitle(title)
            plot.setLabel("left", self.y_axis_name if col == 0 else "")
            plot.setLabel("bottom", self.x_axis_name if row == self.grid_rows - 1 else "")
            plot.getViewBox().invertY(True)
            plot.getViewBox().setBackgroundColor("w")
            
            # Create image items
            background = pg.ImageItem(axisOrder="row-major")
            plot.addItem(background)
            high_res = pg.ImageItem(axisOrder="row-major")
            high_res.hide()
            plot.addItem(high_res)
            layer = pg.ImageItem(axisOrder="row-major")
            layer.hide()
            plot.addItem(layer)
            
            slot.image_layers = {"background": background, "high": high_res, "layer": layer}
            
            # Create single colorbar (switches between background and layer)
            try:
                colorbar = pg.ColorBarItem(
                    label="(dB)",
                    values=(self.args_plot["vmin"], self.args_plot["vmax"]),
                    interactive=True,  # Allow user to drag colorbar
                )
                # Initially attach to background image
                colorbar.setImageItem(background, insert_in=plot)
                if hasattr(colorbar, "setColorMap"):
                    colorbar.setColorMap(self._colormap)
                slot.colorbar = colorbar
                
                # Initialize per-layer levels
                slot.background_levels = (self.args_plot["vmin"], self.args_plot["vmax"])
                slot.layer_levels = (self.args_plot["vmin"], self.args_plot["vmax"])
                
                # Connect colorbar level changes to sync images
                if hasattr(colorbar, 'sigLevelsChanged'):
                    colorbar.sigLevelsChanged.connect(
                        lambda cb=colorbar, s=slot: self._on_colorbar_levels_changed(s, cb)
                    )
            except AttributeError:
                slot.colorbar = None
            
            # No separate layer_colorbar - we use a single colorbar
            slot.layer_colorbar = None
            
            # Create crosshairs
            pen_cross = pg.mkPen(color='r', width=1, style=QtCore.Qt.PenStyle.DashLine)
            slot.crosshair_v = pg.InfiniteLine(angle=90, pen=pen_cross)
            slot.crosshair_h = pg.InfiniteLine(angle=0, pen=pen_cross)
            slot.crosshair_v.hide()
            slot.crosshair_h.hide()
            plot.addItem(slot.crosshair_v)
            plot.addItem(slot.crosshair_h)
            
            # Reset pingline (will be recreated when _update_ping_lines is called)
            slot.pingline = None
            
            # Clear station markers (will be recreated by _recreate_station_markers)
            slot.station_markers = {}
            
            # Link axes to master
            if master_plot is None:
                master_plot = plot
            else:
                plot.setXLink(master_plot)
                plot.setYLink(master_plot)
        
        # Connect scene events
        self._connect_scene_events()
        
        # Update visible slots
        self._update_visible_slots()
        
        # Recreate station markers if they were set
        self._recreate_station_markers()
        
        # Recreate parameter visualization if active
        if hasattr(self, '_param_edit_state') and self._param_edit_state.get('active_param') is not None:
            # Clear old items (they were removed with plots)
            self._param_edit_state['scatter_items'].clear()
            self._param_edit_state['line_items'].clear()
            self._update_param_visualization()
    
    def _connect_scene_events(self) -> None:
        """Connect mouse events for crosshair and click handling."""
        gfx_view = getattr(self.graphics, "gfxView", None)
        scene = gfx_view.scene() if gfx_view is not None else None
        if scene is None:
            return
        
        # Disconnect existing connections if we have them tracked
        if hasattr(self, '_scene_click_connection') and self._scene_click_connection is not None:
            try:
                scene.sigMouseClicked.disconnect(self._handle_scene_click)
            except (TypeError, RuntimeError):
                pass
        if hasattr(self, '_scene_move_connection') and self._scene_move_connection is not None:
            try:
                scene.sigMouseMoved.disconnect(self._handle_scene_move)
            except (TypeError, RuntimeError):
                pass
        
        # Connect and track
        self._scene_click_connection = scene.sigMouseClicked.connect(self._handle_scene_click)
        self._scene_move_connection = scene.sigMouseMoved.connect(self._handle_scene_move)
        
        # Note: Keyboard shortcuts are handled via jupyter_rfb handle_event hook
        # (see _setup_rfb_event_handler) since pyqtgraph.jupyter doesn't forward
        # key events to Qt by default.
    
    def _setup_rfb_event_handler(self) -> None:
        """Hook into jupyter_rfb's handle_event to capture keyboard events.
        
        The pyqtgraph.jupyter GraphicsLayoutWidget is based on jupyter_rfb.RemoteFrameBuffer,
        which receives events from the browser. However, it only forwards mouse/wheel events
        to Qt, not keyboard events. We hook into handle_event to capture key_down/key_up.
        """
        if not hasattr(self.graphics, 'handle_event'):
            return
        
        # Skip if already hooked
        if hasattr(self, '_rfb_event_hooked') and self._rfb_event_hooked:
            return
        
        # Store the original handle_event
        original_handle_event = self.graphics.handle_event
        viewer = self
        
        def hooked_handle_event(event):
            """Extended handle_event that also processes keyboard events."""
            event_type = event.get('event_type', '')
            
            # Handle keyboard events for parameter editing
            if event_type == 'key_down':
                key = event.get('key', '')
                viewer._handle_rfb_key_down(key, event.get('modifiers', ()))
            
            # Track mouse position for crosshair (use pointer_move events)
            if event_type == 'pointer_move':
                viewer._last_pointer_position = (event.get('x', 0), event.get('y', 0))
            
            # Always call original handler
            return original_handle_event(event)
        
        self.graphics.handle_event = hooked_handle_event
        self._rfb_event_hooked = True
        self._last_pointer_position = None
    
    def _handle_rfb_key_down(self, key: str, modifiers: tuple) -> None:
        """Handle key_down event from jupyter_rfb.
        
        Parameters
        ----------
        key : str
            The key name as per jupyter_rfb spec (e.g., 'a', 'Delete', 'Backspace')
        modifiers : tuple
            Tuple of modifier keys being held (e.g., ('Shift', 'Control'))
        """
        # Only process keys if we have an active parameter being edited
        if self._param_edit_state.get('active_param') is None:
            return
        
        # Delete/Backspace - remove last point
        if key in ('Delete', 'Backspace'):
            self._delete_selected_point()
            return
        
        # 'a' or 'A' - add point at cursor
        if key.lower() == 'a':
            self._add_point_at_cursor()
            return
    

    
    def _get_master_plot(self) -> Optional[pg.PlotItem]:
        """Get the first visible plot item (master for axis linking)."""
        for slot in self.slots:
            if slot.is_visible and slot.plot_item is not None:
                return slot.plot_item
        return None
    
    def _get_visible_slots(self) -> List[EchogramSlot]:
        """Get list of currently visible slots."""
        return [s for s in self.slots if s.is_visible and s.echogram_key is not None]
    
    def _update_visible_slots(self) -> None:
        """Update only the visible slots that need updating."""
        for slot in self._get_visible_slots():
            if slot.needs_update or slot.background_image is None:
                self._update_slot(slot)
    
    def _update_slot(self, slot: EchogramSlot) -> None:
        """Update a single slot's display."""
        if slot.plot_item is None or slot.echogram_key is None:
            return
        
        echogram = slot.get_echogram()
        if echogram is None:
            return
        
        # Disable auto-range during update to prevent view changes
        vb = slot.plot_item.getViewBox()
        old_auto_range = vb.autoRangeEnabled()
        vb.disableAutoRange()
        
        try:
            # Update title
            slot.plot_item.setTitle(str(slot.echogram_key) if slot.echogram_key is not None else "")
            
            # Show background image
            if slot.background_image is not None and slot.background_extent is not None:
                self._update_slot_image(slot, "background", slot.background_image, slot.background_extent)
            
            # Show high-res image if available
            if slot.high_res_image is not None and slot.high_res_extent is not None:
                self._update_slot_image(slot, "high", slot.high_res_image, slot.high_res_extent)
            else:
                slot.image_layers.get("high", pg.ImageItem()).hide()
            
            # Show layer image if available
            if slot.layer_image is not None and slot.layer_extent is not None:
                self._update_slot_image(slot, "layer", slot.layer_image, slot.layer_extent)
            else:
                slot.image_layers.get("layer", pg.ImageItem()).hide()
        finally:
            # Restore auto-range state
            if old_auto_range[0] or old_auto_range[1]:
                vb.enableAutoRange(x=old_auto_range[0], y=old_auto_range[1])
        
        slot.needs_update = False
    
    def _update_slot_image(
        self,
        slot: EchogramSlot,
        key: str,
        data: np.ndarray,
        extent: Tuple[float, float, float, float]
    ) -> None:
        """Update a specific image layer in a slot."""
        image_item = slot.image_layers.get(key)
        if image_item is None:
            return
        
        array = data.transpose()
        image_item.setImage(array, autoLevels=False)
        
        x0, x1, y0, y1 = self._numeric_extent(extent)
        plot = slot.plot_item
        vb = plot.getViewBox()
        if vb.yInverted():
            y0, y1 = y1, y0
        
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        image_item.setRect(rect)
        
        # Set colormap
        is_layer = (key == "layer")
        colormap = self._colormap_layer if is_layer else self._colormap
        if hasattr(image_item, "setColorMap"):
            image_item.setColorMap(colormap)
        else:
            lut = colormap.getLookupTable(256)
            image_item.setLookupTable(lut)
        
        # Use stored per-layer levels (preserve user's colorbar adjustments)
        if is_layer and slot.layer_levels is not None:
            vmin, vmax = slot.layer_levels
        elif not is_layer and slot.background_levels is not None:
            vmin, vmax = slot.background_levels
        elif slot.colorbar is not None:
            vmin, vmax = slot.colorbar.levels()
        else:
            # Fallback to global sliders
            offset = self.voffsets.get(slot.echogram_key, 0.0) if slot.echogram_key else 0.0
            vmin = float(self.w_vmin.value + offset)
            vmax = float(self.w_vmax.value + offset)
        
        image_item.setLevels((vmin, vmax))
        image_item.show()
    
    def _load_all_backgrounds(self) -> None:
        """Load background images for all echograms."""
        for name, echogram in self.echograms.items():
            slot = self._get_slot_for_echogram(name)
            if slot is None:
                continue
            
            self.progress.set_description(f"Loading {name}...")
            
            if len(echogram.layers) == 0 and echogram.main_layer is None:
                image, extent = echogram.build_image(progress=self.progress)
                slot.background_image = image
                slot.background_extent = extent
                slot.layer_image = None
                slot.layer_extent = None
            else:
                image, layer_img, extent = echogram.build_image_and_layer_image(progress=self.progress)
                slot.background_image = image
                slot.background_extent = extent
                slot.layer_image = layer_img
                slot.layer_extent = extent
            
            # Add to global cache
            self._global_image_cache[name] = {
                'background_image': slot.background_image,
                'background_extent': slot.background_extent,
                'layer_image': slot.layer_image,
                'layer_extent': slot.layer_extent,
            }
            
            slot.needs_update = True
        
        self.progress.set_description("Idle")
        self._update_visible_slots()
        self._reset_view()
    
    def _get_slot_for_echogram(self, echogram_key: str) -> Optional[EchogramSlot]:
        """Find the slot assigned to a given echogram."""
        for slot in self.slots:
            if slot.echogram_key == echogram_key:
                return slot
        return None
    
    # =========================================================================
    # UI Event Handlers
    # =========================================================================
    
    def _on_layout_change(self, change: Dict[str, Any]) -> None:
        """Handle grid layout change."""
        new_rows, new_cols = change['new']
        
        # Skip if no actual change
        if (self.grid_rows, self.grid_cols) == (new_rows, new_cols):
            return
        
        # Capture current view range before changing layout
        current_range = self._capture_current_view_range()
        
        # Block range changes during layout update
        self._ignore_range_changes = True
        
        try:
            self.grid_rows, self.grid_cols = new_rows, new_cols
            self._update_grid_layout()
        finally:
            self._ignore_range_changes = False
        
        # Restore view range after layout change
        if current_range is not None:
            self._restore_view_range(current_range)
        
        # Update UI to show correct number of slot selectors
        self._update_selector_visibility()
        self._request_remote_draw()
        
        # Update ping lines if connected to a pingviewer
        if self.pingviewer is not None:
            self._update_ping_lines()
        
        # Trigger high-res update for visible slots
        if self._auto_update_enabled:
            self._schedule_debounced_update()
    
    def _capture_current_view_range(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Capture the current view range from master plot."""
        master = self._get_master_plot()
        if master is not None:
            vb = master.getViewBox()
            return tuple(vb.viewRange())
        return None
    
    def _restore_view_range(self, view_range: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        """Restore view range to master plot."""
        master = self._get_master_plot()
        if master is not None:
            self._ignore_range_changes = True
            try:
                x_range, y_range = view_range
                master.setXRange(x_range[0], x_range[1], padding=0)
                master.setYRange(y_range[0], y_range[1], padding=0)
            finally:
                self._ignore_range_changes = False
    
    def _update_selector_visibility(self) -> None:
        """Update the selector row to show correct number of slots."""
        if hasattr(self, 'selector_row'):
            n_visible = self.grid_rows * self.grid_cols
            visible_selectors = [self.slot_selectors[i] for i in range(n_visible)]
            self.selector_row.children = visible_selectors
    
    def _on_slot_change(self, slot_idx: int, change: Dict[str, Any]) -> None:
        """Handle slot echogram assignment change."""
        new_key = change['new']
        slot = self.slots[slot_idx]
        
        # Capture current view range before change
        current_range = self._capture_current_view_range()
        
        # Block view range changes during update
        self._ignore_range_changes = True
        
        try:
            # Allow same echogram in multiple slots - no swap logic
            # (assign_echogram now checks both local and global cache)
            slot.assign_echogram(new_key)
            
            # If still not loaded after cache checks, load from echogram
            if new_key and slot.background_image is None:
                echogram = self.echograms.get(new_key)
                if echogram:
                    self.progress.set_description(f"Loading {new_key}...")
                    if len(echogram.layers) == 0 and echogram.main_layer is None:
                        slot.background_image, slot.background_extent = echogram.build_image(progress=self.progress)
                    else:
                        slot.background_image, slot.layer_image, slot.background_extent = \
                            echogram.build_image_and_layer_image(progress=self.progress)
                        slot.layer_extent = slot.background_extent
                    self.progress.set_description("Idle")
                    # Add to global cache
                    self._global_image_cache[new_key] = {
                        'background_image': slot.background_image,
                        'background_extent': slot.background_extent,
                        'layer_image': slot.layer_image,
                        'layer_extent': slot.layer_extent,
                    }
            
            if slot.is_visible:
                self._update_slot(slot)
                self._process_qt_events()
        finally:
            self._ignore_range_changes = False
        
        # Restore view range after slot change
        if current_range is not None:
            self._restore_view_range(current_range)
        
        self._request_remote_draw()
        
        # Trigger high-res update for the new echogram (it may not have high-res for current view)
        if self._auto_update_enabled and slot.is_visible:
            self._schedule_debounced_update()
    
    def _on_global_color_change(self, change: Dict[str, Any]) -> None:
        """Handle global color scale change - applies to ALL colorbars."""
        new_vmin = self.w_vmin.value
        new_vmax = self.w_vmax.value
        
        # Update all slot colorbars and stored levels
        for slot in self.slots:
            slot.background_levels = (new_vmin, new_vmax)
            slot.layer_levels = (new_vmin, new_vmax)
            if slot.colorbar is not None:
                slot.colorbar.setLevels((new_vmin, new_vmax))
        
        self._request_remote_draw()
    
    def _on_colorbar_layer_change(self, change: Dict[str, Any]) -> None:
        """Handle colorbar layer selection change - applies to ALL slots."""
        new_layer = change['new']
        for slot in self._get_visible_slots():
            self._switch_colorbar_layer(slot, new_layer)
        self._request_remote_draw()
    
    def _switch_colorbar_layer(self, slot: EchogramSlot, new_layer: str) -> None:
        """Switch a slot's colorbar to display a different layer."""
        if slot.colorbar is None:
            slot.active_colorbar_layer = new_layer
            return
        
        old_layer = slot.active_colorbar_layer
        if old_layer == new_layer:
            return
        
        # Save current levels for the old layer
        current_levels = slot.colorbar.levels()
        if old_layer == 'background':
            slot.background_levels = current_levels
        else:
            slot.layer_levels = current_levels
        
        slot.active_colorbar_layer = new_layer
        
        # Switch to new layer
        if new_layer == 'layer' and slot.layer_image is not None:
            # Switch to layer image
            layer_img = slot.image_layers.get('layer')
            if layer_img is not None:
                slot.colorbar.setImageItem(layer_img)
                if hasattr(slot.colorbar, "setColorMap"):
                    slot.colorbar.setColorMap(slot.parent._colormap_layer)
                # Restore layer levels
                if slot.layer_levels is not None:
                    slot.colorbar.setLevels(slot.layer_levels)
        else:
            # Switch to background image
            bg_img = slot.image_layers.get('background')
            if bg_img is not None:
                slot.colorbar.setImageItem(bg_img)
                if hasattr(slot.colorbar, "setColorMap"):
                    slot.colorbar.setColorMap(slot.parent._colormap)
                # Restore background levels
                if slot.background_levels is not None:
                    slot.colorbar.setLevels(slot.background_levels)
    
    def _on_colorbar_levels_changed(self, slot: EchogramSlot, colorbar: pg.ColorBarItem) -> None:
        """Handle colorbar level change from user interaction."""
        # Get levels from the colorbar
        vmin, vmax = colorbar.levels()
        
        # Store levels for current layer
        if slot.active_colorbar_layer == 'layer':
            slot.layer_levels = (vmin, vmax)
            # Apply to layer image
            layer_img = slot.image_layers.get('layer')
            if layer_img is not None:
                layer_img.setLevels((vmin, vmax))
        else:
            slot.background_levels = (vmin, vmax)
            # Apply to background and high-res images
            for key in ['background', 'high']:
                img = slot.image_layers.get(key)
                if img is not None:
                    img.setLevels((vmin, vmax))
    
    def _on_auto_update_toggle(self, change: Dict[str, Any]) -> None:
        """Handle auto-update checkbox toggle."""
        self._auto_update_enabled = change['new']
        if not self._auto_update_enabled and self._debounce_task is not None:
            self._debounce_task.cancel()
            self._debounce_task = None
    
    def _on_update_click(self, _: Any = None) -> None:
        """Handle manual update button click."""
        self._trigger_high_res_update()
    
    def _on_reset_click(self, _: Any = None) -> None:
        """Handle reset view button click."""
        self._reset_view()
    
    def _show_single(self, echogram_name: str) -> None:
        """Show a single echogram full-size."""
        # Capture current view range
        current_range = self._capture_current_view_range()
        
        # Block view range changes during update
        self._ignore_range_changes = True
        
        try:
            # Check if we need to change grid layout
            need_grid_change = (self.grid_rows, self.grid_cols) != (1, 1)
            
            if need_grid_change:
                # Set grid to 1x1 (this will trigger _on_layout_change)
                self.w_layout.value = (1, 1)
            
            # Assign the echogram to slot 0 if different
            # (assign_echogram now checks both local and global cache)
            if self.slots[0].echogram_key != echogram_name:
                self.slots[0].assign_echogram(echogram_name)
                self.slot_selectors[0].value = echogram_name
            
            # Update slot display without rebuilding grid
            if not need_grid_change:
                self._update_slot(self.slots[0])
        finally:
            self._ignore_range_changes = False
        
        # Restore view range (don't reset to full extent)
        if current_range is not None:
            self._restore_view_range(current_range)
        
        self._request_remote_draw()
        
        # Trigger high-res update for new echogram
        if self._auto_update_enabled:
            self._schedule_debounced_update()
    
    def _reset_view(self) -> None:
        """Reset view to show full extent of all visible echograms."""
        minx, maxx = np.inf, -np.inf
        miny, maxy = np.inf, -np.inf
        
        for slot in self._get_visible_slots():
            if slot.background_extent is not None:
                x0, x1, y0, y1 = self._numeric_extent(slot.background_extent)
                minx = min(minx, x0)
                maxx = max(maxx, x1)
                miny = min(miny, y0)
                maxy = max(maxy, y1)
        
        master = self._get_master_plot()
        if master and np.all(np.isfinite([minx, maxx, miny, maxy])):
            self._ignore_range_changes = True
            try:
                master.setXRange(minx, maxx, padding=0)
                master.setYRange(miny, maxy, padding=0)
            finally:
                self._ignore_range_changes = False
        
        self._request_remote_draw()
    
    # =========================================================================
    # Mouse Event Handlers
    # =========================================================================
    
    def _handle_scene_click(self, event: Any) -> None:
        """Handle mouse click on scene."""
        # Ensure widget has focus for keyboard events (important for shortcuts)
        gfx_view = getattr(self.graphics, "gfxView", None)
        if gfx_view is not None:
            gfx_view.setFocus()
        
        pos = event.scenePos()
        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            vb = slot.plot_item.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                point = vb.mapSceneToView(pos)
                # Update pingviewer if connected
                if self.pingviewer is not None:
                    self._update_pingviewer_from_coordinate(point.x())
                    self._update_ping_lines()
                break
    
    def _handle_scene_move(self, pos: QtCore.QPointF) -> None:
        """Handle mouse move over scene - update crosshairs."""
        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            vb = slot.plot_item.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                point = vb.mapSceneToView(pos)
                x, y = point.x(), point.y()
                
                # Update hover label
                value = self._sample_value(slot, x, y)
                self._update_hover_label(x, y, value, slot.echogram_key)
                
                # Update crosshairs on all visible plots
                if self._crosshair_enabled:
                    self._update_crosshairs(x, y)
                return
        
        # Mouse not over any plot
        self.hover_label.value = "&nbsp;"
        if self._crosshair_enabled:
            self._hide_crosshairs()
    
    def _update_crosshairs(self, x: float, y: float) -> None:
        """Update crosshair position on all visible plots."""
        self._crosshair_position = (x, y)
        self._last_crosshair_position = (x, y)  # Store for add point button
        for slot in self._get_visible_slots():
            if slot.crosshair_v and slot.crosshair_h:
                slot.crosshair_v.setValue(x)
                slot.crosshair_h.setValue(y)
                slot.crosshair_v.show()
                slot.crosshair_h.show()
    
    def _hide_crosshairs(self) -> None:
        """Hide all crosshairs."""
        self._crosshair_position = None
        for slot in self.slots:
            if slot.crosshair_v:
                slot.crosshair_v.hide()
            if slot.crosshair_h:
                slot.crosshair_h.hide()
    
    def _sample_value(self, slot: EchogramSlot, x: float, y: float) -> Optional[float]:
        """Sample value at coordinates from slot's image."""
        sources = [
            (slot.high_res_image, slot.high_res_extent),
            (slot.background_image, slot.background_extent),
        ]
        for image, extent in sources:
            if image is None or extent is None:
                continue
            x0, x1, y0, y1 = self._numeric_extent(extent)
            dx, dy = x1 - x0, y1 - y0
            if dx == 0 or dy == 0:
                continue
            col = (x - x0) / dx * (image.shape[1] - 1)
            row = (y - y0) / dy * (image.shape[0] - 1)
            if 0 <= col < image.shape[1] and 0 <= row < image.shape[0]:
                return float(image[int(row), int(col)])
        return None
    
    def _update_hover_label(self, x: float, y: float, value: Optional[float], name: Optional[str]) -> None:
        """Update the hover label with current position and value."""
        x_text = self._format_x_value(x)
        y_text = f"{y:0.2f}"
        value_text = f"{value:0.2f}" if value is not None else "--"
        name_text = f" [{name}]" if name else ""
        self.hover_label.value = (
            f"<b>x</b>: {x_text} | <b>y</b>: {y_text} | <b>value</b>: {value_text}{name_text}"
        )
    
    # =========================================================================
    # Background Loading
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
    
    def _trigger_high_res_update(self) -> None:
        """Trigger high-res image loading for visible slots."""
        if self._is_shutting_down:
            return
        
        self._cancel_pending_load()
        
        # Capture view params
        view_params = self._capture_view_params()
        
        # Get visible slots with echograms
        visible_slots = self._get_visible_slots()
        if not visible_slots:
            return
        
        self._is_loading = True
        self._view_changed_during_load = False
        self._cancel_flag.clear()
        self.progress.set_description('Loading...')
        
        viewer = self
        
        def load_images():
            results = {}
            for slot in visible_slots:
                if viewer._cancel_flag.is_set():
                    return None
                
                echogram = slot.get_echogram()
                if echogram is None:
                    continue
                
                # Apply axis limits
                params = view_params.get(slot.slot_idx, {})
                if params:
                    viewer._apply_axis_limits(
                        echogram,
                        params['xmin'], params['xmax'],
                        params['ymin'], params['ymax']
                    )
                
                # Build high-res image
                if len(echogram.layers) == 0 and echogram.main_layer is None:
                    image, extent = echogram.build_image(progress=None)
                    results[slot.slot_idx] = {'high': image, 'extent': extent}
                else:
                    image, layer_img, extent = echogram.build_image_and_layer_image(progress=None)
                    results[slot.slot_idx] = {
                        'high': image, 'extent': extent,
                        'layer': layer_img, 'layer_extent': extent
                    }
            
            return results
        
        def apply_results(results):
            viewer._is_loading = False
            if results is None:
                viewer.progress.set_description('Cancelled')
                if viewer._view_changed_during_load:
                    viewer._view_changed_during_load = False
                    viewer._schedule_debounced_update()
                return
            
            for slot_idx, data in results.items():
                slot = viewer.slots[slot_idx]
                slot.high_res_image = data.get('high')
                slot.high_res_extent = data.get('extent')
                if 'layer' in data:
                    slot.layer_image = data['layer']
                    slot.layer_extent = data['layer_extent']
                viewer._update_slot(slot)
                
                # Update global cache with high-res data
                if slot.echogram_key and slot.echogram_key in viewer._global_image_cache:
                    viewer._global_image_cache[slot.echogram_key]['high_res_image'] = slot.high_res_image
                    viewer._global_image_cache[slot.echogram_key]['high_res_extent'] = slot.high_res_extent
            
            viewer._process_qt_events()
            viewer._request_remote_draw()
            viewer.progress.set_description('Idle')
            
            if viewer._view_changed_during_load:
                viewer._view_changed_during_load = False
                viewer._schedule_debounced_update()
        
        async def run_async():
            try:
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(viewer._executor, load_images)
                apply_results(results)
            except Exception as e:
                viewer._is_loading = False
                with viewer.output:
                    print(f"Error: {e}")
                viewer.progress.set_description('Error')
        
        try:
            loop = asyncio.get_running_loop()
            self._loading_future = loop.create_task(run_async())
        except RuntimeError:
            results = load_images()
            apply_results(results)
    
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
    
    def _capture_view_params(self) -> Dict[int, Dict[str, float]]:
        """Capture current view parameters for visible slots."""
        params = {}
        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            vb = slot.plot_item.getViewBox()
            xmin, xmax = vb.viewRange()[0]
            ymin, ymax = vb.viewRange()[1]
            params[slot.slot_idx] = {
                'xmin': xmin, 'xmax': xmax,
                'ymin': ymin, 'ymax': ymax
            }
        return params
    
    # =========================================================================
    # Navigation
    # =========================================================================
    
    def pan_view(self, direction: str, fraction: float = 0.25) -> None:
        """Pan the view in a direction."""
        master = self._get_master_plot()
        if not master:
            return
        
        vb = master.getViewBox()
        x_range = vb.viewRange()[0]
        y_range = vb.viewRange()[1]
        
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        
        dx, dy = 0.0, 0.0
        if direction == 'left':
            dx = -x_span * fraction
        elif direction == 'right':
            dx = x_span * fraction
        elif direction == 'up':
            dy = y_span * fraction
        elif direction == 'down':
            dy = -y_span * fraction
        
        self._ignore_range_changes = True
        try:
            vb.setXRange(x_range[0] + dx, x_range[1] + dx, padding=0)
            vb.setYRange(y_range[0] + dy, y_range[1] + dy, padding=0)
        finally:
            self._ignore_range_changes = False
        
        self._request_remote_draw()
        self._last_range_change_time = time.time()
        self._schedule_debounced_update()
    
    # =========================================================================
    # Pingviewer Integration
    # =========================================================================
    
    def connect_pingviewer(self, pingviewer: Any) -> None:
        """Connect to a pingviewer for synchronized display.
        
        If the pingviewer supports ping change callbacks (e.g., WCIViewerMultiChannel),
        automatically registers to update pinglines when the ping changes.
        """
        # Disconnect from any existing pingviewer first
        if self.pingviewer is not None:
            self.disconnect_pingviewer()
        
        self.pingviewer = pingviewer
        self._update_ping_lines()
        
        # Auto-register for ping change callbacks if supported
        if hasattr(pingviewer, 'register_ping_change_callback'):
            pingviewer.register_ping_change_callback(self._update_ping_lines)
    
    def disconnect_pingviewer(self) -> None:
        """Disconnect from pingviewer."""
        # Unregister callback if we were registered
        if self.pingviewer is not None and hasattr(self.pingviewer, 'unregister_ping_change_callback'):
            self.pingviewer.unregister_ping_change_callback(self._update_ping_lines)
        
        self.pingviewer = None
        for slot in self.slots:
            if slot.pingline:
                slot.pingline.hide()
    
    def update_ping_lines(self) -> None:
        """Public method to update ping lines from connected pingviewer."""
        self._update_ping_lines()
    
    def _get_pingviewer_pings(self) -> Optional[Sequence[Any]]:
        """Get pings from pingviewer (supports both single and multi-channel viewers)."""
        if self.pingviewer is None:
            return None
        # Multi-channel viewer: get pings from first visible slot
        if hasattr(self.pingviewer, 'slots'):
            for slot in self.pingviewer.slots:
                if slot.is_visible and slot.get_pings() is not None:
                    return slot.get_pings()
            return None
        # Single-channel viewer
        if hasattr(self.pingviewer, 'imagebuilder'):
            return self.pingviewer.imagebuilder.pings
        return None
    
    def _get_pingviewer_current_index(self) -> int:
        """Get current ping index from pingviewer."""
        if self.pingviewer is None:
            return 0
        # Multi-channel viewer
        if hasattr(self.pingviewer, 'slots'):
            for slot in self.pingviewer.slots:
                if slot.is_visible:
                    return slot.ping_index
            return 0
        # Single-channel viewer
        return self.pingviewer.w_index.value
    
    def _update_pingviewer_from_coordinate(self, coord: float) -> None:
        """Update pingviewer from x coordinate."""
        if self.pingviewer is None:
            return
        
        pings = self._get_pingviewer_pings()
        if pings is None:
            return
        
        match self.x_axis_name:
            case "Ping number" | "Ping index":
                new_idx = int(max(0, min(coord, len(pings) - 1)))
                self._set_pingviewer_index(new_idx)
            case "Date time":
                target = self._mpl_num_to_datetime(coord).timestamp()
                for idx, ping in enumerate(pings):
                    ping_obj = ping if not isinstance(ping, dict) else next(iter(ping.values()))
                    if ping_obj.get_datetime().timestamp() > target:
                        self._set_pingviewer_index(max(0, idx - 1))
                        return
                self._set_pingviewer_index(len(pings) - 1)
            case "Ping time":
                target = coord
                for idx, ping in enumerate(pings):
                    ping_obj = ping if not isinstance(ping, dict) else next(iter(ping.values()))
                    if ping_obj.get_timestamp() > target:
                        self._set_pingviewer_index(max(0, idx - 1))
                        return
                self._set_pingviewer_index(len(pings) - 1)
    
    def _set_pingviewer_index(self, idx: int) -> None:
        """Set ping index on pingviewer."""
        if self.pingviewer is None:
            return
        # Multi-channel viewer
        if hasattr(self.pingviewer, 'ping_sliders'):
            # Set on first visible slot (will trigger sync)
            for i, slot in enumerate(self.pingviewer.slots):
                if slot.is_visible:
                    self.pingviewer.ping_sliders[i].value = idx
                    break
        # Single-channel viewer
        elif hasattr(self.pingviewer, 'w_index'):
            self.pingviewer.w_index.value = idx
    
    def _update_ping_lines(self) -> None:
        """Update ping lines on all visible plots.
        
        If auto-follow is enabled, also smoothly scrolls the view to keep
        the pingline visible (triggers when pingline is out of view or 
        within 10% of either edge).
        """
        if self.pingviewer is None:
            return
        
        ping = self._get_current_ping()
        if ping is None:
            return
        
        match self.x_axis_name:
            case "Ping number" | "Ping index":
                value = float(self._get_pingviewer_current_index())
            case "Date time":
                value = self._datetime_to_mpl_num(ping.get_datetime())
            case "Ping time":
                value = ping.get_timestamp()
            case _:
                return
        
        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            if slot.pingline is None:
                line = pg.InfiniteLine(angle=90, pen=pg.mkPen(color='k', style=QtCore.Qt.PenStyle.DashLine))
                slot.plot_item.addItem(line)
                slot.pingline = line
            slot.pingline.setValue(value)
            slot.pingline.show()
        
        # Auto-follow: smoothly scroll if pingline is near edge or out of view
        if self.w_auto_follow.value:
            self._auto_follow_pingline(value)
        
        self._request_remote_draw()
    
    def _auto_follow_pingline(self, pingline_x: float) -> None:
        """Smoothly scroll view to keep pingline visible.
        
        Triggers when pingline is:
        - Out of view, OR
        - Within 10% of either edge
        
        Re-centers to place pingline at 30% from left (shows more "future" data).
        Uses smooth animated transition.
        
        Args:
            pingline_x: The x coordinate of the pingline.
        """
        master = self._get_master_plot()
        if master is None:
            return
        
        vb = master.getViewBox()
        (xmin, xmax), _ = vb.viewRange()
        x_extent = xmax - xmin
        
        if x_extent <= 0:
            return
        
        # Calculate pingline position as fraction of visible range
        position_fraction = (pingline_x - xmin) / x_extent
        
        # Check if we need to scroll:
        # - Out of view: position_fraction < 0 or > 1
        # - Near left edge: position_fraction < 0.10
        # - Near right edge: position_fraction > 0.90
        edge_threshold = 0.10
        needs_scroll = (
            position_fraction < 0 or 
            position_fraction > 1 or
            position_fraction < edge_threshold or 
            position_fraction > (1 - edge_threshold)
        )
        
        if not needs_scroll:
            return
        
        # Target position: pingline at 30% from left edge
        target_position = 0.30
        new_xmin = pingline_x - (target_position * x_extent)
        new_xmax = new_xmin + x_extent
        
        # Smooth animated scroll
        self._ignore_range_changes = True
        try:
            # PyQtGraph supports animated range changes
            vb.setXRange(new_xmin, new_xmax, padding=0)
        finally:
            self._ignore_range_changes = False
        
        # Trigger high-res update if auto-update is enabled
        if self._auto_update_enabled:
            self._schedule_debounced_update()
    
    def _goto_pingline(self) -> None:
        """Center the view on the current pingline position.
        
        Updates pingline first, then centers view on its x coordinate
        while keeping current x and y extent.
        """
        # First update pinglines to get latest position
        self._update_ping_lines()
        
        # Get pingline x coordinate
        ping = self._get_current_ping()
        if ping is None:
            return
        
        match self.x_axis_name:
            case "Ping number" | "Ping index":
                pingline_x = float(self._get_pingviewer_current_index())
            case "Date time":
                pingline_x = self._datetime_to_mpl_num(ping.get_datetime())
            case "Ping time":
                pingline_x = ping.get_timestamp()
            case _:
                return
        
        # Get current view range
        master = self._get_master_plot()
        if master is None:
            return
        
        vb = master.getViewBox()
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        
        # Calculate current extent
        x_extent = xmax - xmin
        
        # Center on pingline x coordinate, keep y extent
        new_xmin = pingline_x - x_extent / 2
        new_xmax = pingline_x + x_extent / 2
        
        # Apply new range
        self._ignore_range_changes = True
        try:
            master.setXRange(new_xmin, new_xmax, padding=0)
        finally:
            self._ignore_range_changes = False
        
        self._request_remote_draw()
        
        # Trigger high-res update if auto-update is enabled
        if self._auto_update_enabled:
            self._schedule_debounced_update()
    
    def _get_current_ping(self) -> Optional[Any]:
        """Get current ping from pingviewer."""
        pings = self._get_pingviewer_pings()
        if pings is None:
            return None
        idx = self._get_pingviewer_current_index()
        if 0 <= idx < len(pings):
            ping = pings[idx]
            if isinstance(ping, dict):
                return next(iter(ping.values()))
            return ping
        return None
    
    # =========================================================================
    # Axis Limits
    # =========================================================================
    
    def _apply_axis_limits(self, echogram: Any, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        """Apply axis limits to an echogram builder."""
        x_kwargs = echogram.get_x_kwargs()
        y_kwargs = echogram.get_y_kwargs()
        
        match self.x_axis_name:
            case "Date time":
                tmin, tmax = self._mpl_num_to_datetime([xmin, xmax])
                x_kwargs["min_ping_time"] = tmin
                x_kwargs["max_ping_time"] = tmax
                echogram.set_x_axis_date_time(**x_kwargs)
            case "Ping number":
                x_kwargs["min_ping_nr"] = xmin
                x_kwargs["max_ping_nr"] = xmax
                echogram.set_x_axis_ping_nr(**x_kwargs)
            case "Ping index":
                x_kwargs["min_ping_index"] = xmin
                x_kwargs["max_ping_index"] = xmax
                echogram.set_x_axis_ping_index(**x_kwargs)
            case "Ping time":
                x_kwargs["min_timestamp"] = xmin
                x_kwargs["max_timestamp"] = xmax
                echogram.set_x_axis_ping_time(**x_kwargs)
        
        match self.y_axis_name:
            case "Depth (m)":
                y_kwargs["min_depth"] = ymin
                y_kwargs["max_depth"] = ymax
                echogram.set_y_axis_depth(**y_kwargs)
            case "Range (m)":
                y_kwargs["min_range"] = ymin
                y_kwargs["max_range"] = ymax
                echogram.set_y_axis_range(**y_kwargs)
            case "Sample number":
                y_kwargs["min_sample_nr"] = ymin
                y_kwargs["max_sample_nr"] = ymax
                echogram.set_y_axis_sample_nr(**y_kwargs)
            case "Y indice":
                y_kwargs["min_sample_nr"] = ymin
                y_kwargs["max_sample_nr"] = ymax
                echogram.set_y_axis_y_indice(**y_kwargs)
    
    # =========================================================================
    # Display
    # =========================================================================
    
    def show(self) -> None:
        """Display the viewer widget."""
        # Build layout
        tab_row = ipywidgets.HBox(self.tab_buttons)
        
        # Create selector row (just slot selectors)
        n_visible = self.grid_rows * self.grid_cols
        visible_selectors = [self.slot_selectors[i] for i in range(n_visible)]
        self.selector_row = ipywidgets.HBox(visible_selectors)
        
        controls_row = ipywidgets.HBox([
            self.w_layout,
            self.w_colorbar_layer,
            self.w_vmin, self.w_vmax,
            self.w_auto_update, self.w_crosshair,
        ])
        
        buttons_row = ipywidgets.HBox([
            self.btn_update, self.btn_reset, self.w_auto_follow, self.btn_goto_pingline,
            ipywidgets.Label('  Nav:'),
            self.btn_nav_left, self.btn_nav_up, self.btn_nav_down, self.btn_nav_right,
        ])
        
        widgets = [
            tab_row,
            self.selector_row,
            ipywidgets.HBox([self.graphics]),
            controls_row,
            buttons_row,
            self._build_param_editor_row(),
            self.hover_label,
        ]
        
        if self.display_progress:
            widgets.append(ipywidgets.HBox([self.progress]))
        
        widgets.append(self.output)
        
        self.layout = ipywidgets.VBox(widgets)
        display(self.layout)
    
    def set_widget_size(self, width_px: int, height_px: int) -> None:
        """Set widget dimensions."""
        self.widget_width_px = width_px
        self.widget_height_px = height_px
        pgh.apply_widget_layout(self.graphics, width_px, height_px)
        self._request_remote_draw()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @staticmethod
    def _mpl_num_to_datetime(value: Union[float, Sequence[float]]) -> Union[datetime, List[datetime]]:
        base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return [base + timedelta(days=float(v)) for v in value]
        return base + timedelta(days=float(value))
    
    @staticmethod
    def _datetime_to_mpl_num(value: datetime) -> float:
        base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        dt_value = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        delta = dt_value - base
        return delta.total_seconds() / 86400.0
    
    def _format_x_value(self, coord: float) -> str:
        match self.x_axis_name:
            case "Date time":
                return self._mpl_num_to_datetime(coord).isoformat(sep=" ")
            case "Ping time":
                return f"{coord:0.2f} s"
            case "Ping index":
                return f"{coord:0.0f}"
            case _:
                return f"{coord:0.2f}"
    
    def _numeric_extent(self, extent: Tuple[Any, Any, Any, Any]) -> Tuple[float, float, float, float]:
        return tuple(self._extent_value_to_float(v) for v in extent)
    
    def _extent_value_to_float(self, value: Any) -> float:
        if isinstance(value, datetime):
            return self._datetime_to_mpl_num(value)
        if isinstance(value, timedelta):
            return value.total_seconds() / 86400.0
        if isinstance(value, np.datetime64):
            delta = value - np.datetime64("1970-01-01T00:00:00Z")
            seconds = delta / np.timedelta64(1, "s")
            return float(seconds) / 86400.0
        if isinstance(value, np.timedelta64):
            seconds = value / np.timedelta64(1, "s")
            return float(seconds) / 86400.0
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)
    
    @staticmethod
    def _process_qt_events() -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
    
    def _request_remote_draw(self) -> None:
        request_draw = getattr(self.graphics, "request_draw", None)
        if callable(request_draw):
            request_draw()
    
    # =========================================================================
    # Station Time Markers
    # =========================================================================
    
    def add_station_times(
        self,
        stations: Dict[str, Tuple[Any, Any]],
        line_color: str = '#424242',  # Dark grey
        line_width: int = 2,
        line_style: str = 'dash',
        region_color: Optional[str] = None,  # Auto-derive from line_color if None
        region_alpha: float = 0.15,
        label_color: Optional[str] = None,  # Auto-derive from line_color if None
        label_size: str = '10pt',
        label_position: str = 'top',  # 'top' or 'bottom'
    ) -> None:
        """Add station time markers to all visible echograms.
        
        Draws vertical lines at station start/end times with a semi-transparent
        region between them and a label showing the station name.
        
        Args:
            stations: Dict mapping station names to (start_time, end_time) tuples.
                      Times can be datetime objects or unix timestamps (float/int).
            line_color: Color for the vertical lines (hex or named color).
            line_width: Width of the vertical lines in pixels.
            line_style: Line style - 'solid', 'dash', 'dot', or 'dashdot'.
            region_color: Fill color for the region between lines. If None,
                         uses line_color with region_alpha transparency.
            region_alpha: Alpha (0-1) for the region fill color.
            label_color: Color for the station label text. If None, uses line_color.
            label_size: Font size for labels (e.g., '10pt', '12pt').
            label_position: Where to place the label - 'top' or 'bottom'.
        
        Example:
            >>> viewer.add_station_times({
            ...     'Station A': (datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 30)),
            ...     'Station B': (1704110400.0, 1704112200.0),  # unix timestamps
            ... })
        """
        # Clear existing stations first
        self.clear_station_times()
        
        # Store station data for recreation after layout changes
        self._station_data = {
            'stations': stations,
            'line_color': line_color,
            'line_width': line_width,
            'line_style': line_style,
            'region_color': region_color,
            'region_alpha': region_alpha,
            'label_color': label_color,
            'label_size': label_size,
            'label_position': label_position,
        }
        
        # Convert line style to Qt pen style
        style_map = {
            'solid': QtCore.Qt.PenStyle.SolidLine,
            'dash': QtCore.Qt.PenStyle.DashLine,
            'dot': QtCore.Qt.PenStyle.DotLine,
            'dashdot': QtCore.Qt.PenStyle.DashDotLine,
        }
        pen_style = style_map.get(line_style, QtCore.Qt.PenStyle.SolidLine)
        
        # Use line color for region and label if not specified
        effective_label_color = label_color if label_color is not None else line_color
        
        for station_name, (start_time, end_time) in stations.items():
            # Convert times to x-axis coordinates
            start_x = self._time_to_x_coord(start_time)
            end_x = self._time_to_x_coord(end_time)
            
            if start_x is None or end_x is None:
                continue
            
            # Add markers to all visible slots
            for slot in self._get_visible_slots():
                self._add_station_marker_to_slot(
                    slot, station_name, start_x, end_x,
                    line_color, line_width, pen_style,
                    region_color, region_alpha,
                    effective_label_color, label_size, label_position
                )
        
        self._request_remote_draw()
    
    def _time_to_x_coord(self, time_value: Any) -> Optional[float]:
        """Convert a time value to x-axis coordinate.
        
        Automatically detects whether time_value is a datetime or unix timestamp.
        """
        # Check if it's a datetime object
        if isinstance(time_value, datetime):
            if self.x_axis_name == "Date time":
                return self._datetime_to_mpl_num(time_value)
            elif self.x_axis_name == "Ping time":
                return time_value.timestamp()
            else:
                # For ping number/index axes, we can't directly convert datetime
                # Would need to find the ping index at this time
                return self._find_ping_index_for_time(time_value.timestamp())
        
        # Check if it's a numeric value (unix timestamp)
        elif isinstance(time_value, (int, float)):
            if self.x_axis_name == "Date time":
                # Convert unix timestamp to matplotlib num
                dt = datetime.fromtimestamp(time_value, tz=timezone.utc)
                return self._datetime_to_mpl_num(dt)
            elif self.x_axis_name == "Ping time":
                return float(time_value)
            else:
                # For ping number/index axes
                return self._find_ping_index_for_time(time_value)
        
        return None
    
    def _find_ping_index_for_time(self, timestamp: float) -> Optional[float]:
        """Find the ping index closest to a given unix timestamp."""
        # Try to get pings from the first echogram
        if not self.echograms:
            return None
        
        first_echogram = next(iter(self.echograms.values()))
        if hasattr(first_echogram, 'pings') and first_echogram.pings:
            pings = first_echogram.pings
            for idx, ping in enumerate(pings):
                ping_obj = ping if not isinstance(ping, dict) else next(iter(ping.values()))
                if ping_obj.get_timestamp() >= timestamp:
                    return float(idx)
            return float(len(pings) - 1)
        
        return None
    
    def _add_station_marker_to_slot(
        self,
        slot: EchogramSlot,
        station_name: str,
        start_x: float,
        end_x: float,
        line_color: str,
        line_width: int,
        pen_style: Any,
        region_color: Optional[str],
        region_alpha: float,
        label_color: str,
        label_size: str,
        label_position: str,
    ) -> None:
        """Add station markers to a single slot."""
        if slot.plot_item is None:
            return
        
        # Create pen for lines
        pen = pg.mkPen(color=line_color, width=line_width, style=pen_style)
        
        # Create start line
        start_line = pg.InfiniteLine(pos=start_x, angle=90, pen=pen)
        slot.plot_item.addItem(start_line)
        
        # Create end line
        end_line = pg.InfiniteLine(pos=end_x, angle=90, pen=pen)
        slot.plot_item.addItem(end_line)
        
        # Create region between lines
        if region_color is None:
            # Parse line_color and apply alpha
            from pyqtgraph.Qt.QtGui import QColor
            qcolor = QColor(line_color)
            qcolor.setAlphaF(region_alpha)
            brush = pg.mkBrush(qcolor)
        else:
            from pyqtgraph.Qt.QtGui import QColor
            qcolor = QColor(region_color)
            qcolor.setAlphaF(region_alpha)
            brush = pg.mkBrush(qcolor)
        
        region = pg.LinearRegionItem(
            values=(start_x, end_x),
            orientation='vertical',
            brush=brush,
            pen=pg.mkPen(None),  # No border (we have lines)
            movable=False,
        )
        # Insert region behind images but in front of background
        slot.plot_item.addItem(region)
        region.setZValue(-100)  # Behind images
        
        # Create label
        label = pg.TextItem(
            text=station_name,
            color=label_color,
            anchor=(0.5, 1.0 if label_position == 'top' else 0.0),
        )
        label.setFont(pg.QtGui.QFont('Arial', int(label_size.replace('pt', ''))))
        slot.plot_item.addItem(label)
        
        # Position label at center of region
        center_x = (start_x + end_x) / 2
        self._update_label_position(slot, label, center_x, label_position)
        
        # Store references for cleanup and updates
        slot.station_markers[station_name] = {
            'start_line': start_line,
            'end_line': end_line,
            'region': region,
            'label': label,
            'label_position': label_position,
            'center_x': center_x,
        }
        
        # Connect to view range changes to update label position
        vb = slot.plot_item.getViewBox()
        vb.sigRangeChanged.connect(
            lambda _, s=slot, lbl=label, cx=center_x, pos=label_position: 
            self._update_label_position(s, lbl, cx, pos)
        )
    
    def _update_label_position(
        self, 
        slot: EchogramSlot, 
        label: pg.TextItem, 
        center_x: float,
        label_position: str
    ) -> None:
        """Update label position based on current view range."""
        if slot.plot_item is None:
            return
        
        vb = slot.plot_item.getViewBox()
        y_range = vb.viewRange()[1]
        
        # Position at top or bottom of view
        if label_position == 'top':
            # Near top with small margin
            y_pos = y_range[0] + (y_range[1] - y_range[0]) * 0.02
        else:
            # Near bottom with small margin  
            y_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.02
        
        label.setPos(center_x, y_pos)
    
    def clear_station_times(self, station_name: Optional[str] = None) -> None:
        """Remove station time markers.
        
        Args:
            station_name: If specified, remove only this station's markers.
                         If None, remove all station markers.
        """
        for slot in self.slots:
            if station_name is not None:
                # Remove specific station
                if station_name in slot.station_markers:
                    self._remove_station_marker_from_slot(slot, station_name)
            else:
                # Remove all stations
                for name in list(slot.station_markers.keys()):
                    self._remove_station_marker_from_slot(slot, name)
        
        # Clear stored data if clearing all
        if station_name is None:
            self._station_data = None
        
        self._request_remote_draw()
    
    def _remove_station_marker_from_slot(self, slot: EchogramSlot, station_name: str) -> None:
        """Remove a station marker from a slot."""
        if station_name not in slot.station_markers:
            return
        
        marker = slot.station_markers[station_name]
        if slot.plot_item is not None:
            for item_key in ['start_line', 'end_line', 'region', 'label']:
                item = marker.get(item_key)
                if item is not None:
                    try:
                        slot.plot_item.removeItem(item)
                    except Exception:
                        pass
        
        del slot.station_markers[station_name]
    
    def _recreate_station_markers(self) -> None:
        """Recreate station markers after layout change."""
        if hasattr(self, '_station_data') and self._station_data is not None:
            data = self._station_data
            # Temporarily clear the data to avoid clearing in add_station_times
            self._station_data = None
            self.add_station_times(
                data['stations'],
                line_color=data['line_color'],
                line_width=data['line_width'],
                line_style=data['line_style'],
                region_color=data['region_color'],
                region_alpha=data['region_alpha'],
                label_color=data['label_color'],
                label_size=data['label_size'],
                label_position=data['label_position'],
            )
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._is_shutting_down = True
        self._cancel_pending_load()
        
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
            self._debounce_task = None
        
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)
    
    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
