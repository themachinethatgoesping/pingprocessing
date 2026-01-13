"""Enhanced PyQtGraph-based multi-echogram viewer with grid layout and lazy updates.

Features:
- Grid layout selector (1, 2, 2x2, 3x2, 4x2)
- Per-slot dropdown to select which echogram/frequency to display
- Visibility-based updates (inactive echograms don't update until shown)
- Synchronized crosshair for target investigation across frequencies
- Tab-based quick access for single echogram view
- Lazy loading pattern for performance
"""
from __future__ import annotations

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import ipywidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.jupyter import GraphicsLayoutWidget
from pyqtgraph.Qt import QtCore, QtWidgets

import themachinethatgoesping as theping
from . import pyqtgraph_helpers as pgh


def _get_axis_names(echogram):
    """Get x_axis_name and y_axis_name from echogram (old or new builder)."""
    if hasattr(echogram, 'coord_system'):
        return echogram.coord_system.x_axis_name, echogram.coord_system.y_axis_name
    return echogram.x_axis_name, echogram.y_axis_name


class EchogramSlot:
    """Manages a single echogram display slot with lazy loading."""
    
    def __init__(self, slot_idx: int, parent: 'MultiEchogramViewer'):
        self.slot_idx = slot_idx
        self.parent = parent
        self.echogram_key: Optional[str] = None  # Key into parent.echograms dict
        self.is_visible = False
        self.needs_update = False  # Dirty flag
        
        # Image data
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
            self.echogram_key = echogram_key
            self.background_image = None
            self.background_extent = None
            self.high_res_image = None
            self.high_res_extent = None
            self.layer_image = None
            self.layer_extent = None
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


class MultiEchogramViewer:
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
        else:
            echogramdata = list(echogramdata)
            if names is not None:
                self.echogram_names = [n if n else f"Echogram {i}" for i, n in enumerate(names)]
            else:
                self.echogram_names = [f"Echogram {i}" for i in range(len(echogramdata))]
            self.echograms = {name: eg for name, eg in zip(self.echogram_names, echogramdata)}
        
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
        
        # Grid layout state
        self.grid_rows, self.grid_cols = initial_grid
        self.max_slots = 8  # Maximum number of slots
        
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
            btn = ipywidgets.Button(
                description=name[:15],  # Truncate long names
                tooltip=f"Show {name} full-size",
                layout=ipywidgets.Layout(width='auto', min_width='60px'),
            )
            btn.on_click(lambda _, n=name: self._show_single(n))
            self.tab_buttons.append(btn)
        
        # Color scale sliders
        self.w_vmin = ipywidgets.FloatSlider(
            description="vmin", min=-150, max=100, step=5,
            value=self.args_plot["vmin"],
            layout=ipywidgets.Layout(width='250px'),
        )
        self.w_vmax = ipywidgets.FloatSlider(
            description="vmax", min=-150, max=100, step=5,
            value=self.args_plot["vmax"],
            layout=ipywidgets.Layout(width='250px'),
        )
        self.w_vmin.observe(self._on_color_change, names='value')
        self.w_vmax.observe(self._on_color_change, names='value')
        
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
    
    def _make_graphics_widget(self) -> None:
        """Create the PyQtGraph graphics widget."""
        self.graphics = GraphicsLayoutWidget(
            css_width=f"{self.widget_width_px}px",
            css_height=f"{self.widget_height_px}px"
        )
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, self.widget_height_px)
        if hasattr(self.graphics, "gfxView"):
            self.graphics.gfxView.setBackground("w")
        
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
            title = slot.echogram_key or f"Slot {i+1}"
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
            
            # Create colorbar
            try:
                colorbar = pg.ColorBarItem(
                    label="(dB)",
                    values=(self.args_plot["vmin"], self.args_plot["vmax"])
                )
                colorbar.setImageItem(background, insert_in=plot)
                if hasattr(colorbar, "setColorMap"):
                    colorbar.setColorMap(self._colormap)
                slot.colorbar = colorbar
            except AttributeError:
                slot.colorbar = None
            
            # Create crosshairs
            pen_cross = pg.mkPen(color='r', width=1, style=QtCore.Qt.PenStyle.DashLine)
            slot.crosshair_v = pg.InfiniteLine(angle=90, pen=pen_cross)
            slot.crosshair_h = pg.InfiniteLine(angle=0, pen=pen_cross)
            slot.crosshair_v.hide()
            slot.crosshair_h.hide()
            plot.addItem(slot.crosshair_v)
            plot.addItem(slot.crosshair_h)
            
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
    
    def _connect_scene_events(self) -> None:
        """Connect mouse events for crosshair and click handling."""
        gfx_view = getattr(self.graphics, "gfxView", None)
        scene = gfx_view.scene() if gfx_view is not None else None
        if scene is None:
            return
        
        try:
            scene.sigMouseClicked.disconnect()
            scene.sigMouseMoved.disconnect()
        except (TypeError, RuntimeError):
            pass
        
        scene.sigMouseClicked.connect(self._handle_scene_click)
        scene.sigMouseMoved.connect(self._handle_scene_move)
    
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
        
        # Update title
        slot.plot_item.setTitle(slot.echogram_key)
        
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
        
        colormap = self._colormap_layer if key == "layer" else self._colormap
        if hasattr(image_item, "setColorMap"):
            image_item.setColorMap(colormap)
        else:
            lut = colormap.getLookupTable(256)
            image_item.setLookupTable(lut)
        
        vmin, vmax = self._current_levels(slot.echogram_key)
        image_item.setLevels((vmin, vmax))
        image_item.show()
        
        if slot.colorbar is not None and key == "background":
            if hasattr(slot.colorbar, "setColorMap"):
                slot.colorbar.setColorMap(colormap)
            slot.colorbar.setLevels((vmin, vmax))
    
    def _current_levels(self, echogram_key: Optional[str]) -> Tuple[float, float]:
        """Get current color levels for an echogram."""
        offset = self.voffsets.get(echogram_key, 0.0) if echogram_key else 0.0
        return float(self.w_vmin.value + offset), float(self.w_vmax.value + offset)
    
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
            else:
                image, layer_img, extent = echogram.build_image_and_layer_image(progress=self.progress)
                slot.background_image = image
                slot.background_extent = extent
                slot.layer_image = layer_img
                slot.layer_extent = extent
            
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
        self.grid_rows, self.grid_cols = change['new']
        self._update_grid_layout()
        self._request_remote_draw()
    
    def _on_slot_change(self, slot_idx: int, change: Dict[str, Any]) -> None:
        """Handle slot echogram assignment change."""
        new_key = change['new']
        slot = self.slots[slot_idx]
        
        # Find if this echogram was assigned elsewhere and swap
        for other_slot in self.slots:
            if other_slot != slot and other_slot.echogram_key == new_key:
                other_slot.assign_echogram(slot.echogram_key)
                # Update selector
                self.slot_selectors[other_slot.slot_idx].value = other_slot.echogram_key
                break
        
        slot.assign_echogram(new_key)
        
        # Load background if needed
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
        
        if slot.is_visible:
            self._update_slot(slot)
            self._process_qt_events()
            self._request_remote_draw()
    
    def _on_color_change(self, change: Dict[str, Any]) -> None:
        """Handle color scale change."""
        for slot in self._get_visible_slots():
            self._update_slot(slot)
        self._request_remote_draw()
    
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
        # Set grid to 1x1
        self.w_layout.value = (1, 1)
        # Assign the echogram to slot 0
        self.slots[0].assign_echogram(echogram_name)
        self.slot_selectors[0].value = echogram_name
        self._update_grid_layout()
        self._update_slot(self.slots[0])
        self._reset_view()
    
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
        """Connect to a pingviewer for synchronized display."""
        self.pingviewer = pingviewer
        self._update_ping_lines()
    
    def disconnect_pingviewer(self) -> None:
        """Disconnect from pingviewer."""
        self.pingviewer = None
        for slot in self.slots:
            if slot.pingline:
                slot.pingline.hide()
    
    def _update_pingviewer_from_coordinate(self, coord: float) -> None:
        """Update pingviewer from x coordinate."""
        if self.pingviewer is None:
            return
        match self.x_axis_name:
            case "Ping number" | "Ping index":
                self.pingviewer.w_index.value = int(max(0, coord))
            case "Date time":
                target = self._mpl_num_to_datetime(coord).timestamp()
                for idx, ping in enumerate(self.pingviewer.imagebuilder.pings):
                    ping_obj = ping if not isinstance(ping, dict) else next(iter(ping.values()))
                    if ping_obj.get_datetime().timestamp() > target:
                        self.pingviewer.w_index.value = max(0, idx - 1)
                        return
                self.pingviewer.w_index.value = len(self.pingviewer.imagebuilder.pings) - 1
            case "Ping time":
                target = coord
                for idx, ping in enumerate(self.pingviewer.imagebuilder.pings):
                    ping_obj = ping if not isinstance(ping, dict) else next(iter(ping.values()))
                    if ping_obj.get_timestamp() > target:
                        self.pingviewer.w_index.value = max(0, idx - 1)
                        return
                self.pingviewer.w_index.value = len(self.pingviewer.imagebuilder.pings) - 1
    
    def _update_ping_lines(self) -> None:
        """Update ping lines on all visible plots."""
        if self.pingviewer is None:
            return
        
        match self.x_axis_name:
            case "Ping number" | "Ping index":
                value = float(self.pingviewer.w_index.value)
            case "Date time":
                ping = self._get_current_ping()
                value = self._datetime_to_mpl_num(ping.get_datetime())
            case "Ping time":
                ping = self._get_current_ping()
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
        
        self._request_remote_draw()
    
    def _get_current_ping(self) -> Any:
        """Get current ping from pingviewer."""
        ping = self.pingviewer.imagebuilder.pings[self.pingviewer.w_index.value]
        if isinstance(ping, dict):
            return next(iter(ping.values()))
        return ping
    
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
        
        n_visible = self.grid_rows * self.grid_cols
        visible_selectors = self.slot_selectors[:n_visible]
        selector_row = ipywidgets.HBox(visible_selectors)
        
        controls_row = ipywidgets.HBox([
            self.w_layout,
            self.w_vmin, self.w_vmax,
            self.w_auto_update, self.w_crosshair,
        ])
        
        buttons_row = ipywidgets.HBox([
            self.btn_update, self.btn_reset,
            ipywidgets.Label('  Nav:'),
            self.btn_nav_left, self.btn_nav_up, self.btn_nav_down, self.btn_nav_right,
        ])
        
        widgets = [
            tab_row,
            selector_row,
            ipywidgets.HBox([self.graphics]),
            controls_row,
            buttons_row,
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
