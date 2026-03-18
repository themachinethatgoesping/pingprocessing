"""Core echogram viewer logic independent of the UI toolkit.

Manages pyqtgraph scene-graph items (PlotItem, ImageItem, ColorBarItem, …)
and reads / writes control state through a :class:`ControlPanel` abstraction.
Adapters (``echogramviewer_jupyter``, ``echogramviewer_qt``) create the
concrete controls and wire up observers.
"""
from __future__ import annotations

import time as time_module
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

import themachinethatgoesping as theping

from .control_spec import (
    ControlPanel,
    GRID_LAYOUTS,
)
from . import pyqtgraph_helpers as pgh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_axis_names(echogram):
    """Get x_axis_name and y_axis_name from echogram (old or new builder)."""
    if hasattr(echogram, 'coord_system'):
        return echogram.coord_system.x_axis_name, echogram.coord_system.y_axis_name
    return echogram.x_axis_name, echogram.y_axis_name


def normalise_echograms(
    echogramdata: Union[Dict[str, Any], Sequence[Any], Any],
    names: Optional[Sequence[Optional[str]]],
) -> Tuple[Dict[str, Any], List[str]]:
    """Normalise echogram input to ``(dict, names_list)``."""
    if isinstance(echogramdata, dict):
        return dict(echogramdata), list(echogramdata.keys())

    if hasattr(echogramdata, '__iter__') and not isinstance(echogramdata, (str, bytes)):
        echogramdata_list = list(echogramdata)
        if len(echogramdata_list) > 0:
            first_item = echogramdata_list[0]
            if hasattr(first_item, 'layers') or hasattr(first_item, 'build_image'):
                return {"default": first_item}, ["default"]
            if names is not None:
                echo_names = [n if n else f"Echogram {i}" for i, n in enumerate(names)]
            else:
                echo_names = [f"Echogram {i}" for i in range(len(echogramdata_list))]
            return {n: eg for n, eg in zip(echo_names, echogramdata_list)}, echo_names
        return {}, []

    # Single echogram
    return {"default": echogramdata}, ["default"]


def auto_select_grid(
    initial_grid: Tuple[int, int],
    n_items: int,
) -> Tuple[int, int]:
    """Choose an appropriate grid size based on the number of items."""
    if initial_grid != (2, 2):
        return initial_grid
    if n_items == 1:
        return (1, 1)
    elif n_items == 2:
        return (1, 2)
    elif n_items <= 4:
        return (2, 2)
    elif n_items <= 6:
        return (3, 2)
    else:
        return (4, 2)


# ---------------------------------------------------------------------------
# DraggableScatterPlotItem
# ---------------------------------------------------------------------------

class DraggableScatterPlotItem(pg.ScatterPlotItem):
    """ScatterPlotItem that supports dragging individual points."""

    sigPointDragged = QtCore.Signal(int, float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dragging_point_idx: Optional[int] = None
        self._drag_start_pos: Optional[QtCore.QPointF] = None

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        pos = ev.pos()
        pts = self.pointsAt(pos)
        if len(pts) > 0:
            self._dragging_point_idx = pts[0].index()
            self._drag_start_pos = pos
            ev.accept()
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        if self._dragging_point_idx is None:
            ev.ignore()
            return
        vb = self.getViewBox()
        if vb is None:
            ev.ignore()
            return
        scene_pos = ev.scenePos()
        view_pos = vb.mapSceneToView(scene_pos)
        self.sigPointDragged.emit(self._dragging_point_idx, view_pos.x(), view_pos.y())
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._dragging_point_idx is not None:
            self._dragging_point_idx = None
            self._drag_start_pos = None
            ev.accept()
        else:
            ev.ignore()


# ---------------------------------------------------------------------------
# StationOverlayItem
# ---------------------------------------------------------------------------

class StationOverlayItem(pg.GraphicsObject):
    """Lightweight graphics item that draws station markers in one paint().

    Uses a *draw_mode* to control what is rendered:
      - ``'background'``: translucent region fills only (behind echogram).
      - ``'foreground'``: vertical lines and text labels (above echogram).
    """

    def __init__(self, draw_mode: str = 'foreground', parent=None):
        super().__init__(parent)
        self._draw_mode = draw_mode
        self._stations: List[dict] = []
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents, False)

    def add_station(self, name, start_x, end_x, pen, brush,
                    label_color, font, label_position):
        self._stations.append({
            'name': name, 'start_x': start_x, 'end_x': end_x,
            'pen': pen, 'brush': brush, 'label_color': label_color,
            'font': font, 'label_position': label_position,
        })
        self._picture = None
        self.prepareGeometryChange()
        self.update()

    def remove_station(self, name: str):
        self._stations = [s for s in self._stations if s['name'] != name]
        self._picture = None
        self.prepareGeometryChange()
        self.update()

    def clear_stations(self):
        self._stations.clear()
        self._picture = None
        self.prepareGeometryChange()
        self.update()

    def station_names(self) -> List[str]:
        return [s['name'] for s in self._stations]

    def stations_at_x(self, x: float) -> List[str]:
        return [s['name'] for s in self._stations if s['start_x'] <= x <= s['end_x']]

    def boundingRect(self):
        vb = self.getViewBox()
        if vb is None:
            return QtCore.QRectF()
        return vb.viewRect()

    def paint(self, painter, option, widget=None):
        if not self._stations:
            return
        vb = self.getViewBox()
        if vb is None:
            return

        view_rect = vb.viewRect()
        y_min = view_rect.top()
        y_max = view_rect.bottom()
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        y_span = y_max - y_min

        for s in self._stations:
            sx, ex = s['start_x'], s['end_x']
            if ex < view_rect.left() or sx > view_rect.right():
                continue

            if self._draw_mode == 'background':
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(s['brush'])
                painter.drawRect(QtCore.QRectF(sx, y_min, ex - sx, y_span))
            else:
                painter.setPen(s['pen'])
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawLine(QtCore.QLineF(sx, y_min, sx, y_max))
                painter.drawLine(QtCore.QLineF(ex, y_min, ex, y_max))

                center_x = (sx + ex) / 2.0
                if s['label_position'] == 'top':
                    label_y = y_min + y_span * 0.15
                else:
                    label_y = y_max - y_span * 0.15

                device_transform = painter.deviceTransform()
                px = device_transform.map(QtCore.QPointF(center_x, label_y))

                painter.save()
                painter.resetTransform()
                painter.setPen(pg.mkPen(s['label_color']))
                painter.setFont(s['font'])
                painter.translate(px)
                painter.rotate(-45)
                fm = QtGui.QFontMetrics(s['font'])
                text_rect = fm.boundingRect(s['name'])
                painter.drawText(-text_rect.width() // 2, fm.ascent() // 2, s['name'])
                painter.restore()

    def viewRangeChanged(self):
        self.prepareGeometryChange()
        self.update()


# ---------------------------------------------------------------------------
# SafePolyLineROI
# ---------------------------------------------------------------------------

class SafePolyLineROI(pg.PolyLineROI):
    """PolyLineROI subclass that disables right-click context menu on handles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disable_handle_context_menus()

    def _disable_handle_context_menus(self):
        for handle in self.handles:
            handle_item = handle.get('item')
            if handle_item is not None:
                original_click = handle_item.mouseClickEvent
                def safe_click(ev, orig=original_click):
                    if ev.button() == QtCore.Qt.MouseButton.RightButton:
                        ev.accept()
                        return
                    orig(ev)
                handle_item.mouseClickEvent = safe_click

    def addHandle(self, *args, **kwargs):
        result = super().addHandle(*args, **kwargs)
        self._disable_handle_context_menus()
        return result

    def setPoints(self, points, closed=None):
        result = super().setPoints(points, closed)
        self._disable_handle_context_menus()
        return result


# ---------------------------------------------------------------------------
# EchogramSlot
# ---------------------------------------------------------------------------

class EchogramSlot:
    """Manages a single echogram display slot with lazy loading."""

    def __init__(
        self,
        slot_idx: int,
        echograms: Dict[str, Any],
        global_image_cache: Dict[str, Dict[str, Any]],
    ) -> None:
        self.slot_idx = slot_idx
        self._echograms = echograms
        self._global_image_cache = global_image_cache
        self.echogram_key: Optional[str] = None
        self.is_visible = False
        self.needs_update = False

        # Per-layer color scales
        self.background_levels: Optional[Tuple[float, float]] = None
        self.layer_levels: Optional[Tuple[float, float]] = None
        self.active_colorbar_layer: str = 'background'

        # Image data cache
        self._image_cache: Dict[str, Dict[str, Any]] = {}

        # Image data (current)
        self.background_image: Optional[np.ndarray] = None
        self.background_extent: Optional[Tuple[float, float, float, float]] = None
        self.high_res_image: Optional[np.ndarray] = None
        self.high_res_extent: Optional[Tuple[float, float, float, float]] = None
        self.layer_image: Optional[np.ndarray] = None
        self.layer_extent: Optional[Tuple[float, float, float, float]] = None

        # PyQtGraph items (set by core when creating plots)
        self.plot_item: Optional[pg.PlotItem] = None
        self.image_layers: Dict[str, pg.ImageItem] = {}
        self.colorbar: Optional[pg.ColorBarItem] = None
        self.layer_colorbar: Optional[pg.ColorBarItem] = None
        self.crosshair_v: Optional[pg.InfiniteLine] = None
        self.crosshair_h: Optional[pg.InfiniteLine] = None
        self.pingline: Optional[pg.InfiniteLine] = None
        self.station_overlay_bg: Optional[StationOverlayItem] = None
        self.station_overlay_fg: Optional[StationOverlayItem] = None

    def mark_dirty(self):
        self.needs_update = True

    def set_visible(self, visible: bool):
        self.is_visible = visible

    def assign_echogram(self, echogram_key: Optional[str]):
        if echogram_key != self.echogram_key:
            # Cache current images
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
            # Then check global cache
            elif echogram_key is not None and echogram_key in self._global_image_cache:
                cached = self._global_image_cache[echogram_key]
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
        if self.echogram_key is None:
            return None
        return self._echograms.get(self.echogram_key)

    def clear_high_res(self):
        self.high_res_image = None
        self.high_res_extent = None


# ---------------------------------------------------------------------------
# EchogramCore
# ---------------------------------------------------------------------------

class EchogramCore:
    """Backend-agnostic echogram viewer core.

    Parameters
    ----------
    echograms : dict
        ``{name: echogram_object}``
    echogram_names : list[str]
        Ordered echogram names.
    panel : ControlPanel
        Named control handles (must already contain all echogram specs
        plus per-slot controls ``"slot_selector_<i>"``).
    graphics : pg.GraphicsLayoutWidget
        The pyqtgraph widget (jupyter-rfb or native).
    progress : any
        Progress bar (TqdmWidget or similar).
    """

    MAX_SLOTS = 8

    def __init__(
        self,
        echograms: Dict[str, Any],
        echogram_names: List[str],
        panel: ControlPanel,
        graphics: pg.GraphicsLayoutWidget,
        progress: Any,
        cmap: str = "Greys_r",
        cmap_layer: str = "YlGnBu_r",
        initial_grid: Tuple[int, int] = (2, 2),
        voffsets: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        self.echograms = echograms
        self.echogram_names = echogram_names
        self.panel = panel
        self.graphics = graphics
        self.progress = progress
        self.cmap_name = cmap
        self.cmap_layer_name = cmap_layer
        self._colormap = pgh.resolve_colormap(cmap)
        self._colormap_layer = pgh.resolve_colormap(cmap_layer)

        self.args_plot: Dict[str, Any] = {
            "vmin": kwargs.pop("vmin", -100),
            "vmax": kwargs.pop("vmax", -25),
        }
        self.args_plot.update(kwargs)
        self.args_plot_layer = dict(self.args_plot)

        # Vertical offsets
        self.voffsets: Dict[str, float] = {}
        if voffsets is not None:
            if isinstance(voffsets, dict):
                self.voffsets = dict(voffsets)
            else:
                for name, off in zip(self.echogram_names, voffsets):
                    self.voffsets[name] = float(off)
        for name in self.echogram_names:
            if name not in self.voffsets:
                self.voffsets[name] = 0.0

        # Axis info from first echogram
        if self.echograms:
            first_eg = next(iter(self.echograms.values()))
            self.x_axis_name, self.y_axis_name = _get_axis_names(first_eg)
        else:
            self.x_axis_name = "Ping number"
            self.y_axis_name = "Depth (m)"
        self._x_axis_is_datetime = self.x_axis_name == "Date time"

        self._x_axis_format = None
        self._x_axis_max_seconds = 60.0
        if self.echograms:
            first_eg = next(iter(self.echograms.values()))
            if hasattr(first_eg, '_coord_system'):
                self._x_axis_format = getattr(first_eg._coord_system, '_custom_x_format', None)
                ppc = getattr(first_eg._coord_system, '_custom_x_per_ping', None)
                if ppc is not None and len(ppc) > 0:
                    self._x_axis_max_seconds = float(ppc[-1] - ppc[0])

        # Grid state
        self.grid_rows, self.grid_cols = initial_grid
        self._ignore_range_changes = False
        self._last_view_range = None

        # Crosshair
        self._crosshair_enabled = True
        self._crosshair_position: Optional[Tuple[float, float]] = None
        self._last_crosshair_position: Optional[Tuple[float, float]] = None

        # Pingviewer
        self.pingviewer = None
        self._ping_timestamps: Optional[np.ndarray] = None

        # Drag-throttle state for pingline dragging
        self._drag_timer = QtCore.QTimer()
        self._drag_timer.setSingleShot(True)
        self._drag_timer.setInterval(50)
        self._drag_timer.timeout.connect(self._on_drag_timer_fired)
        self._drag_coord: Optional[float] = None
        self._drag_updating = False

        # Station data
        self._station_data_list: List[Dict[str, Any]] = []

        # Param editor state
        self._param_edit_state: Dict[str, Any] = {
            'active_param': None,
            'editing_data': None,
            'native_data': None,
            'has_unsaved_changes': False,
            'selected_point_idx': None,
            'synced_params': set(),
            'roi_items': {},
            'line_items': {},
            'display_indices': None,
            '_updating_roi': False,
        }

        # Global image cache
        self._global_image_cache: Dict[str, Dict[str, Any]] = {}

        # Create slots
        self.slots: List[EchogramSlot] = []
        for i in range(self.MAX_SLOTS):
            self.slots.append(
                EchogramSlot(i, self.echograms, self._global_image_cache)
            )
        for i, name in enumerate(self.echogram_names[:self.MAX_SLOTS]):
            self.slots[i].assign_echogram(name)

        # Adapter-provided callbacks
        self._schedule_update: Callable[[], None] = lambda: None
        self._cancel_load: Callable[[], None] = lambda: None
        self._report_error: Callable[[str], None] = lambda msg: print(msg)
        self._auto_update_enabled: bool = True

        # Build initial graphics
        self.update_grid_layout()

        # Refresh param master list
        self._refresh_param_master_list()

    # =====================================================================
    # Observer wiring
    # =====================================================================

    def wire_observers(
        self,
        *,
        layout_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Connect panel controls to core methods.

        Parameters
        ----------
        layout_callback
            Called after grid layout change (e.g. to refresh UI selectors).
        """
        p = self.panel

        def _on_layout(new_val):
            self.on_layout_change(*new_val)
            if layout_callback is not None:
                layout_callback()

        p["layout"].on_change(_on_layout)

        p["vmin"].on_change(lambda _: self.on_global_color_change())
        p["vmax"].on_change(lambda _: self.on_global_color_change())
        p["colorbar_layer"].on_change(lambda v: self.on_colorbar_layer_change(v))
        p["auto_update"].on_change(lambda v: setattr(self, '_auto_update_enabled', v))
        p["crosshair"].on_change(lambda v: setattr(self, '_crosshair_enabled', v))

        p["btn_update"].on_click(lambda _: self._schedule_update())
        p["btn_reset"].on_click(lambda _: self.reset_view())
        p["btn_autoscale_y"].on_click(lambda _: self.autoscale_y())
        p["btn_goto_pingline"].on_click(lambda _: self.goto_pingline())

        p["btn_nav_left"].on_click(lambda _: self.pan_view('left'))
        p["btn_nav_right"].on_click(lambda _: self.pan_view('right'))
        p["btn_nav_up"].on_click(lambda _: self.pan_view('up'))
        p["btn_nav_down"].on_click(lambda _: self.pan_view('down'))

        p["btn_set_x_interval"].on_click(lambda _: self.set_x_interval_from_panel())

        # Param editor observers
        p["param_master"].on_change(lambda _: self._on_param_master_change())
        p["param_select"].on_change(lambda v: self._on_param_select_change(v))
        p["btn_refresh_params"].on_click(lambda _: self._refresh_param_master_list())
        p["btn_new_param"].on_click(lambda _: self._on_new_param_click())
        p["btn_copy_param"].on_click(lambda _: self._on_copy_param_click())
        p["btn_copy_to_all"].on_click(lambda _: self._on_copy_to_all_click())
        p["param_sync"].on_change(lambda v: self._on_param_sync_change(v))
        p["btn_apply_param"].on_click(lambda _: self._on_apply_param_click())
        p["btn_discard_param"].on_click(lambda _: self._on_discard_param_click())
        p["btn_add_point"].on_click(lambda _: self._add_point_at_cursor())
        p["btn_del_point"].on_click(lambda _: self._delete_selected_point())

    # =====================================================================
    # Grid layout
    # =====================================================================

    def update_grid_layout(self) -> None:
        """(Re-)create PlotItems / ImageItems for the current grid size."""
        self.graphics.clear()
        n_visible = self.grid_rows * self.grid_cols

        for i, slot in enumerate(self.slots):
            slot.set_visible(i < n_visible)

        master_plot: Optional[pg.PlotItem] = None
        for i in range(n_visible):
            row = i // self.grid_cols
            col = i % self.grid_cols
            slot = self.slots[i]

            axis_items = None
            if self._x_axis_is_datetime:
                axis_items = {"bottom": pgh.MatplotlibDateAxis(
                    self._mpl_num_to_datetime, orientation="bottom")}
            elif self._x_axis_format == "timedelta":
                axis_items = {"bottom": pgh.TimedeltaAxis(
                    max_seconds=self._x_axis_max_seconds, orientation="bottom")}

            plot: pg.PlotItem = self.graphics.addPlot(
                row=row, col=col * 2, axisItems=axis_items)
            slot.plot_item = plot

            title = str(slot.echogram_key) if slot.echogram_key is not None else f"Slot {i+1}"
            plot.setTitle(title)
            plot.setLabel("left", self.y_axis_name if col == 0 else "")
            plot.setLabel("bottom", self.x_axis_name if row == self.grid_rows - 1 else "")
            plot.getViewBox().invertY(True)
            plot.getViewBox().setBackgroundColor("w")

            # Image layers
            background = pg.ImageItem(axisOrder="row-major")
            plot.addItem(background)
            high_res = pg.ImageItem(axisOrder="row-major")
            high_res.hide()
            plot.addItem(high_res)
            layer = pg.ImageItem(axisOrder="row-major")
            layer.hide()
            plot.addItem(layer)
            slot.image_layers = {"background": background, "high": high_res, "layer": layer}

            # Colorbar
            try:
                colorbar = pg.ColorBarItem(
                    label="(dB)",
                    values=(self.args_plot["vmin"], self.args_plot["vmax"]),
                    interactive=True,
                )
                colorbar.setImageItem(background, insert_in=plot)
                if hasattr(colorbar, "setColorMap"):
                    colorbar.setColorMap(self._colormap)
                slot.colorbar = colorbar
                slot.background_levels = (self.args_plot["vmin"], self.args_plot["vmax"])
                slot.layer_levels = (self.args_plot["vmin"], self.args_plot["vmax"])
                if hasattr(colorbar, 'sigLevelsChanged'):
                    colorbar.sigLevelsChanged.connect(
                        lambda cb=colorbar, s=slot: self._on_colorbar_levels_changed(s, cb))
            except AttributeError:
                slot.colorbar = None

            slot.layer_colorbar = None

            # Crosshairs
            pen_cross = pg.mkPen(color='r', width=1, style=QtCore.Qt.PenStyle.DashLine)
            slot.crosshair_v = pg.InfiniteLine(angle=90, pen=pen_cross)
            slot.crosshair_h = pg.InfiniteLine(angle=0, pen=pen_cross)
            slot.crosshair_v.hide()
            slot.crosshair_h.hide()
            plot.addItem(slot.crosshair_v)
            plot.addItem(slot.crosshair_h)

            slot.pingline = None
            slot.station_overlay_bg = None
            slot.station_overlay_fg = None

            # Link axes
            if master_plot is None:
                master_plot = plot
            else:
                plot.setXLink(master_plot)
                plot.setYLink(master_plot)

        self._connect_scene_events()
        self._update_visible_slots()
        self._recreate_station_markers()

        # Recreate param visualization
        if self._param_edit_state.get('active_param') is not None:
            self._param_edit_state['roi_items'] = {}
            self._param_edit_state['line_items'] = {}
            self._update_param_visualization()

    # =====================================================================
    # Event handlers (called by adapters or wire_observers)
    # =====================================================================

    def on_layout_change(self, new_rows: int, new_cols: int) -> None:
        if (new_rows, new_cols) == (self.grid_rows, self.grid_cols):
            return
        current_range = self._capture_current_view_range()
        self._ignore_range_changes = True
        try:
            self.grid_rows, self.grid_cols = new_rows, new_cols
            self.update_grid_layout()
        finally:
            self._ignore_range_changes = False
        if current_range is not None:
            self._restore_view_range(current_range)
        self._request_remote_draw()
        if self.pingviewer is not None:
            self._update_ping_lines()
        if self._auto_update_enabled:
            self._schedule_update()

    def on_slot_change(self, slot_idx: int, new_key: Optional[str]) -> None:
        slot = self.slots[slot_idx]
        current_range = self._capture_current_view_range()
        self._ignore_range_changes = True
        try:
            slot.assign_echogram(new_key)
            if new_key and slot.background_image is None:
                echogram = self.echograms.get(new_key)
                if echogram:
                    self.progress.set_description(f"Loading {new_key}...")
                    if len(echogram.layers) == 0 and echogram.main_layer is None:
                        slot.background_image, slot.background_extent = \
                            echogram.build_image(progress=self.progress)
                    else:
                        slot.background_image, slot.layer_image, slot.background_extent = \
                            echogram.build_image_and_layer_image(progress=self.progress)
                        slot.layer_extent = slot.background_extent
                    self.progress.set_description("Idle")
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
        if current_range is not None:
            self._restore_view_range(current_range)
        self._request_remote_draw()
        if self._auto_update_enabled and slot.is_visible:
            self._schedule_update()

    def on_global_color_change(self) -> None:
        new_vmin = float(self.panel["vmin"].value)
        new_vmax = float(self.panel["vmax"].value)
        for slot in self.slots:
            slot.background_levels = (new_vmin, new_vmax)
            slot.layer_levels = (new_vmin, new_vmax)
            if slot.colorbar is not None:
                slot.colorbar.setLevels((new_vmin, new_vmax))
        self._request_remote_draw()

    def on_colorbar_layer_change(self, new_layer: str) -> None:
        for slot in self._get_visible_slots():
            self._switch_colorbar_layer(slot, new_layer)
        self._request_remote_draw()

    def show_single(self, echogram_name: str) -> None:
        current_range = self._capture_current_view_range()
        self._ignore_range_changes = True
        try:
            need_grid_change = (self.grid_rows, self.grid_cols) != (1, 1)
            if need_grid_change:
                self.panel["layout"].value = (1, 1)
            if self.slots[0].echogram_key != echogram_name:
                self.slots[0].assign_echogram(echogram_name)
                self.panel["slot_selector_0"].value = echogram_name
            if not need_grid_change:
                self._update_slot(self.slots[0])
        finally:
            self._ignore_range_changes = False
        if current_range is not None:
            self._restore_view_range(current_range)
        self._request_remote_draw()
        if self._auto_update_enabled:
            self._schedule_update()

    def handle_key_down(self, key: str, modifiers: tuple = ()) -> None:
        """Handle keyboard events for parameter editing."""
        if self._param_edit_state.get('active_param') is None:
            return
        if key in ('Delete', 'Backspace'):
            self._delete_selected_point()
        elif key.lower() == 'a':
            self._add_point_at_cursor()

    # =====================================================================
    # Slot updates
    # =====================================================================

    def _get_visible_slots(self) -> List[EchogramSlot]:
        return [s for s in self.slots if s.is_visible and s.echogram_key is not None]

    def _update_visible_slots(self) -> None:
        for slot in self._get_visible_slots():
            if slot.needs_update or slot.background_image is None:
                self._update_slot(slot)

    def _update_slot(self, slot: EchogramSlot) -> None:
        if slot.plot_item is None or slot.echogram_key is None:
            return
        echogram = slot.get_echogram()
        if echogram is None:
            return

        vb = slot.plot_item.getViewBox()
        old_auto_range = vb.autoRangeEnabled()
        vb.disableAutoRange()

        try:
            slot.plot_item.setTitle(
                str(slot.echogram_key) if slot.echogram_key is not None else "")

            if slot.background_image is not None and slot.background_extent is not None:
                self._update_slot_image(
                    slot, "background", slot.background_image, slot.background_extent)
            if slot.high_res_image is not None and slot.high_res_extent is not None:
                self._update_slot_image(
                    slot, "high", slot.high_res_image, slot.high_res_extent)
            else:
                slot.image_layers.get("high", pg.ImageItem()).hide()
            if slot.layer_image is not None and slot.layer_extent is not None:
                self._update_slot_image(
                    slot, "layer", slot.layer_image, slot.layer_extent)
            else:
                slot.image_layers.get("layer", pg.ImageItem()).hide()
        finally:
            if old_auto_range[0] or old_auto_range[1]:
                vb.enableAutoRange(x=old_auto_range[0], y=old_auto_range[1])

        slot.needs_update = False

    def _update_slot_image(self, slot: EchogramSlot, key: str,
                           data: np.ndarray,
                           extent: Tuple[float, float, float, float]) -> None:
        image_item = slot.image_layers.get(key)
        if image_item is None:
            return

        offset = self.voffsets.get(slot.echogram_key, 0.0) if slot.echogram_key else 0.0
        array = (data + offset).transpose() if offset != 0.0 else data.transpose()
        image_item.setImage(array, autoLevels=False)

        x0, x1, y0, y1 = self._numeric_extent(extent)
        vb = slot.plot_item.getViewBox()
        if vb.yInverted():
            y0, y1 = y1, y0

        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        image_item.setRect(rect)

        is_layer = (key == "layer")
        colormap = self._colormap_layer if is_layer else self._colormap
        if hasattr(image_item, "setColorMap"):
            image_item.setColorMap(colormap)
        else:
            lut = colormap.getLookupTable(256)
            image_item.setLookupTable(lut)

        if is_layer and slot.layer_levels is not None:
            vmin, vmax = slot.layer_levels
        elif not is_layer and slot.background_levels is not None:
            vmin, vmax = slot.background_levels
        elif slot.colorbar is not None:
            vmin, vmax = slot.colorbar.levels()
        else:
            vmin = float(self.panel["vmin"].value)
            vmax = float(self.panel["vmax"].value)
        image_item.setLevels((vmin, vmax))
        image_item.show()

    def _switch_colorbar_layer(self, slot: EchogramSlot, new_layer: str) -> None:
        if slot.colorbar is None:
            slot.active_colorbar_layer = new_layer
            return
        old_layer = slot.active_colorbar_layer
        if old_layer == new_layer:
            return
        current_levels = slot.colorbar.levels()
        if old_layer == 'background':
            slot.background_levels = current_levels
        else:
            slot.layer_levels = current_levels

        slot.active_colorbar_layer = new_layer

        if new_layer == 'layer' and slot.layer_image is not None:
            layer_img = slot.image_layers.get('layer')
            if layer_img is not None:
                slot.colorbar.setImageItem(layer_img)
                if hasattr(slot.colorbar, "setColorMap"):
                    slot.colorbar.setColorMap(self._colormap_layer)
                if slot.layer_levels is not None:
                    slot.colorbar.setLevels(slot.layer_levels)
        else:
            bg_img = slot.image_layers.get('background')
            if bg_img is not None:
                slot.colorbar.setImageItem(bg_img)
                if hasattr(slot.colorbar, "setColorMap"):
                    slot.colorbar.setColorMap(self._colormap)
                if slot.background_levels is not None:
                    slot.colorbar.setLevels(slot.background_levels)

    def _on_colorbar_levels_changed(self, slot: EchogramSlot,
                                    colorbar: pg.ColorBarItem) -> None:
        vmin, vmax = colorbar.levels()
        if slot.active_colorbar_layer == 'layer':
            slot.layer_levels = (vmin, vmax)
            layer_img = slot.image_layers.get('layer')
            if layer_img is not None:
                layer_img.setLevels((vmin, vmax))
        else:
            slot.background_levels = (vmin, vmax)
            for key in ['background', 'high']:
                img = slot.image_layers.get(key)
                if img is not None:
                    img.setLevels((vmin, vmax))

    # =====================================================================
    # Image loading (synchronous — adapters handle scheduling)
    # =====================================================================

    def load_all_backgrounds(self) -> None:
        """Load background images for all echograms (synchronous)."""
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
                image, layer_img, extent = echogram.build_image_and_layer_image(
                    progress=self.progress)
                slot.background_image = image
                slot.background_extent = extent
                slot.layer_image = layer_img
                slot.layer_extent = extent
            self._global_image_cache[name] = {
                'background_image': slot.background_image,
                'background_extent': slot.background_extent,
                'layer_image': slot.layer_image,
                'layer_extent': slot.layer_extent,
            }
            slot.needs_update = True
        self.progress.set_description("Idle")
        self._update_visible_slots()
        self.reset_view()

    def build_high_res_sync(
        self,
        view_params: Dict[int, Dict[str, float]],
        cancel_flag=None,
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        """Build high-res images for visible slots (synchronous).

        Run in a background thread/executor.  Returns results dict or None
        if cancelled.
        """
        results = {}
        for slot in self._get_visible_slots():
            if cancel_flag is not None and cancel_flag.is_set():
                return None
            echogram = slot.get_echogram()
            if echogram is None:
                continue
            params = view_params.get(slot.slot_idx, {})
            if params:
                self._apply_axis_limits(
                    echogram,
                    params['xmin'], params['xmax'],
                    params['ymin'], params['ymax'])
            if len(echogram.layers) == 0 and echogram.main_layer is None:
                image, extent = echogram.build_image(progress=None)
                results[slot.slot_idx] = {'high': image, 'extent': extent}
            else:
                image, layer_img, extent = echogram.build_image_and_layer_image(
                    progress=None)
                results[slot.slot_idx] = {
                    'high': image, 'extent': extent,
                    'layer': layer_img, 'layer_extent': extent,
                }
        return results

    def apply_high_res_results(self, results: Dict[int, Dict[str, Any]]) -> None:
        """Apply loaded high-res results to slots and refresh display."""
        for slot_idx, data in results.items():
            slot = self.slots[slot_idx]
            slot.high_res_image = data.get('high')
            slot.high_res_extent = data.get('extent')
            if 'layer' in data:
                slot.layer_image = data['layer']
                slot.layer_extent = data['layer_extent']
            self._update_slot(slot)
            if slot.echogram_key and slot.echogram_key in self._global_image_cache:
                self._global_image_cache[slot.echogram_key]['high_res_image'] = \
                    slot.high_res_image
                self._global_image_cache[slot.echogram_key]['high_res_extent'] = \
                    slot.high_res_extent
        self._process_qt_events()
        self._request_remote_draw()
        self.progress.set_description('Idle')

    def _get_slot_for_echogram(self, echogram_key: str) -> Optional[EchogramSlot]:
        for slot in self.slots:
            if slot.echogram_key == echogram_key:
                return slot
        return None

    # =====================================================================
    # View range helpers
    # =====================================================================

    def _get_master_plot(self) -> Optional[pg.PlotItem]:
        for slot in self.slots:
            if slot.is_visible and slot.plot_item is not None:
                return slot.plot_item
        return None

    def _capture_current_view_range(self):
        master = self._get_master_plot()
        if master is not None:
            return tuple(master.getViewBox().viewRange())
        return None

    def get_xlim(self) -> Optional[Tuple[float, float]]:
        vr = self._capture_current_view_range()
        return tuple(vr[0]) if vr is not None else None

    def get_ylim(self) -> Optional[Tuple[float, float]]:
        vr = self._capture_current_view_range()
        return tuple(vr[1]) if vr is not None else None

    def _restore_view_range(self, view_range) -> None:
        master = self._get_master_plot()
        if master is None:
            return
        self._ignore_range_changes = True
        try:
            x_range, y_range = view_range
            master.setXRange(x_range[0], x_range[1], padding=0)
            master.setYRange(y_range[0], y_range[1], padding=0)
        finally:
            self._ignore_range_changes = False

    def reset_view(self) -> None:
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

    def capture_view_params(self) -> Dict[int, Dict[str, float]]:
        """Capture current view parameters for visible slots (for high-res loading)."""
        params = {}
        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            vb = slot.plot_item.getViewBox()
            xmin, xmax = vb.viewRange()[0]
            ymin, ymax = vb.viewRange()[1]
            params[slot.slot_idx] = {
                'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
            }
        return params

    # =====================================================================
    # Navigation
    # =====================================================================

    def pan_view(self, direction: str, fraction: float = 0.25) -> None:
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
        self._schedule_update()

    def autoscale_y(self) -> None:
        """Scale Y axis to fit the visible data range in the current X view.

        Uses the per-ping vec_min_y / vec_max_y from the echogrambuilder's
        coordinate system rather than scanning rendered pixels.
        """
        master = self._get_master_plot()
        if not master:
            return
        vb = master.getViewBox()
        x_range = vb.viewRange()[0]

        ymin_global, ymax_global = np.inf, -np.inf
        for slot in self._get_visible_slots():
            eg = slot.get_echogram()
            if eg is None or not hasattr(eg, '_coord_system'):
                continue
            cs = eg._coord_system

            if not hasattr(cs, 'vec_min_y') or cs.vec_min_y is None:
                continue

            # Convert x view boundaries to ping indices
            fm = cs.feature_mapper
            x_axis = cs.x_axis_name
            if x_axis == "Date time":
                # View range is in matplotlib day-numbers; convert to
                # unix-epoch seconds which is what the feature mapper uses.
                x_lo_feat = x_range[0] * 86400.0
                x_hi_feat = x_range[1] * 86400.0
            else:
                x_lo_feat = x_range[0]
                x_hi_feat = x_range[1]

            idx_lo = int(fm.feature_to_index(x_axis, float(x_lo_feat)))
            idx_hi = int(fm.feature_to_index(x_axis, float(x_hi_feat)))
            if idx_lo > idx_hi:
                idx_lo, idx_hi = idx_hi, idx_lo
            idx_hi = min(idx_hi + 1, len(cs.vec_min_y))
            if idx_lo >= idx_hi:
                continue

            slice_min = cs.vec_min_y[idx_lo:idx_hi]
            slice_max = cs.vec_max_y[idx_lo:idx_hi]
            finite_min = slice_min[np.isfinite(slice_min)]
            finite_max = slice_max[np.isfinite(slice_max)]
            if finite_min.size == 0 or finite_max.size == 0:
                continue

            ymin_global = min(ymin_global, float(np.nanmin(finite_min)))
            ymax_global = max(ymax_global, float(np.nanmax(finite_max)))

        if not np.isfinite(ymin_global) or not np.isfinite(ymax_global):
            return
        if ymin_global >= ymax_global:
            return

        # Add a small margin
        margin = (ymax_global - ymin_global) * 0.02
        self._ignore_range_changes = True
        try:
            master.setYRange(ymin_global - margin, ymax_global + margin, padding=0)
        finally:
            self._ignore_range_changes = False
        self._request_remote_draw()
        self._schedule_update()

    def set_x_interval_from_panel(self) -> None:
        """Set X axis width from the x_interval panel text field.

        Supported formats:
        - ``"2 min"`` or ``"2min"`` — 2 minutes (datetime axis only)
        - ``"30 s"`` or ``"30s"`` — 30 seconds (datetime axis only)
        - ``"1 h"`` or ``"1h"`` — 1 hour (datetime axis only)
        - ``"500"`` — 500 in native x-axis units (ping number, etc.)
        """
        text = self.panel["x_interval"].value.strip()
        if not text:
            return
        width = self._parse_x_interval(text)
        if width is None or width <= 0:
            return
        self.set_x_interval(width)

    def _parse_x_interval(self, text: str) -> Optional[float]:
        """Parse an interval string and return the width in x-axis units."""
        import re
        text = text.strip()

        # Try "<number> <unit>" patterns
        m = re.match(r'^([0-9]*\.?[0-9]+)\s*(h|hr|hours?|min|minutes?|m|s|sec|seconds?)$',
                     text, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith('h'):
                seconds = value * 3600
            elif unit.startswith('m') and not unit.startswith('ms'):
                seconds = value * 60
            else:
                seconds = value

            if self._x_axis_is_datetime:
                # DateTime axis uses matplotlib day numbers
                return seconds / 86400.0
            elif self._x_axis_format == "timedelta":
                return seconds
            else:
                return seconds

        # Plain number
        try:
            return float(text)
        except ValueError:
            return None

    def set_x_interval(self, width: float) -> None:
        """Set X axis to a given width, centered on the current view center."""
        master = self._get_master_plot()
        if not master:
            return
        vb = master.getViewBox()
        x_range = vb.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2.0
        new_xmin = center - width / 2.0
        new_xmax = center + width / 2.0
        self._ignore_range_changes = True
        try:
            master.setXRange(new_xmin, new_xmax, padding=0)
        finally:
            self._ignore_range_changes = False
        self._request_remote_draw()
        self._schedule_update()

    # =====================================================================
    # Mouse / crosshair
    # =====================================================================

    def handle_scene_click(self, event: Any) -> None:
        gfx_view = getattr(self.graphics, "gfxView", self.graphics)
        if gfx_view is not None:
            gfx_view.setFocus()
        pos = event.scenePos()
        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            vb = slot.plot_item.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                point = vb.mapSceneToView(pos)
                if self.pingviewer is not None:
                    self._update_pingviewer_from_coordinate(point.x())
                    self._update_ping_lines()
                break

    def handle_scene_move(self, pos: QtCore.QPointF) -> None:
        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            vb = slot.plot_item.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                point = vb.mapSceneToView(pos)
                x, y = point.x(), point.y()
                value = self._sample_value(slot, x, y)
                self._update_hover_label(x, y, value, slot.echogram_key)
                if self._crosshair_enabled:
                    self._update_crosshairs(x, y)
                return
        self.panel["hover_label"].value = "&nbsp;"
        if self._crosshair_enabled:
            self._hide_crosshairs()

    def _update_crosshairs(self, x: float, y: float) -> None:
        self._crosshair_position = (x, y)
        self._last_crosshair_position = (x, y)
        for slot in self._get_visible_slots():
            if slot.crosshair_v and slot.crosshair_h:
                slot.crosshair_v.setValue(x)
                slot.crosshair_h.setValue(y)
                slot.crosshair_v.show()
                slot.crosshair_h.show()

    def _hide_crosshairs(self) -> None:
        self._crosshair_position = None
        for slot in self.slots:
            if slot.crosshair_v:
                slot.crosshair_v.hide()
            if slot.crosshair_h:
                slot.crosshair_h.hide()

    def _sample_value(self, slot: EchogramSlot, x: float, y: float) -> Optional[float]:
        layers_and_images = [
            (slot.image_layers.get("high"), slot.high_res_image),
            (slot.image_layers.get("bg"), slot.background_image),
        ]
        for layer_item, orig_image in layers_and_images:
            if layer_item is None or layer_item.image is None or orig_image is None:
                continue
            inv, ok = layer_item.transform().inverted()
            if not ok:
                continue
            pt = inv.map(QtCore.QPointF(x, y))
            # After transpose, local x spans original rows, y spans original cols
            r, c = int(pt.x()), int(pt.y())
            nrows, ncols = orig_image.shape
            if 0 <= r < nrows and 0 <= c < ncols:
                return float(orig_image[r, c])
        return None

    def _update_hover_label(self, x: float, y: float,
                            value: Optional[float], name: Optional[str]) -> None:
        x_text = self._format_x_value(x)
        y_text = f"{y:0.2f}"
        value_text = f"{value:0.2f}" if value is not None else "--"
        name_text = f" [{name}]" if name else ""
        station_names = self._stations_at_x(x)
        stations_text = (" | <b>stations</b>: " + ", ".join(station_names)
                         if station_names else "")
        self.panel["hover_label"].value = (
            f"<b>x</b>: {x_text} | <b>y</b>: {y_text} | "
            f"<b>value</b>: {value_text}{name_text}{stations_text}"
        )

    def _stations_at_x(self, x: float) -> List[str]:
        names: List[str] = []
        seen = set()
        for slot in self._get_visible_slots():
            overlay = slot.station_overlay_fg
            if overlay is None:
                continue
            for n in overlay.stations_at_x(x):
                if n not in seen:
                    seen.add(n)
                    names.append(n)
            break
        return names

    # =====================================================================
    # Scene events
    # =====================================================================

    def _connect_scene_events(self) -> None:
        # Jupyter wrapper exposes the QGraphicsView as .gfxView;
        # native pg.GraphicsLayoutWidget IS the QGraphicsView.
        gfx_view = getattr(self.graphics, "gfxView", self.graphics)
        scene = gfx_view.scene() if gfx_view is not None else None
        if scene is None:
            return
        if hasattr(self, '_scene_click_conn') and self._scene_click_conn:
            try:
                scene.sigMouseClicked.disconnect(self.handle_scene_click)
            except (TypeError, RuntimeError):
                pass
        if hasattr(self, '_scene_move_conn') and self._scene_move_conn:
            try:
                scene.sigMouseMoved.disconnect(self.handle_scene_move)
            except (TypeError, RuntimeError):
                pass
        self._scene_click_conn = scene.sigMouseClicked.connect(self.handle_scene_click)
        self._scene_move_conn = scene.sigMouseMoved.connect(self.handle_scene_move)

    # =====================================================================
    # Parameter editor
    # =====================================================================

    def _refresh_param_master_list(self) -> None:
        options = [(str(key), key) for key in self.echograms.keys()]
        if not options:
            self.panel["param_master"].options = []
            self.panel["param_master"].value = None
            self._refresh_param_list()
            return
        old_value = self.panel["param_master"].value
        self.panel["param_master"].options = options
        valid_keys = [k for _, k in options]
        if old_value in valid_keys:
            self.panel["param_master"].value = old_value
        else:
            self.panel["param_master"].value = options[0][1] if options else None
        self._refresh_param_list()

    def _on_param_master_change(self) -> None:
        self._refresh_param_list()
        self._param_edit_state['active_param'] = None
        self._param_edit_state['editing_data'] = None
        self._param_edit_state['native_data'] = None
        self._param_edit_state['has_unsaved_changes'] = False
        self._param_edit_state['selected_point_idx'] = None
        self._clear_param_visualization()
        self.panel["param_status"].value = ""

    def _get_master_echogram(self) -> Optional[Any]:
        master_key = self.panel["param_master"].value
        if master_key is None:
            return None
        return self.echograms.get(master_key)

    def _refresh_param_list(self) -> None:
        master_eg = self._get_master_echogram()
        if master_eg is None or not hasattr(master_eg, '_coord_system'):
            self.panel["param_select"].options = [("(none)", None)]
            self.panel["param_select"].value = None
            self.panel["param_sync"].disabled = True
            return
        cs = master_eg._coord_system
        if not hasattr(cs, 'param'):
            self.panel["param_select"].options = [("(none)", None)]
            self.panel["param_select"].value = None
            self.panel["param_sync"].disabled = True
            return
        params = set(cs.param.keys())
        options = [("(none)", None)]
        for name in sorted(params):
            options.append((name, name))
        old_value = self.panel["param_select"].value
        self.panel["param_select"].options = options
        if old_value in params:
            self.panel["param_select"].value = old_value
        else:
            self.panel["param_select"].value = None
        self._update_sync_checkbox_state()

    def _update_sync_checkbox_state(self) -> None:
        param_name = self.panel["param_select"].value
        if param_name is None:
            self.panel["param_sync"].disabled = True
            self.panel["param_sync"].value = False
            return
        all_have_param = True
        for eg in self.echograms.values():
            if not hasattr(eg, '_coord_system') or not hasattr(eg._coord_system, 'param'):
                all_have_param = False
                break
            if param_name not in eg._coord_system.param:
                all_have_param = False
                break
        self.panel["param_sync"].disabled = not all_have_param
        if not all_have_param:
            self.panel["param_sync"].value = False

    def _on_param_select_change(self, new_param) -> None:
        if self._param_edit_state['has_unsaved_changes']:
            self.panel["param_status"].value = \
                "<span style='color:orange'>⚠ Unsaved changes exist</span>"
        self._param_edit_state['active_param'] = new_param
        self._param_edit_state['selected_point_idx'] = None
        self._update_sync_checkbox_state()
        if new_param is not None:
            self._load_param_for_editing(new_param)
        else:
            self._param_edit_state['editing_data'] = None
            self._param_edit_state['native_data'] = None
            self._param_edit_state['has_unsaved_changes'] = False
            self.panel["param_status"].value = ""
        self._update_param_visualization()

    def _load_param_for_editing(self, param_name: str) -> None:
        master_eg = self._get_master_echogram()
        if master_eg is None:
            self.panel["param_status"].value = \
                "<span style='color:red'>No master echogram selected</span>"
            return
        if not hasattr(master_eg, '_coord_system'):
            self.panel["param_status"].value = \
                "<span style='color:red'>Echogram has no coordinate system</span>"
            return
        cs = master_eg._coord_system
        if param_name not in cs.param:
            self.panel["param_status"].value = \
                f"<span style='color:red'>Parameter '{param_name}' not found</span>"
            return

        y_reference, param_data = cs.param[param_name]
        is_sparse = isinstance(param_data, tuple) and len(param_data) == 2

        try:
            if is_sparse:
                sparse_x_ping_time, sparse_y_native = param_data
                sparse_x_ping_time = np.asarray(sparse_x_ping_time, dtype=np.float64)
                sparse_y_native = np.asarray(sparse_y_native, dtype=np.float64)
                self._param_edit_state['native_data'] = \
                    (y_reference, (sparse_x_ping_time.copy(), sparse_y_native.copy()))

                all_ping_times = np.array(cs.ping_times)
                if cs.x_axis_name == "Date time":
                    all_view_x = all_ping_times / 86400.0
                elif cs.x_axis_name == "Ping time":
                    all_view_x = all_ping_times
                elif (cs._custom_x_per_ping is not None
                      and cs._custom_x_axis_name == cs.x_axis_name):
                    all_view_x = np.asarray(cs._custom_x_per_ping, dtype=np.float64)
                else:
                    all_view_x = np.arange(len(all_ping_times), dtype=np.float64)

                sort_idx = np.argsort(all_ping_times)
                sorted_ping_times = all_ping_times[sort_idx]
                sorted_view_x = all_view_x[sort_idx]

                x_view = np.interp(sparse_x_ping_time, sorted_ping_times, sorted_view_x)
                y_view = self._convert_native_to_view_y(cs, sparse_y_native, y_reference)

                self._param_edit_state['editing_data'] = (x_view, y_view)
                self._param_edit_state['has_unsaved_changes'] = False
                self.panel["param_status"].value = \
                    f"<span style='color:green'>Loaded '{param_name}' ({len(x_view)} control points)</span>"
            else:
                self._param_edit_state['native_data'] = \
                    (y_reference, np.array(param_data).copy())
                x_coords, y_coords = cs.get_ping_param(
                    param_name, use_x_coordinates=False)
                x_coords = np.array(
                    [self._extent_value_to_float(x) for x in x_coords])
                y_coords = np.array(y_coords, dtype=np.float64)
                valid_mask = np.isfinite(y_coords)
                x_valid = x_coords[valid_mask]
                y_valid = y_coords[valid_mask]
                MAX_EDIT_POINTS = 500
                if len(x_valid) > MAX_EDIT_POINTS:
                    step = len(x_valid) // MAX_EDIT_POINTS
                    x_valid = x_valid[::step]
                    y_valid = y_valid[::step]
                    self.panel["param_status"].value = \
                        f"<span style='color:green'>Loaded '{param_name}' ({len(x_valid)} points, downsampled from dense)</span>"
                else:
                    n_valid = len(x_valid)
                    if n_valid == 0:
                        self.panel["param_status"].value = \
                            f"<span style='color:green'>Loaded '{param_name}' (empty - use 'a' to add)</span>"
                    else:
                        self.panel["param_status"].value = \
                            f"<span style='color:green'>Loaded '{param_name}' ({n_valid} dense points)</span>"
                self._param_edit_state['editing_data'] = (x_valid, y_valid)
                self._param_edit_state['has_unsaved_changes'] = False
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.panel["param_status"].value = \
                f"<span style='color:red'>Error: {e}</span>"
            self._param_edit_state['editing_data'] = None

    def _on_copy_param_click(self) -> None:
        source_param = self.panel["param_select"].value
        new_name = self.panel["new_param_name"].value.strip()
        if source_param is None:
            self.panel["param_status"].value = \
                "<span style='color:red'>Select a parameter to copy</span>"
            return
        if not new_name:
            self.panel["param_status"].value = \
                "<span style='color:red'>Enter a name for the copy</span>"
            return
        master_eg = self._get_master_echogram()
        if master_eg is None or not hasattr(master_eg, '_coord_system'):
            self.panel["param_status"].value = \
                "<span style='color:red'>No valid master echogram</span>"
            return
        cs = master_eg._coord_system
        if source_param not in cs.param:
            self.panel["param_status"].value = \
                f"<span style='color:red'>Parameter '{source_param}' not found</span>"
            return
        if new_name in cs.param:
            self.panel["param_status"].value = \
                f"<span style='color:red'>Parameter '{new_name}' already exists</span>"
            return
        y_ref, vec_y_val = cs.param[source_param]
        cs.param[new_name] = (y_ref, np.array(vec_y_val).copy())
        self.panel["param_status"].value = \
            f"<span style='color:green'>Copied '{source_param}' to '{new_name}'</span>"
        self._refresh_param_list()
        self.panel["param_select"].value = new_name
        self.panel["new_param_name"].value = ""

    def _on_new_param_click(self) -> None:
        new_name = self.panel["new_param_name"].value.strip()
        if not new_name:
            self.panel["param_status"].value = \
                "<span style='color:red'>Enter a name for the new parameter</span>"
            return
        master_eg = self._get_master_echogram()
        if master_eg is None or not hasattr(master_eg, '_coord_system'):
            self.panel["param_status"].value = \
                "<span style='color:red'>No valid master echogram</span>"
            return
        cs = master_eg._coord_system
        if new_name in cs.param:
            self.panel["param_status"].value = \
                f"<span style='color:red'>Parameter '{new_name}' already exists</span>"
            return
        n_pings = cs.n_pings
        empty_values = np.full(n_pings, np.nan, dtype=np.float64)
        y_reference = cs.y_axis_name if cs.y_axis_name else "Y indice"
        cs.param[new_name] = (y_reference, empty_values)
        self.panel["param_status"].value = \
            f"<span style='color:green'>Created empty '{new_name}'</span>"
        self._refresh_param_list()
        self.panel["param_select"].value = new_name
        self.panel["new_param_name"].value = ""

    def _on_copy_to_all_click(self) -> None:
        param_name = self.panel["param_select"].value
        if param_name is None:
            self.panel["param_status"].value = \
                "<span style='color:red'>Select a parameter first</span>"
            return
        master_eg = self._get_master_echogram()
        if master_eg is None or not hasattr(master_eg, '_coord_system'):
            self.panel["param_status"].value = \
                "<span style='color:red'>No valid master echogram</span>"
            return
        master_cs = master_eg._coord_system
        if param_name not in master_cs.param:
            self.panel["param_status"].value = \
                f"<span style='color:red'>Parameter '{param_name}' not in master</span>"
            return
        y_ref, vec_y_val = master_cs.param[param_name]
        master_n_pings = len(vec_y_val)
        vec_y_val_np = np.asarray(vec_y_val, dtype=np.float64)
        copied_count = 0
        for key, eg in self.echograms.items():
            if eg is master_eg:
                continue
            if not hasattr(eg, '_coord_system'):
                continue
            cs = eg._coord_system
            n_pings = cs.n_pings
            new_values = (vec_y_val_np.copy() if n_pings == master_n_pings
                          else np.full(n_pings, np.nan, dtype=np.float64))
            cs.param[param_name] = (y_ref, new_values)
            copied_count += 1
        if copied_count > 0:
            self.panel["param_status"].value = \
                f"<span style='color:green'>Copied to {copied_count} echogram(s)</span>"
            self.panel["param_sync"].disabled = False
        else:
            self.panel["param_status"].value = \
                "<span style='color:orange'>No other echograms to copy to</span>"

    def _on_param_sync_change(self, new_value) -> None:
        param_name = self._param_edit_state['active_param']
        if param_name is None:
            return
        if new_value:
            self._param_edit_state['synced_params'].add(param_name)
        else:
            self._param_edit_state['synced_params'].discard(param_name)

    def _on_apply_param_click(self) -> None:
        param_name = self._param_edit_state['active_param']
        if param_name is None:
            return
        editing_data = self._param_edit_state['editing_data']
        native_data = self._param_edit_state['native_data']
        if editing_data is None or native_data is None:
            self.panel["param_status"].value = \
                "<span style='color:red'>No data to apply</span>"
            return

        x_coords, y_coords = editing_data
        y_reference, _ = native_data

        if (self.panel["param_sync"].value
                and param_name in self._param_edit_state['synced_params']):
            echograms_to_update = list(self.echograms.values())
        else:
            master_eg = self._get_master_echogram()
            echograms_to_update = [master_eg] if master_eg is not None else []

        updated_count = 0
        x_as_ping_times = None
        native_y = None
        for eg in echograms_to_update:
            if not hasattr(eg, '_coord_system'):
                continue
            cs = eg._coord_system
            try:
                native_y = self._convert_view_to_native_y(cs, y_coords, y_reference)
                all_ping_times = np.array(cs.ping_times)
                if cs.x_axis_name == "Date time":
                    all_view_x = all_ping_times / 86400.0
                elif cs.x_axis_name == "Ping time":
                    all_view_x = all_ping_times
                elif (cs._custom_x_per_ping is not None
                      and cs._custom_x_axis_name == cs.x_axis_name):
                    all_view_x = np.asarray(cs._custom_x_per_ping, dtype=np.float64)
                else:
                    all_view_x = np.arange(len(all_ping_times), dtype=np.float64)

                sort_idx = np.argsort(all_view_x)
                sorted_view_x = all_view_x[sort_idx]
                sorted_ping_times = all_ping_times[sort_idx]
                x_as_ping_times = np.interp(x_coords, sorted_view_x, sorted_ping_times)

                sort_order = np.argsort(x_as_ping_times)
                x_as_ping_times = x_as_ping_times[sort_order]
                native_y = native_y[sort_order]

                unique_x, indices = np.unique(x_as_ping_times, return_inverse=True)
                if len(unique_x) < len(x_as_ping_times):
                    sums = np.bincount(indices, weights=native_y)
                    counts = np.bincount(indices)
                    unique_y = sums / counts
                    x_as_ping_times = unique_x
                    native_y = unique_y

                cs.param[param_name] = (
                    y_reference, (x_as_ping_times.copy(), native_y.copy()))
                updated_count += 1
            except Exception as e:
                self._report_error(f"Error updating {eg}: {e}")
                import traceback
                traceback.print_exc()

        if updated_count > 0:
            self._param_edit_state['has_unsaved_changes'] = False
            self._param_edit_state['native_data'] = \
                (y_reference, (x_as_ping_times.copy(), native_y.copy()))
            self.panel["param_status"].value = \
                f"<span style='color:green'>Applied {len(x_coords)} points to {updated_count} echogram(s)</span>"
        else:
            self.panel["param_status"].value = \
                "<span style='color:red'>Failed to apply changes</span>"

    def _on_discard_param_click(self) -> None:
        param_name = self._param_edit_state['active_param']
        if param_name is not None:
            self._load_param_for_editing(param_name)
            self._update_param_visualization()
            self.panel["param_status"].value = \
                "<span style='color:blue'>Changes discarded</span>"

    def _convert_view_to_native_y(self, cs, view_y: np.ndarray,
                                  y_reference: str) -> np.ndarray:
        if cs._affine_sample_to_y is None:
            return view_y
        a_y, b_y = cs._affine_sample_to_y
        a_y_mean = np.nanmean(a_y)
        b_y_mean = np.nanmean(b_y)
        sample_indices = ((view_y - a_y_mean) / b_y_mean
                          if b_y_mean != 0 else view_y)

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

    def _convert_native_to_view_y(self, cs, native_y: np.ndarray,
                                  y_reference: str) -> np.ndarray:
        match y_reference:
            case "Y indice":
                sample_indices = native_y
            case "Sample number":
                if not cs.has_sample_nrs:
                    sample_indices = native_y
                else:
                    a, b = cs._affine_sample_to_sample_nr
                    a_m, b_m = np.nanmean(a), np.nanmean(b)
                    sample_indices = (native_y - a_m) / b_m if b_m != 0 else native_y
            case "Depth (m)":
                if not cs.has_depths:
                    sample_indices = native_y
                else:
                    a, b = cs._affine_sample_to_depth
                    a_m, b_m = np.nanmean(a), np.nanmean(b)
                    sample_indices = (native_y - a_m) / b_m if b_m != 0 else native_y
            case "Range (m)":
                if not cs.has_ranges:
                    sample_indices = native_y
                else:
                    a, b = cs._affine_sample_to_range
                    a_m, b_m = np.nanmean(a), np.nanmean(b)
                    sample_indices = (native_y - a_m) / b_m if b_m != 0 else native_y
            case _:
                sample_indices = native_y

        if cs._affine_sample_to_y is None:
            return sample_indices
        a_y, b_y = cs._affine_sample_to_y
        return np.nanmean(a_y) + np.nanmean(b_y) * sample_indices

    # -- parameter visualization --

    def _update_param_visualization(self) -> None:
        self._clear_param_visualization()
        param_name = self._param_edit_state['active_param']
        editing_data = self._param_edit_state['editing_data']
        if param_name is None:
            return
        if editing_data is None:
            x_coords = np.array([], dtype=np.float64)
            y_coords = np.array([], dtype=np.float64)
        else:
            x_coords, y_coords = editing_data
        n_points = len(x_coords)

        MAX_ROI_HANDLES = 500
        if n_points > MAX_ROI_HANDLES:
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

        self._param_edit_state['display_indices'] = display_indices

        for slot in self._get_visible_slots():
            if slot.plot_item is None:
                continue
            if is_downsampled and n_points > 0:
                pen = pg.mkPen(color='#FF6600', width=2,
                               style=QtCore.Qt.PenStyle.DashLine)
                line_item = pg.PlotCurveItem(x_coords, y_coords, pen=pen)
                slot.plot_item.addItem(line_item)
                self._param_edit_state['line_items'][slot.slot_idx] = line_item

            if len(x_display) > 0:
                positions = [[float(x_display[i]), float(y_display[i])]
                             for i in range(len(x_display))]
                roi = SafePolyLineROI(
                    positions, closed=False,
                    pen=pg.mkPen('#FF6600', width=2),
                    hoverPen=pg.mkPen('#FFAA00', width=3),
                    handlePen=pg.mkPen('#FF6600', width=1),
                    handleHoverPen=pg.mkPen('#FF0000', width=2),
                    movable=False, removable=False,
                )
                roi.sigRegionChanged.connect(
                    lambda r=roi, s=slot: self._on_roi_changed(r, s))
                roi.sigRegionChangeFinished.connect(
                    lambda r=roi, s=slot: self._on_roi_change_finished(r, s))
                slot.plot_item.addItem(roi)
                self._param_edit_state['roi_items'][slot.slot_idx] = roi

        if is_downsampled:
            self.panel["param_status"].value = \
                f"<span style='color:orange'>Showing {len(x_display)}/{n_points} handles (downsampled)</span>"
        self._request_remote_draw()

    def _on_roi_changed(self, roi, slot: EchogramSlot) -> None:
        if self._param_edit_state.get('_updating_roi'):
            return
        self._param_edit_state['_is_dragging_roi'] = True
        self._param_edit_state['_active_drag_slot'] = slot.slot_idx
        self._param_edit_state['has_unsaved_changes'] = True

    def _on_roi_change_finished(self, roi, slot: EchogramSlot) -> None:
        if not self._param_edit_state.get('_is_dragging_roi'):
            return
        self._param_edit_state['_is_dragging_roi'] = False
        self._param_edit_state['_active_drag_slot'] = None
        editing_data = self._param_edit_state['editing_data']
        if editing_data is None:
            return
        try:
            handle_positions = roi.getLocalHandlePositions()
            if len(handle_positions) == 0:
                return
            x_coords, y_coords = editing_data
            display_indices = self._param_edit_state.get('display_indices')

            for i, (name, pos) in enumerate(handle_positions):
                actual_idx = (display_indices[i]
                              if display_indices is not None and i < len(display_indices)
                              else i)
                if 0 <= actual_idx < len(x_coords):
                    x_coords[actual_idx] = pos.x()
                    y_coords[actual_idx] = pos.y()

            needs_resort = any(
                x_coords[i] > x_coords[i + 1] for i in range(len(x_coords) - 1))
            if needs_resort:
                sort_indices = np.argsort(x_coords)
                x_coords = x_coords[sort_indices].copy()
                y_coords = y_coords[sort_indices].copy()
                self._param_edit_state['editing_data'] = (x_coords, y_coords)
                self._clear_param_visualization()
                self._update_param_visualization()
                self.panel["param_status"].value = \
                    "<span style='color:orange'>Points reordered (unsaved)</span>"
            else:
                self._param_edit_state['editing_data'] = (x_coords, y_coords)
                for slot_idx, line_item in list(
                        self._param_edit_state['line_items'].items()):
                    try:
                        line_item.setData(x_coords, y_coords)
                    except Exception:
                        pass
                # Sync other ROIs
                self._param_edit_state['_updating_roi'] = True
                try:
                    for other_idx, other_roi in list(
                            self._param_edit_state['roi_items'].items()):
                        if other_idx != slot.slot_idx:
                            try:
                                if display_indices is not None:
                                    x_d = x_coords[display_indices]
                                    y_d = y_coords[display_indices]
                                else:
                                    x_d = x_coords
                                    y_d = y_coords
                                positions = [[float(x_d[j]), float(y_d[j])]
                                             for j in range(len(x_d))]
                                other_roi.setPoints(positions)
                            except Exception:
                                pass
                finally:
                    self._param_edit_state['_updating_roi'] = False
                self.panel["param_status"].value = \
                    "<span style='color:orange'>Modified (unsaved)</span>"
        except Exception as e:
            self._report_error(f"ROI finish error: {e}")
            self.panel["param_status"].value = \
                f"<span style='color:red'>Error: {e}</span>"

    def _clear_param_visualization(self) -> None:
        for slot_idx, roi in list(self._param_edit_state.get('roi_items', {}).items()):
            slot = self.slots[slot_idx] if slot_idx < len(self.slots) else None
            if slot and slot.plot_item:
                try:
                    slot.plot_item.removeItem(roi)
                except Exception:
                    pass
        self._param_edit_state['roi_items'] = {}
        for slot_idx, line in list(self._param_edit_state['line_items'].items()):
            slot = self.slots[slot_idx] if slot_idx < len(self.slots) else None
            if slot and slot.plot_item:
                try:
                    slot.plot_item.removeItem(line)
                except Exception:
                    pass
        self._param_edit_state['line_items'].clear()

    def _delete_selected_point(self) -> bool:
        if self._param_edit_state.get('_is_dragging_roi'):
            self.panel["param_status"].value = \
                "<span style='color:red'>Cannot delete while dragging</span>"
            return False
        if self._param_edit_state.get('_deletion_in_progress'):
            return False
        try:
            self._param_edit_state['_deletion_in_progress'] = True
            editing_data = self._param_edit_state['editing_data']
            if editing_data is None:
                self.panel["param_status"].value = \
                    "<span style='color:red'>No parameter data to delete from</span>"
                return False
            x_coords, y_coords = editing_data
            if len(x_coords) == 0:
                self.panel["param_status"].value = \
                    "<span style='color:red'>No points to delete</span>"
                return False
            cursor_pos = self._last_crosshair_position
            if cursor_pos is not None:
                cursor_x = cursor_pos[0]
                x_distances = np.abs(x_coords - cursor_x)
                idx = int(np.argmin(x_distances))
            else:
                idx = len(x_coords) - 1
            new_x = np.delete(x_coords, idx)
            new_y = np.delete(y_coords, idx)
            self._param_edit_state['editing_data'] = (new_x, new_y)
            self._param_edit_state['selected_point_idx'] = None
            self._param_edit_state['has_unsaved_changes'] = True
            self._clear_param_visualization()
            self._update_param_visualization()
            self.panel["param_status"].value = \
                "<span style='color:orange'>Point deleted (unsaved)</span>"
            return True
        except Exception as e:
            self._report_error(f"Delete point error: {e}")
            self.panel["param_status"].value = \
                f"<span style='color:red'>Delete error: {e}</span>"
            return False
        finally:
            self._param_edit_state['_deletion_in_progress'] = False

    def _add_point_at_cursor(self) -> bool:
        cursor_pos = self._last_crosshair_position
        if cursor_pos is None:
            self.panel["param_status"].value = \
                "<span style='color:red'>Move cursor over plot first</span>"
            return False
        if self._param_edit_state['active_param'] is None:
            self.panel["param_status"].value = \
                "<span style='color:red'>Select a parameter first</span>"
            return False
        editing_data = self._param_edit_state['editing_data']
        cursor_x, cursor_y = cursor_pos
        if editing_data is None:
            x_coords = np.array([], dtype=np.float64)
            y_coords = np.array([], dtype=np.float64)
        else:
            x_coords, y_coords = editing_data
        insert_idx = np.searchsorted(x_coords, cursor_x)
        new_x = np.insert(x_coords, insert_idx, cursor_x)
        new_y = np.insert(y_coords, insert_idx, cursor_y)
        self._param_edit_state['editing_data'] = (new_x, new_y)
        self._param_edit_state['selected_point_idx'] = insert_idx
        self._param_edit_state['has_unsaved_changes'] = True
        self._update_param_visualization()
        self.panel["param_status"].value = \
            "<span style='color:orange'>Point added (unsaved)</span>"
        return True

    # =====================================================================
    # Pingviewer integration
    # =====================================================================

    def connect_pingviewer(self, pingviewer: Any,
                           progress: bool = False) -> None:
        if self.pingviewer is not None:
            self.disconnect_pingviewer()
        self.pingviewer = pingviewer
        self._build_ping_index(progress=progress)
        self._update_ping_lines()
        if hasattr(pingviewer, 'register_ping_change_callback'):
            pingviewer.register_ping_change_callback(self._update_ping_lines)

    def disconnect_pingviewer(self) -> None:
        if self.pingviewer is not None:
            if hasattr(self.pingviewer, 'unregister_ping_change_callback'):
                self.pingviewer.unregister_ping_change_callback(self._update_ping_lines)
        self.pingviewer = None
        self._ping_timestamps = None
        for slot in self.slots:
            if slot.pingline:
                slot.pingline.hide()

    def _build_ping_index(self, progress: bool = False) -> None:
        """Pre-extract ping timestamps into a sorted numpy array.

        Converts once to avoid per-click Python/nanobind overhead.
        """
        pings = self._get_pingviewer_pings()
        if pings is None or len(pings) == 0:
            self._ping_timestamps = None
            return
        n = len(pings)
        timestamps = np.empty(n, dtype=np.float64)
        for i in tqdm(range(n), desc="Building ping index",
                      disable=not progress):
            ping = pings[i]
            if isinstance(ping, dict):
                ping = next(iter(ping.values()))
            timestamps[i] = ping.get_timestamp()
        self._ping_timestamps = timestamps

    def update_ping_lines(self) -> None:
        self._update_ping_lines()

    def _get_pingviewer_pings(self) -> Optional[Sequence[Any]]:
        if self.pingviewer is None:
            return None
        if hasattr(self.pingviewer, 'slots'):
            for slot in self.pingviewer.slots:
                if slot.is_visible and slot.get_pings() is not None:
                    return slot.get_pings()
            return None
        if hasattr(self.pingviewer, 'imagebuilder'):
            return self.pingviewer.imagebuilder.pings
        return None

    def _get_pingviewer_current_index(self) -> int:
        if self.pingviewer is None:
            return 0
        if hasattr(self.pingviewer, 'slots'):
            for slot in self.pingviewer.slots:
                if slot.is_visible:
                    return slot.ping_index
            return 0
        return self.pingviewer.w_index.value

    def _coord_to_ping_index(self, coord: float) -> Optional[int]:
        """Convert an x-axis coordinate to a ping index (O(log n))."""
        pings = self._get_pingviewer_pings()
        if pings is None or len(pings) == 0:
            return None
        n = len(pings)
        match self.x_axis_name:
            case "Ping number" | "Ping index":
                return int(max(0, min(coord, n - 1)))
            case "Date time":
                if self._ping_timestamps is None:
                    return None
                target = self._mpl_num_to_datetime(coord).timestamp()
                idx = int(np.searchsorted(self._ping_timestamps, target,
                                          side='right')) - 1
                return max(0, min(idx, n - 1))
            case "Ping time":
                if self._ping_timestamps is None:
                    return None
                idx = int(np.searchsorted(self._ping_timestamps, coord,
                                          side='right')) - 1
                return max(0, min(idx, n - 1))
            case _:
                first_eg = next(iter(self.echograms.values()), None)
                if first_eg is not None and hasattr(first_eg, '_coord_system'):
                    cs = first_eg._coord_system
                    if cs._custom_x_per_ping is not None:
                        idx = int(np.searchsorted(cs._custom_x_per_ping,
                                                  coord))
                        return max(0, min(idx, n - 1))
        return None

    def _update_pingviewer_from_coordinate(self, coord: float) -> None:
        if self.pingviewer is None:
            return
        idx = self._coord_to_ping_index(coord)
        if idx is not None:
            self._set_pingviewer_index(idx)

    def _set_pingviewer_index(self, idx: int) -> None:
        if self.pingviewer is None:
            return
        # Jupyter WCI viewer: public ping_sliders (ipywidgets with .value)
        if hasattr(self.pingviewer, 'ping_sliders'):
            for i, slot in enumerate(self.pingviewer.slots):
                if slot.is_visible:
                    self.pingviewer.ping_sliders[i].value = idx
                    break
        # Qt WCI viewer: private _ping_sliders (QSliders with .setValue)
        elif hasattr(self.pingviewer, '_ping_sliders'):
            for i, slot in enumerate(self.pingviewer.slots):
                if slot.is_visible:
                    self.pingviewer._ping_sliders[i].setValue(idx)
                    break
        # Single-channel Jupyter viewer
        elif hasattr(self.pingviewer, 'w_index'):
            self.pingviewer.w_index.value = idx

    def _update_ping_lines(self) -> None:
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
                line = pg.InfiniteLine(
                    angle=90, movable=True, pen=pg.mkPen(
                        color='k', style=QtCore.Qt.PenStyle.DashLine))
                line.sigPositionChanged.connect(self._on_pingline_dragged)
                slot.plot_item.addItem(line)
                slot.pingline = line
            slot.pingline.blockSignals(True)
            slot.pingline.setValue(value)
            slot.pingline.blockSignals(False)
            slot.pingline.show()
        if self.panel["auto_follow"].value:
            self._auto_follow_pingline(value)
        self._request_remote_draw()

    def _auto_follow_pingline(self, pingline_x: float) -> None:
        master = self._get_master_plot()
        if master is None:
            return
        vb = master.getViewBox()
        (xmin, xmax), _ = vb.viewRange()
        x_extent = xmax - xmin
        if x_extent <= 0:
            return
        position_fraction = (pingline_x - xmin) / x_extent
        edge_threshold = 0.10
        needs_scroll = (
            position_fraction < 0 or position_fraction > 1
            or position_fraction < edge_threshold
            or position_fraction > (1 - edge_threshold))
        if not needs_scroll:
            return
        target_position = 0.30
        new_xmin = pingline_x - (target_position * x_extent)
        new_xmax = new_xmin + x_extent
        self._ignore_range_changes = True
        try:
            vb.setXRange(new_xmin, new_xmax, padding=0)
        finally:
            self._ignore_range_changes = False
        if self._auto_update_enabled:
            self._schedule_update()

    # =====================================================================
    # Pingline drag handling (throttled)
    # =====================================================================

    def _on_pingline_dragged(self, line: pg.InfiniteLine) -> None:
        """Called on every sigPositionChanged during a drag.

        Immediately syncs the other slots' pinglines (cheap) and
        schedules a throttled WCI update so the image rebuild doesn't
        block every pixel of the drag.
        """
        coord = line.value()
        # Sync all other pinglines instantly (visual only, cheap)
        for slot in self._get_visible_slots():
            pl = slot.pingline
            if pl is not None and pl is not line:
                pl.blockSignals(True)
                pl.setValue(coord)
                pl.blockSignals(False)
        # Store latest coord and (re)start the timer
        self._drag_coord = coord
        if not self._drag_updating:
            self._drag_timer.start()

    def _on_drag_timer_fired(self) -> None:
        """Dispatch the most recent drag coordinate to the pingviewer."""
        coord = self._drag_coord
        if coord is None or self.pingviewer is None:
            return
        self._drag_updating = True
        try:
            self._update_pingviewer_from_coordinate(coord)
        finally:
            self._drag_updating = False
        # If new drag events arrived during the (synchronous) WCI build,
        # schedule one more update so the final position is honoured.
        if self._drag_coord is not None and self._drag_coord != coord:
            self._drag_timer.start()

    def goto_pingline(self) -> None:
        """Center the view on the current pingline position."""
        self._update_ping_lines()
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
        master = self._get_master_plot()
        if master is None:
            return
        vb = master.getViewBox()
        (xmin, xmax), _ = vb.viewRange()
        x_extent = xmax - xmin
        new_xmin = pingline_x - x_extent / 2
        new_xmax = pingline_x + x_extent / 2
        self._ignore_range_changes = True
        try:
            master.setXRange(new_xmin, new_xmax, padding=0)
        finally:
            self._ignore_range_changes = False
        self._request_remote_draw()
        if self._auto_update_enabled:
            self._schedule_update()

    def _get_current_ping(self) -> Optional[Any]:
        pings = self._get_pingviewer_pings()
        if pings is None:
            return None
        idx = self._get_pingviewer_current_index()
        if 0 <= idx < len(pings):
            ping = pings[idx]
            return ping if not isinstance(ping, dict) else next(iter(ping.values()))
        return None

    # =====================================================================
    # Axis limits
    # =====================================================================

    def _apply_axis_limits(self, echogram, xmin, xmax, ymin, ymax) -> None:
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
            case _:
                x_kwargs["min_value"] = xmin
                x_kwargs["max_value"] = xmax
                echogram.set_x_axis_custom(**x_kwargs)
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

    # =====================================================================
    # Station overlay management
    # =====================================================================

    def add_station_times(
        self,
        stations: Dict[str, Tuple[Any, Any]],
        line_color: str = '#424242',
        line_width: int = 2,
        line_style: str = 'dash',
        region_color: Optional[str] = None,
        region_alpha: float = 0.15,
        label_color: Optional[str] = None,
        label_size: str = '10pt',
        label_position: str = 'top',
    ) -> None:
        """Add station time markers to all visible echograms."""
        self._station_data_list.append({
            'stations': stations,
            'line_color': line_color, 'line_width': line_width,
            'line_style': line_style,
            'region_color': region_color, 'region_alpha': region_alpha,
            'label_color': label_color, 'label_size': label_size,
            'label_position': label_position,
        })
        style_map = {
            'solid': QtCore.Qt.PenStyle.SolidLine,
            'dash': QtCore.Qt.PenStyle.DashLine,
            'dot': QtCore.Qt.PenStyle.DotLine,
            'dashdot': QtCore.Qt.PenStyle.DashDotLine,
        }
        pen_style = style_map.get(line_style, QtCore.Qt.PenStyle.SolidLine)
        effective_label_color = label_color if label_color is not None else line_color

        for station_name, (start_time, end_time) in stations.items():
            start_x = self._time_to_x_coord(start_time)
            end_x = self._time_to_x_coord(end_time)
            if start_x is None or end_x is None:
                continue
            for slot in self._get_visible_slots():
                self._add_station_marker_to_slot(
                    slot, station_name, start_x, end_x,
                    line_color, line_width, pen_style,
                    region_color, region_alpha,
                    effective_label_color, label_size, label_position)
        self._request_remote_draw()

    def clear_station_times(self, station_name: Optional[str] = None) -> None:
        for slot in self.slots:
            if slot.station_overlay_bg is None:
                continue
            if station_name is not None:
                slot.station_overlay_bg.remove_station(station_name)
                slot.station_overlay_fg.remove_station(station_name)
            else:
                slot.station_overlay_bg.clear_stations()
                slot.station_overlay_fg.clear_stations()
        if station_name is None:
            self._station_data_list.clear()
        else:
            for data in self._station_data_list:
                data['stations'].pop(station_name, None)
            self._station_data_list = [
                d for d in self._station_data_list if d['stations']]
        self._request_remote_draw()

    def _time_to_x_coord(self, time_value) -> Optional[float]:
        if isinstance(time_value, datetime):
            if self.x_axis_name == "Date time":
                return self._datetime_to_mpl_num(time_value)
            elif self.x_axis_name == "Ping time":
                return time_value.timestamp()
            else:
                return self._timestamp_to_x_coord(time_value.timestamp())
        elif isinstance(time_value, (int, float)):
            if self.x_axis_name == "Date time":
                dt = datetime.fromtimestamp(time_value, tz=timezone.utc)
                return self._datetime_to_mpl_num(dt)
            elif self.x_axis_name == "Ping time":
                return float(time_value)
            else:
                return self._timestamp_to_x_coord(time_value)
        return None

    def _timestamp_to_x_coord(self, timestamp: float) -> Optional[float]:
        if not self.echograms:
            return None
        first_echogram = next(iter(self.echograms.values()))
        if hasattr(first_echogram, '_coord_system'):
            cs = first_echogram._coord_system
            if (cs._custom_x_per_ping is not None
                    and cs._custom_x_axis_name == cs.x_axis_name):
                ping_times = np.asarray(cs.ping_times, dtype=np.float64)
                custom_x = np.asarray(cs._custom_x_per_ping, dtype=np.float64)
                return float(np.interp(timestamp, ping_times, custom_x))
        return self._find_ping_index_for_time(timestamp)

    def _find_ping_index_for_time(self, timestamp: float) -> Optional[float]:
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

    def _ensure_station_overlay(self, slot):
        if slot.station_overlay_bg is None and slot.plot_item is not None:
            bg = StationOverlayItem(draw_mode='background')
            bg.setZValue(-100)
            slot.plot_item.addItem(bg)
            slot.station_overlay_bg = bg
            fg = StationOverlayItem(draw_mode='foreground')
            fg.setZValue(200)
            slot.plot_item.addItem(fg)
            slot.station_overlay_fg = fg
        return slot.station_overlay_bg, slot.station_overlay_fg

    def _add_station_marker_to_slot(self, slot, station_name, start_x, end_x,
                                    line_color, line_width, pen_style,
                                    region_color, region_alpha,
                                    label_color, label_size,
                                    label_position) -> None:
        if slot.plot_item is None:
            return
        bg, fg = self._ensure_station_overlay(slot)
        if bg is None or fg is None:
            return
        pen = pg.mkPen(color=line_color, width=line_width, style=pen_style)
        from pyqtgraph.Qt.QtGui import QColor
        qcolor = QColor(region_color if region_color else line_color)
        qcolor.setAlphaF(region_alpha)
        brush = pg.mkBrush(qcolor)
        font = pg.QtGui.QFont('Arial', int(label_size.replace('pt', '')))
        lbl_qcolor = QColor(label_color)
        data = dict(name=station_name, start_x=start_x, end_x=end_x,
                    pen=pen, brush=brush, label_color=lbl_qcolor,
                    font=font, label_position=label_position)
        bg.add_station(**data)
        fg.add_station(**data)

    def _recreate_station_markers(self) -> None:
        if not getattr(self, '_station_data_list', None):
            return
        saved = list(self._station_data_list)
        self._station_data_list.clear()
        for data in saved:
            self.add_station_times(
                data['stations'],
                line_color=data['line_color'],
                line_width=data['line_width'],
                line_style=data['line_style'],
                region_color=data['region_color'],
                region_alpha=data['region_alpha'],
                label_color=data['label_color'],
                label_size=data['label_size'],
                label_position=data['label_position'])

    # =====================================================================
    # Scene export
    # =====================================================================

    def get_scene(self) -> QtWidgets.QGraphicsScene:
        return self._get_gfx_view().scene()

    def save_scene(self, filename: str = "scene.svg") -> None:
        import pyqtgraph.exporters
        exporter = pg.exporters.SVGExporter(self.get_scene())
        exporter.export(filename)

    def get_matplotlib(self, dpi: int = 150):
        import matplotlib.pyplot as plt
        scene = self.get_scene()
        rect = scene.sceneRect()
        w = int(rect.width()) or 1000
        h = int(rect.height()) or 600
        image = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.white)
        painter = QtGui.QPainter(image)
        scene.render(painter)
        painter.end()
        ptr = image.bits()
        if hasattr(ptr, 'setsize'):
            ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()
        arr = arr[..., [2, 1, 0, 3]]
        fig, ax = plt.subplots(dpi=dpi)
        ax.imshow(arr)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        return fig

    # =====================================================================
    # Qt helpers
    # =====================================================================

    def _get_gfx_view(self):
        return getattr(self.graphics, "gfxView", self.graphics)

    @staticmethod
    def _process_qt_events() -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def _request_remote_draw(self) -> None:
        fn = getattr(self.graphics, "request_draw", None)
        if callable(fn):
            fn()

    # =====================================================================
    # Utility methods
    # =====================================================================

    @staticmethod
    def _mpl_num_to_datetime(value):
        base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return [base + timedelta(days=float(v)) for v in value]
        return base + timedelta(days=float(value))

    @staticmethod
    def _datetime_to_mpl_num(value: datetime) -> float:
        base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        dt_value = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return (dt_value - base).total_seconds() / 86400.0

    def _format_x_value(self, coord: float) -> str:
        match self.x_axis_name:
            case "Date time":
                return self._mpl_num_to_datetime(coord).isoformat(sep=" ")
            case "Ping time":
                return f"{coord:0.2f} s"
            case "Ping index":
                return f"{coord:0.0f}"
            case _:
                if self._x_axis_format == "timedelta":
                    return pgh.TimedeltaAxis._format_seconds(
                        coord, self._x_axis_max_seconds)
                return f"{coord:0.2f}"

    def _numeric_extent(self, extent):
        return tuple(self._extent_value_to_float(v) for v in extent)

    def _extent_value_to_float(self, value) -> float:
        if isinstance(value, datetime):
            return self._datetime_to_mpl_num(value)
        if isinstance(value, timedelta):
            return value.total_seconds() / 86400.0
        if isinstance(value, np.datetime64):
            delta = value - np.datetime64("1970-01-01T00:00:00Z")
            return float(delta / np.timedelta64(1, "s")) / 86400.0
        if isinstance(value, np.timedelta64):
            return float(value / np.timedelta64(1, "s")) / 86400.0
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)

    # =====================================================================
    # Ping change callbacks (for connected viewers)
    # =====================================================================

    def register_ping_change_callback(self, callback: Any) -> None:
        pass  # Echogram viewer doesn't have ping callbacks (it's not WCI)

    def unregister_ping_change_callback(self, callback: Any) -> None:
        pass
