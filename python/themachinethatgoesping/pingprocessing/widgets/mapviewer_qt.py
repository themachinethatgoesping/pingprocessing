"""Native Qt map viewer using pyqtgraph DockArea + extracted MapCore.

Provides a standalone Qt application for viewing maps without Jupyter.
The layout uses ``pyqtgraph.dockarea`` so the user can drag, float,
and rearrange panels.

Usage::

    from themachinethatgoesping.pingprocessing.widgets.mapviewer_qt import MapViewerQt
    viewer = MapViewerQt(builder)
    viewer.show()
    viewer.run()   # blocks in QApplication.exec()
"""
from __future__ import annotations

import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.dockarea import Dock, DockArea

from . import pyqtgraph_helpers as pgh
from .control_spec import (
    MAP_NAV_SPECS,
    MAP_MISC_SPECS,
    MAP_COLORBAR_SPECS,
    MAP_COLORMAPS,
    MAP_TAB_LAYOUT,
    DropdownSpec,
    CheckboxSpec,
)
from .control_qt import QtControlPanel, QtControlHandle, create_qt_control
from .mapviewer_core import MapCore, LayerRenderSettings


class MapViewerQt(QtWidgets.QMainWindow):
    """Standalone native Qt map viewer window.

    Parameters match :class:`MapViewerJupyter` so the two viewers are
    interchangeable at construction time.
    """

    def __init__(
        self,
        builder: Any = None,
        tile_builder: Any = None,
        width: int = 1000,
        height: int = 700,
        show_controls: bool = True,
        max_render_size: Tuple[int, int] = (2000, 2000),
        auto_update: bool = True,
        auto_update_delay_ms: int = 300,
        show: bool = True,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        pgh.ensure_qapp()
        super().__init__(parent)
        self.setWindowTitle("Map Viewer")
        self.resize(width, height)

        pg.setConfigOptions(imageAxisOrder='row-major')

        # Auto-update state
        self._auto_update_enabled = auto_update
        self._auto_update_delay_ms = auto_update_delay_ms
        self._startup_complete = False
        self._last_view_range = None

        # Loading state
        self._is_loading = False
        self._is_shutting_down = False
        self._view_changed_during_load = False
        self._cancel_flag = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Tile loading state
        self._is_loading_tiles = False
        self._tile_cancel_flag = threading.Event()
        self._tile_view_changed_during_load = False
        self._pending_tile_bounds = None
        self._tile_executor = ThreadPoolExecutor(max_workers=2)

        # Build control panel
        self.panel = QtControlPanel.from_specs(
            MAP_NAV_SPECS,
            MAP_MISC_SPECS,
            MAP_COLORBAR_SPECS,
        )

        # Graphics widget (native Qt, not jupyter_rfb)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")

        # Create core
        self.core = MapCore(
            builder=builder,
            tile_builder=tile_builder,
            panel=self.panel,
            graphics=self.graphics,
            max_render_size=max_render_size,
        )

        # Wire adapter callbacks into the core
        self.core._schedule_update = self._schedule_debounced_update
        self.core._request_draw = lambda: None  # native Qt repaints automatically
        self.core._trigger_tile_load = self._trigger_tile_load
        self.core._report_error = lambda msg: print(msg)

        # Wire observers
        self.core.wire_observers()
        self.panel["auto_update"].on_change(self._on_auto_update_toggle)

        # Debounce timer
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._trigger_high_res_update)

        # View-range change detection timer
        self._range_check_timer = QtCore.QTimer(self)
        self._range_check_timer.setInterval(100)
        self._range_check_timer.timeout.connect(self._check_view_range_changed)

        # Build dock layout
        self._build_dock_layout(builder, tile_builder, show_controls)

        # Wire track legend
        self.core._update_track_legend = self._update_track_legend

        # Initial render
        self.core.update_view()
        self.core.create_colorbar()
        self._startup_complete = True

        if auto_update:
            self._range_check_timer.start()

        if show:
            self.show()
            self.core.zoom_to_fit()

    # =====================================================================
    # Auto-update
    # =====================================================================

    def _on_auto_update_toggle(self, value) -> None:
        self._auto_update_enabled = value
        if value:
            self._range_check_timer.start()
        else:
            self._range_check_timer.stop()
            self._debounce_timer.stop()

    def _check_view_range_changed(self) -> None:
        if not self._startup_complete or not self._auto_update_enabled:
            return
        if self.core._ignore_range_changes or self._is_loading:
            return
        vb = self.core._plot.getViewBox()
        current_range = vb.viewRange()
        if self._last_view_range is not None:
            old_x, old_y = self._last_view_range
            new_x, new_y = current_range
            if not (np.allclose(old_x, new_x, rtol=1e-6) and
                    np.allclose(old_y, new_y, rtol=1e-6)):
                self._last_view_range = current_range
                self._schedule_debounced_update()
        else:
            self._last_view_range = current_range

    def _schedule_debounced_update(self) -> None:
        if self._is_loading:
            self._view_changed_during_load = True
            return
        self._debounce_timer.start(self._auto_update_delay_ms)

    def _trigger_high_res_update(self) -> None:
        if self._is_shutting_down:
            return
        self._cancel_pending_load()

        # Trigger tile loading
        if self.core._tile_builder is not None and self.core.tile_visible:
            self.core.render_tiles()

        if self.core._builder is None:
            if self.core._tracks or self.core._overview_tracks:
                self.core._update_tracks()
                self.core._update_ping_marker()
            return

        if self.core._current_bounds is None:
            self.core._current_bounds = self.core._builder.combined_bounds
        if self.core._current_bounds is None:
            return

        self._is_loading = True
        self._view_changed_during_load = False
        self._cancel_flag.clear()

        core = self.core
        cancel_flag = self._cancel_flag
        viewer = self

        def load_data():
            return core.build_high_res_sync(cancel_flag)

        def on_done(future):
            try:
                results = future.result()
            except Exception as e:
                viewer._is_loading = False
                print(f"Map update error: {e}")
                return
            viewer._is_loading = False
            if results is None:
                if viewer._view_changed_during_load:
                    viewer._view_changed_during_load = False
                    QtCore.QTimer.singleShot(0, viewer._schedule_debounced_update)
                return
            core.apply_high_res_results(results)
            if viewer._view_changed_during_load:
                viewer._view_changed_during_load = False
                QtCore.QTimer.singleShot(0, viewer._schedule_debounced_update)

        future = self._executor.submit(load_data)
        future.add_done_callback(lambda f: QtCore.QMetaObject.invokeMethod(
            self, "_apply_future_result", QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG("QVariant", f)))
        self._pending_future = future
        self._pending_on_done = on_done

    @QtCore.Slot("QVariant")
    def _apply_future_result(self, future) -> None:
        if hasattr(self, '_pending_on_done') and self._pending_on_done:
            self._pending_on_done(future)

    def _cancel_pending_load(self) -> None:
        self._cancel_flag.set()
        self._debounce_timer.stop()
        self._is_loading = False

    # =====================================================================
    # Tile loading (threaded)
    # =====================================================================

    def _trigger_tile_load(self, bbox, target_size, cache_key) -> None:
        """Load tiles in a thread pool, apply on main thread."""
        self._cancel_tile_load()

        self._is_loading_tiles = True
        self._tile_view_changed_during_load = False
        self._tile_cancel_flag.clear()

        viewer = self
        tile_builder = self.core._tile_builder

        def load_tiles():
            if viewer._tile_cancel_flag.is_set():
                return None
            try:
                tile_image, actual_bounds = tile_builder.get_image_with_bounds(
                    bounds=bbox, target_size=target_size,
                )
                if viewer._tile_cancel_flag.is_set():
                    return None
                return {
                    'image': tile_image, 'bounds': actual_bounds,
                    'requested_bounds': bbox, 'cache_key': cache_key,
                }
            except AttributeError:
                tile_image, _ = tile_builder.get_image(
                    bounds=bbox, target_size=target_size,
                )
                if viewer._tile_cancel_flag.is_set():
                    return None
                return {
                    'image': tile_image, 'bounds': bbox,
                    'requested_bounds': bbox, 'cache_key': cache_key,
                }
            except Exception as e:
                warnings.warn(f"Tile loading error: {e}")
                return None

        def on_tile_done(future):
            viewer._is_loading_tiles = False
            try:
                result = future.result()
            except Exception:
                return
            if result is not None and result['image'] is not None:
                viewer.core.apply_tile_result(result)
            if viewer._tile_view_changed_during_load:
                viewer._tile_view_changed_during_load = False
                if viewer._pending_tile_bounds is not None:
                    viewer.core.render_tiles()

        future = self._tile_executor.submit(load_tiles)
        future.add_done_callback(lambda f: QtCore.QMetaObject.invokeMethod(
            viewer, "_apply_tile_future", QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG("QVariant", f)))
        self._pending_tile_future = future
        self._pending_tile_on_done = on_tile_done

    @QtCore.Slot("QVariant")
    def _apply_tile_future(self, future) -> None:
        if hasattr(self, '_pending_tile_on_done') and self._pending_tile_on_done:
            self._pending_tile_on_done(future)

    def _cancel_tile_load(self) -> None:
        self._tile_cancel_flag.set()
        self._is_loading_tiles = False

    # =====================================================================
    # DockArea layout
    # =====================================================================

    def _build_dock_layout(self, builder, tile_builder, show_controls) -> None:
        area = DockArea()
        self.setCentralWidget(area)

        # -- Graphics dock --
        d_graphics = Dock("Map", size=(800, 500))
        d_graphics.addWidget(self.graphics)

        # -- Controls dock --
        d_controls = Dock("Controls", size=(800, 80))
        ctrl_widget = QtWidgets.QWidget()
        ctrl_vlayout = QtWidgets.QVBoxLayout(ctrl_widget)
        ctrl_vlayout.setContentsMargins(4, 2, 4, 2)

        # Navigation row
        nav_row = QtWidgets.QHBoxLayout()
        for name in ["btn_zoom_fit", "btn_zoom_track", "btn_zoom_wci", "btn_refresh_tracks"]:
            if name in self.panel:
                nav_row.addWidget(self.panel[name].widget)
        for name in ["auto_update", "auto_center_wci"]:
            if name in self.panel:
                nav_row.addWidget(self.panel[name].widget)
        nav_row.addStretch()
        ctrl_vlayout.addLayout(nav_row)

        # Colorbar layer selector
        colorbar_row = QtWidgets.QHBoxLayout()
        if "colorbar_layer" in self.panel:
            colorbar_row.addWidget(self.panel["colorbar_layer"].widget)
        colorbar_row.addStretch()
        ctrl_vlayout.addLayout(colorbar_row)

        d_controls.addWidget(ctrl_widget)

        # -- Layers dock --
        d_layers = Dock("Layers", size=(300, 200))
        layers_widget = QtWidgets.QWidget()
        layers_vlayout = QtWidgets.QVBoxLayout(layers_widget)
        layers_vlayout.setContentsMargins(4, 4, 4, 4)

        # Tile controls
        self._tile_source_combo: Optional[QtWidgets.QComboBox] = None
        self._tile_visible_cb: Optional[QtWidgets.QCheckBox] = None
        if tile_builder is not None:
            from ..overview.map_builder.tile_builder import TILE_SOURCES

            self._tile_visible_cb = QtWidgets.QCheckBox("Show background tiles")
            self._tile_visible_cb.setChecked(self.core.tile_visible)
            self._tile_visible_cb.toggled.connect(self._on_tile_visibility_qt)

            self._tile_source_combo = QtWidgets.QComboBox()
            self._tile_source_combo.addItem("None")
            for src in TILE_SOURCES.keys():
                self._tile_source_combo.addItem(src)
            current_source = getattr(tile_builder, '_current_source_name', None)
            if current_source:
                idx = self._tile_source_combo.findText(current_source)
                if idx >= 0:
                    self._tile_source_combo.setCurrentIndex(idx)
            self._tile_source_combo.currentTextChanged.connect(self._on_tile_source_qt)

            tile_row = QtWidgets.QHBoxLayout()
            tile_row.addWidget(self._tile_visible_cb)
            tile_row.addWidget(self._tile_source_combo)
            tile_row.addStretch()
            layers_vlayout.addLayout(tile_row)

        # Per-layer controls
        self._layer_checkboxes_qt: Dict[str, QtWidgets.QCheckBox] = {}
        self._layer_sliders_qt: Dict[str, QtWidgets.QSlider] = {}
        self._layer_colormap_combos: Dict[str, QtWidgets.QComboBox] = {}

        if builder is not None:
            for layer in builder.layers:
                settings = self.core._layer_render_settings.get(layer.name, LayerRenderSettings())

                row = QtWidgets.QHBoxLayout()

                cb = QtWidgets.QCheckBox(layer.name)
                cb.setChecked(layer.visible)
                cb.toggled.connect(lambda checked, n=layer.name: self.core.set_layer_visibility(n, checked))
                self._layer_checkboxes_qt[layer.name] = cb
                row.addWidget(cb)

                slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(int(settings.opacity * 100))
                slider.setFixedWidth(80)
                slider.valueChanged.connect(
                    lambda val, n=layer.name: self.core.set_layer_opacity_image(n, val / 100.0)
                )
                self._layer_sliders_qt[layer.name] = slider
                row.addWidget(slider)

                cmap_combo = QtWidgets.QComboBox()
                for cm in MAP_COLORMAPS:
                    cmap_combo.addItem(cm)
                idx = cmap_combo.findText(settings.colormap)
                if idx >= 0:
                    cmap_combo.setCurrentIndex(idx)
                cmap_combo.currentTextChanged.connect(
                    lambda text, n=layer.name: self.core.set_layer_colormap_image(n, text)
                )
                self._layer_colormap_combos[layer.name] = cmap_combo
                row.addWidget(cmap_combo)

                row.addStretch()
                layers_vlayout.addLayout(row)

        # Update colorbar dropdown options
        if builder is not None:
            layer_names = [l.name for l in builder.layers]
            if layer_names and "colorbar_layer" in self.panel:
                handle = self.panel["colorbar_layer"]
                w = handle.widget
                if isinstance(w, QtWidgets.QComboBox):
                    w.clear()
                    w.addItem("None", None)
                    for n in layer_names:
                        w.addItem(n, n)
                    w.setCurrentIndex(1)
                    self.core._active_colorbar_layer = layer_names[0]

        layers_vlayout.addStretch()
        d_layers.addWidget(layers_widget)

        # -- Track legend dock --
        d_tracks = Dock("Tracks", size=(300, 150))
        self._track_scroll = QtWidgets.QScrollArea()
        self._track_scroll.setWidgetResizable(True)
        self._track_container = QtWidgets.QWidget()
        self._track_layout = QtWidgets.QVBoxLayout(self._track_container)
        self._track_layout.setContentsMargins(4, 4, 4, 4)
        self._track_layout.addStretch()
        self._track_scroll.setWidget(self._track_container)
        d_tracks.addWidget(self._track_scroll)

        # -- Coord label dock --
        d_coords = Dock("Info", size=(800, 30))
        if "lbl_coords" in self.panel:
            d_coords.addWidget(self.panel["lbl_coords"].widget)

        # -- Assemble --
        area.addDock(d_graphics, "top")
        area.addDock(d_controls, "bottom", d_graphics)
        area.addDock(d_coords, "bottom", d_controls)
        area.addDock(d_layers, "right")
        area.addDock(d_tracks, "bottom", d_layers)

        self._dock_area = area

    # =====================================================================
    # Tile callbacks (Qt)
    # =====================================================================

    def _on_tile_visibility_qt(self, checked: bool) -> None:
        self.core.tile_visible = checked

    def _on_tile_source_qt(self, source_name: str) -> None:
        self.core.change_tile_source(source_name)
        if self._tile_visible_cb is not None and source_name != 'None':
            self._tile_visible_cb.setChecked(True)

    # =====================================================================
    # Track legend (Qt)
    # =====================================================================

    def _update_track_legend(self) -> None:
        """Rebuild the track legend with checkboxes (Qt native)."""
        # Clear old widgets
        while self._track_layout.count() > 0:
            item = self._track_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        all_tracks = {**self.core._tracks, **self.core._overview_tracks}
        for name, track in all_tracks.items():
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(2, 1, 2, 1)

            cb = QtWidgets.QCheckBox()
            cb.setChecked(track.visible)
            cb.toggled.connect(lambda checked, n=name: self.core.set_track_visibility(n, checked))
            row_layout.addWidget(cb)

            color_btn = QtWidgets.QPushButton()
            color_btn.setFixedSize(20, 20)
            color_btn.setStyleSheet(f"background-color: {track.color}; border: 1px solid #888;")
            row_layout.addWidget(color_btn)

            label = QtWidgets.QLabel(name)
            row_layout.addWidget(label)
            row_layout.addStretch()

            self._track_layout.addWidget(row_widget)

        self._track_layout.addStretch()

    # =====================================================================
    # Public API (forwarded to core)
    # =====================================================================

    def run(self) -> None:
        """Enter the Qt event loop (blocking)."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.exec()

    def set_layer_colormap(self, layer_name: str, colormap: str) -> "MapViewerQt":
        self.core.set_layer_colormap(layer_name, colormap)
        if layer_name in self._layer_colormap_combos:
            idx = self._layer_colormap_combos[layer_name].findText(colormap)
            if idx >= 0:
                self._layer_colormap_combos[layer_name].setCurrentIndex(idx)
        return self

    def set_layer_opacity(self, layer_name: str, opacity: float) -> "MapViewerQt":
        self.core.set_layer_opacity(layer_name, opacity)
        if layer_name in self._layer_sliders_qt:
            self._layer_sliders_qt[layer_name].setValue(int(opacity * 100))
        return self

    def set_layer_range(self, layer_name: str, vmin: float, vmax: float) -> "MapViewerQt":
        self.core.set_layer_range(layer_name, vmin, vmax)
        return self

    def set_layer_blend_mode(self, layer_name: str, blend_mode: str) -> "MapViewerQt":
        self.core.set_layer_blend_mode(layer_name, blend_mode)
        return self

    def set_tile_source(self, source_name: str) -> "MapViewerQt":
        self.core.change_tile_source(source_name)
        if self._tile_source_combo is not None:
            idx = self._tile_source_combo.findText(source_name)
            if idx >= 0:
                self._tile_source_combo.setCurrentIndex(idx)
        return self

    def set_tile_visible(self, visible: bool) -> "MapViewerQt":
        self.core.tile_visible = visible
        if self._tile_visible_cb is not None:
            self._tile_visible_cb.setChecked(visible)
        return self

    @property
    def tile_builder(self):
        return self.core.tile_builder

    @tile_builder.setter
    def tile_builder(self, builder):
        self.core.tile_builder = builder

    def list_tile_sources(self) -> List[str]:
        return self.core.list_tile_sources()

    def add_geotiff(self, path: str, name: Optional[str] = None, band: int = 1, **kwargs) -> "MapViewerQt":
        self.core.add_geotiff(path, name=name, band=band, **kwargs)
        return self

    def add_layer(self, backend: Any, name: Optional[str] = None,
                  visible: bool = True, z_order: Optional[int] = None) -> "MapViewerQt":
        self.core.add_layer(backend, name=name, visible=visible, z_order=z_order)
        return self

    def add_track(self, latitudes, longitudes, name: str = "Track",
                  color: Optional[str] = None, line_width: float = 2.0,
                  is_active: bool = False, slot_idx: Optional[int] = None) -> None:
        self.core.add_track(latitudes, longitudes, name, color, line_width, is_active, slot_idx)

    def add_overview_track(self, overview, name: str = "Overview",
                           color: Optional[str] = None, **kwargs) -> None:
        self.core.add_overview_track(overview, name, color=color, **kwargs)

    def set_active_track(self, name: str) -> None:
        self.core.set_active_track(name)

    def clear_tracks(self) -> None:
        self.core.clear_tracks()

    def set_track_color(self, name: str, color: str) -> None:
        self.core.set_track_color(name, color)

    def add_markers(self, latitudes, longitudes, name: str = "markers", **kwargs):
        return self.core.add_markers(latitudes, longitudes, name, **kwargs)

    def add_markers_tuples(self, positions, name: str = "markers", **kwargs):
        return self.core.add_markers_tuples(positions, name, **kwargs)

    def remove_markers(self, name: str) -> None:
        self.core.remove_markers(name)

    def clear_markers(self) -> None:
        self.core.clear_markers()

    def update_ping_position(self, lat: float, lon: float) -> None:
        self.core.update_ping_position(lat, lon)

    def zoom_to_fit(self) -> None:
        self.core.zoom_to_fit()

    def zoom_to_track(self) -> None:
        self.core.zoom_to_track()

    def zoom_to_position(self, lat: float, lon: float, radius_deg: float = 0.01) -> None:
        self.core.zoom_to_position(lat, lon, radius_deg)

    def pan_to_wci_position(self) -> None:
        self.core.pan_to_wci_position()

    def pan_to_position(self, lat: float, lon: float) -> None:
        self.core.pan_to_position(lat, lon)

    def is_position_near_edge(self, lat: float, lon: float, edge_fraction: float = 0.2) -> bool:
        return self.core.is_position_near_edge(lat, lon, edge_fraction)

    def pan_to_position_if_near_edge(self, lat: float, lon: float, edge_fraction: float = 0.2) -> None:
        self.core.pan_to_position_if_near_edge(lat, lon, edge_fraction)

    def connect_echogram_viewer(self, echogram_viewer) -> None:
        self.core.connect_echogram_viewer(echogram_viewer)

    def connect_wci_viewer(self, wci_viewer) -> None:
        self.core.connect_wci_viewer(wci_viewer)

    def refresh_tracks(self) -> None:
        self.core.refresh_tracks()

    def register_click_callback(self, callback: Callable[[float, float], None]) -> None:
        self.core.register_click_callback(callback)

    def register_view_change_callback(self, callback: Callable) -> None:
        self.core.register_view_change_callback(callback)

    # =====================================================================
    # Cleanup
    # =====================================================================

    def cleanup(self) -> None:
        self._is_shutting_down = True
        self._cancel_pending_load()
        self._cancel_tile_load()
        self._range_check_timer.stop()
        self._debounce_timer.stop()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)
        try:
            self._tile_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._tile_executor.shutdown(wait=False)

    def _cancel_tile_load(self) -> None:
        self._tile_cancel_flag.set()
        self._is_loading_tiles = False

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
