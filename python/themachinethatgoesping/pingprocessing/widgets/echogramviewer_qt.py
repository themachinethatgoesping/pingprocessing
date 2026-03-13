"""Native Qt echogram viewer using pyqtgraph DockArea + extracted EchogramCore.

This module provides a standalone Qt application for viewing echograms
without Jupyter.  The layout uses ``pyqtgraph.dockarea`` so the user can
drag, float, and rearrange panels.

Usage::

    from themachinethatgoesping.pingprocessing.widgets.echogramviewer_qt import EchogramViewerQt
    viewer = EchogramViewerQt(echogramdata)
    viewer.show()
    viewer.run()   # blocks in QApplication.exec()
"""
from __future__ import annotations

import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.dockarea import Dock, DockArea

from . import pyqtgraph_helpers as pgh
from .control_spec import (
    GRID_LAYOUTS,
    ECHO_RENDER_SPECS,
    ECHO_NAV_SPECS,
    ECHO_MISC_SPECS,
    ECHO_PARAM_SPECS,
    ECHO_TAB_LAYOUT,
    DropdownSpec,
)
from .control_qt import QtControlPanel, QtControlHandle, create_qt_control
from .echogramviewer_core import EchogramCore, normalise_echograms, auto_select_grid
from .wciviewer_qt import _QtProgressBar


class EchogramViewerQt(QtWidgets.QMainWindow):
    """Standalone native Qt echogram viewer window.

    Parameters match :class:`echogramviewer_jupyter.EchogramViewerJupyter`
    so the two viewers are interchangeable at construction time.
    """

    def __init__(
        self,
        echogramdata: Union[Dict[str, Any], Sequence[Any], Any],
        name: str = "Multi-Echogram Viewer",
        names: Optional[Sequence[Optional[str]]] = None,
        progress: Optional[Any] = None,
        cmap: str = "Greys_r",
        cmap_layer: str = "YlGnBu_r",
        show: bool = True,
        voffsets: Optional[Dict[str, float]] = None,
        widget_height_px: int = 800,
        widget_width_px: int = 1200,
        auto_update: bool = True,
        auto_update_delay_ms: int = 300,
        initial_grid: Tuple[int, int] = (2, 2),
        parent: Optional[QtWidgets.QWidget] = None,
        **kwargs: Any,
    ) -> None:
        pgh.ensure_qapp()
        super().__init__(parent)
        self.setWindowTitle(name)
        self.resize(widget_width_px, widget_height_px)

        pg.setConfigOptions(imageAxisOrder="row-major")

        # -- normalise echogram input --
        self.echograms, self.echogram_names = normalise_echograms(echogramdata, names)

        self.progress = progress or _QtProgressBar()
        self.display_progress = progress is None
        n_eg = len(self.echogram_names)

        # -- auto grid --
        initial_grid = auto_select_grid(initial_grid, n_eg)

        # -- build control panel --
        self.panel = QtControlPanel.from_specs(
            ECHO_RENDER_SPECS,
            ECHO_NAV_SPECS,
            ECHO_MISC_SPECS,
            ECHO_PARAM_SPECS,
        )

        # Grid layout dropdown
        layout_options = [(label, (r, c)) for r, c, label in GRID_LAYOUTS]
        layout_spec = DropdownSpec(
            "layout", "Grid:", options=layout_options, value=initial_grid, width="120px",
        )
        self.panel["layout"] = create_qt_control(layout_spec)

        # Per-slot selectors
        echogram_options = [("(none)", None)] + [(n, n) for n in self.echogram_names]
        self._slot_selectors: List[QtWidgets.QComboBox] = []

        for i in range(EchogramCore.MAX_SLOTS):
            eg_key = self.echogram_names[i] if i < n_eg else None
            sel = QtWidgets.QComboBox()
            for opt_label, opt_val in echogram_options:
                sel.addItem(opt_label, opt_val)
            if eg_key is not None:
                idx = sel.findData(eg_key)
                if idx >= 0:
                    sel.setCurrentIndex(idx)
            sel.currentIndexChanged.connect(
                lambda _, idx=i, s=sel: self._on_slot_change(idx, s.currentData())
            )
            self.panel[f"slot_selector_{i}"] = QtControlHandle(
                sel,
                sel.currentData,
                lambda v, s=sel: s.setCurrentIndex(s.findData(v)),
                sel.currentIndexChanged,
                inner=sel,
            )
            self._slot_selectors.append(sel)

        # -- graphics widget (native) --
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")

        # -- auto-update state --
        self._auto_update_enabled = auto_update
        self._auto_update_delay_ms = auto_update_delay_ms
        self._startup_complete = False
        self._last_view_range = None

        # Background loading state
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="echogram_loader")
        self._cancel_flag = threading.Event()
        self._is_loading = False
        self._is_shutting_down = False
        self._view_changed_during_load = False

        # Debounce timer
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._trigger_high_res_update)

        # View-range change detection timer
        self._range_check_timer = QtCore.QTimer(self)
        self._range_check_timer.setInterval(100)
        self._range_check_timer.timeout.connect(self._check_view_range_changed)

        # -- create core --
        self.core = EchogramCore(
            echograms=self.echograms,
            echogram_names=self.echogram_names,
            panel=self.panel,
            graphics=self.graphics,
            progress=self.progress,
            cmap=cmap,
            cmap_layer=cmap_layer,
            initial_grid=initial_grid,
            voffsets=voffsets,
            **kwargs,
        )

        # Wire adapter callbacks into the core
        self.core._schedule_update = self._schedule_debounced_update
        self.core._cancel_load = self._cancel_pending_load
        self.core._report_error = lambda msg: print(msg)
        self.core._auto_update_enabled = auto_update

        # -- wire shared observers --
        self.core.wire_observers(
            layout_callback=self._update_slot_selector_visibility,
        )

        # -- keyboard handling --
        self.installEventFilter(self)

        # -- assemble DockArea layout --
        self._build_dock_layout()

        # -- load initial backgrounds --
        self.core.load_all_backgrounds()
        self._startup_complete = True

        # Start view-range polling
        if auto_update:
            self._range_check_timer.start()

        if show:
            self.show()

    # =====================================================================
    # Dynamic control callbacks
    # =====================================================================

    def _on_slot_change(self, slot_idx: int, new_key) -> None:
        if hasattr(self, "core"):
            self.core.on_slot_change(slot_idx, new_key)

    def _update_slot_selector_visibility(self) -> None:
        n_visible = self.core.grid_rows * self.core.grid_cols
        for i, sel in enumerate(self._slot_selectors):
            sel.setVisible(i < n_visible)

    # =====================================================================
    # Keyboard event filter
    # =====================================================================

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
                self.core.handle_key_down('Delete')
                return True
            elif key == QtCore.Qt.Key.Key_A:
                self.core.handle_key_down('a')
                return True
        return super().eventFilter(obj, event)

    # =====================================================================
    # Debounced updates (QTimer-based)
    # =====================================================================

    def _check_view_range_changed(self) -> None:
        if not self._startup_complete or not self._auto_update_enabled:
            return
        if self.core._ignore_range_changes or self._is_loading:
            return
        import numpy as np
        master = self.core._get_master_plot()
        if master is None:
            return
        current_range = master.getViewBox().viewRange()
        if self._last_view_range is not None:
            old_x, old_y = self._last_view_range
            new_x, new_y = current_range
            if not (np.allclose(old_x, new_x, rtol=1e-6)
                    and np.allclose(old_y, new_y, rtol=1e-6)):
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

        view_params = self.core.capture_view_params()
        if not view_params:
            return

        self._is_loading = True
        self._view_changed_during_load = False
        self._cancel_flag.clear()
        self.core.progress.set_description('Loading...')

        core = self.core
        cancel_flag = self._cancel_flag
        viewer = self

        def load_images():
            return core.build_high_res_sync(view_params, cancel_flag)

        def on_done(future):
            try:
                results = future.result()
            except Exception as e:
                viewer._is_loading = False
                core.progress.set_description('Error')
                print(f"Error: {e}")
                return
            viewer._is_loading = False
            if results is None:
                core.progress.set_description('Cancelled')
                if viewer._view_changed_during_load:
                    viewer._view_changed_during_load = False
                    QtCore.QTimer.singleShot(0, viewer._schedule_debounced_update)
                return
            core.apply_high_res_results(results)
            if viewer._view_changed_during_load:
                viewer._view_changed_during_load = False
                QtCore.QTimer.singleShot(0, viewer._schedule_debounced_update)

        future = self._executor.submit(load_images)
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
    # DockArea layout
    # =====================================================================

    def _build_dock_layout(self) -> None:
        area = DockArea()
        self.setCentralWidget(area)

        # -- Graphics dock --
        d_graphics = Dock("Echogram", size=(800, 500))
        d_graphics.addWidget(self.graphics)

        # -- Slot selectors dock --
        d_slots = Dock("Echograms", size=(300, 200))
        slot_widget = QtWidgets.QWidget()
        slot_layout = QtWidgets.QVBoxLayout(slot_widget)
        slot_layout.setContentsMargins(4, 4, 4, 4)

        # Layout dropdown + tab buttons
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.panel["layout"].widget)
        for eg_name in self.echogram_names:
            btn = QtWidgets.QPushButton(str(eg_name)[:15])
            btn.clicked.connect(lambda _, n=eg_name: self.core.show_single(n))
            top_row.addWidget(btn)
        slot_layout.addLayout(top_row)

        n_visible = self.core.grid_rows * self.core.grid_cols
        for i in range(n_visible):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self._slot_selectors[i])
            slot_layout.addLayout(row)

        # Progress
        if hasattr(self.progress, "widget"):
            slot_layout.addWidget(self.progress.widget)

        slot_layout.addStretch()
        d_slots.addWidget(slot_widget)

        # -- Controls dock: render + navigation --
        d_controls = Dock("Controls", size=(800, 80))
        ctrl_widget = QtWidgets.QWidget()
        ctrl_vlayout = QtWidgets.QVBoxLayout(ctrl_widget)
        ctrl_vlayout.setContentsMargins(4, 2, 4, 2)
        # Render row
        render_row = QtWidgets.QHBoxLayout()
        for name in ["layout", "colorbar_layer", "vmin", "vmax", "auto_update", "crosshair"]:
            if name in self.panel:
                render_row.addWidget(self.panel[name].widget)
        render_row.addStretch()
        ctrl_vlayout.addLayout(render_row)
        # Nav row
        nav_row = QtWidgets.QHBoxLayout()
        for name in ["btn_update", "btn_reset", "auto_follow", "btn_goto_pingline"]:
            if name in self.panel:
                nav_row.addWidget(self.panel[name].widget)
        nav_row.addWidget(QtWidgets.QLabel("  Nav:"))
        for name in ["btn_nav_left", "btn_nav_up", "btn_nav_down", "btn_nav_right"]:
            if name in self.panel:
                nav_row.addWidget(self.panel[name].widget)
        nav_row.addStretch()
        ctrl_vlayout.addLayout(nav_row)
        d_controls.addWidget(ctrl_widget)

        # -- Settings dock (param editor only) --
        d_settings = Dock("Param Editor", size=(300, 300))
        param_layout_keys = ECHO_TAB_LAYOUT.get("Param Editor", [])
        param_widget = QtWidgets.QWidget()
        param_vlayout = QtWidgets.QVBoxLayout(param_widget)
        param_vlayout.setContentsMargins(4, 4, 4, 4)
        for row_names in param_layout_keys:
            row = QtWidgets.QHBoxLayout()
            for n in row_names:
                if n in self.panel:
                    row.addWidget(self.panel[n].widget)
            row.addStretch()
            param_vlayout.addLayout(row)
        param_vlayout.addStretch()
        d_settings.addWidget(param_widget)

        # -- Hover dock --
        d_hover = Dock("Info", size=(800, 30))
        d_hover.addWidget(self.panel["hover_label"].widget)

        # -- Assemble --
        area.addDock(d_graphics, "top")
        area.addDock(d_controls, "bottom", d_graphics)
        area.addDock(d_hover, "bottom", d_controls)
        area.addDock(d_slots, "bottom", d_hover)
        area.addDock(d_settings, "right", d_slots)

        self._dock_area = area

    # =====================================================================
    # Public API
    # =====================================================================

    def run(self) -> None:
        """Enter the Qt event loop (blocking)."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.exec()

    @property
    def slots(self):
        return self.core.slots

    @property
    def slot_selectors(self):
        return self._slot_selectors

    def get_scene(self):
        return self.core.get_scene()

    def save_scene(self, filename: str = "scene.svg") -> None:
        self.core.save_scene(filename)

    def get_matplotlib(self, dpi: int = 150):
        return self.core.get_matplotlib(dpi)

    def get_xlim(self):
        return self.core.get_xlim()

    def get_ylim(self):
        return self.core.get_ylim()

    def pan_view(self, direction: str, fraction: float = 0.25) -> None:
        self.core.pan_view(direction, fraction)

    def connect_pingviewer(self, pingviewer: Any) -> None:
        self.core.connect_pingviewer(pingviewer)

    def disconnect_pingviewer(self) -> None:
        self.core.disconnect_pingviewer()

    def update_ping_lines(self) -> None:
        self.core.update_ping_lines()

    def show_single(self, echogram_name: str) -> None:
        self.core.show_single(echogram_name)

    def add_station_times(self, stations, **kwargs) -> None:
        self.core.add_station_times(stations, **kwargs)

    def clear_station_times(self, station_name=None) -> None:
        self.core.clear_station_times(station_name)

    def reset_view(self) -> None:
        self.core.reset_view()

    # =====================================================================
    # Cleanup
    # =====================================================================

    def cleanup(self) -> None:
        self._is_shutting_down = True
        self._cancel_pending_load()
        self._range_check_timer.stop()
        self._debounce_timer.stop()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
