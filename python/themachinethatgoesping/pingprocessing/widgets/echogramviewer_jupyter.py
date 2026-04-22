"""Jupyter notebook echogram viewer using the extracted core + ipywidgets controls.

Drop-in replacement for ``EchogramViewerMultiChannel`` in
``echogramviewer_pyqtgraph2.py`` — same constructor signature, same public API
— but built on top of :class:`echogramviewer_core.EchogramCore` and
:class:`control_jupyter.JupyterControlPanel`.
"""
from __future__ import annotations

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ipywidgets
import pyqtgraph as pg
from IPython.display import display
from pyqtgraph.jupyter import GraphicsLayoutWidget

from themachinethatgoesping.pingprocessing.widgets import TqdmWidget

from . import pyqtgraph_helpers as pgh
from .control_spec import (
    GRID_LAYOUTS,
    ECHO_RENDER_SPECS,
    ECHO_NAV_SPECS,
    ECHO_MISC_SPECS,
    ECHO_PARAM_SPECS,
    ECHO_PARAM_DISPLAY_SPECS,
    DropdownSpec,
)
from .control_jupyter import JupyterControlPanel, JupyterControlHandle, create_jupyter_control
from .echogramviewer_core import EchogramCore, normalise_echograms, auto_select_grid


class EchogramViewerJupyter:
    """Multi-echogram viewer for Jupyter notebooks.

    This viewer renders echogram images via pyqtgraph (``jupyter_rfb``)
    and exposes ipywidget controls. Internally all heavy lifting is
    delegated to :class:`echogramviewer_core.EchogramCore`.

    The constructor signature is intentionally compatible with
    ``EchogramViewerMultiChannel`` so you can swap them in notebooks.
    """

    def __init__(
        self,
        echogramdata: Union[Dict[str, Any], Sequence[Any], Any],
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
        embedded: bool = False,
        **kwargs: Any,
    ) -> None:
        pgh.ensure_qapp()
        pg.setConfigOptions(imageAxisOrder="row-major")

        # -- normalise echogram input --
        self.echograms, self.echogram_names = normalise_echograms(echogramdata, names)
        self.name = name
        self.widget_height_px = widget_height_px
        self.widget_width_px = widget_width_px
        self.progress = progress or TqdmWidget()
        self.display_progress = progress is None

        # -- choose initial grid --
        n_eg = len(self.echogram_names)
        initial_grid = auto_select_grid(initial_grid, n_eg)

        # -- build control panel --
        self.panel = JupyterControlPanel.from_specs(
            ECHO_RENDER_SPECS,
            ECHO_NAV_SPECS,
            ECHO_MISC_SPECS,
            ECHO_PARAM_SPECS,
            ECHO_PARAM_DISPLAY_SPECS,
        )

        # Grid layout dropdown (needs dynamic options)
        layout_options = [(label, (r, c)) for r, c, label in GRID_LAYOUTS]
        layout_spec = DropdownSpec(
            "layout", "Grid:", options=layout_options, value=initial_grid, width="120px",
        )
        self.panel["layout"] = create_jupyter_control(layout_spec)

        # Per-slot selectors
        echogram_options = [("(none)", None)] + [(n, n) for n in self.echogram_names]
        self._slot_selectors: List[ipywidgets.Dropdown] = []
        for i in range(EchogramCore.MAX_SLOTS):
            eg_key = self.echogram_names[i] if i < n_eg else None
            sel = ipywidgets.Dropdown(
                description=f"Slot {i + 1}:",
                options=echogram_options,
                value=eg_key,
                layout=ipywidgets.Layout(width="200px"),
            )
            sel.observe(
                lambda change, idx=i: self._on_slot_change(idx, change),
                names="value",
            )
            self.panel[f"slot_selector_{i}"] = JupyterControlHandle(sel)
            self._slot_selectors.append(sel)

        # Tab buttons for quick single-view
        self._tab_buttons: List[ipywidgets.Button] = []
        for eg_name in self.echogram_names:
            btn = ipywidgets.Button(
                description=str(eg_name)[:15],
                tooltip=f"Show {eg_name} full-size",
                layout=ipywidgets.Layout(width="auto", min_width="60px"),
            )
            btn.on_click(lambda _, n=eg_name: self.core.show_single(n))
            self._tab_buttons.append(btn)

        # -- output / hover --
        self.output = ipywidgets.Output()
        self.hover_label = self.panel["hover_label"].widget

        # -- graphics widget --
        self.graphics = GraphicsLayoutWidget(
            css_width=f"{widget_width_px}px",
            css_height=f"{widget_height_px}px",
        )
        pgh.apply_widget_layout(self.graphics, widget_width_px, widget_height_px)
        if hasattr(self.graphics, "gfxView"):
            self.graphics.gfxView.setBackground("w")

        # -- auto-update / async state --
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
        self.core._report_error = self._report_error
        self.core._auto_update_enabled = auto_update

        self._embedded = embedded

        # -- wire shared observers --
        self.core.wire_observers(
            layout_callback=self._update_slot_selector_visibility,
        )

        # -- RFB keyboard hook --
        self._setup_rfb_event_handler()

        # -- auto-update hook --
        self._setup_auto_update_hook()

        # -- assemble layout (skip in embedded mode) --
        if not embedded:
            self._assemble_layout()

        if show and not embedded:
            display(self.layout)

        # -- load initial backgrounds --
        self.core.load_all_backgrounds()
        self._startup_complete = True

    # =====================================================================
    # Adapter callbacks
    # =====================================================================

    def _on_slot_change(self, slot_idx: int, change: Dict[str, Any]) -> None:
        if hasattr(self, "core"):
            self.core.on_slot_change(slot_idx, change["new"])

    def _update_slot_selector_visibility(self) -> None:
        if not hasattr(self, "_slot_selector_box"):
            return
        n_visible = self.core.grid_rows * self.core.grid_cols
        self._slot_selector_box.children = [
            self._slot_selectors[i] for i in range(n_visible)
        ]

    def _report_error(self, msg: str) -> None:
        with self.output:
            print(msg)

    # =====================================================================
    # Auto-update via patched request_draw
    # =====================================================================

    def _setup_auto_update_hook(self) -> None:
        original_request_draw = self.graphics.request_draw
        viewer = self

        def patched_request_draw():
            original_request_draw()
            if not viewer._startup_complete or not viewer._auto_update_enabled:
                return
            if viewer.core._ignore_range_changes or viewer._is_loading:
                return
            # Pingline-only moves don't need an echogram data rebuild
            if getattr(viewer.core, '_pingline_update_in_progress', False):
                return
            master = viewer.core._get_master_plot()
            if master is not None:
                current_range = master.getViewBox().viewRange()
                if viewer._last_view_range is not None:
                    import numpy as np
                    old_x, old_y = viewer._last_view_range
                    new_x, new_y = current_range
                    if not (np.allclose(old_x, new_x, rtol=1e-6) and
                            np.allclose(old_y, new_y, rtol=1e-6)):
                        viewer._last_view_range = current_range
                        viewer._last_range_change_time = time.time()
                        viewer._schedule_debounced_update()
                else:
                    viewer._last_view_range = current_range

        self.graphics.request_draw = patched_request_draw

    # =====================================================================
    # RFB keyboard event handler
    # =====================================================================

    def _setup_rfb_event_handler(self) -> None:
        if not hasattr(self.graphics, 'handle_event'):
            return
        if getattr(self, '_rfb_event_hooked', False):
            return

        original_handle_event = self.graphics.handle_event
        viewer = self

        def hooked_handle_event(event):
            event_type = event.get('event_type', '')
            if event_type == 'key_down':
                key = event.get('key', '')
                viewer.core.handle_key_down(key, event.get('modifiers', ()))
            if event_type == 'pointer_move':
                viewer._last_pointer_position = (event.get('x', 0), event.get('y', 0))
            return original_handle_event(event)

        self.graphics.handle_event = hooked_handle_event
        self._rfb_event_hooked = True
        self._last_pointer_position = None

    # =====================================================================
    # Debounced async updates
    # =====================================================================

    def _schedule_debounced_update(self) -> None:
        if self._is_loading:
            self._view_changed_during_load = True
            return
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()

        delay_s = self._auto_update_delay_ms / 1000.0

        async def debounced():
            try:
                await asyncio.sleep(delay_s)
                elapsed = time.time() - self._last_range_change_time
                if elapsed >= delay_s - 0.01:
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

        def apply_results(results):
            viewer._is_loading = False
            if results is None:
                core.progress.set_description('Cancelled')
                if viewer._view_changed_during_load:
                    viewer._view_changed_during_load = False
                    viewer._schedule_debounced_update()
                return
            core.apply_high_res_results(results)
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
                viewer._report_error(f"Error: {e}")
                core.progress.set_description('Error')

        try:
            loop = asyncio.get_running_loop()
            self._loading_future = loop.create_task(run_async())
        except RuntimeError:
            results = load_images()
            apply_results(results)

    def _cancel_pending_load(self) -> None:
        self._cancel_flag.set()
        if self._loading_future is not None:
            try:
                self._loading_future.cancel()
            except Exception:
                pass
            self._loading_future = None
        self._is_loading = False

    # =====================================================================
    # Embeddable control widget
    # =====================================================================

    def build_control_widget(self) -> ipywidgets.Widget:
        """Return all controls as a single embeddable ipywidget."""
        p = self.panel
        n_visible = self.core.grid_rows * self.core.grid_cols

        slot_box = ipywidgets.HBox(
            [self._slot_selectors[i] for i in range(n_visible)]
        )
        tab_box = ipywidgets.HBox(self._tab_buttons)

        controls_row = ipywidgets.HBox([
            p.widget("layout"), p.widget("colorbar_layer"),
            p.widget("vmin"), p.widget("vmax"),
            p.widget("auto_update"), p.widget("crosshair"),
        ])
        buttons_row = ipywidgets.HBox([
            p.widget("btn_update"), p.widget("btn_reset"),
            p.widget("btn_autoscale_y"),
            p.widget("auto_follow"), p.widget("btn_goto_pingline"),
            ipywidgets.Label("  Nav:"),
            p.widget("btn_nav_left"), p.widget("btn_nav_up"),
            p.widget("btn_nav_down"), p.widget("btn_nav_right"),
            p.widget("x_interval"), p.widget("btn_set_x_interval"),
        ])

        progress_box = (
            ipywidgets.HBox([self.progress]) if self.display_progress
            else ipywidgets.HBox([])
        )

        return ipywidgets.VBox([
            tab_box,
            slot_box,
            controls_row,
            buttons_row,
            self.hover_label,
            progress_box,
        ])

    # =====================================================================
    # Layout assembly
    # =====================================================================

    def _assemble_layout(self) -> None:
        p = self.panel
        n_visible = self.core.grid_rows * self.core.grid_cols

        # Slot selectors
        visible_selectors = [self._slot_selectors[i] for i in range(n_visible)]
        self._slot_selector_box = ipywidgets.HBox(visible_selectors)

        # Tab buttons row (quick single-view)
        tab_box = ipywidgets.HBox(self._tab_buttons)

        # Controls row: grid layout, color settings
        controls_row = ipywidgets.HBox([
            p.widget("layout"),
            p.widget("colorbar_layer"),
            p.widget("vmin"),
            p.widget("vmax"),
            p.widget("auto_update"),
            p.widget("crosshair"),
        ])

        # Navigation buttons row
        buttons_row = ipywidgets.HBox([
            p.widget("btn_update"),
            p.widget("btn_reset"),
            p.widget("btn_autoscale_y"),
            p.widget("auto_follow"),
            p.widget("btn_goto_pingline"),
            ipywidgets.Label('  Nav:'),
            p.widget("btn_nav_left"),
            p.widget("btn_nav_up"),
            p.widget("btn_nav_down"),
            p.widget("btn_nav_right"),
            p.widget("x_interval"),
            p.widget("btn_set_x_interval"),
        ])

        # Param editor in collapsible accordion
        param_rows = ipywidgets.VBox([
            ipywidgets.HBox(p.widgets(
                "param_master", "param_select", "btn_refresh_params")),
            ipywidgets.HBox(p.widgets(
                "new_param_name", "btn_new_param", "btn_copy_param", "btn_copy_to_all")),
            ipywidgets.HBox(p.widgets(
                "btn_add_point", "btn_del_point", "param_sync",
                "btn_apply_param", "btn_discard_param")),
            p.widget("param_status"),
            p.widget("param_help"),
        ])
        param_accordion = ipywidgets.Accordion(children=[param_rows])
        param_accordion.set_title(0, "Parameter Editor")
        param_accordion.selected_index = None  # collapsed by default

        # Param display in its own collapsible accordion
        param_display_rows = ipywidgets.VBox([
            ipywidgets.HBox(p.widgets(
                "param_display", "btn_refresh_param_display")),
            ipywidgets.HBox(p.widgets(
                "param_display_cmap", "param_display_size", "param_display_max_points")),
        ])
        param_display_accordion = ipywidgets.Accordion(children=[param_display_rows])
        param_display_accordion.set_title(0, "Parameter Display")
        param_display_accordion.selected_index = None

        progress_box = (
            ipywidgets.HBox([self.progress]) if self.display_progress
            else ipywidgets.HBox([])
        )

        self.layout = ipywidgets.VBox([
            tab_box,
            self._slot_selector_box,
            ipywidgets.HBox([self.graphics]),
            controls_row,
            buttons_row,
            self.hover_label,
            param_display_accordion,
            param_accordion,
            progress_box,
            self.output,
        ])

    # =====================================================================
    # Public helpers (forwarded to core)
    # =====================================================================

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

    def set_widget_size(self, width_px: int, height_px: int) -> None:
        self.widget_width_px = width_px
        self.widget_height_px = height_px
        pgh.apply_widget_layout(self.graphics, width_px, height_px)
        self.core._request_remote_draw()

    def pan_view(self, direction: str, fraction: float = 0.25) -> None:
        self.core.pan_view(direction, fraction)

    def connect_pingviewer(self, pingviewer: Any, **kwargs) -> None:
        self.core.connect_pingviewer(pingviewer, **kwargs)

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

    def show(self) -> None:
        display(self.layout)

    def redraw(self, force: bool = True) -> None:
        self.core._process_qt_events()
        if force:
            self._force_send_frame()
        else:
            self.core._request_remote_draw()

    def process_events(self) -> None:
        self.core._process_qt_events()

    def _force_send_frame(self) -> None:
        from pyqtgraph.Qt import QtCore as _QtCore
        gfx_view = getattr(self.graphics, "gfxView", None)
        if gfx_view is None:
            return
        img = gfx_view.grab()
        if img.isNull():
            return
        buffer = _QtCore.QBuffer()
        buffer.open(_QtCore.QIODevice.OpenModeFlag.WriteOnly)
        img.save(buffer, "PNG")
        buffer.close()
        if hasattr(self.graphics, "_send_frame"):
            self.graphics._send_frame(buffer.data().data())
        else:
            self.core._request_remote_draw()

    # =====================================================================
    # Cleanup
    # =====================================================================

    def cleanup(self) -> None:
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
