"""Jupyter notebook WCI viewer using the extracted core + ipywidgets controls.

Drop-in replacement for ``WCIViewerMultiChannel`` in
``wciviewer_pyqtgraph2.py`` — same constructor signature, same public API
— but built on top of :class:`wciviewer_core.WCICore` and
:class:`control_jupyter.JupyterControlPanel`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ipywidgets
import pyqtgraph as pg
from IPython.display import display
from pyqtgraph.jupyter import GraphicsLayoutWidget

from themachinethatgoesping.pingprocessing.widgets import TqdmWidget

from . import pyqtgraph_helpers as pgh
from .control_spec import (
    GRID_LAYOUTS,
    WCI_MISC_SPECS,
    WCI_RENDER_SPECS,
    WCI_STACK_SPECS,
    WCI_TIMING_SPECS,
    WCI_PLAYBACK_SPECS,
    WCI_VIDEO_SPECS,
    WCI_TAB_LAYOUT,
    DropdownSpec,
    IntSliderSpec,
)
from .control_jupyter import JupyterControlPanel, create_jupyter_control
from .wciviewer_core import WCICore, normalise_channels, auto_select_grid
from .videoframes import VideoFrames


class WCIViewerJupyter:
    """Multi-channel WCI viewer for Jupyter notebooks.

    This viewer renders water-column images using pyqtgraph (via
    ``jupyter_rfb``) and provides ipywidget controls.  Internally all
    heavy lifting is delegated to :class:`wciviewer_core.WCICore`.

    The constructor signature is intentionally compatible with
    ``WCIViewerMultiChannel`` so you can swap them in notebooks.
    """

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
        pgh.ensure_qapp()
        pg.setConfigOptions(imageAxisOrder="row-major")

        # -- normalise channels input --
        self.channels, self.channel_names = normalise_channels(channels, names)

        self.name = name
        self.widget_height_px = widget_height_px
        self.widget_width_px = widget_width_px
        self.progress = progress or TqdmWidget()
        self.display_progress = progress is None

        # -- choose initial grid --
        n_ch = len(self.channel_names)
        initial_grid = auto_select_grid(initial_grid, n_ch)

        # -- build control panel --
        self.panel = JupyterControlPanel.from_specs(
            WCI_RENDER_SPECS,
            WCI_STACK_SPECS,
            WCI_TIMING_SPECS,
            WCI_PLAYBACK_SPECS,
            WCI_VIDEO_SPECS,
            WCI_MISC_SPECS,
        )

        # Grid layout dropdown (needs dynamic options)
        layout_options = [(label, (r, c)) for r, c, label in GRID_LAYOUTS]
        layout_spec = DropdownSpec(
            "layout", "Grid:", options=layout_options, value=initial_grid, width="120px",
        )
        self.panel["layout"] = create_jupyter_control(layout_spec)

        # Per-slot selectors and ping sliders
        channel_options = [("(none)", None)] + [(n, n) for n in self.channel_names]
        self._slot_selectors: List[ipywidgets.Dropdown] = []
        self._ping_sliders: List[ipywidgets.IntSlider] = []

        for i in range(WCICore.MAX_SLOTS):
            ch_key = self.channel_names[i] if i < n_ch else None
            # --- selector ---
            sel = ipywidgets.Dropdown(
                description=f"Ch {i + 1}:",
                options=channel_options,
                value=ch_key,
                layout=ipywidgets.Layout(width="180px"),
            )
            sel.observe(
                lambda change, idx=i: self._on_slot_change(idx, change),
                names="value",
            )
            from .control_jupyter import JupyterControlHandle
            self.panel[f"slot_selector_{i}"] = JupyterControlHandle(sel)
            self._slot_selectors.append(sel)

            # --- ping slider ---
            slider = ipywidgets.IntSlider(
                description="Ping:",
                min=0,
                max=0,
                value=0,
                layout=ipywidgets.Layout(width="250px"),
            )
            slider.observe(
                lambda change, idx=i: self._on_ping_change(idx, change),
                names="value",
            )
            self.panel[f"ping_slider_{i}"] = JupyterControlHandle(slider)
            self._ping_sliders.append(slider)

        # Tab buttons for quick single-view
        self._tab_buttons: List[ipywidgets.Button] = []
        for ch_name in self.channel_names:
            btn = ipywidgets.Button(
                description=str(ch_name)[:15],
                tooltip=f"Show {ch_name} full-size",
                layout=ipywidgets.Layout(width="auto", min_width="60px"),
            )
            # core's show_single will be called after core is created
            btn.on_click(lambda _, n=ch_name: self.core.show_single(n))
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

        # -- set max for ping sliders before core reads them --
        for i in range(min(n_ch, WCICore.MAX_SLOTS)):
            pings = self.channels.get(self.channel_names[i])
            if pings:
                self._ping_sliders[i].max = max(0, len(pings) - 1)

        # -- create core --
        self.core = WCICore(
            channels=self.channels,
            channel_names=self.channel_names,
            panel=self.panel,
            graphics=self.graphics,
            progress=self.progress,
            cmap=cmap,
            initial_grid=initial_grid,
            time_sync_enabled=time_sync_enabled,
            time_warning_threshold=time_warning_threshold,
            horizontal_pixels=horizontal_pixels,
            **kwargs,
        )

        # Expose frames on the viewer for convenience
        self.frames: VideoFrames = self.core.frames

        # -- wire shared observers (after core exists) --
        self.core.wire_observers(
            layout_callback=self._update_slot_selector_visibility,
        )

        # -- assemble layout --
        self._assemble_layout()

        if show:
            display(self.layout)

    # =====================================================================
    # Dynamic control callbacks
    # =====================================================================

    def _on_slot_change(self, slot_idx: int, change: Dict[str, Any]) -> None:
        if hasattr(self, "core"):
            self.core.on_slot_change(slot_idx, change["new"])

    def _on_ping_change(self, slot_idx: int, change: Dict[str, Any]) -> None:
        if hasattr(self, "core"):
            self.core.on_ping_change(slot_idx, change["new"])

    def _update_slot_selector_visibility(self) -> None:
        n_visible = self.core.grid_rows * self.core.grid_cols
        controls = []
        for i in range(n_visible):
            controls.append(
                ipywidgets.HBox([self._slot_selectors[i], self._ping_sliders[i]])
            )
        self._slot_selector_box.children = controls

    # =====================================================================
    # Layout assembly
    # =====================================================================

    def _assemble_layout(self) -> None:
        p = self.panel
        n_visible = self.core.grid_rows * self.core.grid_cols

        # Slot selectors
        slot_controls = []
        for i in range(n_visible):
            slot_controls.append(
                ipywidgets.HBox([self._slot_selectors[i], self._ping_sliders[i]])
            )
        self._slot_selector_box = ipywidgets.VBox(slot_controls)

        # Tab buttons row
        tab_box = ipywidgets.HBox([p.widget("layout")] + self._tab_buttons)

        # Tabbed settings
        settings_tabs = p.build_tabs(WCI_TAB_LAYOUT)

        main_left = ipywidgets.VBox([
            self._slot_selector_box,
            ipywidgets.HBox([
                p.widget("ref_time"),
                p.widget("fix_xy"),
                p.widget("unfix_xy"),
            ]),
            ipywidgets.HBox([
                p.widget("proctime"),
                p.widget("procrate"),
            ]),
        ])

        main_controls = ipywidgets.HBox([main_left, settings_tabs])

        progress_box = (
            ipywidgets.HBox([self.progress]) if self.display_progress
            else ipywidgets.HBox([])
        )

        self.layout = ipywidgets.VBox([
            ipywidgets.HBox([self.graphics]),
            progress_box,
            tab_box,
            main_controls,
            self.hover_label,
            self.output,
        ])

    # =====================================================================
    # Public helpers (forwarded to core)
    # =====================================================================

    def register_ping_change_callback(self, callback: Any) -> None:
        self.core.register_ping_change_callback(callback)

    def unregister_ping_change_callback(self, callback: Any) -> None:
        self.core.unregister_ping_change_callback(callback)

    @property
    def w_index(self):
        return self.core.w_index

    @property
    def imagebuilder(self):
        return self.core.imagebuilder

    @property
    def slots(self):
        return self.core.slots

    def get_scene(self):
        return self.core.get_scene()

    def save_scene(self, filename: str = "scene.svg") -> None:
        self.core.save_scene(filename)

    def get_matplotlib(self, dpi: int = 150):
        return self.core.get_matplotlib(dpi)

    def set_widget_height(self, height_px: int) -> None:
        self.widget_height_px = height_px
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, height_px)

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
