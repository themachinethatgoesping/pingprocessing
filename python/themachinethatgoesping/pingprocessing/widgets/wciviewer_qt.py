"""Native Qt WCI viewer using pyqtgraph DockArea + extracted WCICore.

This module provides a standalone Qt application for viewing water-column
images without Jupyter.  The layout uses ``pyqtgraph.dockarea`` so the
user can drag, float, and rearrange panels.

Usage (standalone)::

    python -m themachinethatgoesping.pingprocessing.widgets.wciviewer_qt \\
        --help

Or programmatically::

    from themachinethatgoesping.pingprocessing.widgets.wciviewer_qt import WCIViewerQt
    viewer = WCIViewerQt(channels)
    viewer.show()
    viewer.run()   # blocks in QApplication.exec()
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.dockarea import Dock, DockArea

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
from .control_qt import QtControlPanel, QtControlHandle, create_qt_control, _SLIDER_STYLE
from .wciviewer_core import WCICore, normalise_channels, auto_select_grid
from .videoframes import VideoFrames


class _QtProgressBar:
    """Qt progress widget with a real QProgressBar, it/s and ETA like tqdm.

    Compatible with the TqdmWidget interface used by WCICore and ImageBuilder:
    ``set_description(str)``, callable-as-iterator ``progress(iterable)``,
    and ``close()``.
    """

    def __init__(self):
        import time as _time
        self._time = _time

        self._container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self._container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setMinimumHeight(14)
        self._bar.setMaximumHeight(14)
        self._bar.setStyleSheet(
            "QProgressBar {"
            "  border: 1px solid #bbb; border-radius: 4px;"
            "  background: #e8e8e8;"
            "}"
            "QProgressBar::chunk {"
            "  background: #5b9bd5; border-radius: 3px;"
            "}"
        )

        self._label = QtWidgets.QLabel("Idle")
        self._label.setStyleSheet("color: #444; font-size: 11px; padding: 0 2px;")
        self._label.setMaximumHeight(16)

        lay.addWidget(self._bar)
        lay.addWidget(self._label)

        self._total: int = 0
        self._idx: int = 0
        self._t_start: float = 0.0
        self._items: list = []

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._container

    def _refresh(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def _format_stats(self) -> str:
        elapsed = self._time.time() - self._t_start
        if self._idx > 0 and elapsed > 0:
            it_s = self._idx / elapsed
            remaining = (self._total - self._idx) / it_s if it_s > 0 else 0
            return (
                f"{self._idx}/{self._total}"
                f"  [{elapsed:.1f}s < {remaining:.1f}s, {it_s:.1f} it/s]"
            )
        return f"{self._idx}/{self._total}"

    def set_description(self, desc: str) -> None:
        self._label.setText(desc)
        self._refresh()

    def __call__(self, list_like, **kwargs):
        self._items = list(list_like)
        self._idx = 0
        self._total = len(self._items)
        self._t_start = self._time.time()
        self._bar.setRange(0, self._total)
        self._bar.setValue(0)
        desc = kwargs.get("desc", "")
        self._label.setText(f"{desc}  0/{self._total}" if desc else f"0/{self._total}")
        self._refresh()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._total:
            self._bar.setValue(self._total)
            self._label.setText("Idle")
            self._refresh()
            raise StopIteration
        item = self._items[self._idx]
        self._idx += 1
        self._bar.setValue(self._idx)
        self._label.setText(self._format_stats())
        self._refresh()
        return item

    def __len__(self):
        return self._total

    def close(self):
        self._bar.setValue(0)
        self._label.setText("Idle")
        self._refresh()


class WCIViewerQt(QtWidgets.QMainWindow):
    """Standalone native Qt WCI viewer window.

    Parameters match :class:`wciviewer_jupyter.WCIViewerJupyter` so the
    two viewers are interchangeable at construction time.
    """

    def __init__(
        self,
        channels: Union[Dict[str, Sequence[Any]], Sequence[Sequence[Any]]],
        name: str = "Multi-Channel WCI",
        names: Optional[Sequence[Optional[str]]] = None,
        horizontal_pixels: int = 1024,
        progress: Optional[Any] = None,
        cmap: str = "YlGnBu_r",
        show: bool = True,
        widget_height_px: int = 800,
        widget_width_px: int = 1200,
        initial_grid: Tuple[int, int] = (2, 2),
        time_sync_enabled: bool = True,
        time_warning_threshold: float = 5.0,
        parent: Optional[QtWidgets.QWidget] = None,
        **kwargs: Any,
    ) -> None:
        pgh.ensure_qapp()
        super().__init__(parent)
        self.setWindowTitle(name)
        self.resize(widget_width_px, widget_height_px)

        pg.setConfigOptions(imageAxisOrder="row-major")

        # -- normalise channels --
        self.channels, self.channel_names = normalise_channels(channels, names)

        self.progress = progress or _QtProgressBar()
        self.display_progress = progress is None
        n_ch = len(self.channel_names)

        # -- auto grid --
        initial_grid = auto_select_grid(initial_grid, n_ch)

        # -- build control panel --
        self.panel = QtControlPanel.from_specs(
            WCI_RENDER_SPECS,
            WCI_STACK_SPECS,
            WCI_TIMING_SPECS,
            WCI_PLAYBACK_SPECS,
            WCI_VIDEO_SPECS,
            WCI_MISC_SPECS,
        )

        # Grid layout dropdown
        layout_options = [(label, (r, c)) for r, c, label in GRID_LAYOUTS]
        layout_spec = DropdownSpec(
            "layout", "Grid:", options=layout_options, value=initial_grid, width="120px",
        )
        self.panel["layout"] = create_qt_control(layout_spec)

        # Per-slot selectors and ping sliders
        channel_options = [("(none)", None)] + [(n, n) for n in self.channel_names]
        self._slot_selectors: List[QtWidgets.QComboBox] = []
        self._ping_sliders: List[QtWidgets.QSlider] = []

        for i in range(WCICore.MAX_SLOTS):
            ch_key = self.channel_names[i] if i < n_ch else None

            # selector
            sel = QtWidgets.QComboBox()
            for opt_label, opt_val in channel_options:
                sel.addItem(opt_label, opt_val)
            if ch_key is not None:
                idx = sel.findData(ch_key)
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

            # ping slider (QSlider + QSpinBox readout)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 0)
            slider.setValue(0)
            slider.setMinimumHeight(24)
            slider.setStyleSheet(_SLIDER_STYLE)

            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 0)
            spin.setValue(0)
            spin.setFixedWidth(80)

            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)

            slider.valueChanged.connect(
                lambda val, idx=i: self._on_ping_slider_moved(idx, val)
            )

            handle = QtControlHandle(
                slider, slider.value, slider.setValue, slider.valueChanged,
                inner=slider,
            )
            handle._paired_spin = spin
            self.panel[f"ping_slider_{i}"] = handle
            self._ping_sliders.append(slider)
            # store spin for layout
            if not hasattr(self, "_ping_spins"):
                self._ping_spins: List[QtWidgets.QSpinBox] = []
            self._ping_spins.append(spin)

        # -- ping slider debounce (prevent event stacking on arrow keys) --
        self._pending_ping: Dict[int, int] = {}
        self._ping_updating = False

        # -- graphics widget (native, no rfb) --
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")

        # -- set max for initial ping sliders --
        for i in range(min(n_ch, WCICore.MAX_SLOTS)):
            pings = self.channels.get(self.channel_names[i])
            if pings:
                max_val = max(0, len(pings) - 1)
                self._ping_sliders[i].setMaximum(max_val)
                self._ping_spins[i].setMaximum(max_val)

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

        self.frames: VideoFrames = self.core.frames

        # -- wire shared observers (after core exists) --
        self.core.wire_observers(
            play_callback=self._toggle_play_qt,
        )

        # -- assemble DockArea layout --
        self._build_dock_layout()

        if show:
            self.show()

    # =====================================================================
    # Dynamic control callbacks
    # =====================================================================

    def _on_slot_change(self, slot_idx: int, new_channel) -> None:
        if hasattr(self, "core"):
            self.core.on_slot_change(slot_idx, new_channel)

    def _on_ping_slider_moved(self, slot_idx: int, new_ping: int) -> None:
        """Buffer rapid slider changes and process only the latest."""
        self._pending_ping[slot_idx] = new_ping
        if self._ping_updating:
            return  # skip — current update will pick up the latest value
        self._flush_pending_pings()

    def _flush_pending_pings(self) -> None:
        """Process pending ping changes, coalescing rapid events."""
        self._ping_updating = True
        try:
            while self._pending_ping:
                # take a snapshot and clear
                pending = dict(self._pending_ping)
                self._pending_ping.clear()
                for slot_idx, val in pending.items():
                    if hasattr(self, "core"):
                        self.core.on_ping_change(slot_idx, val)
        finally:
            self._ping_updating = False

    def _on_ping_change(self, slot_idx: int, new_ping: int) -> None:
        if hasattr(self, "core"):
            self.core.on_ping_change(slot_idx, new_ping)

    # =====================================================================
    # Qt-native autoplay (QTimer instead of asyncio)
    # =====================================================================

    def _toggle_play_qt(self) -> None:
        if self.core._autoplay_active:
            self._stop_play_qt()
        else:
            self._start_play_qt()

    def _start_play_qt(self) -> None:
        self.core._autoplay_active = True
        self.core.panel["play_button"].description = "Stop"
        self._play_last_time: float | None = None
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        fps = max(0.1, float(self.panel["play_fps"].value))
        self._play_timer.start(int(1000.0 / fps))

    def _play_tick(self) -> None:
        import time as _time

        t0 = _time.time()
        slot = self.core.slots[0]
        pings = slot.get_pings()
        if not pings:
            return
        step = max(1, int(self.panel["ping_step"].value))
        speed_mult = max(0.1, float(self.panel["play_fps"].value))
        use_ping_time = self.panel["use_ping_time"].value
        max_idx = len(pings) - 1
        current_idx = self.panel["ping_slider_0"].value
        new_idx = current_idx + step
        if new_idx > max_idx:
            new_idx = 0

        # Compute next interval
        interval_ms = int(1000.0 / speed_mult)
        if use_ping_time:
            current_ts = slot.get_timestamp(current_idx)
            next_ts = slot.get_timestamp(new_idx)
            if current_ts is not None and next_ts is not None:
                ping_dt = abs(next_ts - current_ts)
                interval_ms = max(5, int(1000.0 * ping_dt / speed_mult))

        self.panel["ping_slider_0"].value = new_idx

        # Real fps display
        if self._play_last_time is not None:
            real_interval = t0 - self._play_last_time
            if real_interval > 0:
                real_fps = 1.0 / real_interval
                self.panel["real_fps"].value = f"real: {real_fps:.1f}"
        self._play_last_time = t0

        # Adapt timer for next tick
        self._play_timer.setInterval(interval_ms)

    def _stop_play_qt(self) -> None:
        self.core._autoplay_active = False
        self.core.panel["play_button"].description = "\u25b6 Play"
        self.core.panel["real_fps"].value = "real: --"
        if hasattr(self, "_play_timer"):
            self._play_timer.stop()

    # =====================================================================
    # DockArea layout
    # =====================================================================

    def _build_dock_layout(self) -> None:
        area = DockArea()
        self.setCentralWidget(area)

        # -- Graphics dock --
        d_graphics = Dock("WCI", size=(800, 500))
        d_graphics.addWidget(self.graphics)

        # -- Slot selectors dock --
        d_slots = Dock("Channels", size=(300, 200))
        slot_widget = QtWidgets.QWidget()
        slot_layout = QtWidgets.QVBoxLayout(slot_widget)
        slot_layout.setContentsMargins(4, 4, 4, 4)

        # layout dropdown + tab buttons
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.panel["layout"].widget)
        for ch_name in self.channel_names:
            btn = QtWidgets.QPushButton(str(ch_name)[:15])
            btn.clicked.connect(lambda _, n=ch_name: self.core.show_single(n))
            top_row.addWidget(btn)
        slot_layout.addLayout(top_row)

        n_visible = self.core.grid_rows * self.core.grid_cols
        for i in range(n_visible):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self._slot_selectors[i])
            row.addWidget(self._ping_sliders[i], 1)
            row.addWidget(self._ping_spins[i])
            slot_layout.addLayout(row)

        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(self.panel["ref_time"].widget)
        ref_row.addWidget(self.panel["fix_xy"].widget)
        ref_row.addWidget(self.panel["unfix_xy"].widget)
        slot_layout.addLayout(ref_row)

        # Timing display (moved from Settings tab)
        timing_row = QtWidgets.QHBoxLayout()
        timing_row.addWidget(self.panel["proctime"].widget)
        timing_row.addWidget(self.panel["procrate"].widget)
        slot_layout.addLayout(timing_row)

        # Progress / status label
        if hasattr(self.progress, "widget"):
            slot_layout.addWidget(self.progress.widget)

        slot_layout.addStretch()
        d_slots.addWidget(slot_widget)

        # -- Settings dock (tabbed) --
        d_settings = Dock("Settings", size=(300, 300))
        settings_tabs = self.panel.build_tab_widget(WCI_TAB_LAYOUT)
        d_settings.addWidget(settings_tabs)

        # -- Hover dock --
        d_hover = Dock("Info", size=(800, 30))
        d_hover.addWidget(self.panel["hover_label"].widget)

        # -- Assemble --
        area.addDock(d_graphics, "top")
        area.addDock(d_hover, "bottom", d_graphics)
        area.addDock(d_slots, "bottom", d_hover)
        area.addDock(d_settings, "right", d_slots)

        self._dock_area = area

    # =====================================================================
    # Public API
    # =====================================================================

    def run(self) -> None:
        """Enter the Qt event loop (blocking).  Call after ``.show()``."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.exec()

    def register_ping_change_callback(self, callback: Any) -> None:
        self.core.register_ping_change_callback(callback)

    def unregister_ping_change_callback(self, callback: Any) -> None:
        self.core.unregister_ping_change_callback(callback)

    @property
    def slots(self):
        return self.core.slots

    @property
    def imagebuilder(self):
        return self.core.imagebuilder
