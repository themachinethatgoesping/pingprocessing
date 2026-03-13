"""Core WCI viewer logic independent of the UI toolkit.

Manages pyqtgraph scene-graph items (PlotItem, ImageItem, ColorBarItem, …)
and reads / writes control state through a :class:`ControlPanel` abstraction.
Adapters (``wciviewer_jupyter``, ``wciviewer_qt``) create the concrete
controls and wire up observers.
"""
from __future__ import annotations

import time as time_module
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from themachinethatgoesping import echosounders
import themachinethatgoesping.pingprocessing.watercolumn.image as mi

from .control_spec import (
    ControlPanel,
    GRID_LAYOUTS,
    WCI_VALUE_CHOICES,
)
from . import pyqtgraph_helpers as pgh
from .videoframes import VideoFrames


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def normalise_channels(
    channels: Union[Dict[str, Sequence[Any]], Sequence[Sequence[Any]]],
    names: Optional[Sequence[Optional[str]]],
) -> Tuple[Dict[str, Sequence[Any]], List[str]]:
    """Normalise various channel input formats to ``(dict, names_list)``."""
    if isinstance(channels, dict):
        return dict(channels), list(channels.keys())

    if hasattr(channels, "__iter__") and not isinstance(channels, (str, bytes)):
        channels_list = list(channels)
        if len(channels_list) > 0:
            first = channels_list[0]
            if hasattr(first, "get_timestamp") or hasattr(first, "file_data"):
                return {"default": channels_list}, ["default"]
            if isinstance(first, dict):
                return {"default": channels_list}, ["default"]
            # list of ping lists
            if names is not None:
                ch_names = [str(n) if n else f"Channel {i}" for i, n in enumerate(names)]
            else:
                ch_names = [f"Channel {i}" for i in range(len(channels_list))]
            return {n: ch for n, ch in zip(ch_names, channels_list)}, ch_names
        return {}, []

    return {"default": [channels]}, ["default"]


def auto_select_grid(
    initial_grid: Tuple[int, int],
    n_channels: int,
) -> Tuple[int, int]:
    """Choose an appropriate grid size based on the number of channels.

    Only overrides when the caller passes the default ``(2, 2)``.
    """
    if initial_grid != (2, 2):
        return initial_grid
    if n_channels == 1:
        return (1, 1)
    elif n_channels == 2:
        return (1, 2)
    elif n_channels <= 4:
        return (2, 2)
    elif n_channels <= 6:
        return (3, 2)
    else:
        return (4, 2)


# ---------------------------------------------------------------------------
# WCISlot
# ---------------------------------------------------------------------------

class WCISlot:
    """Manages a single WCI display slot with per-slot ping and colour levels."""

    def __init__(
        self,
        slot_idx: int,
        channels: Dict[str, Sequence[Any]],
        args_imagebuilder: Dict[str, Any],
        progress: Any,
    ) -> None:
        self.slot_idx = slot_idx
        self._channels = channels
        self._args_imagebuilder = args_imagebuilder
        self._progress = progress

        self.channel_key: Optional[str] = None
        self.is_visible = False

        self.ping_index: int = 0
        self.stored_ping_index: int = 0
        self.color_levels: Optional[Tuple[float, float]] = None
        self.time_offset: float = 0.0

        # pyqtgraph items (set by the core during grid creation)
        self.plot_item: Optional[pg.PlotItem] = None
        self.image_item: Optional[pg.ImageItem] = None
        self.colorbar: Optional[pg.ColorBarItem] = None
        self.time_offset_text: Optional[pg.TextItem] = None
        self.crosshair_v: Optional[pg.InfiniteLine] = None
        self.crosshair_h: Optional[pg.InfiniteLine] = None

        self.wci_image: Optional[np.ndarray] = None
        self.wci_extent: Optional[Tuple[float, float, float, float]] = None
        self._image_cache: Dict[int, Dict[str, Any]] = {}

        self.imagebuilder: Optional[mi.ImageBuilder] = None

    # -- helpers --

    def set_visible(self, visible: bool) -> None:
        self.is_visible = visible

    def assign_channel(self, channel_key: Optional[str]) -> None:
        if channel_key != self.channel_key:
            self.channel_key = channel_key
            self.wci_image = None
            self.wci_extent = None
            self._image_cache.clear()

            if channel_key is not None:
                pings = self._channels.get(channel_key)
                if pings is not None and len(pings) > 0:
                    self.imagebuilder = mi.ImageBuilder(
                        pings,
                        horizontal_pixels=self._args_imagebuilder["horizontal_pixels"],
                        progress=self._progress,
                    )
                else:
                    self.imagebuilder = None
            else:
                self.imagebuilder = None

    def get_pings(self) -> Optional[Sequence[Any]]:
        if self.channel_key is None:
            return None
        return self._channels.get(self.channel_key)

    def get_ping(self, index: Optional[int] = None) -> Optional[Any]:
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
        ping = self.get_ping(index)
        if ping is None:
            return None
        try:
            return ping.get_timestamp()
        except Exception:
            return None

    def find_closest_ping_index(self, target_timestamp: float) -> int:
        pings = self.get_pings()
        if pings is None or len(pings) == 0:
            return 0
        best_idx = 0
        best_diff = float("inf")
        for i in range(len(pings)):
            ts = self.get_timestamp(i)
            if ts is not None:
                diff = abs(ts - target_timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
        return best_idx


# ---------------------------------------------------------------------------
# WCICore
# ---------------------------------------------------------------------------

class WCICore:
    """Backend-agnostic WCI viewer core.

    Parameters
    ----------
    channels : dict
        ``{name: ping_sequence}``
    channel_names : list[str]
        Ordered channel names.
    panel : ControlPanel
        Provides named :class:`ControlHandle` objects for reading / writing
        UI state.  Must already contain all controls listed in
        ``control_spec.WCI_*`` plus per-slot controls registered as
        ``"slot_selector_<i>"`` and ``"ping_slider_<i>"``.
    graphics : pg.GraphicsLayoutWidget
        The pyqtgraph widget (jupyter-rfb or native).
    progress : any
        Progress bar object (TqdmWidget or similar).
    cmap : str
        Matplotlib / pyqtgraph colourmap name.
    initial_grid : tuple[int, int]
        ``(rows, cols)``
    time_sync_enabled : bool
    time_warning_threshold : float
    """

    MAX_SLOTS = 8

    def __init__(
        self,
        channels: Dict[str, Sequence[Any]],
        channel_names: List[str],
        panel: ControlPanel,
        graphics: pg.GraphicsLayoutWidget,
        progress: Any,
        cmap: str = "YlGnBu_r",
        initial_grid: Tuple[int, int] = (2, 2),
        time_sync_enabled: bool = True,
        time_warning_threshold: float = 5.0,
        **kwargs: Any,
    ) -> None:
        self.channels = channels
        self.channel_names = channel_names
        self.panel = panel
        self.graphics = graphics
        self.progress = progress
        self.cmap_name = cmap
        self._colormap = pgh.resolve_colormap(cmap)

        # -- book-keeping from user kwargs --
        self.args_imagebuilder: Dict[str, Any] = {
            "horizontal_pixels": kwargs.pop("horizontal_pixels", 1024),
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
        self.args_plot: Dict[str, Any] = {
            "vmin": kwargs.pop("vmin", -90),
            "vmax": kwargs.pop("vmax", -25),
        }
        for key in list(kwargs.keys()):
            if key in self.args_imagebuilder:
                self.args_imagebuilder[key] = kwargs.pop(key)
        self.args_imagebuilder.update(kwargs)

        # Grid state
        self.grid_rows, self.grid_cols = initial_grid
        self.time_sync_enabled = time_sync_enabled
        self.time_warning_threshold = time_warning_threshold
        self._reference_slot_idx: int = 0
        self._syncing = False
        self._reference_timestamp: Optional[float] = None

        self._crosshair_enabled = True
        self._crosshair_position: Optional[Tuple[float, float]] = None
        self._ping_change_callbacks: List[Any] = []
        self._ignore_range_changes = False
        self._first_draw = True

        self._autoplay_active = False
        self._autoplay_task: Optional[Any] = None
        self._autoplay_last_time: Optional[float] = None

        self._continuous_capture_active = False

        self.frames: VideoFrames = VideoFrames()

        # -- create slots --
        self.slots: List[WCISlot] = []
        for i in range(self.MAX_SLOTS):
            self.slots.append(
                WCISlot(i, self.channels, self.args_imagebuilder, self.progress)
            )
        for i, ch_name in enumerate(self.channel_names[: self.MAX_SLOTS]):
            self.slots[i].assign_channel(ch_name)

        # -- initial graphics build --
        self.update_grid_layout()

        # -- initial data load --
        self._update_all_visible()

        # -- set initial reference time --
        if self.slots[0].channel_key is not None:
            ts = self.slots[0].get_timestamp()
            if ts is not None:
                self._reference_timestamp = ts
                self._update_ref_time_display()

    # =====================================================================
    # Observer wiring (shared between adapters)
    # =====================================================================

    def wire_observers(
        self,
        *,
        play_callback: Optional[Callable[[], Any]] = None,
        layout_callback: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Connect panel controls to core methods.

        Parameters
        ----------
        play_callback
            Called when the play button is clicked.
            Defaults to :meth:`toggle_autoplay` (async, for Jupyter).
        layout_callback
            Called (with no args) **after** the grid layout change has been
            processed.  Use for adapter-specific UI updates such as
            refreshing slot-selector visibility in Jupyter.
        """
        if play_callback is None:
            play_callback = self.toggle_autoplay

        p = self.panel

        def _on_layout(new_val: Any) -> None:
            self.on_layout_change(*new_val)
            if layout_callback is not None:
                layout_callback()

        p["layout"].on_change(_on_layout)

        p["vmin"].on_change(lambda _: self.on_global_color_change())
        p["vmax"].on_change(lambda _: self.on_global_color_change())

        for name in (
            "stack", "stack_step", "mp_cores", "stack_linear",
            "wci_value", "wci_render", "horizontal_pixels",
            "oversampling", "oversampling_mode",
        ):
            p[name].on_change(lambda _, n=name: self.on_global_param_change())

        p["time_sync"].on_change(lambda v: setattr(self, "time_sync_enabled", v))
        p["crosshair"].on_change(lambda v: setattr(self, "_crosshair_enabled", v))
        p["time_warning"].on_change(
            lambda v: setattr(self, "time_warning_threshold", v)
        )

        p["fix_xy"].on_click(lambda _: self.fix_xy())
        p["unfix_xy"].on_click(lambda _: self.unfix_xy())

        p["ping_step"].on_change(lambda v: self.on_ping_step_change(v))
        p["step_prev"].on_click(lambda _: self.step_prev())
        p["step_next"].on_click(lambda _: self.step_next())
        p["play_button"].on_click(lambda _: play_callback())

        p["video_format"].on_change(lambda v: self.on_video_format_change(v))
        p["export_video"].on_click(lambda _: self.export_video())
        p["continuous_capture"].on_click(lambda _: self.toggle_continuous_capture())

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

            plot: pg.PlotItem = self.graphics.addPlot(row=row, col=col * 2)
            slot.plot_item = plot

            title = str(slot.channel_key) if slot.channel_key else f"Slot {i + 1}"
            plot.setTitle(title)
            plot.setLabel("left", "Depth (m)" if col == 0 else "")
            plot.setLabel("bottom", "Horizontal distance" if row == self.grid_rows - 1 else "")
            plot.getViewBox().invertY(True)
            plot.getViewBox().setBackgroundColor("w")
            plot.getViewBox().setAspectLocked(True, ratio=1)

            # image
            image_item = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image_item)
            slot.image_item = image_item

            # colorbar
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
                if hasattr(colorbar, "sigLevelsChanged"):
                    colorbar.sigLevelsChanged.connect(
                        lambda cb=colorbar, s=slot: self._on_colorbar_levels_changed(s, cb)
                    )
            except AttributeError:
                slot.colorbar = None

            # time-offset text
            time_text = pg.TextItem(text="", color=(0, 0, 0), anchor=(0, 0))
            time_text.setPos(0, 0)
            time_text.hide()
            plot.addItem(time_text)
            slot.time_offset_text = time_text

            # crosshairs
            pen = pg.mkPen(color="r", width=1, style=QtCore.Qt.PenStyle.DashLine)
            slot.crosshair_v = pg.InfiniteLine(angle=90, pen=pen)
            slot.crosshair_h = pg.InfiniteLine(angle=0, pen=pen)
            slot.crosshair_v.hide()
            slot.crosshair_h.hide()
            plot.addItem(slot.crosshair_v)
            plot.addItem(slot.crosshair_h)

            # link axes
            if master_plot is None:
                master_plot = plot
            else:
                plot.setXLink(master_plot)
                plot.setYLink(master_plot)

        self._connect_scene_events()

    # =====================================================================
    # Event handlers (called by adapters)
    # =====================================================================

    def on_layout_change(self, new_rows: int, new_cols: int) -> None:
        if (new_rows, new_cols) == (self.grid_rows, self.grid_cols):
            return
        current_range = self._capture_current_view_range()
        self.grid_rows, self.grid_cols = new_rows, new_cols
        self.update_grid_layout()
        if current_range is not None:
            self._restore_view_range(current_range)
        self._update_all_visible()

    def on_slot_change(self, slot_idx: int, new_channel: Optional[str]) -> None:
        slot = self.slots[slot_idx]
        current_range = self._capture_current_view_range()
        self._ignore_range_changes = True
        try:
            slot.assign_channel(new_channel)
            pings = slot.get_pings()
            max_ping = max(0, len(pings) - 1) if pings else 0
            self.panel[f"ping_slider_{slot_idx}"].max = max_ping

            if self.time_sync_enabled and pings and len(pings) > 0:
                ref_ts = self._reference_timestamp
                if ref_ts is not None:
                    closest_idx = slot.find_closest_ping_index(ref_ts)
                    slot.ping_index = closest_idx
                    self._syncing = True
                    try:
                        self.panel[f"ping_slider_{slot_idx}"].value = closest_idx
                    finally:
                        self._syncing = False
                    closest_ts = slot.get_timestamp(closest_idx)
                    slot.time_offset = (closest_ts - ref_ts) if closest_ts is not None else 0.0
                else:
                    self.panel[f"ping_slider_{slot_idx}"].value = min(slot.ping_index, max_ping)
                    slot.time_offset = 0.0
            else:
                self.panel[f"ping_slider_{slot_idx}"].value = min(slot.ping_index, max_ping)

            if slot.is_visible:
                self._update_slot(slot)
                self._update_time_offset_text(slot)
                self._process_qt_events()
        finally:
            self._ignore_range_changes = False

        if current_range is not None:
            self._restore_view_range(current_range)
        self._request_remote_draw()

    def on_ping_change(self, slot_idx: int, new_ping: int) -> None:
        if self._syncing:
            return
        t0 = time_module.time()
        slot = self.slots[slot_idx]
        slot.ping_index = new_ping
        slot.stored_ping_index = new_ping

        new_ts = slot.get_timestamp()
        if new_ts is not None:
            self._reference_timestamp = new_ts
            self._update_ref_time_display()

        if slot.is_visible:
            self._update_slot(slot)

        if self.time_sync_enabled:
            self._sync_other_slots(slot_idx)

        for callback in self._ping_change_callbacks:
            try:
                callback()
            except Exception:
                pass

        t1 = time_module.time()
        self._request_remote_draw()
        t2 = time_module.time()
        self._update_timing_fields(t0, t1, t2)

        if self._continuous_capture_active:
            self._process_qt_events()
            self._capture_current_frame()

    def on_global_color_change(self) -> None:
        vmin = float(self.panel["vmin"].value)
        vmax = float(self.panel["vmax"].value)
        self.args_plot["vmin"] = vmin
        self.args_plot["vmax"] = vmax

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

    def on_global_param_change(self) -> None:
        self._sync_builder_args()
        self._update_all_visible()

    def on_video_format_change(self, fmt: str) -> None:
        if fmt == "frames":
            self.panel["video_filename"].visible = False
            self.panel["video_quality"].visible = False
        elif fmt == "avif":
            self.panel["video_filename"].visible = True
            self.panel["video_quality"].visible = True
        else:
            self.panel["video_filename"].visible = True
            self.panel["video_quality"].visible = False

    def on_ping_step_change(self, new_step: int) -> None:
        new_step = max(1, new_step)
        for i in range(self.MAX_SLOTS):
            if f"ping_slider_{i}" in self.panel:
                self.panel[f"ping_slider_{i}"].step = new_step

    # =====================================================================
    # Time synchronisation
    # =====================================================================

    def _sync_other_slots(self, reference_slot_idx: int) -> None:
        ref_ts = self._reference_timestamp
        if ref_ts is None:
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
                    slot.time_offset = 0.0
                    self._update_time_offset_text(slot)
                    continue
                closest_idx = slot.find_closest_ping_index(ref_ts)
                closest_ts = slot.get_timestamp(closest_idx)
                slot.time_offset = (closest_ts - ref_ts) if closest_ts is not None else 0.0
                slot.ping_index = closest_idx
                self.panel[f"ping_slider_{i}"].value = closest_idx
                if slot.is_visible:
                    self._update_slot(slot)
                self._update_time_offset_text(slot)
        finally:
            self._syncing = False
        self._request_remote_draw()

    def _update_ref_time_display(self) -> None:
        if self._reference_timestamp is not None:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(self._reference_timestamp, tz=timezone.utc)
            self.panel["ref_time"].value = dt.strftime("%H:%M:%S.%f")[:-3]
        else:
            self.panel["ref_time"].value = ""

    def _update_time_offset_text(self, slot: WCISlot) -> None:
        if slot.time_offset_text is None:
            return
        if not self.time_sync_enabled:
            slot.time_offset_text.hide()
            return
        offset = slot.time_offset
        abs_offset = abs(offset)
        if abs_offset < 0.01:
            slot.time_offset_text.hide()
            return
        if abs_offset >= self.time_warning_threshold:
            text = f"\u0394{offset:+.1f}s"
            slot.time_offset_text.setColor((200, 60, 60, 220))
        else:
            text = f"\u0394{offset:+.2f}s"
            slot.time_offset_text.setColor((80, 80, 80, 180))
        slot.time_offset_text.setText(text)
        if slot.wci_extent is not None:
            x0, x1, y0, y1 = slot.wci_extent
            margin_x = (x1 - x0) * 0.02
            slot.time_offset_text.setPos(x1 - margin_x, min(y0, y1))
            slot.time_offset_text.setAnchor((1, 0))
        slot.time_offset_text.show()

    # =====================================================================
    # Image building / colour
    # =====================================================================

    def _sync_builder_args(self) -> None:
        self.args_imagebuilder["linear_mean"] = self.panel["stack_linear"].value
        self.args_imagebuilder["wci_value"] = self.panel["wci_value"].value
        self.args_imagebuilder["wci_render"] = self.panel["wci_render"].value
        self.args_imagebuilder["horizontal_pixels"] = self.panel["horizontal_pixels"].value
        self.args_imagebuilder["mp_cores"] = self.panel["mp_cores"].value
        self.args_imagebuilder["oversampling"] = self.panel["oversampling"].value
        self.args_imagebuilder["oversampling_mode"] = self.panel["oversampling_mode"].value
        for slot in self.slots:
            if slot.imagebuilder is not None:
                slot.imagebuilder.update_args(**self.args_imagebuilder)

    def _update_all_visible(self) -> None:
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
        if slot.imagebuilder is None or slot.plot_item is None:
            return
        if fast_mode:
            try:
                slot.imagebuilder.update_args(**self.args_imagebuilder)
                slot.wci_image, slot.wci_extent = slot.imagebuilder.build(
                    index=slot.ping_index,
                    stack=int(self.panel["stack"].value),
                    stack_step=int(self.panel["stack_step"].value),
                )
            except Exception:
                return
            self._update_slot_image(slot)
            return

        title = str(slot.channel_key) if slot.channel_key else f"Slot {slot.slot_idx + 1}"
        slot.plot_item.setTitle(title)
        try:
            self.progress.set_description(f"Building {slot.channel_key}...")
            slot.imagebuilder.update_args(**self.args_imagebuilder)
            slot.wci_image, slot.wci_extent = slot.imagebuilder.build(
                index=slot.ping_index,
                stack=int(self.panel["stack"].value),
                stack_step=int(self.panel["stack_step"].value),
            )
        except Exception as e:
            self.progress.set_description(f"Error: {e}")
            return
        self.progress.set_description("Idle")
        self._update_slot_image(slot)
        self._update_time_offset_text(slot)

    def _update_slot_image(self, slot: WCISlot) -> None:
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

        if hasattr(slot.image_item, "setColorMap"):
            slot.image_item.setColorMap(self._colormap)
        else:
            lut = self._colormap.getLookupTable(256)
            slot.image_item.setLookupTable(lut)

        if slot.color_levels is not None:
            vmin, vmax = slot.color_levels
        else:
            vmin = float(self.panel["vmin"].value)
            vmax = float(self.panel["vmax"].value)
        slot.image_item.setLevels((vmin, vmax))

        if self._first_draw or self.args_imagebuilder["hmin"] is None:
            vb.setXRange(x0, x1, padding=0)
        if self._first_draw or self.args_imagebuilder["vmin"] is None:
            vb.setYRange(min(y0, y1), max(y0, y1), padding=0)
        self._first_draw = False
        slot.image_item.show()

    def _on_colorbar_levels_changed(self, slot: WCISlot, colorbar: pg.ColorBarItem) -> None:
        levels = colorbar.levels()
        slot.color_levels = levels
        if slot.image_item is not None:
            slot.image_item.setLevels(levels)

    # =====================================================================
    # View range helpers
    # =====================================================================

    def fix_xy(self) -> None:
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

    def unfix_xy(self) -> None:
        for key in ("hmin", "hmax", "vmin", "vmax"):
            self.args_imagebuilder[key] = None
        self._update_all_visible()

    def _get_master_plot(self) -> Optional[pg.PlotItem]:
        for slot in self.slots:
            if slot.is_visible and slot.plot_item is not None:
                return slot.plot_item
        return None

    def _capture_current_view_range(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        master = self._get_master_plot()
        if master is None:
            return None
        return master.getViewBox().viewRange()

    def _restore_view_range(
        self,
        view_range: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> None:
        master = self._get_master_plot()
        if master is None:
            return
        vb = master.getViewBox()
        (xmin, xmax), (ymin, ymax) = view_range
        vb.setXRange(xmin, xmax, padding=0)
        vb.setYRange(ymin, ymax, padding=0)

    # =====================================================================
    # Playback
    # =====================================================================

    def step_prev(self) -> None:
        step = max(1, int(self.panel["ping_step"].value))
        pings = self.slots[0].get_pings()
        if not pings:
            return
        new_idx = max(0, self.panel["ping_slider_0"].value - step)
        self.panel["ping_slider_0"].value = new_idx

    def step_next(self) -> None:
        step = max(1, int(self.panel["ping_step"].value))
        pings = self.slots[0].get_pings()
        if not pings:
            return
        max_idx = len(pings) - 1
        new_idx = min(max_idx, self.panel["ping_slider_0"].value + step)
        self.panel["ping_slider_0"].value = new_idx

    def toggle_autoplay(self) -> None:
        if self._autoplay_active:
            self.stop_autoplay()
        else:
            self.start_autoplay()

    def start_autoplay(self) -> None:
        import asyncio

        self._autoplay_active = True
        self.panel["play_button"].description = "Stop"
        self._autoplay_last_time = None

        async def play_loop() -> None:
            while self._autoplay_active:
                t0 = time_module.time()

                use_ping_time = self.panel["use_ping_time"].value
                speed_mult = max(0.1, float(self.panel["play_fps"].value))
                step = max(1, int(self.panel["ping_step"].value))

                slot = self.slots[0]
                pings = slot.get_pings()
                interval = 1.0 / speed_mult

                if pings:
                    max_idx = len(pings) - 1
                    current_idx = self.panel["ping_slider_0"].value
                    new_idx = current_idx + step
                    if new_idx > max_idx:
                        new_idx = 0

                    if use_ping_time:
                        current_ts = slot.get_timestamp(current_idx)
                        next_ts = slot.get_timestamp(new_idx)
                        if current_ts is not None and next_ts is not None:
                            ping_dt = abs(next_ts - current_ts)
                            interval = ping_dt / speed_mult

                    self.panel["ping_slider_0"].value = new_idx

                elapsed = time_module.time() - t0
                remaining = interval - elapsed

                if self._autoplay_last_time is not None:
                    real_interval = t0 - self._autoplay_last_time
                    if real_interval > 0:
                        real_fps = 1.0 / real_interval
                        self.panel["real_fps"].value = f"real: {real_fps:.1f}"
                self._autoplay_last_time = t0

                await asyncio.sleep(max(0.005, remaining))

        try:
            loop = asyncio.get_running_loop()
            self._autoplay_task = loop.create_task(play_loop())
        except RuntimeError:
            self._autoplay_task = asyncio.ensure_future(play_loop())

    def stop_autoplay(self) -> None:
        self._autoplay_active = False
        self.panel["play_button"].description = "\u25b6 Play"
        self.panel["real_fps"].value = "real: --"
        if self._autoplay_task is not None:
            self._autoplay_task.cancel()
            self._autoplay_task = None

    # =====================================================================
    # Video export
    # =====================================================================

    def export_video(self) -> None:
        slot = self.slots[0]
        pings = slot.get_pings()
        if not pings:
            self.panel["video_status"].value = "Error: No pings available"
            return

        num_frames = int(self.panel["video_frames"].value)
        if num_frames <= 0:
            num_frames = len(pings)
        video_fps = max(1, float(self.panel["video_fps"].value))
        step = max(1, int(self.panel["ping_step"].value))
        fmt = self.panel["video_format"].value
        filename = (self.panel["video_filename"].value or "").strip() or "wci_video"
        use_ping_time = self.panel["video_ping_time"].value

        start_idx = self.panel["ping_slider_0"].value
        max_idx = len(pings) - 1
        actual_frames = min(num_frames, (max_idx - start_idx) // step + 1)
        self.panel["video_status"].value = f"Capturing {actual_frames} frames..."

        try:
            gfx_view = self._get_gfx_view()
            if not hasattr(gfx_view, "grab"):
                self.panel["video_status"].value = "Error: No graphics view available"
                return

            self.frames.clear()
            current_idx = start_idx

            show_live = self.panel["video_live"].value if "video_live" in self.panel else False
            old_syncing = self._syncing
            if not show_live:
                self._syncing = True

            t_start = time_module.time()
            for i in range(actual_frames):
                t0 = time_module.time()
                current_ts = slot.get_timestamp(current_idx)

                for slot_i, s in enumerate(self.slots):
                    if s.is_visible and s.get_pings():
                        s.ping_index = current_idx
                        self._update_slot(s, fast_mode=not show_live)
                        if f"ping_slider_{slot_i}" in self.panel:
                            self.panel[f"ping_slider_{slot_i}"].value = current_idx

                self._process_qt_events()

                try:
                    pixmap = gfx_view.grab()
                    image = pixmap.toImage()
                    w = image.width()
                    h = image.height()
                    ptr = image.bits()
                    nbytes = image.sizeInBytes() if hasattr(image, "sizeInBytes") else image.byteCount()
                    if hasattr(ptr, "tobytes"):
                        arr = np.frombuffer(ptr.tobytes(), dtype=np.uint8)
                    else:
                        ptr.setsize(nbytes)
                        arr = np.array(ptr, dtype=np.uint8)
                    arr = arr.reshape(h, w, 4)
                    frame = arr[:, :, [2, 1, 0]].copy()
                    self.frames.append(frame, timestamp=current_ts)
                except Exception as e:
                    self._syncing = old_syncing
                    self.panel["video_status"].value = f"Frame capture error: {e}"
                    return

                current_idx += step
                if current_idx > max_idx:
                    break

                t1 = time_module.time()
                fps_cur = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
                elapsed = t1 - t_start
                remaining = (actual_frames - i - 1) * (elapsed / (i + 1)) if i > 0 else 0
                self.panel["video_status"].value = (
                    f"Frame {i + 1}/{actual_frames} "
                    f"({fps_cur:.1f} fps, ~{remaining:.0f}s left)"
                )

            self._syncing = old_syncing

            if len(self.frames) == 0:
                self.panel["video_status"].value = "Error: No frames captured"
                return

            if fmt == "frames":
                self.panel["video_status"].value = (
                    f"Captured {len(self.frames)} frames "
                    f"(use core.frames.export_avif() / .export_mp4())"
                )
            elif fmt == "avif":
                if not filename.endswith(".avif"):
                    filename = f"{filename}.avif"
                self.panel["video_status"].value = f"Writing {filename}..."
                self._process_qt_events()
                try:
                    kw: Dict[str, Any] = {"quality": int(self.panel["video_quality"].value)}
                    if use_ping_time:
                        kw["ping_time_speed"] = video_fps
                    else:
                        kw["fps"] = video_fps
                    self.frames.export_avif(filename, **kw)
                    self.panel["video_status"].value = f"Saved: {filename} ({len(self.frames)} frames)"
                except Exception as e:
                    self.panel["video_status"].value = f"AVIF error: {e}"
                    return
            else:  # mp4
                if not filename.endswith(".mp4"):
                    filename = f"{filename}.mp4"
                self.panel["video_status"].value = f"Writing {filename}..."
                self._process_qt_events()
                try:
                    kw_mp4: Dict[str, Any] = {}
                    if use_ping_time:
                        kw_mp4["ping_time_speed"] = video_fps
                    else:
                        kw_mp4["fps"] = video_fps
                    self.frames.export_mp4(filename, **kw_mp4)
                    self.panel["video_status"].value = f"Saved: {filename} ({len(self.frames)} frames)"
                except Exception as e:
                    self.panel["video_status"].value = f"MP4 error: {e}"
                    return

            self.panel["ping_slider_0"].value = start_idx

        except Exception as e:
            self.panel["video_status"].value = f"Export error: {e}"

    # =====================================================================
    # Continuous capture
    # =====================================================================

    def toggle_continuous_capture(self) -> None:
        """Toggle infinite capture mode.

        While active, every ping change grabs a frame.  When toggled off,
        the captured frames are exported with the current video settings.
        """
        if self._continuous_capture_active:
            self._stop_and_save_continuous_capture()
        else:
            self._start_continuous_capture()

    def _start_continuous_capture(self) -> None:
        self._continuous_capture_active = True
        self.frames.clear()
        self.panel["continuous_capture"].description = "Stop Capture"
        self.panel["video_status"].value = "Continuous capture started..."
        # capture first frame immediately
        self._capture_current_frame()

    def _capture_current_frame(self) -> None:
        """Grab a single frame from the graphics view and append to frames."""
        if not self._continuous_capture_active:
            return
        gfx_view = self._get_gfx_view()
        if gfx_view is None or not hasattr(gfx_view, "grab"):
            return
        try:
            slot = self.slots[0]
            ts = slot.get_timestamp()
            pixmap = gfx_view.grab()
            image = pixmap.toImage()
            w = image.width()
            h = image.height()
            ptr = image.bits()
            nbytes = image.sizeInBytes() if hasattr(image, "sizeInBytes") else image.byteCount()
            if hasattr(ptr, "tobytes"):
                arr = np.frombuffer(ptr.tobytes(), dtype=np.uint8)
            else:
                ptr.setsize(nbytes)
                arr = np.array(ptr, dtype=np.uint8)
            arr = arr.reshape(h, w, 4)
            frame = arr[:, :, [2, 1, 0]].copy()
            self.frames.append(frame, timestamp=ts)
            self.panel["video_status"].value = f"Capturing... {len(self.frames)} frames"
        except Exception as e:
            self.panel["video_status"].value = f"Capture error: {e}"

    def _stop_and_save_continuous_capture(self) -> None:
        self._continuous_capture_active = False
        self.panel["continuous_capture"].description = "Start Capture"

        if len(self.frames) == 0:
            self.panel["video_status"].value = "No frames captured"
            return

        video_fps = max(1, float(self.panel["video_fps"].value))
        fmt = self.panel["video_format"].value
        filename = (self.panel["video_filename"].value or "").strip() or "wci_video"
        use_ping_time = self.panel["video_ping_time"].value

        self.panel["video_status"].value = f"Saving {len(self.frames)} frames..."
        self._process_qt_events()

        try:
            if fmt == "frames":
                self.panel["video_status"].value = (
                    f"Captured {len(self.frames)} frames "
                    f"(use core.frames.export_avif() / .export_mp4())"
                )
            elif fmt == "avif":
                if not filename.endswith(".avif"):
                    filename = f"{filename}.avif"
                kw: Dict[str, Any] = {"quality": int(self.panel["video_quality"].value)}
                if use_ping_time:
                    kw["ping_time_speed"] = video_fps
                else:
                    kw["fps"] = video_fps
                self.frames.export_avif(filename, **kw)
                self.panel["video_status"].value = f"Saved: {filename} ({len(self.frames)} frames)"
            else:  # mp4
                if not filename.endswith(".mp4"):
                    filename = f"{filename}.mp4"
                kw_mp4: Dict[str, Any] = {}
                if use_ping_time:
                    kw_mp4["ping_time_speed"] = video_fps
                else:
                    kw_mp4["fps"] = video_fps
                self.frames.export_mp4(filename, **kw_mp4)
                self.panel["video_status"].value = f"Saved: {filename} ({len(self.frames)} frames)"
        except Exception as e:
            self.panel["video_status"].value = f"Save error: {e}"

    # =====================================================================
    # Quick-select single channel
    # =====================================================================

    def show_single(self, channel_name: str) -> None:
        current_range = self._capture_current_view_range()
        self._ignore_range_changes = True
        try:
            need_grid_change = (self.grid_rows, self.grid_cols) != (1, 1)
            if need_grid_change:
                self.panel["layout"].value = (1, 1)

            if self.slots[0].channel_key != channel_name:
                self.slots[0].assign_channel(channel_name)
                self.panel["slot_selector_0"].value = channel_name
                pings = self.slots[0].get_pings()
                max_ping = max(0, len(pings) - 1) if pings else 0
                self.panel["ping_slider_0"].max = max_ping

            if not need_grid_change:
                self._update_slot(self.slots[0])
        finally:
            self._ignore_range_changes = False

        if current_range is not None:
            self._restore_view_range(current_range)
        self._request_remote_draw()

    # =====================================================================
    # Mouse / crosshair
    # =====================================================================

    def handle_scene_move(self, pos: QtCore.QPointF) -> None:
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
                self.panel["hover_label"].value = label
                break

        if self._crosshair_enabled and found_slot is not None and data_pos is not None:
            self._crosshair_position = data_pos
            self._update_crosshairs()
        else:
            self._hide_crosshairs()

        if found_slot is None:
            self.panel["hover_label"].value = "&nbsp;"

    def _update_crosshairs(self) -> None:
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
        for slot in self.slots:
            if slot.crosshair_v is not None:
                slot.crosshair_v.hide()
            if slot.crosshair_h is not None:
                slot.crosshair_h.hide()

    def _sample_value(self, slot: WCISlot, x: float, y: float) -> Optional[float]:
        if slot.wci_image is None or slot.wci_extent is None:
            return None
        x0, x1, y0, y1 = slot.wci_extent
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0 or dy == 0:
            return None
        col = (x - x0) / dx * (slot.wci_image.shape[0] - 1)
        row = (y - y0) / dy * (slot.wci_image.shape[1] - 1)
        if 0 <= col < slot.wci_image.shape[0] and 0 <= row < slot.wci_image.shape[1]:
            return float(slot.wci_image[int(col), int(row)])
        return None

    # =====================================================================
    # Timing display
    # =====================================================================

    def _update_timing_fields(self, t0: float, t1: float, t2: float) -> None:
        build = t1 - t0
        draw = t2 - t1
        total = t2 - t0
        self.panel["proctime"].value = f"{build:0.3f} / {draw:0.3f} / [{total:0.3f}] s"
        r1 = 1.0 / build if build > 0 else 0
        r2 = 1.0 / draw if draw > 0 else 0
        r3 = 1.0 / total if total > 0 else 0
        self.panel["procrate"].value = f"r1: {r1:0.1f} / r2: {r2:0.1f} / r3: [{r3:0.1f}] Hz"

    # =====================================================================
    # Qt helpers
    # =====================================================================

    def _get_gfx_view(self):
        """Return the underlying QGraphicsView (works for both native and jupyter_rfb)."""
        return getattr(self.graphics, "gfxView", self.graphics)

    def _connect_scene_events(self) -> None:
        gfx_view = self._get_gfx_view()
        scene = gfx_view.scene() if gfx_view is not None else None
        if scene is None:
            return
        if hasattr(self, "_scene_move_connection") and self._scene_move_connection:
            try:
                scene.sigMouseMoved.disconnect(self.handle_scene_move)
            except (TypeError, RuntimeError):
                pass
        self._scene_move_connection = scene.sigMouseMoved.connect(self.handle_scene_move)

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
    # Ping change callbacks (for connected viewers)
    # =====================================================================

    def register_ping_change_callback(self, callback: Any) -> None:
        if callback not in self._ping_change_callbacks:
            self._ping_change_callbacks.append(callback)

    def unregister_ping_change_callback(self, callback: Any) -> None:
        if callback in self._ping_change_callbacks:
            self._ping_change_callbacks.remove(callback)

    # =====================================================================
    # Compatibility properties
    # =====================================================================

    @property
    def w_index(self):
        """Compatibility: ping slider handle for first visible slot."""
        return self.panel["ping_slider_0"]

    @property
    def imagebuilder(self) -> Optional[mi.ImageBuilder]:
        for slot in self.slots:
            if slot.is_visible and slot.imagebuilder is not None:
                return slot.imagebuilder
        return None

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
        if hasattr(ptr, "setsize"):
            ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()
        arr = arr[..., [2, 1, 0, 3]]
        fig, ax = plt.subplots(dpi=dpi)
        ax.imshow(arr)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        return fig
