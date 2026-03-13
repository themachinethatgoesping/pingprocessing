"""Declarative control specifications for WCI viewer widgets.

Defines dataclasses that describe UI controls (sliders, dropdowns, etc.)
without binding to any specific toolkit.  Factory modules
(``control_jupyter``, ``control_qt``) read these specs and create concrete
widgets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FloatSliderSpec:
    name: str
    description: str
    min: float = 0.0
    max: float = 100.0
    step: float = 1.0
    value: float = 0.0
    width: str = "220px"


@dataclass
class IntSliderSpec:
    name: str
    description: str
    min: int = 0
    max: int = 100
    step: int = 1
    value: int = 0
    width: str = "200px"


@dataclass
class DropdownSpec:
    name: str
    description: str
    options: list = field(default_factory=list)
    value: Any = None
    width: str = "180px"


@dataclass
class CheckboxSpec:
    name: str
    description: str
    value: bool = False
    tooltip: str = ""


@dataclass
class IntTextSpec:
    name: str
    description: str
    value: int = 0
    width: str = "140px"


@dataclass
class FloatTextSpec:
    name: str
    description: str
    value: float = 0.0
    width: str = "140px"


@dataclass
class ButtonSpec:
    name: str
    description: str
    tooltip: str = ""
    width: str = "80px"


@dataclass
class LabelSpec:
    name: str
    value: str = ""
    width: str = "100px"


@dataclass
class TextSpec:
    name: str
    description: str
    value: str = ""
    disabled: bool = False
    width: str = "200px"


@dataclass
class HTMLSpec:
    name: str
    value: str = "&nbsp;"


ControlSpecType = Union[
    FloatSliderSpec, IntSliderSpec, DropdownSpec, CheckboxSpec,
    IntTextSpec, FloatTextSpec, ButtonSpec, LabelSpec, TextSpec, HTMLSpec,
]


# ---------------------------------------------------------------------------
# ControlHandle – unified interface to a single control
# ---------------------------------------------------------------------------

class ControlHandle:
    """Base class for a UI-agnostic control handle."""

    @property
    def value(self) -> Any:
        raise NotImplementedError

    @value.setter
    def value(self, v: Any) -> None:
        raise NotImplementedError

    def on_change(self, callback: Callable[[Any], None]) -> None:
        """Register a callback for value changes.  *callback* receives the
        new value."""
        raise NotImplementedError

    def on_click(self, callback: Callable) -> None:
        """Register a click handler (buttons only)."""
        raise NotImplementedError

    @property
    def visible(self) -> bool:
        return True

    @visible.setter
    def visible(self, v: bool) -> None:
        pass

    @property
    def max(self) -> Any:
        raise NotImplementedError

    @max.setter
    def max(self, v: Any) -> None:
        raise NotImplementedError

    @property
    def step(self) -> Any:
        raise NotImplementedError

    @step.setter
    def step(self, v: Any) -> None:
        raise NotImplementedError

    @property
    def description(self) -> str:
        return ""

    @description.setter
    def description(self, v: str) -> None:
        pass

    @property
    def options(self) -> Any:
        raise NotImplementedError

    @options.setter
    def options(self, v: Any) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ControlPanel – dict-like container of named handles
# ---------------------------------------------------------------------------

class ControlPanel:
    """Container mapping control names to :class:`ControlHandle` objects."""

    def __init__(self) -> None:
        self._controls: Dict[str, ControlHandle] = {}

    def __getitem__(self, name: str) -> ControlHandle:
        return self._controls[name]

    def __setitem__(self, name: str, handle: ControlHandle) -> None:
        self._controls[name] = handle

    def __contains__(self, name: str) -> bool:
        return name in self._controls

    def keys(self):
        return self._controls.keys()


# ---------------------------------------------------------------------------
# WCI control definitions (shared across backends)
# ---------------------------------------------------------------------------

WCI_VALUE_CHOICES = [
    "sv/av/pv/rv", "sv/av/pv", "sv/av",
    "sp/ap/pp/rp", "sp/ap/pp", "sp/ap",
    "power/amp", "av", "ap", "amp", "sv", "sp",
    "pv", "pp", "rv", "rp", "power",
    "sv_vs_av", "sp_vs_ap",
]

GRID_LAYOUTS: List[Tuple[int, int, str]] = [
    (1, 1, "1"),
    (1, 2, "1×2"),
    (2, 1, "2×1"),
    (2, 2, "2×2"),
    (3, 2, "3×2"),
    (4, 2, "4×2"),
]

# -- Tab: Render --
WCI_RENDER_SPECS: List[ControlSpecType] = [
    FloatSliderSpec("vmin", "vmin", min=-150, max=100, step=0.5, value=-90, width="220px"),
    FloatSliderSpec("vmax", "vmax", min=-150, max=100, step=0.5, value=-25, width="220px"),
    DropdownSpec("wci_value", "value", options=WCI_VALUE_CHOICES, value="sv/av/pv/rv", width="180px"),
    DropdownSpec("wci_render", "render", options=["linear", "beamsample"], value="linear", width="150px"),
    IntSliderSpec("horizontal_pixels", "h_pixels", min=2, max=2048, step=1, value=1024, width="200px"),
    DropdownSpec("oversampling", "oversample", options=[1, 2, 3, 4], value=1, width="140px"),
    DropdownSpec("oversampling_mode", "avg", options=["linear_mean", "db_mean"], value="linear_mean", width="170px"),
    CheckboxSpec("time_sync", "Sync time", value=True),
    CheckboxSpec("crosshair", "Crosshair", value=True),
    FloatTextSpec("time_warning", "Warn \u0394t (s):", value=5.0, width="130px"),
]

# -- Tab: Stack --
WCI_STACK_SPECS: List[ControlSpecType] = [
    IntTextSpec("stack", "stack", value=1, width="140px"),
    IntTextSpec("stack_step", "step", value=1, width="140px"),
    IntTextSpec("mp_cores", "cores", value=1, width="140px"),
    CheckboxSpec("stack_linear", "linear stack", value=True),
]

# -- Tab: Timing --
WCI_TIMING_SPECS: List[ControlSpecType] = [
    TextSpec("proctime", "time", disabled=True, width="280px"),
    TextSpec("procrate", "rate", disabled=True, width="280px"),
]

# -- Tab: Playback --
WCI_PLAYBACK_SPECS: List[ControlSpecType] = [
    IntTextSpec("ping_step", "ping step", value=1, width="140px"),
    ButtonSpec("step_prev", "\u25c0 Prev", width="80px"),
    ButtonSpec("step_next", "Next \u25b6", width="80px"),
    ButtonSpec("play_button", "\u25b6 Play", width="80px"),
    FloatTextSpec("play_fps", "fps", value=2.0, width="160px"),
    CheckboxSpec("use_ping_time", "ping time", value=False,
                 tooltip="Use actual ping timestamps for timing (fps becomes speed multiplier)"),
    LabelSpec("real_fps", "real: --", width="100px"),
]

# -- Tab: Video --
WCI_VIDEO_SPECS: List[ControlSpecType] = [
    IntTextSpec("video_frames", "frames", value=100, width="140px"),
    FloatTextSpec("video_fps", "video fps", value=10.0, width="140px"),
    DropdownSpec("video_format", "format", options=["avif", "mp4", "frames"], value="avif", width="140px"),
    IntSliderSpec("video_quality", "quality", min=1, max=100, step=1, value=75, width="200px"),
    TextSpec("video_filename", "filename", value="wci_video", width="200px"),
    ButtonSpec("export_video", "Capture", tooltip="Capture frames (and optionally export)", width="120px"),
    ButtonSpec("continuous_capture", "Start Capture", tooltip="Capture continuously until pressed again", width="130px"),
    LabelSpec("video_status", "", width="300px"),
    CheckboxSpec("video_ping_time", "ping time", value=False,
                 tooltip="Use ping timestamps for video timing"),
    CheckboxSpec("video_live", "live", value=True,
                 tooltip="Show live preview during capture"),
]

# -- Non-tabbed controls --
WCI_MISC_SPECS: List[ControlSpecType] = [
    TextSpec("ref_time", "Ref time:", disabled=True, width="220px"),
    ButtonSpec("fix_xy", "Fix view", width="80px"),
    ButtonSpec("unfix_xy", "Unfix", width="70px"),
    HTMLSpec("hover_label", "&nbsp;"),
]

# Spec groups keyed by tab name
WCI_TABS: Dict[str, List[ControlSpecType]] = {
    "Render": WCI_RENDER_SPECS,
    "Stack": WCI_STACK_SPECS,
    "Playback": WCI_PLAYBACK_SPECS,
    "Video": WCI_VIDEO_SPECS,
}

# Tab layout: maps tab names to rows of control names.
# Shared between Jupyter and Qt adapters.
WCI_TAB_LAYOUT: Dict[str, List[List[str]]] = {
    "Render": [
        ["vmin", "vmax"],
        ["wci_value", "wci_render"],
        ["horizontal_pixels", "oversampling", "oversampling_mode"],
        ["time_sync", "crosshair", "time_warning"],
    ],
    "Stack": [
        ["stack", "stack_step", "mp_cores", "stack_linear"],
    ],
    "Playback": [
        ["ping_step", "step_prev", "step_next"],
        ["play_button", "play_fps", "use_ping_time", "real_fps"],
    ],
    "Video": [
        ["video_frames", "video_fps", "video_format", "video_quality"],
        ["video_filename", "video_ping_time", "video_live", "export_video"],
        ["continuous_capture", "video_status"],
    ],
}
