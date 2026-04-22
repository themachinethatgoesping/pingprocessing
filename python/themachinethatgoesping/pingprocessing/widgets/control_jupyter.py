"""ipywidgets backend for the declarative control specifications.

Creates ipywidgets from :mod:`.control_spec` dataclasses and wraps
them in :class:`ControlHandle` / :class:`ControlPanel` objects so the
WCI core can read / write controls without knowing about ipywidgets.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import ipywidgets

from .control_spec import (
    ButtonSpec,
    CheckboxSpec,
    ControlHandle,
    ControlPanel,
    ControlSpecType,
    DropdownSpec,
    FloatSliderSpec,
    FloatTextSpec,
    HTMLSpec,
    IntSliderSpec,
    IntTextSpec,
    LabelSpec,
    MultiSelectSpec,
    TextSpec,
)


# ---------------------------------------------------------------------------
# JupyterControlHandle
# ---------------------------------------------------------------------------

class JupyterControlHandle(ControlHandle):
    """Wraps a single ipywidgets widget."""

    def __init__(self, widget: ipywidgets.Widget) -> None:
        self._widget = widget

    # -- value --
    @property
    def value(self) -> Any:
        return self._widget.value

    @value.setter
    def value(self, v: Any) -> None:
        self._widget.value = v

    # -- callbacks --
    def on_change(self, callback: Callable[[Any], None]) -> None:
        self._widget.observe(lambda change: callback(change["new"]), names="value")

    def on_click(self, callback: Callable) -> None:
        if isinstance(self._widget, ipywidgets.Button):
            self._widget.on_click(callback)

    # -- visibility --
    @property
    def visible(self) -> bool:
        return self._widget.layout.display != "none"

    @visible.setter
    def visible(self, v: bool) -> None:
        self._widget.layout.display = None if v else "none"

    # -- slider / dropdown helpers --
    @property
    def max(self) -> Any:
        return getattr(self._widget, "max", None)

    @max.setter
    def max(self, v: Any) -> None:
        if hasattr(self._widget, "max"):
            self._widget.max = v

    @property
    def step(self) -> Any:
        return getattr(self._widget, "step", None)

    @step.setter
    def step(self, v: Any) -> None:
        if hasattr(self._widget, "step"):
            self._widget.step = v

    @property
    def description(self) -> str:
        return getattr(self._widget, "description", "")

    @description.setter
    def description(self, v: str) -> None:
        if hasattr(self._widget, "description"):
            self._widget.description = v

    @property
    def disabled(self) -> bool:
        return getattr(self._widget, "disabled", False)

    @disabled.setter
    def disabled(self, v: bool) -> None:
        if hasattr(self._widget, "disabled"):
            self._widget.disabled = v

    @property
    def options(self) -> Any:
        return getattr(self._widget, "options", None)

    @options.setter
    def options(self, v: Any) -> None:
        if hasattr(self._widget, "options"):
            self._widget.options = v

    @property
    def widget(self) -> ipywidgets.Widget:
        """The underlying ipywidgets widget (useful for layout assembly)."""
        return self._widget


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_jupyter_control(spec: ControlSpecType) -> JupyterControlHandle:
    """Create a :class:`JupyterControlHandle` from a specification."""
    if isinstance(spec, FloatSliderSpec):
        w = ipywidgets.FloatSlider(
            description=spec.description, min=spec.min, max=spec.max,
            step=spec.step, value=spec.value,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, IntSliderSpec):
        w = ipywidgets.IntSlider(
            description=spec.description, min=spec.min, max=spec.max,
            step=spec.step, value=spec.value,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, DropdownSpec):
        w = ipywidgets.Dropdown(
            description=spec.description, options=spec.options,
            value=spec.value,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, MultiSelectSpec):
        w = ipywidgets.SelectMultiple(
            description=spec.description, options=spec.options,
            value=tuple(spec.value),
            rows=spec.rows,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, CheckboxSpec):
        w = ipywidgets.Checkbox(
            description=spec.description, value=spec.value,
            tooltip=spec.tooltip, indent=False,
        )
    elif isinstance(spec, IntTextSpec):
        w = ipywidgets.IntText(
            description=spec.description, value=spec.value,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, FloatTextSpec):
        w = ipywidgets.FloatText(
            description=spec.description, value=spec.value,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, ButtonSpec):
        w = ipywidgets.Button(
            description=spec.description, tooltip=spec.tooltip,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, LabelSpec):
        w = ipywidgets.Label(
            value=spec.value,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, TextSpec):
        w = ipywidgets.Text(
            description=spec.description, value=spec.value,
            disabled=spec.disabled,
            layout=ipywidgets.Layout(width=spec.width),
        )
    elif isinstance(spec, HTMLSpec):
        w = ipywidgets.HTML(value=spec.value)
    else:
        raise ValueError(f"Unknown spec type: {type(spec)}")

    return JupyterControlHandle(w)


# ---------------------------------------------------------------------------
# JupyterControlPanel
# ---------------------------------------------------------------------------

class JupyterControlPanel(ControlPanel):
    """A :class:`ControlPanel` backed by ipywidgets.

    After construction every registered control can be accessed by name
    through ``panel["name"]``.  The underlying ipywidgets widget for
    layout assembly is available via ``panel["name"].widget``.
    """

    @classmethod
    def from_specs(cls, *spec_lists: List[ControlSpecType]) -> "JupyterControlPanel":
        """Build a panel from one or more flat lists of specs."""
        panel = cls()
        for spec_list in spec_lists:
            for spec in spec_list:
                panel[spec.name] = create_jupyter_control(spec)
        return panel

    # -- helpers for ipywidgets layout assembly --

    def widget(self, name: str) -> ipywidgets.Widget:
        """Short-hand: return the underlying ipywidgets.Widget by control name."""
        return self[name].widget

    def widgets(self, *names: str) -> List[ipywidgets.Widget]:
        """Return a list of underlying ipywidgets.Widgets."""
        return [self[n].widget for n in names]

    def hbox(self, *names: str) -> ipywidgets.HBox:
        """Return an HBox containing the named widgets."""
        return ipywidgets.HBox(self.widgets(*names))

    def vbox(self, *names: str) -> ipywidgets.VBox:
        """Return a VBox containing the named widgets."""
        return ipywidgets.VBox(self.widgets(*names))

    def build_tabs(
        self,
        tab_layouts: Dict[str, List[List[str]]],
    ) -> ipywidgets.Tab:
        """Build an ``ipywidgets.Tab`` widget.

        Parameters
        ----------
        tab_layouts
            ``{tab_name: [[row0_ctrl_names], [row1_ctrl_names], ...]}``
        """
        children = []
        tab_names = []
        for tab_name, rows in tab_layouts.items():
            row_boxes = []
            for row_names in rows:
                ws = [self[n].widget for n in row_names if n in self]
                if ws:
                    row_boxes.append(ipywidgets.HBox(ws))
            children.append(ipywidgets.VBox(row_boxes))
            tab_names.append(tab_name)

        tab = ipywidgets.Tab(children=children)
        for i, name in enumerate(tab_names):
            tab.set_title(i, name)
        return tab
