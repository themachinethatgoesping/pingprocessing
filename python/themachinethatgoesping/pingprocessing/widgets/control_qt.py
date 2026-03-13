"""Qt (native) backend for the declarative control specifications.

Creates Qt widgets from :mod:`.control_spec` dataclasses and wraps
them in :class:`ControlHandle` / :class:`ControlPanel` objects so
:class:`wciviewer_core.WCICore` can be driven from a native Qt application.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from pyqtgraph.Qt import QtCore, QtWidgets

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
    TextSpec,
)


# ---------------------------------------------------------------------------
# QtControlHandle
# ---------------------------------------------------------------------------

class QtControlHandle(ControlHandle):
    """Wraps a single Qt widget and exposes the unified ControlHandle API.

    Parameters
    ----------
    widget : QWidget
        The outermost widget (may be a labelled container).
    value_getter, value_setter
        Callables to read/write the control value.
    change_signal
        Qt signal emitted when the value changes.
    inner : QWidget, optional
        The actual editable widget inside *widget* (e.g. the QSlider or
        QSpinBox inside a labelled container).  If *None*, *widget* is
        treated as both outer and inner widget.
    """

    def __init__(self, widget: QtWidgets.QWidget, value_getter, value_setter,
                 change_signal=None, *, inner: QtWidgets.QWidget | None = None) -> None:
        self._widget = widget
        self._inner = inner or widget
        self._getter = value_getter
        self._setter = value_setter
        self._change_signal = change_signal

    @property
    def value(self) -> Any:
        return self._getter()

    @value.setter
    def value(self, v: Any) -> None:
        self._setter(v)

    def on_change(self, callback: Callable[[Any], None]) -> None:
        if self._change_signal is not None:
            self._change_signal.connect(lambda *a: callback(self.value))

    def on_click(self, callback: Callable) -> None:
        btn = self._inner
        if isinstance(btn, QtWidgets.QPushButton):
            btn.clicked.connect(lambda: callback(None))

    @property
    def visible(self) -> bool:
        return self._widget.isVisible()

    @visible.setter
    def visible(self, v: bool) -> None:
        self._widget.setVisible(v)

    @property
    def max(self) -> Any:
        w = self._inner
        if isinstance(w, (QtWidgets.QSlider, QtWidgets.QSpinBox,
                          QtWidgets.QDoubleSpinBox)):
            return w.maximum()
        return None

    @max.setter
    def max(self, v: Any) -> None:
        w = self._inner
        if isinstance(w, (QtWidgets.QSlider, QtWidgets.QSpinBox,
                          QtWidgets.QDoubleSpinBox)):
            w.setMaximum(int(v))
        # also sync a paired spin box if present
        paired = getattr(self, "_paired_spin", None)
        if paired is not None:
            paired.setMaximum(int(v))

    @property
    def step(self) -> Any:
        w = self._inner
        if isinstance(w, (QtWidgets.QSlider, QtWidgets.QSpinBox,
                          QtWidgets.QDoubleSpinBox)):
            return w.singleStep()
        return None

    @step.setter
    def step(self, v: Any) -> None:
        w = self._inner
        if isinstance(w, (QtWidgets.QSlider, QtWidgets.QSpinBox,
                          QtWidgets.QDoubleSpinBox)):
            w.setSingleStep(int(v))
        paired = getattr(self, "_paired_spin", None)
        if paired is not None:
            paired.setSingleStep(int(v))

    @property
    def description(self) -> str:
        w = self._inner
        if isinstance(w, QtWidgets.QPushButton):
            return w.text()
        return ""

    @description.setter
    def description(self, v: str) -> None:
        w = self._inner
        if isinstance(w, QtWidgets.QPushButton):
            w.setText(v)

    @property
    def options(self) -> Any:
        w = self._inner
        if isinstance(w, QtWidgets.QComboBox):
            return [w.itemData(i) for i in range(w.count())]
        return None

    @options.setter
    def options(self, v: Any) -> None:
        w = self._inner
        if isinstance(w, QtWidgets.QComboBox):
            w.clear()
            for item in v:
                if isinstance(item, tuple) and len(item) == 2:
                    w.addItem(str(item[0]), item[1])
                else:
                    w.addItem(str(item), item)

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._widget


# ---------------------------------------------------------------------------
# Slider stylesheet
# ---------------------------------------------------------------------------

_SLIDER_STYLE = """
QSlider::groove:horizontal {
    border: 1px solid #bbb;
    background: #e0e0e0;
    height: 8px;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #5b9bd5;
    border: 1px solid #4a8bc2;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #4a8bc2;
}
QSlider::sub-page:horizontal {
    background: #5b9bd5;
    border-radius: 4px;
}
"""


# ---------------------------------------------------------------------------
# Labelled widget helper
# ---------------------------------------------------------------------------

def _labelled(widget: QtWidgets.QWidget, description: str) -> QtWidgets.QWidget:
    """Wrap *widget* in an HBox with a label if *description* is non-empty."""
    if not description:
        return widget
    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    lbl = QtWidgets.QLabel(description)
    layout.addWidget(lbl)
    layout.addWidget(widget)
    return container


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_qt_control(spec: ControlSpecType) -> QtControlHandle:
    """Create a :class:`QtControlHandle` from a specification."""

    if isinstance(spec, FloatSliderSpec):
        # Float slider: QDoubleSpinBox with a label.
        # (QSlider is int-only, so QDoubleSpinBox is the best we can do for
        # continuous float values.)
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(spec.min, spec.max)
        w.setSingleStep(spec.step)
        w.setValue(spec.value)
        w.setKeyboardTracking(False)
        container = _labelled(w, spec.description)
        return QtControlHandle(
            container, w.value, w.setValue, w.editingFinished, inner=w,
        )

    if isinstance(spec, IntSliderSpec):
        # Real QSlider (horizontal) paired with a QSpinBox for readout.
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(spec.min, spec.max)
        slider.setSingleStep(spec.step)
        slider.setValue(spec.value)
        slider.setMinimumHeight(24)
        slider.setStyleSheet(_SLIDER_STYLE)

        spin = QtWidgets.QSpinBox()
        spin.setRange(spec.min, spec.max)
        spin.setSingleStep(spec.step)
        spin.setValue(spec.value)
        spin.setFixedWidth(70)

        # keep both in sync
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)

        container = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(container)
        hl.setContentsMargins(0, 0, 0, 0)
        if spec.description:
            hl.addWidget(QtWidgets.QLabel(spec.description))
        hl.addWidget(slider, 1)
        hl.addWidget(spin)

        handle = QtControlHandle(
            container, slider.value, slider.setValue, slider.valueChanged,
            inner=slider,
        )
        handle._paired_spin = spin
        return handle

    if isinstance(spec, DropdownSpec):
        w = QtWidgets.QComboBox()
        for opt in spec.options:
            if isinstance(opt, tuple) and len(opt) == 2:
                w.addItem(str(opt[0]), opt[1])
            else:
                w.addItem(str(opt), opt)
        if spec.value is not None:
            idx = w.findData(spec.value)
            if idx >= 0:
                w.setCurrentIndex(idx)
        container = _labelled(w, spec.description)
        return QtControlHandle(
            container,
            lambda: w.currentData(),
            lambda v: w.setCurrentIndex(w.findData(v)),
            w.currentIndexChanged,
            inner=w,
        )

    if isinstance(spec, CheckboxSpec):
        w = QtWidgets.QCheckBox(spec.description)
        w.setChecked(spec.value)
        if spec.tooltip:
            w.setToolTip(spec.tooltip)
        return QtControlHandle(
            w, w.isChecked, w.setChecked, w.stateChanged, inner=w,
        )

    if isinstance(spec, IntTextSpec):
        w = QtWidgets.QSpinBox()
        w.setRange(-999999, 999999)
        w.setValue(spec.value)
        w.setKeyboardTracking(False)
        container = _labelled(w, spec.description)
        return QtControlHandle(
            container, w.value, w.setValue, w.editingFinished, inner=w,
        )

    if isinstance(spec, FloatTextSpec):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(-1e9, 1e9)
        w.setDecimals(3)
        w.setValue(spec.value)
        w.setKeyboardTracking(False)
        container = _labelled(w, spec.description)
        return QtControlHandle(
            container, w.value, w.setValue, w.editingFinished, inner=w,
        )

    if isinstance(spec, ButtonSpec):
        w = QtWidgets.QPushButton(spec.description)
        if spec.tooltip:
            w.setToolTip(spec.tooltip)
        return QtControlHandle(
            w, lambda: None, lambda v: None, None, inner=w,
        )

    if isinstance(spec, LabelSpec):
        w = QtWidgets.QLabel(spec.value)
        return QtControlHandle(
            w, w.text, w.setText, None, inner=w,
        )

    if isinstance(spec, TextSpec):
        w = QtWidgets.QLineEdit(spec.value)
        if spec.disabled:
            w.setReadOnly(True)
        container = _labelled(w, spec.description)
        return QtControlHandle(
            container, w.text, w.setText, w.textChanged, inner=w,
        )

    if isinstance(spec, HTMLSpec):
        w = QtWidgets.QLabel(spec.value)
        w.setTextFormat(QtCore.Qt.TextFormat.RichText)
        return QtControlHandle(w, w.text, w.setText, None, inner=w)

    raise ValueError(f"Unknown spec type: {type(spec)}")


# ---------------------------------------------------------------------------
# QtControlPanel
# ---------------------------------------------------------------------------

class QtControlPanel(ControlPanel):
    """A :class:`ControlPanel` backed by native Qt widgets."""

    @classmethod
    def from_specs(cls, *spec_lists: List[ControlSpecType]) -> "QtControlPanel":
        panel = cls()
        for spec_list in spec_lists:
            for spec in spec_list:
                panel[spec.name] = create_qt_control(spec)
        return panel

    def widget(self, name: str) -> QtWidgets.QWidget:
        return self[name].widget

    def widgets(self, *names: str) -> List[QtWidgets.QWidget]:
        return [self[n].widget for n in names]

    def hbox_widget(self, *names: str) -> QtWidgets.QWidget:
        """Return a QWidget with an HBoxLayout containing the named widgets."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for n in names:
            if n in self:
                layout.addWidget(self[n].widget)
        return container

    def build_tab_widget(
        self,
        tab_layouts: Dict[str, List[List[str]]],
    ) -> QtWidgets.QTabWidget:
        """Build a ``QTabWidget`` with the given layout."""
        tab_widget = QtWidgets.QTabWidget()
        for tab_name, rows in tab_layouts.items():
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            for row_names in rows:
                row = QtWidgets.QWidget()
                rl = QtWidgets.QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                for name in row_names:
                    if name in self:
                        rl.addWidget(self[name].widget)
                page_layout.addWidget(row)
            tab_widget.addTab(page, tab_name)
        return tab_widget
