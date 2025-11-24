"""Utility helpers shared by PyQtGraph-based widget viewers."""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional, Union

import ipywidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

__all__ = [
    "MatplotlibDateAxis",
    "ensure_qapp",
    "resolve_colormap",
    "apply_widget_layout",
]


class MatplotlibDateAxis(pg.AxisItem):
    """AxisItem that formats matplotlib-style ordinal dates."""

    def __init__(self, converter: Callable[[float], datetime], orientation: str = "bottom") -> None:
        super().__init__(orientation=orientation)
        self._converter = converter

    def tickStrings(self, values: List[float], scale: float, spacing: float) -> List[str]:  # noqa: N802
        labels: List[str] = []
        for value in values:
            try:
                dt_value = self._converter(float(value))
                labels.append(dt_value.strftime("%Y-%m-%d\n%H:%M:%S"))
            except Exception:  # pragma: no cover - formatting failure should not crash UI
                labels.append("")
        return labels


def ensure_qapp() -> None:
    """Ensure a QApplication exists for PyQtGraph widgets."""

    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication([])


def resolve_colormap(cmap: Union[str, pg.ColorMap]) -> pg.ColorMap:
    """Return a PyQtGraph ColorMap from either a name or a ColorMap instance."""

    if isinstance(cmap, pg.ColorMap):
        return cmap
    if isinstance(cmap, str):
        try:
            return pg.colormap.get(cmap)
        except Exception:
            fallback = _matplotlib_colormap(cmap)
            if fallback is not None:
                return fallback
    return pg.colormap.get("viridis")


def apply_widget_layout(widget: ipywidgets.Widget, width_px: int, height_px: int) -> None:
    """Attach a resizable layout to the GraphicsLayoutWidget wrapper."""

    width = f"{width_px}px"
    height = f"{height_px}px"
    layout = getattr(widget, "layout", None)
    if layout is None:
        layout = ipywidgets.Layout(
            width=width,
            height=height,
            min_height="0px",
            resize="vertical",
            overflow="auto",
        )
        widget.layout = layout
    else:
        layout.width = width
        layout.height = height
        layout.min_height = "0px"
        layout.resize = "vertical"
        layout.overflow = "auto"


def _matplotlib_colormap(name: str) -> Optional[pg.ColorMap]:
    try:
        import matplotlib.cm as mpl_cm
    except Exception:  # pragma: no cover - matplotlib optional
        return None
    try:
        cmap = mpl_cm.get_cmap(name)
    except ValueError:  # pragma: no cover - invalid name fallback
        return None
    positions = np.linspace(0.0, 1.0, 256)
    colors = (cmap(positions) * 255).astype(np.uint8)
    return pg.ColorMap(positions, colors)
