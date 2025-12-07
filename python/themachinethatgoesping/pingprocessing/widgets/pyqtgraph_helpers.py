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
    "list_colormaps",
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
    """Return a PyQtGraph ColorMap from either a name or a ColorMap instance.
    
    Parameters
    ----------
    cmap : str or pg.ColorMap
        Colormap name (pyqtgraph or matplotlib) or ColorMap instance.
        Use :func:`list_colormaps` to see all available names.
    
    Returns
    -------
    pg.ColorMap
        Resolved colormap. Falls back to 'viridis' if not found.
    
    Examples
    --------
    >>> cmap = resolve_colormap('cubehelix_r')  # matplotlib colormap
    >>> cmap = resolve_colormap('viridis')      # pyqtgraph colormap
    """

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


def list_colormaps(source: Optional[str] = None) -> List[str]:
    """List available colormap names.
    
    Parameters
    ----------
    source : str, optional
        Filter by source: 'pyqtgraph', 'matplotlib', or None for all.
    
    Returns
    -------
    List[str]
        Sorted list of colormap names that can be passed to :func:`resolve_colormap`.
    
    Examples
    --------
    >>> list_colormaps()                    # All colormaps
    >>> list_colormaps('matplotlib')        # Only matplotlib colormaps
    >>> list_colormaps('pyqtgraph')         # Only pyqtgraph colormaps
    """
    names: List[str] = []
    
    # PyQtGraph colormaps
    if source is None or source == "pyqtgraph":
        try:
            pg_names = pg.colormap.listMaps()
            if isinstance(pg_names, dict):
                # listMaps() returns dict with categories
                for category_maps in pg_names.values():
                    names.extend(category_maps)
            else:
                names.extend(pg_names)
        except Exception:  # pragma: no cover
            pass
    
    # Matplotlib colormaps
    if source is None or source == "matplotlib":
        try:
            import matplotlib
            mpl_names = list(matplotlib.colormaps)
            names.extend(mpl_names)
        except Exception:  # pragma: no cover - matplotlib optional
            pass
    
    return sorted(set(names))


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
    """Convert a matplotlib colormap to a PyQtGraph ColorMap."""
    try:
        import matplotlib
    except Exception:  # pragma: no cover - matplotlib optional
        return None
    try:
        # Use modern API (matplotlib >= 3.7)
        cmap = matplotlib.colormaps.get_cmap(name)
    except (KeyError, AttributeError):
        # Fallback for older matplotlib or invalid name
        try:
            import matplotlib.cm as mpl_cm
            cmap = mpl_cm.get_cmap(name)
        except (ValueError, AttributeError):  # pragma: no cover
            return None
    positions = np.linspace(0.0, 1.0, 256)
    colors = (cmap(positions) * 255).astype(np.uint8)
    return pg.ColorMap(positions, colors)
