"""Jupyter notebook combined viewer using ipywidgets layout.

Thin wrapper that hosts existing viewer instances (created with
``embedded=True``) in a shared layout with a tabbed control panel.

Usage::

    from themachinethatgoesping.pingprocessing.widgets import (
        CombinedViewerJupyter, WCIViewerJupyter, EchogramViewerJupyter,
        MapViewerJupyter,
    )

    wci = WCIViewerJupyter(channels, embedded=True, show=False)
    echo = EchogramViewerJupyter(echogramdata, embedded=True, show=False)
    map_ = MapViewerJupyter(builder=builder, tile_builder=tb, embedded=True, show=False)

    cv = CombinedViewerJupyter()
    cv.add(wci, name="WCI")
    cv.add(echo, name="Echogram")
    cv.add(map_, name="Map")

    # manual cross-wiring
    echo.core.connect_pingviewer(wci.core)
    map_.core.connect_echogram_viewer(echo.core)
    map_.core.connect_wci_viewer(wci.core)

    cv.show()
"""
from __future__ import annotations

from typing import Any

import ipywidgets
from IPython.display import display

from .combinedviewer_core import CombinedViewerCore, ViewerEntry


class CombinedViewerJupyter:
    """Jupyter combined viewer hosting multiple embedded viewer instances."""

    def __init__(self) -> None:
        self._core = CombinedViewerCore()
        self._graphics_widgets: list[ipywidgets.Widget] = []
        self._tab = ipywidgets.Tab()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def core(self) -> CombinedViewerCore:
        return self._core

    @property
    def entries(self):
        return self._core.entries

    def add(
        self,
        viewer: Any,
        name: str = "",
        position: str = "auto",
    ) -> ViewerEntry:
        """Add an existing viewer instance (created with ``embedded=True``).

        Parameters
        ----------
        viewer
            A viewer instance with ``.graphics`` and ``.build_control_widget()``
            attributes.
        name
            Tab label for controls.
        position
            Unused for now; reserved for future grid layout control.
        """
        entry = self._core.add(viewer, name=name, position=position)

        # Store graphics widget
        self._graphics_widgets.append(viewer.graphics)

        # Add controls tab
        ctrl = viewer.build_control_widget()
        children = list(self._tab.children) + [ctrl]
        titles = [self._tab.get_title(i) for i in range(len(self._tab.children))]
        titles.append(entry.name)
        self._tab.children = children
        for i, t in enumerate(titles):
            self._tab.set_title(i, t)

        return entry

    def remove(self, entry: ViewerEntry) -> None:
        """Remove a viewer from the combined widget."""
        idx = next(
            (i for i, e in enumerate(self._core.entries) if e.uid == entry.uid),
            None,
        )
        if idx is not None:
            self._graphics_widgets.pop(idx)
            children = list(self._tab.children)
            children.pop(idx)
            titles = [
                self._tab.get_title(i)
                for i in range(len(self._tab.children))
                if i != idx
            ]
            self._tab.children = children
            for i, t in enumerate(titles):
                self._tab.set_title(i, t)
        self._core.remove(entry)

    def show(self) -> None:
        """Display the combined viewer widget."""
        graphics_row = ipywidgets.HBox(self._graphics_widgets)
        layout = ipywidgets.VBox([graphics_row, self._tab])
        display(layout)
