"""Native Qt combined viewer using pyqtgraph DockArea.

Thin wrapper that hosts existing viewer instances (created with
``embedded=True``) in a shared :class:`QMainWindow` with dockable
graphics panels and a dockable control sidebar.

All docks (graphics **and** controls) are movable and can optionally
be floated to a second monitor.

Usage::

    from themachinethatgoesping.pingprocessing.widgets import (
        CombinedViewerQt, WCIViewerQt, EchogramViewerQt, MapViewerQt,
    )

    wci = WCIViewerQt(channels, embedded=True, show=False)
    echo = EchogramViewerQt(echogramdata, embedded=True, show=False)
    map_ = MapViewerQt(builder=builder, tile_builder=tb, embedded=True, show=False)

    cv = CombinedViewerQt()
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

import json
import warnings
from typing import Any, Optional

from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.dockarea import Dock, DockArea

from . import pyqtgraph_helpers as pgh
from .combinedviewer_core import CombinedViewerCore, ViewerEntry


class _FlowLayout(QtWidgets.QLayout):
    """Horizontal flow layout that wraps children to new rows."""

    def __init__(self, parent=None, margin=4, spacing=4):
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect, test_only=False):
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y = effective.x(), effective.y()
        row_height = 0
        for item in self._items:
            sz = item.sizeHint()
            next_x = x + sz.width() + self._spacing
            if next_x - self._spacing > effective.right() and row_height > 0:
                x = effective.x()
                y += row_height + self._spacing
                next_x = x + sz.width() + self._spacing
                row_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), sz))
            x = next_x
            row_height = max(row_height, sz.height())
        return y + row_height - rect.y() + m.bottom()


class CombinedViewerQt(QtWidgets.QMainWindow):
    """QMainWindow hosting multiple embedded viewer instances.

    Every panel — graphics *and* the shared controls tab — lives in a
    pyqtgraph :class:`Dock` so it can be dragged, re-arranged, or
    floated to a secondary screen.
    """

    def __init__(
        self,
        title: str = "Combined Viewer",
        width: int = 1600,
        height: int = 900,
        parent: Optional[QtWidgets.QWidget] = None,
        settings_key: str = "CombinedViewerQt",
    ) -> None:
        pgh.ensure_qapp()
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)

        self._core = CombinedViewerCore()
        self._settings_key = settings_key

        # Single dock area for everything (graphics + controls)
        self._dock_area = DockArea()
        self.setCentralWidget(self._dock_area)

        # Tab widget for per-viewer controls
        self._tab_widget = QtWidgets.QTabWidget()

        # Put the tab widget inside a dock so it's movable / floatable
        self._controls_dock = Dock("Controls", size=(400, 600))
        self._controls_dock.addWidget(self._tab_widget)
        self._dock_area.addDock(self._controls_dock, "right")

        # Track graphics docks per entry
        self._docks: dict[int, Dock] = {}

        # Combined hover info label
        self._info_label = QtWidgets.QLabel("&nbsp;")
        self._info_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._info_label.setWordWrap(True)
        self._info_dock = Dock("Info", size=(800, 30))
        self._info_dock.addWidget(self._info_label)
        self._dock_area.addDock(self._info_dock, "bottom")
        self._info_parts: dict[str, str] = {}

        # Settings tab
        self._settings_widget = self._build_settings_tab()
        self._tab_widget.addTab(self._settings_widget, "Settings")

        # Restore window geometry from QSettings
        self._restore_window_geometry()

    # ------------------------------------------------------------------
    # Settings tab
    # ------------------------------------------------------------------

    def _build_settings_tab(self) -> QtWidgets.QWidget:
        """Build the shared Settings tab with cross-viewer toggles."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Crosshair sync ---
        grp = QtWidgets.QGroupBox("Crosshair")
        grp_lay = QtWidgets.QVBoxLayout(grp)

        self._cb_sync_depth = QtWidgets.QCheckBox("Sync depth crosshair")
        self._cb_sync_depth.setChecked(True)
        self._cb_sync_depth.toggled.connect(self._on_depth_sync_toggled)
        grp_lay.addWidget(self._cb_sync_depth)

        layout.addWidget(grp)
        layout.addStretch()
        return widget

    def _on_depth_sync_toggled(self, enabled: bool) -> None:
        """Enable / disable depth crosshair sync on all connected echogram viewers."""
        for entry in self._core.entries:
            viewer = entry.viewer
            core = getattr(viewer, "core", viewer)
            if hasattr(core, "set_depth_sync_enabled"):
                core.set_depth_sync_enabled(enabled)

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
            Dock / tab label.
        position
            Dock position hint: ``"top"``, ``"bottom"``, ``"left"``,
            ``"right"``, or ``"auto"`` (stacks below existing graphics docks).
        """
        entry = self._core.add(viewer, name=name, position=position)

        # -- Graphics dock --
        dock = Dock(entry.name, size=(600, 400))
        dock.addWidget(viewer.graphics)
        rel_pos = position if position != "auto" else "bottom"
        if self._docks:
            last_dock = list(self._docks.values())[-1]
            self._dock_area.addDock(dock, rel_pos, last_dock)
        else:
            self._dock_area.addDock(dock, "left", self._controls_dock)
        self._docks[entry.uid] = dock

        # -- Controls tab (scrollable, with flow-friendly layout) --
        ctrl = viewer.build_control_widget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # Wrap the control widget in a FlowLayout container so
        # child rows re-wrap when the panel width changes.
        wrapper = _FlowWrapper(ctrl)
        scroll.setWidget(wrapper)
        self._tab_widget.addTab(scroll, entry.name)

        # Wire viewer's hover label to the combined info bar
        self._hook_hover_label(viewer, entry.name)

        # Let WCI viewers grab the entire combined dock area for video
        core = getattr(viewer, "core", None)
        if core is not None and hasattr(core, "_combined_grab_widget"):
            core._combined_grab_widget = self._dock_area

        return entry

    def _hook_hover_label(self, viewer: Any, name: str) -> None:
        """Intercept a viewer's info label setter to feed the combined bar."""
        core = getattr(viewer, "core", None)
        panel = getattr(core, "panel", None) if core else getattr(viewer, "panel", None)
        if panel is None:
            return


        for key in ("hover_label", "lbl_coords"):
            if key not in panel:
                continue
            handle = panel[key]
            original_setter = handle._setter

            def _make_wrapped(orig, label_name):
                def wrapped(text):
                    orig(text)
                    self._update_combined_info(label_name, text)
                return wrapped

            handle._setter = _make_wrapped(original_setter, name)

    def _update_combined_info(self, name: str, html: str) -> None:
        """Update one viewer's section in the combined info label."""
        stripped = html.strip()
        if stripped and stripped != "&nbsp;":
            self._info_parts[name] = f"<b>{name}</b>: {html}"
        else:
            self._info_parts.pop(name, None)

        if self._info_parts:
            combined = "<br>".join(self._info_parts.values())
        else:
            combined = "&nbsp;"
        self._info_label.setText(combined)

    def remove(self, entry: ViewerEntry) -> None:
        """Remove a viewer from the combined window."""
        uid = entry.uid
        if uid in self._docks:
            dock = self._docks.pop(uid)
            dock.close()

        for i in range(self._tab_widget.count()):
            if self._tab_widget.tabText(i) == entry.name:
                self._tab_widget.removeTab(i)
                break

        self._core.remove(entry)

    def show(self) -> None:
        """Show the window and restore saved dock arrangement."""
        super().show()
        self._restore_dock_state()

    def closeEvent(self, event) -> None:
        """Auto-save window geometry and dock arrangement on close."""
        self._save_state()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _qsettings(self) -> QtCore.QSettings:
        return QtCore.QSettings("themachinethatgoesping", self._settings_key)

    def save_state(self) -> None:
        """Explicitly save window geometry and dock arrangement."""
        self._save_state()

    def _save_state(self) -> None:
        settings = self._qsettings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        try:
            dock_state = self._dock_area.saveState()
            settings.setValue("dockState", json.dumps(dock_state))
            settings.setValue("dockNames", json.dumps(self._current_dock_names()))
        except Exception:
            pass

    def _current_dock_names(self) -> list[str]:
        """Sorted list of dock names currently in the area."""
        names = [self._controls_dock.name(), self._info_dock.name()]
        for d in self._docks.values():
            names.append(d.name())
        return sorted(names)

    def restore_state(self) -> None:
        """Explicitly restore window geometry and dock arrangement."""
        self._restore_window_geometry()
        self._restore_dock_state()

    def _restore_window_geometry(self) -> None:
        settings = self._qsettings()
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        window_state = settings.value("windowState")
        if window_state is not None:
            super().restoreState(window_state)

    def _restore_dock_state(self) -> None:
        settings = self._qsettings()
        dock_state_json = settings.value("dockState")
        if dock_state_json is None:
            return

        # Skip restore when the set of docks has changed since the
        # state was saved — avoids stale layouts that leave docks
        # with area=None.
        saved_names_json = settings.value("dockNames")
        if saved_names_json is not None:
            try:
                saved_names = sorted(json.loads(saved_names_json))
            except Exception:
                saved_names = None
            if saved_names is not None and saved_names != self._current_dock_names():
                return

        try:
            dock_state = json.loads(dock_state_json)
            # Suppress pyqtgraph's harmless "Failed to disconnect"
            # RuntimeWarning that fires when old containers are torn
            # down during restoreState.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self._dock_area.restoreState(
                    dock_state, missing="ignore", extra="bottom"
                )
            # After restore, some docks may have lost their area
            # reference (pyqtgraph sets area=None when a container is
            # destroyed).  Re-assign so double-click-to-float works.
            for dock in self._all_managed_docks():
                if dock.area is None:
                    dock.area = self._dock_area
        except Exception:
            pass

    def _all_managed_docks(self):
        """Yield every Dock managed by this combined viewer."""
        yield self._controls_dock
        yield self._info_dock
        yield from self._docks.values()

    def run(self) -> None:
        """Start the Qt event loop (convenience for scripts)."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.exec()


class _FlowWrapper(QtWidgets.QWidget):
    """Thin wrapper that gives a control widget a *height-for-width*
    size policy so ``QScrollArea`` can adapt its scroll range as the
    panel is made narrower or wider.

    The inner widget is placed inside a ``QVBoxLayout`` that stretches
    horizontally; the wrapper then reports its sizeHint height as the
    inner widget's actual height for the current width.
    """

    def __init__(self, inner: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(inner)
        lay.addStretch()
        self._inner = inner
        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        # Ask the inner widget how tall it wants to be at this width.
        h = self._inner.heightForWidth(w)
        if h < 0:
            h = self._inner.sizeHint().height()
        return h + 8

    def sizeHint(self):
        return self._inner.sizeHint()

