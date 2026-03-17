"""Backend-agnostic core for the combined viewer.

Thin registry that manages a flat list of viewer entries.  Each entry
wraps an existing viewer instance (created with ``embedded=True``) and
carries a layout hint.  Backends resolve the hints into concrete
layouts.

Cross-wiring between viewers is intentionally left to the user — call
the viewers' own ``connect_*`` or ``core.connect_*`` methods directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ViewerEntry:
    """One viewer inside the combined window."""

    viewer: Any          # e.g. WCIViewerQt / EchogramViewerJupyter / …
    name: str
    position: str = "auto"   # layout hint ("top-left", "bottom", …)
    uid: int = 0


class CombinedViewerCore:
    """Backend-agnostic list of viewer entries.

    Usage::

        core = CombinedViewerCore()
        entry = core.add(viewer_instance, name="WCI", position="top-left")
    """

    def __init__(self) -> None:
        self._entries: List[ViewerEntry] = []
        self._next_uid: int = 1

    # ------------------------------------------------------------------
    # Add / remove
    # ------------------------------------------------------------------

    def add(
        self,
        viewer: Any,
        name: str = "",
        position: str = "auto",
    ) -> ViewerEntry:
        """Register *viewer* and return its :class:`ViewerEntry`."""
        entry = ViewerEntry(
            viewer=viewer,
            name=name or f"Viewer {self._next_uid}",
            position=position,
            uid=self._next_uid,
        )
        self._next_uid += 1
        self._entries.append(entry)
        return entry

    def remove(self, entry: ViewerEntry) -> None:
        """Unregister *entry*."""
        if entry in self._entries:
            self._entries.remove(entry)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def entries(self) -> List[ViewerEntry]:
        return list(self._entries)
