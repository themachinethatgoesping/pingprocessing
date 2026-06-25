"""Resolution and caching of :class:`Layer` specifications for one echogram.

The :class:`LayerStore` owns the named layers attached to an echogram's
coordinate system and turns the portable :class:`Layer` specs into concrete
per-ping sample-index bands (and, on demand, display-grid bands).

Caching strategy
----------------
* Sample-index bands depend only on the per-ping geometry, tracked by
  ``coordinate_system.geometry_version``. They are cached and only recomputed
  when that version changes (e.g. new extents / ping times). Crucially they do
  **not** depend on the display y-axis, so switching axes is free.
* Display-grid bands additionally depend on ``coordinate_system.display_version``
  and are recomputed lazily when the display axis changes.

A *named* layer is a list of one or more :class:`Layer` specs that are combined
by **intersection** in sample space. This makes the common "valid mask = several
constraints" pattern natural and supports mixing references (e.g. a depth band
intersected with a bottom-relative band).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .layer import Layer


class ResolvedBand:
    """A resolved per-ping sample-index band, with lazy display-grid projection.

    Attributes
    ----------
    i0, i1:
        Per-ping sample-index bounds (``i0`` inclusive, ``i1`` exclusive).
    y0, y1:
        Per-ping display-grid index bounds for the *current* y-axis, computed
        lazily and refreshed when the display axis changes.
    """

    __slots__ = ("_cs", "i0", "i1", "_y0", "_y1", "_grid_display_version")

    def __init__(self, cs, i0: np.ndarray, i1: np.ndarray):
        self._cs = cs
        self.i0 = i0
        self.i1 = i1
        self._y0: Optional[np.ndarray] = None
        self._y1: Optional[np.ndarray] = None
        self._grid_display_version: Optional[int] = None

    def _ensure_grid(self):
        dv = self._cs.display_version
        if self._y0 is None or self._grid_display_version != dv:
            self._y0, self._y1 = self._cs.samples_to_grid(self.i0, self.i1)
            self._grid_display_version = dv

    @property
    def y0(self) -> np.ndarray:
        self._ensure_grid()
        return self._y0

    @property
    def y1(self) -> np.ndarray:
        self._ensure_grid()
        return self._y1

    def bounds(self, reference: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-ping ``(lower, upper)`` values in ``reference`` units."""
        lo = self._cs.sample_index_to_value(reference, self.i0)
        hi = self._cs.sample_index_to_value(reference, self.i1 - 1)
        return lo, hi


class LayerStore:
    """Manages named layers for one echogram coordinate system."""

    def __init__(self, cs):
        self._cs = cs
        self._groups: Dict[str, List[Layer]] = {}
        self._cache: Dict[str, Tuple[int, ResolvedBand]] = {}

    # -- mutation --------------------------------------------------------
    def add(self, name: str, layer, *, combine: bool = True) -> None:
        """Add a layer spec under ``name``.

        If ``name`` already exists and ``combine`` is True, the new spec is
        intersected with the existing ones; otherwise it replaces them. A list
        of specs may be passed to set several intersected constraints at once.
        """
        specs = list(layer) if isinstance(layer, (list, tuple)) else [layer]
        for s in specs:
            if not isinstance(s, Layer):
                raise TypeError(f"Expected Layer, got {type(s).__name__}")
        if combine and name in self._groups:
            self._groups[name].extend(specs)
        else:
            self._groups[name] = list(specs)
        self._cache.pop(name, None)

    def remove(self, name: str) -> None:
        self._groups.pop(name, None)
        self._cache.pop(name, None)

    def clear(self) -> None:
        self._groups.clear()
        self._cache.clear()

    def rename(self, old: str, new: str) -> None:
        if old in self._groups:
            self._groups[new] = self._groups.pop(old)
            self._cache.pop(old, None)

    # -- mapping-like access --------------------------------------------
    def names(self) -> List[str]:
        return list(self._groups.keys())

    def keys(self):
        return self._groups.keys()

    def values(self):
        for name in self._groups:
            yield self.resolve(name)

    def specs(self, name: str) -> List[Layer]:
        return list(self._groups[name])

    def __contains__(self, name: str) -> bool:
        return name in self._groups

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self):
        return iter(self._groups)

    def __getitem__(self, name: str) -> ResolvedBand:
        return self.resolve(name)

    def get(self, name: str) -> Optional[ResolvedBand]:
        if name not in self._groups:
            return None
        return self.resolve(name)

    def items(self):
        for name in self._groups:
            yield name, self.resolve(name)

    # -- resolution ------------------------------------------------------
    def _resolve_group(self, specs: List[Layer]) -> Tuple[np.ndarray, np.ndarray]:
        i0_acc: Optional[np.ndarray] = None
        i1_acc: Optional[np.ndarray] = None
        for layer in specs:
            i0, i1 = layer.resolve_sample_indices(self._cs)
            if i0_acc is None:
                i0_acc, i1_acc = i0, i1
            else:
                i0_acc = np.maximum(i0_acc, i0)
                i1_acc = np.minimum(i1_acc, i1)
        if i0_acc is None:
            n = self._cs.n_pings
            i0_acc = np.zeros(n, dtype=np.int64)
            i1_acc = np.zeros(n, dtype=np.int64)
        i1_acc = np.maximum(i1_acc, i0_acc)
        return i0_acc, i1_acc

    def resolve(self, name: str) -> ResolvedBand:
        """Resolve ``name`` to a :class:`ResolvedBand` (cached by geometry)."""
        if name not in self._groups:
            raise KeyError(f"Layer '{name}' not found")
        version = self._cs.geometry_version
        cached = self._cache.get(name)
        if cached is not None and cached[0] == version:
            return cached[1]
        i0, i1 = self._resolve_group(self._groups[name])
        band = ResolvedBand(self._cs, i0, i1)
        self._cache[name] = (version, band)
        return band

    def sample_indices(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        band = self.resolve(name)
        return band.i0, band.i1

    def grid_indices(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        band = self.resolve(name)
        return band.y0, band.y1

    def bounds(self, name: str, reference: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.resolve(name).bounds(reference)
