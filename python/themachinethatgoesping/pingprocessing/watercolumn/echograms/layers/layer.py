"""Portable, echogram-independent layer (region-of-interest) specifications.

A :class:`Layer` describes a vertical band (a region of interest) per ping by a
*lower* and an *upper* boundary curve expressed in a fixed **reference frame**
(``'Depth (m)'``, ``'Range (m)'``, ``'Sample number'`` or ``'Y indice'``).

Key design properties
----------------------
* A layer owns **no** echogram. It is a lightweight, reusable spec.
* Binding a layer to an echogram resolves the boundaries to **sample indices**
  (the data-anchored canonical representation) using that echogram's per-ping
  geometry. This is independent of the current display y-axis, so changing the
  display axis never invalidates a layer.
* Because resolution goes through a physical reference, the same layer can be
  applied to *different* echograms, and a band resolved on one echogram can be
  read back in any reference (e.g. depth) and transferred to another echogram.

Boundaries
----------
Each boundary (lower/upper) is one of:

* a constant value (in the layer's reference units),
* a per-ping array (already aligned to the echogram's pings),
* a time series ``(timestamps, values)`` interpolated onto the ping times,
* a ping-parameter reference (e.g. ``'bottom'``) with ``scale``/``offset``,
* ``None`` meaning "open" (the top resp. bottom of the data).
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence, Tuple, Union

import numpy as np

REFERENCES = ("Depth (m)", "Range (m)", "Sample number", "Y indice")

BoundaryLike = Union[float, int, Sequence[float], np.ndarray, "Boundary", None]


def _to_timestamps(values) -> np.ndarray:
    """Convert a sequence that may contain datetimes to float POSIX timestamps."""
    arr = list(values)
    if len(arr) > 0 and isinstance(arr[0], dt.datetime):
        arr = [v.timestamp() for v in arr]
    return np.asarray(arr, dtype=np.float64)


class Boundary:
    """One boundary curve (lower or upper) of a :class:`Layer`.

    Resolves, against an ``EchogramCoordinateSystem``, to a per-ping float array
    (length ``n_pings``) expressed in the layer's reference units. ``None``
    values mark an *open* boundary that the layer maps to the data extent.
    """

    __slots__ = ("kind", "value", "values", "timestamps", "series",
                 "param", "scale", "offset")

    def __init__(self, kind, *, value=None, values=None, timestamps=None,
                 series=None, param=None, scale=1.0, offset=0.0):
        self.kind = kind
        self.value = value
        self.values = values
        self.timestamps = timestamps
        self.series = series
        self.param = param
        self.scale = float(scale)
        self.offset = float(offset)

    # -- constructors ----------------------------------------------------
    @classmethod
    def const(cls, value: float) -> "Boundary":
        return cls("const", value=float(value))

    @classmethod
    def open(cls) -> "Boundary":
        return cls("open")

    @classmethod
    def per_ping(cls, values: Sequence[float]) -> "Boundary":
        return cls("per_ping", values=np.asarray(values, dtype=np.float64))

    @classmethod
    def time_series(cls, timestamps, values, *, scale=1.0, offset=0.0) -> "Boundary":
        return cls("time_series", timestamps=_to_timestamps(timestamps),
                   series=np.asarray(values, dtype=np.float64),
                   scale=scale, offset=offset)

    @classmethod
    def from_param(cls, param: str, *, scale=1.0, offset=0.0) -> "Boundary":
        return cls("param", param=param, scale=scale, offset=offset)

    @classmethod
    def coerce(cls, obj: BoundaryLike) -> "Boundary":
        """Turn a user value into a :class:`Boundary`.

        * ``None`` -> open boundary
        * scalar -> constant
        * array-like -> per-ping
        * :class:`Boundary` -> returned unchanged
        """
        if obj is None:
            return cls.open()
        if isinstance(obj, Boundary):
            return obj
        if np.isscalar(obj):
            return cls.const(float(obj))
        return cls.per_ping(obj)

    # -- resolution ------------------------------------------------------
    def resolve(self, cs, reference: str) -> np.ndarray:
        """Resolve to a per-ping array in ``reference`` units (NaN where open)."""
        n = cs.n_pings
        match self.kind:
            case "open":
                return np.full(n, np.nan, dtype=np.float64)
            case "const":
                return np.full(n, self.value, dtype=np.float64)
            case "per_ping":
                vals = np.asarray(self.values, dtype=np.float64)
                if len(vals) != n:
                    raise ValueError(
                        f"Boundary per-ping length {len(vals)} != n_pings {n}")
                return vals
            case "time_series":
                from themachinethatgoesping import tools
                ts = np.asarray(self.timestamps, dtype=np.float64)
                ser = np.asarray(self.series, dtype=np.float64)
                finite = np.isfinite(ts) & np.isfinite(ser)
                ts, ser = ts[finite], ser[finite]
                if len(ts) == 0:
                    return np.full(n, np.nan, dtype=np.float64)
                order = np.argsort(ts)
                ts, ser = ts[order], ser[order]
                uniq, inv = np.unique(ts, return_inverse=True)
                if len(uniq) < len(ts):
                    ser = np.bincount(inv, weights=ser) / np.bincount(inv)
                    ts = uniq
                interp = tools.vectorinterpolators.LinearInterpolator(
                    ts, ser, extrapolation_mode="nearest")
                out = interp(np.asarray(cs.ping_times, dtype=np.float64))
                return self.scale * out + self.offset
            case "param":
                base = cs.get_param_values(self.param, reference)
                return self.scale * np.asarray(base, dtype=np.float64) + self.offset
            case _:
                raise RuntimeError(f"Unknown boundary kind '{self.kind}'")

    def is_open(self) -> bool:
        return self.kind == "open"


class Layer:
    """A region of interest defined by lower/upper boundaries in a reference frame.

    Parameters
    ----------
    reference:
        One of ``'Depth (m)'``, ``'Range (m)'``, ``'Sample number'``,
        ``'Y indice'`` -- the units the boundaries are expressed in.
    lower, upper:
        Boundary specifications. Each accepts a scalar, a per-ping array, a
        :class:`Boundary`, or ``None`` (open). ``lower`` is the shallower /
        smaller-value edge, ``upper`` the deeper / larger-value edge.
    name:
        Optional layer name (informational; the echogram stores layers by name).
    """

    def __init__(self, reference: str, lower: BoundaryLike, upper: BoundaryLike,
                 name: Optional[str] = None):
        if reference not in REFERENCES:
            raise ValueError(
                f"Invalid reference '{reference}'. Must be one of {REFERENCES}.")
        self.reference = reference
        self.lower = Boundary.coerce(lower)
        self.upper = Boundary.coerce(upper)
        self.name = name
        # When set (via from_sample_indices) the layer resolves directly to
        # these per-ping sample-index bounds, bypassing boundary resolution.
        self._explicit_samples: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __repr__(self) -> str:
        nm = f" '{self.name}'" if self.name else ""
        return f"<Layer{nm} reference='{self.reference}'>"

    # -- convenience constructors ---------------------------------------
    @classmethod
    def depth(cls, lower: BoundaryLike, upper: BoundaryLike, name=None) -> "Layer":
        """A band between two depths (meters)."""
        return cls("Depth (m)", lower, upper, name)

    @classmethod
    def range(cls, lower: BoundaryLike, upper: BoundaryLike, name=None) -> "Layer":
        """A band between two ranges (meters)."""
        return cls("Range (m)", lower, upper, name)

    @classmethod
    def sample_number(cls, lower: BoundaryLike, upper: BoundaryLike, name=None) -> "Layer":
        """A band between two sample numbers."""
        return cls("Sample number", lower, upper, name)

    @classmethod
    def y_indice(cls, lower: BoundaryLike, upper: BoundaryLike, name=None) -> "Layer":
        """A band between two raw sample indices."""
        return cls("Y indice", lower, upper, name)

    @classmethod
    def from_param_absolute(cls, param: str, offset_lower: Optional[float],
                            offset_upper: Optional[float], *,
                            reference: str = "Depth (m)", name=None) -> "Layer":
        """Band at ``param + offset`` for each edge (``None`` -> open).

        Example: ``Layer.from_param_absolute('bottom', -1.0, +1.0)`` is the
        1 m band straddling the bottom.
        """
        lower = None if offset_lower is None else Boundary.from_param(param, offset=offset_lower)
        upper = None if offset_upper is None else Boundary.from_param(param, offset=offset_upper)
        return cls(reference, lower, upper, name)

    @classmethod
    def from_param_relative(cls, param: str, scale_lower: Optional[float],
                            scale_upper: Optional[float], *,
                            reference: str = "Depth (m)", name=None) -> "Layer":
        """Band at ``param * scale`` for each edge (``None`` -> open).

        Example: ``Layer.from_param_relative('bottom', 0.0, 1.2)`` covers from
        the surface to 1.2x the bottom value.
        """
        lower = None if scale_lower is None else Boundary.from_param(param, scale=scale_lower)
        upper = None if scale_upper is None else Boundary.from_param(param, scale=scale_upper)
        return cls(reference, lower, upper, name)

    @classmethod
    def from_param(cls, param: str, *, scale_lower=1.0, offset_lower=0.0,
                   scale_upper=1.0, offset_upper=0.0,
                   reference: str = "Depth (m)", name=None) -> "Layer":
        """General param band: ``param*scale + offset`` for each edge."""
        lower = Boundary.from_param(param, scale=scale_lower, offset=offset_lower)
        upper = Boundary.from_param(param, scale=scale_upper, offset=offset_upper)
        return cls(reference, lower, upper, name)

    @classmethod
    def from_time_series(cls, reference: str, timestamps, values, *,
                         offset_lower: float = 0.0, offset_upper: float = 0.0,
                         name=None) -> "Layer":
        """Band straddling a time-referenced value series (e.g. a sensor depth).

        Example (sensor depth +/- 1 m)::

            Layer.from_time_series('Depth (m)', times, depths,
                                   offset_lower=-1.0, offset_upper=+1.0)
        """
        lower = Boundary.time_series(timestamps, values, offset=offset_lower)
        upper = Boundary.time_series(timestamps, values, offset=offset_upper)
        return cls(reference, lower, upper, name)

    @classmethod
    def from_sample_indices(cls, i0, i1, name=None) -> "Layer":
        """Build a layer that resolves to exact per-ping sample-index bounds.

        Mainly used for persistence (exact round-trips). The bounds are tied to
        a specific ping count, so such a layer is not portable across echograms
        with a different number of pings.
        """
        obj = cls("Y indice", 0, 0, name)
        obj._explicit_samples = (
            np.asarray(i0, dtype=np.int64), np.asarray(i1, dtype=np.int64))
        return obj

    # -- resolution ------------------------------------------------------
    def resolve_bounds(self, cs) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve to per-ping ``(lower_value, upper_value)`` in reference units.

        Open boundaries are returned as the data extent (sample 0 resp. the last
        sample mapped into the reference), so the result always spans the full
        available band where a boundary is open.
        """
        lo = self.lower.resolve(cs, self.reference)
        hi = self.upper.resolve(cs, self.reference)
        if self.lower.is_open():
            lo = cs.sample_index_to_value(self.reference, np.zeros(cs.n_pings))
        if self.upper.is_open():
            max_samples = np.asarray(cs.max_number_of_samples, dtype=np.float64)
            hi = cs.sample_index_to_value(self.reference, max_samples)
        return lo, hi

    def resolve_sample_indices(self, cs) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve to per-ping ``(i0, i1)`` sample-index bounds (i1 exclusive).

        ``i0`` is inclusive, ``i1`` exclusive. Degenerate / undefined pings map
        to ``i0 == i1 == 0`` (empty band).
        """
        n = cs.n_pings
        max_samples = np.asarray(cs.max_number_of_samples, dtype=np.int64) + 1
        if self._explicit_samples is not None:
            i0 = np.clip(self._explicit_samples[0], 0, max_samples)
            i1 = np.clip(self._explicit_samples[1], 0, max_samples)
            i1 = np.maximum(i1, i0)
            return i0, i1
        # Open edges resolve directly to the sample extent to avoid a redundant
        # value->sample round trip (and to stay exact at the boundaries).
        if self.lower.is_open():
            s_lo = np.zeros(n, dtype=np.float64)
        else:
            s_lo = cs.value_to_sample_index(
                self.reference, self.lower.resolve(cs, self.reference))
        if self.upper.is_open():
            s_hi = np.asarray(cs.max_number_of_samples, dtype=np.float64)
        else:
            s_hi = cs.value_to_sample_index(
                self.reference, self.upper.resolve(cs, self.reference))

        lo = np.minimum(s_lo, s_hi)
        hi = np.maximum(s_lo, s_hi)
        finite = np.isfinite(lo) & np.isfinite(hi)
        i0 = np.where(finite, np.round(lo), 0).astype(np.int64)
        i1 = np.where(finite, np.round(hi), 0).astype(np.int64)
        i1 = np.where(finite, i1 + 1, 0)  # make upper edge inclusive -> exclusive
        # Clamp to valid sample range per ping.
        i0 = np.clip(i0, 0, max_samples)
        i1 = np.clip(i1, 0, max_samples)
        i1 = np.maximum(i1, i0)
        return i0, i1
