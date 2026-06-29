"""Build a cross-calibration dataset from a base + many beam echograms.

:class:`CalibrationBuilder` takes one *base* echogram (the reference, e.g. a
vertical single-beam echosounder) and lets you add *beam* echograms one at a
time (e.g. individual multibeam beams). For each beam it:

1. defines the range-band layers on the beam echogram (intersected with the
   beam's ``valid`` mask),
2. transfers each band to the base echogram through a shared physical
   reference (depth) so the base measures the *same physical extent*
   (intersected with the base's ``valid`` mask),
3. pools both echograms' layer samples into shared time blocks (fast), and
4. writes that beam's long table to disk immediately.

Because every beam is written as soon as it is computed, the process is
**resumable**: stop any time and re-run later with more (or finer) beam angles;
already-computed beams are skipped. Shared per-block parameters (temperature,
focus range, station label, ...) are written once and reused by every beam.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..layers.layer import Layer
from ..layers.transfer import transfer_layer
from . import _pooling
from .data import CalibrationData, CalibrationStore


def _parse_delta_seconds(deltaT: Union[str, float]) -> float:
    if isinstance(deltaT, str):
        import pytimeparse2
        return float(pytimeparse2.parse(deltaT, as_timedelta=False))
    return float(deltaT)


class CalibrationBuilder:
    """Incrementally build a cross-calibration dataset on disk.

    Parameters
    ----------
    path:
        Output directory (created/extended). Re-opening an existing directory
        resumes it.
    base_echogram:
        The reference echogram, shared by every beam comparison.
    base_name:
        Display name for the base channel (stored in metadata).
    ranges:
        Range-band centres (m) on the beam echogram, e.g. ``np.arange(1, 31)``.
    layer_size:
        Range-band thickness (m); each band is ``r +/- layer_size/2``.
    deltaT:
        Time-block size (``'1min'`` or seconds).
    base_mask, beam_mask:
        Layer names of the per-echogram ``valid`` masks to intersect into every
        band (``None`` to skip).
    layer_reference:
        Reference the range bands are defined in on the *beam* echogram
        (``'Range (m)'`` by default -- multibeam beams are banded in slant
        range; use ``'Depth (m)'`` for vertical setups).
    reference:
        Physical reference used to transfer bands base<->beam (depth).
    step:
        Ping stride for pooling (``1`` = every ping).
    reduce:
        Block reducer for pooled samples (default median).
    """

    def __init__(
        self,
        path,
        base_echogram,
        *,
        base_name: str = "base",
        ranges: Sequence[float] = tuple(range(1, 31)),
        layer_size: float = 1.0,
        deltaT: Union[str, float] = "1min",
        base_mask: Optional[str] = "valid",
        beam_mask: Optional[str] = "valid",
        layer_reference: str = "Range (m)",
        reference: str = "Depth (m)",
        step: int = 1,
        reduce=np.nanmedian,
        time_extent: Optional[Tuple[float, float]] = None,
        show_progress: bool = True,
    ):
        self._base = base_echogram
        self._base_name = base_name
        self._ranges = [float(r) for r in ranges]
        self._layer_size = float(layer_size)
        self._deltaT = deltaT
        self._delta_seconds = _parse_delta_seconds(deltaT)
        self._base_mask = base_mask
        self._beam_mask = beam_mask
        self._layer_reference = layer_reference
        self._reference = reference
        self._step = max(1, int(step))
        self._reduce = reduce
        self._show_progress = show_progress

        if time_extent is None:
            t = np.asarray(base_echogram.ping_times, dtype=np.float64)
            time_extent = (float(np.min(t)), float(np.max(t)))
        self._block_edges, self._block_centers = _pooling.make_time_blocks(
            time_extent[0], time_extent[1], self._delta_seconds)

        self._layer_names = [f"{r:g}m" for r in self._ranges]

        self._store = CalibrationStore(path)
        self._store.init(self._meta())

        # Per-block params table (full grid); seeded with the time columns.
        existing = self._store.read_params()
        if existing is not None:
            params = existing.reset_index(drop=True).copy()
            if "unixtime" not in params.columns:
                raise ValueError("Existing params.parquet is missing required column 'unixtime'")

            # Resume against the persisted time grid (one row per block).
            persisted_centers = np.asarray(params["unixtime"], dtype=np.float64)
            self._block_centers = persisted_centers
            self._block_edges = _pooling.block_edges_from_centers(
                persisted_centers, self._delta_seconds)
            self._params = params
        else:
            self._params = pd.DataFrame({"unixtime": self._block_centers})
            self._store.write_params(self._params)

        # Keep metadata aligned with the effective (possibly resumed) block grid.
        self._store.write_meta(self._meta())

    # -- metadata --------------------------------------------------------
    def _meta(self) -> dict:
        return {
            "base_name": self._base_name,
            "deltaT": self._deltaT if isinstance(self._deltaT, str) else None,
            "delta_seconds": self._delta_seconds,
            "reference": self._reference,
            "layer_size": self._layer_size,
            "ranges": self._ranges,
            "layers": self._layer_names,
            "block_start": float(self._block_centers[0]) if len(self._block_centers) else 0.0,
            "n_blocks": int(len(self._block_centers)),
        }

    # -- beams -----------------------------------------------------------
    def has_beam(self, channel: str, angle: float) -> bool:
        return self._store.has_beam(channel, angle)

    def add_beam(self, channel: str, angle: float, beam_echogram, *,
                 overwrite: bool = False, show_progress: Union[bool, object, None] = None) -> None:
        """Extract and store one beam comparison (skipped if already present)."""
        if show_progress is None:
            show_progress = self._show_progress
        if not overwrite and self._store.has_beam(channel, angle):
            return

        base = self._base
        beam = beam_echogram

        base_specs = list(base.layers.specs(self._base_mask)) if self._base_mask else []
        beam_specs = list(beam.layers.specs(self._beam_mask)) if self._beam_mask else []

        # 1+2. Define range bands on the beam, transfer the same depths to base.
        for r, name in zip(self._ranges, self._layer_names):
            band = Layer(self._layer_reference, r - self._layer_size * 0.5,
                         r + self._layer_size * 0.5, name=name)
            beam.add_layer(name, [band] + beam_specs, combine=False)
            transfer_layer(beam, base, name, reference=self._reference)
            for spec in base_specs:
                base.add_layer(name, spec, combine=True)

        # 3. Geometry per layer (median band centres over valid pings).
        geom = {name: self._layer_geometry(base, beam, name) for name in self._layer_names}

        # 4. Pool both echograms into the shared time blocks.
        progress = self._make_progress(show_progress, channel, angle)
        base_pool = _pooling.pool_layers(
            base, self._layer_names, self._block_edges,
            step=self._step, reduce=self._reduce, progress=progress)
        beam_pool = _pooling.pool_layers(
            beam, self._layer_names, self._block_edges,
            step=self._step, reduce=self._reduce, progress=progress)
        self._close_progress(show_progress, progress)

        df = self._assemble(channel, angle, geom, base_pool, beam_pool)
        self._store.write_beam(channel, angle, df)
        self._beams_dirty = True

    def add_beams(self, channel: str, beam_echograms: Dict[float, object], *,
                  overwrite: bool = False, show_progress: Optional[bool] = None) -> None:
        """Convenience: add several ``{angle: echogram}`` beams of one channel."""
        if show_progress is None:
            show_progress = self._show_progress
        items = beam_echograms.items()
        if show_progress:
            items = tqdm(list(items), desc=f"beams [{channel}]")
        for angle, echo in items:
            self.add_beam(channel, angle, echo, overwrite=overwrite, show_progress=False)

    # -- params ----------------------------------------------------------
    def add_param(self, name: str, times, values, *, reduce=None, interpolate: bool = True) -> None:
        """Pool an external ``(times, values)`` series into the block grid.

        Parameters are first binned by time block (using ``reduce`` for blocks
        with multiple samples) and then linearly interpolated across block
        centers by default.
        """
        reducer = reduce if reduce is not None else self._reduce
        pooled = _pooling.pool_values(
            times,
            values,
            self._block_edges,
            reduce=reducer,
            centers=self._block_centers,
            interpolate=interpolate,
        )
        self._params[name] = pooled
        self._store.write_params(self._params)

    def add_intervals(self, name: str, intervals: Iterable[Tuple[str, object, object]]) -> None:
        """Label each block by the first ``(label, t0, t1)`` interval it falls in."""
        centers = self._block_centers
        labels = np.full(len(centers), None, dtype=object)
        for label, t0, t1 in intervals:
            u0, u1 = _as_unix(t0), _as_unix(t1)
            mask = (centers >= u0) & (centers <= u1) & (labels == None)  # noqa: E711
            labels[mask] = label
        self._params[name] = labels
        self._store.write_params(self._params)

    # -- result ----------------------------------------------------------
    def result(self) -> CalibrationData:
        """Open the (current) on-disk dataset as a :class:`CalibrationData`."""
        return CalibrationData.open(self._store.root)

    # -- internals -------------------------------------------------------
    def _layer_geometry(self, base, beam, name) -> Tuple[float, float, float]:
        depth = _median_center(beam.get_layer_bounds(name, "Depth (m)"))
        range_beam = _safe_center(beam, name, "Range (m)", fallback=depth)
        range_base = _safe_center(base, name, "Range (m)", fallback=depth)
        return range_beam, range_base, depth

    def _assemble(self, channel, angle, geom, base_pool, beam_pool) -> pd.DataFrame:
        centers = self._block_centers
        frames = []
        for name in self._layer_names:
            bv, bc = base_pool[name]
            mv, mc = beam_pool[name]
            keep = (bc > 0) | (mc > 0)
            if not keep.any():
                continue
            idx = np.where(keep)[0]
            rb, rbase, depth = geom[name]
            frames.append(pd.DataFrame({
                "unixtime": centers[idx],
                "layer": name,
                "range_beam": np.float32(rb),
                "range_base": np.float32(rbase),
                "depth": np.float32(depth),
                "base_value": bv[idx].astype(np.float32),
                "base_count": bc[idx].astype(np.int32),
                "beam_value": mv[idx].astype(np.float32),
                "beam_count": mc[idx].astype(np.int32),
            }))
        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.DataFrame({c: [] for c in [
                "unixtime", "layer", "range_beam", "range_base", "depth",
                "base_value", "base_count", "beam_value", "beam_count"]})
        df["layer"] = df["layer"].astype(str)
        return df

    @staticmethod
    def _make_progress(show_progress, channel, angle):
        if show_progress is True:
            return tqdm(desc=f"{channel} | {angle:+.2f}")
        if show_progress is False or show_progress is None:
            return None
        show_progress.set_description(f"{channel} | {angle:+.2f}")
        return show_progress

    @staticmethod
    def _close_progress(show_progress, progress):
        if progress is not None and show_progress is True:
            progress.close()

    def __repr__(self) -> str:
        return (f"CalibrationBuilder(base={self._base_name!r}, "
                f"ranges={self._ranges[0]}..{self._ranges[-1]}, deltaT={self._deltaT!r}, "
                f"n_beams={len(self._store.read_manifest())})")


def _median_center(bounds) -> float:
    lo, hi = bounds
    center = 0.5 * (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64))
    finite = center[np.isfinite(center)]
    return float(np.median(finite)) if len(finite) else np.nan


def _safe_center(echogram, name: str, reference: str, *, fallback: float) -> float:
    """Median band centre in ``reference`` units; ``fallback`` if unavailable."""
    try:
        value = _median_center(echogram.get_layer_bounds(name, reference))
    except Exception:
        value = np.nan
    return value if np.isfinite(value) else fallback


def _as_unix(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    return float(pd.Timestamp(t).timestamp())
