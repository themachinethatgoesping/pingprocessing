"""Tidy, on-disk cross-calibration dataset and its analysis API.

A cross-calibration dataset compares a *base* echogram (e.g. a vertical
single-beam echosounder) against many *beam* echograms (e.g. individual
multibeam beams) over fixed time blocks and named range layers. This module
stores that comparison in a compact, portable, **append-able** layout and
exposes a clean analysis API (per-range calibration with bootstrap CIs,
cross-plot data, and cheap splitting by station / environmental parameter).

On-disk layout (a directory, written incrementally so processing is resumable)::

    <root>/
        meta.json            # base name, deltaT, reference, layer config, ...
        params.parquet       # per-time-block params (temperature, station, ...)
        beams/
            manifest.json     # [{file, channel, angle}, ...]
            <channel>=<angle>.parquet   # one long table per (channel, angle)

Each beam table is long over ``(time, layer)`` with columns
``unixtime, layer, range_beam, range_base, depth,
base_value, base_count, beam_value, beam_count``.

The base/beam *values* are per-block reductions (median by default) of the
acoustic samples inside the layer; the *counts* are the number of contributing
samples and drive the fill-based filtering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BEAM_COLUMNS = [
    "unixtime", "layer", "range_beam", "range_base", "depth",
    "base_value", "base_count", "beam_value", "beam_count",
]


# ---------------------------------------------------------------------------
# Low-level directory store (shared by builder for writing and data for reading)
# ---------------------------------------------------------------------------

def beam_filename(channel: str, angle: float) -> str:
    """Filesystem-safe ``<channel>=<angle>.parquet`` name for one beam."""
    safe = str(channel).replace("/", "_").replace(" ", "_")
    return f"{safe}={angle:+08.2f}.parquet"


class CalibrationStore:
    """Read/write helper for the on-disk calibration directory layout."""

    def __init__(self, root):
        self.root = Path(root)
        self.beams_dir = self.root / "beams"

    # -- creation --------------------------------------------------------
    def init(self, meta: dict) -> None:
        self.beams_dir.mkdir(parents=True, exist_ok=True)
        self.write_meta(meta)
        if not self._manifest_path.exists():
            self._write_manifest([])

    # -- meta ------------------------------------------------------------
    @property
    def _meta_path(self) -> Path:
        return self.root / "meta.json"

    def write_meta(self, meta: dict) -> None:
        self._meta_path.write_text(json.dumps(meta, indent=2))

    def read_meta(self) -> dict:
        return json.loads(self._meta_path.read_text())

    # -- params ----------------------------------------------------------
    @property
    def _params_path(self) -> Path:
        return self.root / "params.parquet"

    def write_params(self, df: pd.DataFrame) -> None:
        df.to_parquet(self._params_path, index=False)

    def read_params(self) -> Optional[pd.DataFrame]:
        if not self._params_path.exists():
            return None
        return pd.read_parquet(self._params_path)

    # -- beams + manifest -----------------------------------------------
    @property
    def _manifest_path(self) -> Path:
        return self.beams_dir / "manifest.json"

    def _write_manifest(self, entries: List[dict]) -> None:
        self._manifest_path.write_text(json.dumps(entries, indent=2))

    def read_manifest(self) -> List[dict]:
        if not self._manifest_path.exists():
            return []
        return json.loads(self._manifest_path.read_text())

    def write_beam(self, channel: str, angle: float, df: pd.DataFrame) -> None:
        self.beams_dir.mkdir(parents=True, exist_ok=True)
        fname = beam_filename(channel, angle)
        df.to_parquet(self.beams_dir / fname, index=False)
        entries = [e for e in self.read_manifest() if e["file"] != fname]
        entries.append({"file": fname, "channel": str(channel), "angle": float(angle)})
        self._write_manifest(entries)

    def has_beam(self, channel: str, angle: float) -> bool:
        return (self.beams_dir / beam_filename(channel, angle)).exists()

    def read_beam(self, fname: str) -> pd.DataFrame:
        return pd.read_parquet(self.beams_dir / fname)


# ---------------------------------------------------------------------------
# Per-range calibration result container
# ---------------------------------------------------------------------------

@dataclass
class _Beam:
    channel: str
    angle: float
    file: str


class CalibrationData:
    """Analysis view over a cross-calibration dataset.

    Open an existing dataset with :meth:`open`. Splitting methods
    (:meth:`filter_time`, :meth:`split_by_param`, :meth:`split_by_label`)
    return lightweight views that share the underlying (cached) beam tables but
    restrict the time blocks considered -- so comparing e.g. per-station or
    per-temperature calibration is cheap and never copies the heavy data.
    """

    def __init__(self, store: CalibrationStore, meta: dict,
                 params: Optional[pd.DataFrame], beams: Dict[Tuple[str, float], _Beam]):
        self._store = store
        self._meta = meta
        self._beams = beams
        self._beam_cache: Dict[Tuple[str, float], pd.DataFrame] = {}
        # params: indexed by block datetime, with a 'unixtime' float column.
        if params is None:
            params = pd.DataFrame({"unixtime": []})
        self._params = params
        self._allowed_unix: Optional[np.ndarray] = None  # None == all blocks
        self.label: Optional[str] = None

    # -- construction ----------------------------------------------------
    @classmethod
    def open(cls, path) -> "CalibrationData":
        """Open a dataset directory written by :class:`~.builder.CalibrationBuilder`."""
        store = CalibrationStore(path)
        meta = store.read_meta()
        params = store.read_params()
        if params is not None and "unixtime" in params:
            params = params.set_index(
                pd.to_datetime(params["unixtime"], unit="s", utc=True), drop=False)
            params.index.name = "time"
        beams = {}
        for e in store.read_manifest():
            key = (str(e["channel"]), float(e["angle"]))
            beams[key] = _Beam(key[0], key[1], e["file"])
        return cls(store, meta, params, beams)

    def _view(self, allowed_unix: Optional[np.ndarray], label: Optional[str]) -> "CalibrationData":
        v = CalibrationData(self._store, self._meta, self._params, self._beams)
        v._beam_cache = self._beam_cache  # share the cache
        v._allowed_unix = allowed_unix
        v.label = label
        return v

    # -- metadata --------------------------------------------------------
    @property
    def base_name(self) -> str:
        return self._meta.get("base_name", "base")

    @property
    def deltaT(self) -> str:
        return self._meta.get("deltaT", "")

    @property
    def reference(self) -> str:
        return self._meta.get("reference", "Depth (m)")

    @property
    def channels(self) -> List[str]:
        return sorted({c for c, _ in self._beams})

    def angles(self, channel: str) -> np.ndarray:
        return np.array(sorted(a for c, a in self._beams if c == channel))

    @property
    def beams(self) -> List[Tuple[str, float]]:
        return sorted(self._beams.keys())

    @property
    def layers(self) -> List[str]:
        layers = self._meta.get("layers")
        if layers:
            return list(layers)
        if self._beams:
            ch, an = next(iter(self._beams))
            return list(dict.fromkeys(self._beam_full(ch, an)["layer"]))
        return []

    @property
    def params(self) -> pd.DataFrame:
        return self._masked_params()

    @property
    def time(self) -> pd.DatetimeIndex:
        return self._masked_params().index

    def param(self, name: str) -> pd.Series:
        return self._masked_params()[name]

    # -- beam access -----------------------------------------------------
    def _beam_full(self, channel: str, angle: float) -> pd.DataFrame:
        key = (str(channel), float(angle))
        if key not in self._beam_cache:
            beam = self._beams.get(key)
            if beam is None:
                raise KeyError(f"No beam for channel={channel!r} angle={angle}")
            self._beam_cache[key] = self._store.read_beam(beam.file)
        return self._beam_cache[key]

    def beam(self, channel: str, angle: float) -> pd.DataFrame:
        """Long table for one beam, restricted to the current time mask."""
        df = self._beam_full(channel, angle)
        if self._allowed_unix is not None:
            df = df[df["unixtime"].isin(self._allowed_unix)]
        return df

    def find_closest_angle(self, channel: str, angle: float) -> float:
        angles = self.angles(channel)
        if len(angles) == 0:
            raise KeyError(f"No beams for channel {channel!r}")
        return float(angles[int(np.argmin(np.abs(angles - angle)))])

    def find_closest_angles(self, channel: str, angles: Iterable[float]) -> List[float]:
        return [self.find_closest_angle(channel, a) for a in angles]

    # -- splitting (cheap views) ----------------------------------------
    def _masked_params(self) -> pd.DataFrame:
        if self._allowed_unix is None:
            return self._params
        return self._params[self._params["unixtime"].isin(self._allowed_unix)]

    def filter_time(self, t0=None, t1=None) -> "CalibrationData":
        """View restricted to ``t0 <= time <= t1`` (each bound optional)."""
        u = self._params["unixtime"].to_numpy()
        keep = np.ones(len(u), dtype=bool)
        if t0 is not None:
            keep &= u >= _to_unix_scalar(t0)
        if t1 is not None:
            keep &= u <= _to_unix_scalar(t1)
        allowed = u[keep]
        if self._allowed_unix is not None:
            allowed = np.intersect1d(allowed, self._allowed_unix)
        return self._view(allowed, self.label)

    def split_by_param(self, name: str, ranges: Sequence[Tuple[str, float, float]]
                       ) -> Dict[str, "CalibrationData"]:
        """Split into views by value ranges of a numeric param.

        ``ranges`` is a list of ``(label, low, high)``; a block is included when
        ``low <= param <= high``.
        """
        p = self._params
        out: Dict[str, "CalibrationData"] = {}
        for label, lo, hi in ranges:
            mask = (p[name] >= lo) & (p[name] <= hi)
            allowed = p["unixtime"].to_numpy()[mask.to_numpy()]
            if self._allowed_unix is not None:
                allowed = np.intersect1d(allowed, self._allowed_unix)
            out[label] = self._view(allowed, label)
        return out

    def split_by_label(self, name: str) -> Dict[str, "CalibrationData"]:
        """Split into one view per distinct value of a categorical param."""
        p = self._params
        out: Dict[str, "CalibrationData"] = {}
        for label in pd.unique(p[name].dropna()):
            allowed = p["unixtime"].to_numpy()[(p[name] == label).to_numpy()]
            if self._allowed_unix is not None:
                allowed = np.intersect1d(allowed, self._allowed_unix)
            out[str(label)] = self._view(allowed, str(label))
        return out

    def exclude_label(self, name: str, label) -> "CalibrationData":
        """View with all blocks of ``param == label`` removed."""
        p = self._params
        allowed = p["unixtime"].to_numpy()[(p[name] != label).to_numpy()]
        if self._allowed_unix is not None:
            allowed = np.intersect1d(allowed, self._allowed_unix)
        return self._view(allowed, self.label)

    # -- analysis --------------------------------------------------------
    @staticmethod
    def _count_threshold(counts: np.ndarray, fraction: float) -> float:
        pos = counts[counts > 0]
        if len(pos) == 0:
            return 0.0
        return float(fraction) * float(np.median(pos))

    def calibration_per_range(
        self,
        channel: str,
        angle: float,
        *,
        min_count_fraction: float = 0.66,
        iqr_filter: bool = True,
        iqr_k: float = 1.5,
        bootstrap: int = 100,
        confidence_level: float = 0.95,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """Per-layer calibration offset ``C = base - beam`` for one beam.

        Pipeline (matching the validated notebook approach):

        1. **Fill filter** -- per side, drop blocks whose sample count is below
           ``min_count_fraction * median(positive counts)`` (computed on the
           full, unsplit beam so splits stay comparable).
        2. **Outlier filter** -- optional 1.5*IQR fence on the per-block
           difference ``C`` within each layer.
        3. **Reduce** -- median ``C`` per layer with a bootstrap CI.

        Returns a frame indexed by layer with ``range_beam, range_base, depth,
        csv`` (median offset), ``ci_low, ci_high, n``.
        """
        from scipy import stats

        full = self._beam_full(channel, angle)
        thr_base = self._count_threshold(full["base_count"].to_numpy(), min_count_fraction)
        thr_beam = self._count_threshold(full["beam_count"].to_numpy(), min_count_fraction)

        df = full
        if self._allowed_unix is not None:
            df = df[df["unixtime"].isin(self._allowed_unix)]

        rows = []
        for layer, g in df.groupby("layer", sort=False):
            base = g["base_value"].to_numpy(dtype=np.float64).copy()
            beam = g["beam_value"].to_numpy(dtype=np.float64).copy()
            base[g["base_count"].to_numpy() < thr_base] = np.nan
            beam[g["beam_count"].to_numpy() < thr_beam] = np.nan

            C = base - beam
            C = C[np.isfinite(C)]
            if iqr_filter and len(C) > 0:
                q1, q3 = np.nanquantile(C, 0.25), np.nanquantile(C, 0.75)
                med = np.nanmedian(C)
                iqr = q3 - q1
                C = C[(C >= med - iqr * iqr_k) & (C <= med + iqr * iqr_k)]
            n = int(len(C))
            if n == 0:
                continue

            csv = float(np.nanmedian(C))
            low, high = csv, csv
            if n > 1 and bootstrap > 0:
                res = stats.bootstrap(
                    (C,), np.nanmedian, n_resamples=bootstrap,
                    confidence_level=confidence_level, method="percentile",
                    random_state=random_state)
                low = float(res.confidence_interval.low)
                high = float(res.confidence_interval.high)

            rows.append({
                "layer": layer,
                "range_beam": float(np.nanmedian(g["range_beam"])),
                "range_base": float(np.nanmedian(g["range_base"])),
                "depth": float(np.nanmedian(g["depth"])),
                "csv": csv,
                "ci_low": low,
                "ci_high": high,
                "n": n,
            })

        out = pd.DataFrame(rows, columns=[
            "layer", "range_beam", "range_base", "depth", "csv", "ci_low", "ci_high", "n"])
        return out.sort_values("range_beam").reset_index(drop=True)

    def cross_data(
        self,
        channel: str,
        angle: float,
        layer: str,
        *,
        min_count_fraction: float = 0.66,
        iqr_filter: bool = True,
        iqr_k: float = 1.5,
    ) -> pd.DataFrame:
        """Per-block ``base``/``beam`` values for one layer (for cross-plots).

        Returns a frame with ``base, beam, diff, inlier`` columns (fill-filtered;
        ``inlier`` marks the 1.5*IQR-accepted points used for the offset).
        """
        full = self._beam_full(channel, angle)
        thr_base = self._count_threshold(full["base_count"].to_numpy(), min_count_fraction)
        thr_beam = self._count_threshold(full["beam_count"].to_numpy(), min_count_fraction)

        df = full[full["layer"] == layer]
        if self._allowed_unix is not None:
            df = df[df["unixtime"].isin(self._allowed_unix)]

        base = df["base_value"].to_numpy(dtype=np.float64).copy()
        beam = df["beam_value"].to_numpy(dtype=np.float64).copy()
        base[df["base_count"].to_numpy() < thr_base] = np.nan
        beam[df["beam_count"].to_numpy() < thr_beam] = np.nan
        diff = base - beam

        inlier = np.isfinite(diff)
        if iqr_filter and inlier.any():
            finite = diff[inlier]
            med = np.nanmedian(finite)
            iqr = np.nanquantile(finite, 0.75) - np.nanquantile(finite, 0.25)
            inlier = inlier & (diff >= med - iqr * iqr_k) & (diff <= med + iqr * iqr_k)

        return pd.DataFrame({
            "unixtime": df["unixtime"].to_numpy(),
            "base": base, "beam": beam, "diff": diff, "inlier": inlier,
        })

    def series(self, channel: str, angle: float, layer: str) -> pd.DataFrame:
        """Raw per-block time series for one layer (base/beam value+count)."""
        df = self.beam(channel, angle)
        df = df[df["layer"] == layer]
        return df[["unixtime", "base_value", "base_count", "beam_value", "beam_count"]].copy()

    def __repr__(self) -> str:
        lbl = f" label={self.label!r}" if self.label else ""
        return (f"CalibrationData(base={self.base_name!r}, channels={self.channels}, "
                f"n_beams={len(self._beams)}, deltaT={self.deltaT!r}{lbl})")


def _to_unix_scalar(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    return float(pd.Timestamp(t).timestamp())
