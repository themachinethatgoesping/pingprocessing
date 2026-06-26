"""Beam-pattern modelling from a cross-calibration dataset.

:class:`CalibrationPattern` turns a :class:`~.data.CalibrationData` into a
per-channel 2-D calibration surface ``offset(beam_angle, range)``. For each beam
angle it fits a 1-D :mod:`.models` curve to the per-range offset, then stacks
those curves (one row per angle) into a bivariate Akima interpolator. The
result can be evaluated anywhere, plotted as a far-field pattern, and applied
back onto multibeam pings as an Sv calibration offset.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .data import CalibrationData
from .models import PchipBlendChangePoint


def _default_model() -> PchipBlendChangePoint:
    return PchipBlendChangePoint(blend_width=0.5, window_frac=0.5, mode="constant")


class CalibrationPattern:
    """Fit and evaluate a per-channel ``offset(angle, range)`` calibration surface.

    Parameters
    ----------
    x:
        Which range column to model against -- ``'depth'`` (default),
        ``'range_beam'`` or ``'range_base'``.
    model_factory:
        Zero-arg callable returning a fresh model with ``fit``/``get_fit``
        (default :class:`PchipBlendChangePoint`).
    min_n:
        Minimum accepted blocks for a range point to be used.
    min_points:
        Minimum range points required to fit a beam.
    range_grid:
        ``(r0, r1, n)`` grid the fitted curve is sampled on before stacking.
    """

    def __init__(
        self,
        *,
        x: str = "depth",
        model_factory: Callable[[], object] = _default_model,
        min_n: int = 20,
        min_points: int = 4,
        range_grid=(0.0, 30.0, 31),
    ):
        self._x = x
        self._model_factory = model_factory
        self._min_n = int(min_n)
        self._min_points = int(min_points)
        self._range_grid = range_grid
        self.models: Dict[str, Dict[float, object]] = {}
        self._interp: Dict[str, object] = {}

    # -- fitting ---------------------------------------------------------
    def fit(
        self,
        data: CalibrationData,
        *,
        channels: Optional[Sequence[str]] = None,
        show_progress: bool = True,
        **calib_kwargs,
    ) -> "CalibrationPattern":
        """Fit a model per beam angle and build the per-channel interpolators."""
        from themachinethatgoesping.tools.vectorinterpolators.bivectorinterpolators import (
            BiAkimaInterpolatorF,
        )

        channels = list(channels) if channels is not None else data.channels
        iterator = channels
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(channels, desc="pattern")

        r0, r1, n = self._range_grid
        for channel in iterator:
            interp = BiAkimaInterpolatorF(extrapolation_mode="nearest")
            self.models[channel] = {}
            for angle in data.angles(channel):
                table = data.calibration_per_range(channel, angle, **calib_kwargs)
                sel = table[table["n"] >= self._min_n]
                if len(sel) < self._min_points:
                    continue
                xv = sel[self._x].to_numpy(dtype=float)
                yv = sel["csv"].to_numpy(dtype=float)
                model = self._model_factory().fit(xv, yv)
                xg, yg = model.get_fit(r0, r1, n)
                interp.append_row(float(angle), xg, yg)
                self.models[channel][float(angle)] = model
            self._interp[channel] = interp
        return self

    # -- evaluation ------------------------------------------------------
    def channels(self) -> List[str]:
        return list(self._interp.keys())

    def interpolator(self, channel: str):
        """The bivariate ``offset(angle, range)`` interpolator for a channel."""
        return self._interp[channel]

    def angles(self, channel: str) -> np.ndarray:
        return np.asarray(self._interp[channel].get_row_coordinates(), dtype=float)

    def evaluate(self, channel: str, angles, ranges) -> np.ndarray:
        """Offset surface for ``angles`` x ``ranges`` (2-D)."""
        return self._interp[channel](np.asarray(angles, dtype=float),
                                     np.asarray(ranges, dtype=float))

    def far_field(self, channel: str, far_range: float = 5000.0):
        """``(angles, offset)`` at a large (far-field) range for one channel."""
        a = self.angles(channel)
        offsets = self._interp[channel](a, [float(far_range)])[:, 0]
        return a, offsets

    # -- apply to pings --------------------------------------------------
    def build_calibrations(self, pings) -> dict:
        """Build ``{channel: WaterColumnCalibration}`` with the offset applied.

        Mirrors the notebook workflow: take each channel's existing Av
        calibration, set the per-beamangle-and-range offset from this pattern,
        and store it back as the Sv calibration.
        """
        cals: dict = {}
        for ping in pings:
            ch = ping.get_channel_id()
            if ch in cals or ch not in self._interp:
                continue
            cal = ping.file_data.get_watercolumn_calibration()
            av = cal.get_av_calibration()
            av.set_offset_per_beamangle_and_range(self._interp[ch])
            cal.set_sv_calibration(av)
            cals[ch] = cal
        return cals

    def apply_to_pings(self, pings, *, show_progress: bool = True) -> dict:
        """Apply the pattern as an Sv calibration onto ``pings`` (in place)."""
        cals = self.build_calibrations(pings)
        for ch, cal in cals.items():
            try:
                cals[ch] = cal.pre_hashed()
            except Exception:
                pass
        iterator = pings
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(pings, desc="apply calibration")
        for ping in iterator:
            ch = ping.get_channel_id()
            if ch in cals:
                ping.watercolumn.update_calibration(cals[ch])
        return cals
