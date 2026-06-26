"""Range-dependent calibration-offset models.

These 1-D models describe how the cross-calibration offset ``C`` (dB) varies
with range/depth for a single beam. They share a tiny common interface so the
:class:`~.pattern.CalibrationPattern` can fit and evaluate them
interchangeably:

* ``fit(x, y)`` where ``x`` is range/depth and ``y`` is the offset ``C`` --
  returns ``self``.
* ``__call__(x)`` -> predicted offset at ``x``.
* ``get_fit(x0, x1, n)`` -> ``(x_grid, y_grid)`` for plotting.

Two models are provided:

* :class:`PchipBlendChangePoint` -- a shape-preserving PCHIP spline blended
  into a constant (or linear) tail beyond a detected change point ``xc``. This
  is the model used to build the beam pattern (the near-field rolls off, the
  far field is ~constant).
* :class:`LogisticSTR` -- a smooth-transition regression with a logistic gate
  between a linear near-field trend and a constant/linear far-field trend.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


class CalibrationModel:
    """Common interface for 1-D calibration-offset models."""

    def fit(self, x, y) -> "CalibrationModel":  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, x):  # pragma: no cover - interface
        raise NotImplementedError

    def get_fit(self, x0: float, x1: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        x_grid = np.linspace(x0, x1, n)
        return x_grid, np.asarray(self(x_grid), dtype=np.float64)


class PchipBlendChangePoint(CalibrationModel):
    """PCHIP spline blended into a constant/linear tail past a change point.

    The near field is interpolated by a shape-preserving PCHIP spline; beyond a
    detected change point ``xc`` the model blends (via a logistic weight of
    width ``blend_width``) into a constant (mean of the tail) or a linear tail.

    Parameters
    ----------
    mode:
        ``'auto'`` (decide constant vs linear from the tail slope),
        ``'constant'`` or ``'linear'``.
    blend_width:
        Width of the logistic blend between spline and tail (same units as x).
    window_frac:
        Fraction of points in the sliding window used to locate the change
        point from local linear-fit residuals.
    """

    def __init__(self, mode: str = "auto", blend_width: float = 5.0,
                 window_frac: float = 0.15):
        if mode not in ("auto", "constant", "linear"):
            raise ValueError("mode must be 'auto', 'constant' or 'linear'")
        self.mode = mode
        self.blend_width = float(blend_width)
        self.window_frac = float(window_frac)

    def fit(self, x, y) -> "PchipBlendChangePoint":
        # x is range/depth, y is the offset C; we fit C(range) sorted by range
        # and locate a change point past which the offset is ~constant.
        rng = np.asarray(x, dtype=np.float64)
        val = np.asarray(y, dtype=np.float64)
        order = np.argsort(rng)
        rng, val = rng[order], val[order]

        self._x = rng
        self._y = val
        self.pchip = PchipInterpolator(rng, val)

        n = len(rng)
        win = max(3, int(n * self.window_frac))
        slopes, residuals = [], []
        for i in range(n - win + 1):
            xs = rng[i:i + win]
            ys = val[i:i + win]
            A = np.vstack([xs, np.ones_like(xs)]).T
            slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
            slopes.append(slope)
            residuals.append(float(np.mean(np.abs(ys - (slope * xs + intercept)))))
        slopes = np.asarray(slopes)
        residuals = np.asarray(residuals)

        if self.mode == "constant":
            cand = np.where((np.abs(slopes) < 1e-3) & (residuals < np.median(residuals)))[0]
        else:
            cand = np.where(residuals < np.median(residuals))[0]
        self.xc = float(rng[cand[0]]) if len(cand) else float(rng[-win])

        tail = rng >= self.xc
        A = np.vstack([rng[tail], np.ones(int(tail.sum()))]).T
        slope, intercept = np.linalg.lstsq(A, val[tail], rcond=None)[0]
        if self.mode == "constant" or (self.mode == "auto" and abs(slope) < 1e-3):
            self.tail_mode = "constant"
            self.const_val = float(np.mean(val[tail]))
            self.slope = self.intercept = None
        else:
            self.tail_mode = "linear"
            self.slope, self.intercept = float(slope), float(intercept)
        return self

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        base = self.pchip(x)
        w = 1.0 / (1.0 + np.exp(-(x - self.xc) / self.blend_width))
        if self.tail_mode == "constant":
            tail = np.full_like(x, self.const_val)
        else:
            tail = self.slope * x + self.intercept
        return (1.0 - w) * base + w * tail


class LogisticSTR(CalibrationModel):
    """Smooth-transition regression with a logistic gate between two regimes.

    Near field: linear trend ``b0 + b1*x``. Far field: constant ``a0`` (or
    linear ``a0 + a1*x``). The transition is a logistic gate centred at
    ``transition_center`` with steepness ``gamma``.
    """

    def __init__(self, tail: str = "constant"):
        if tail not in ("constant", "linear"):
            raise ValueError("tail must be 'constant' or 'linear'")
        self.tail = tail

    @staticmethod
    def _logistic(x, c, gamma):
        return 1.0 / (1.0 + np.exp(-gamma * (x - c)))

    def _model(self, x, params):
        b0, b1, c, log_gamma = params[:4]
        gate = self._logistic(x, c, np.exp(log_gamma))
        near = b0 + b1 * x
        if self.tail == "constant":
            far = np.full_like(x, params[4])
        else:
            far = params[4] + params[5] * x
        return (1.0 - gate) * near + gate * far

    def fit(self, x, y) -> "LogisticSTR":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size < 3:
            raise ValueError("need at least three points to fit the STR model")
        order = np.argsort(x)
        self._x, self._y = x[order], y[order]

        span = max(2, int(0.15 * self._x.size))
        slope_pre, intercept_pre = np.polyfit(self._x[:span + 1], self._y[:span + 1], 1)
        scale = max(np.std(self._x), 1e-3)
        if self.tail == "constant":
            p0 = np.array([intercept_pre, slope_pre, float(np.median(self._x)),
                           np.log(1.0 / scale), float(np.median(self._y[-(span + 1):]))])
        else:
            slope_post, intercept_post = np.polyfit(self._x[-(span + 1):], self._y[-(span + 1):], 1)
            p0 = np.array([intercept_pre, slope_pre, float(np.median(self._x)),
                           np.log(1.0 / scale), intercept_post, slope_post])

        res = least_squares(lambda p: self._model(self._x, p) - self._y, p0,
                            jac="2-point", method="trf", max_nfev=100000)
        if not res.success:
            raise RuntimeError(f"STR optimisation failed: {res.message}")
        self.params = res.x
        self.transition_center = float(self.params[2])
        self.gamma = float(np.exp(self.params[3]))
        self.transition_width = float(2 * np.log(9) / self.gamma)
        return self

    def __call__(self, x):
        if not hasattr(self, "params"):
            raise RuntimeError("fit must be called before predicting")
        return self._model(np.asarray(x, dtype=np.float64), self.params)
