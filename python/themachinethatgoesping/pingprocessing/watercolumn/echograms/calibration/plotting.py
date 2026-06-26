"""Plotting helpers for cross-calibration results.

These operate on the tidy frames returned by :class:`~.data.CalibrationData`
(``calibration_per_range`` and ``cross_data``) so notebooks stay short. Colour
schemes are Paul Tol's qualitative palettes (colour-blind friendly).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Paul Tol qualitative palettes (repeated so long angle lists don't run out).
TOL_MUTED = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
    "#DDCC77", "#CC6677", "#882255", "#AA4499",
] * 4
TOL_LIGHT = [
    "#77AADD", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00",
    "#EEDD88", "#EE8866", "#FFAABB", "#DDDDDD",
] * 4
TOL_BRIGHT = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#000000", "#FFA500", "#44AA99",
] * 4


def plot_calibration(
    ax,
    table,
    *,
    y: str = "depth",
    min_n: int = 20,
    color: str = "black",
    color_excluded: str = "grey",
    label: Optional[str] = None,
    lines: bool = False,
    plot_excluded: bool = True,
    zorder: int = 3,
    markersize: float = 3.0,
):
    """Plot ``csv`` vs ``y`` with bootstrap-CI error bars.

    ``table`` is the frame from :meth:`CalibrationData.calibration_per_range`.
    Points with fewer than ``min_n`` accepted blocks are drawn faded (or hidden
    when ``plot_excluded=False``).
    """
    c = table["csv"].to_numpy(dtype=float)
    yv = table[y].to_numpy(dtype=float)
    low = table["ci_low"].to_numpy(dtype=float)
    high = table["ci_high"].to_numpy(dtype=float)
    n = table["n"].to_numpy()

    inc = n >= min_n
    exc = ~inc
    capsize = markersize / 2

    if lines and inc.any():
        ax.plot(c[inc], yv[inc], c=color, zorder=zorder)
    if inc.any():
        ax.errorbar(c[inc], yv[inc], xerr=(c[inc] - low[inc], high[inc] - c[inc]),
                    fmt="o", markersize=markersize, linewidth=1, capsize=capsize,
                    label=label, color=color, zorder=zorder)
    if plot_excluded and exc.any():
        ax.errorbar(c[exc], yv[exc], xerr=(c[exc] - low[exc], high[exc] - c[exc]),
                    fmt="o", markersize=markersize, linewidth=1, capsize=capsize,
                    color=color_excluded, zorder=1)
    return ax


def plot_cross(
    ax,
    cross,
    *,
    voffset: float = 0.0,
    color: str = "black",
    color_excluded: str = "grey",
    s: float = 1.0,
    draw_offset: bool = True,
    label: Optional[str] = None,
):
    """Scatter base vs beam values for one layer (from ``cross_data``).

    ``voffset`` shifts the beam axis (useful to overlay the nominal offset).
    The dashed red line marks the median offset of the accepted (inlier) points.
    """
    base = cross["base"].to_numpy(dtype=float)
    beam = cross["beam"].to_numpy(dtype=float) + voffset
    inlier = cross["inlier"].to_numpy(dtype=bool)

    if inlier.any():
        ax.scatter(base[inlier], beam[inlier], marker=".", c=color, s=s, zorder=10, label=label)
    out = ~inlier & np.isfinite(base) & np.isfinite(beam)
    if out.any():
        ax.scatter(base[out], beam[out], marker=".", c=color_excluded, s=s, zorder=1)

    if draw_offset and inlier.any():
        c = float(np.nanmedian(base[inlier] - beam[inlier]))
        ax.plot([-100, 0], [-100 - c, 0 - c], c="red", marker="o",
                linestyle="--", zorder=9)
    return ax
