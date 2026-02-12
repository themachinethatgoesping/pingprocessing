"""Split pings into groups by their true heading (yaw).

Divide the compass into *n* evenly spaced directions and assign each
ping to the nearest one.  An optional *mirror* mode treats headings
180° apart as equivalent—useful for survey lines that alternate
direction.
"""

import math
from typing import Dict, List

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

from themachinethatgoesping.echosounders import filetemplates

I_Ping = filetemplates.I_Ping


# ──────────────────────────── helpers ────────────────────────────

def _angular_distance(a: float, b: float) -> float:
    """Unsigned shortest angular distance in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _make_bin_centers(
    num_directions: int,
    heading_offset: float,
    mirror: bool,
) -> List[float]:
    """Return the bin-center headings.

    Without *mirror* the full 360° circle is divided.
    With *mirror* only the 0°–180° half-circle is divided (the other
    half is folded back).
    """
    span = 180.0 if mirror else 360.0
    step = span / num_directions
    return [(heading_offset + i * step) % 360.0 for i in range(num_directions)]


def _nearest_bin(
    heading: float,
    bin_centers: List[float],
    mirror: bool,
) -> float:
    """Return the bin center closest to *heading*.

    When *mirror* is ``True``, headings 180° apart from a bin center
    are considered equally close (i.e. 225° maps to the 45° bin).
    """
    best_center = bin_centers[0]
    best_dist = float("inf")

    for center in bin_centers:
        if mirror:
            dist = min(
                _angular_distance(heading, center),
                _angular_distance(heading, (center + 180.0) % 360.0),
            )
        else:
            dist = _angular_distance(heading, center)

        if dist < best_dist:
            best_dist = dist
            best_center = center

    return best_center


def _format_label(degrees: float) -> str:
    """Human-readable label for a bin center, e.g. ``'heading_045'``."""
    return f"heading_{degrees:05.1f}"


# ──────────────────────────── public API ─────────────────────────

def by_heading(
    pings: List[I_Ping],
    num_directions: int = 4,
    heading_offset: float = 0.0,
    mirror: bool = False,
    progress: bool = False,
) -> Dict[str, List[I_Ping]]:
    """Split pings into groups by true heading (yaw).

    The compass is divided into *num_directions* evenly spaced bins.
    Each ping is placed in the bin whose centre is closest to the
    ping's ``yaw`` (true heading).

    Parameters
    ----------
    pings : list of I_Ping
        Input ping sequence.
    num_directions : int, optional
        Number of heading bins.  Default **4** (N / E / S / W when
        *heading_offset* is 0).
    heading_offset : float, optional
        Rotation of the bin grid in degrees.  The first bin centre is
        placed at this angle.  Default **0** (north).
    mirror : bool, optional
        If ``True``, opposite headings (180° apart) are treated as the
        same direction.  The *num_directions* bins then span only 180°
        instead of 360°.  For example ``num_directions=4, mirror=True``
        yields bins at 0°, 45°, 90°, 135°; headings 180°–359° are
        folded onto the corresponding opposite bin.  Default **False**.
    progress : bool, optional
        Show a progress bar.  Default **False**.

    Returns
    -------
    dict[str, list[I_Ping]]
        Keys are ``"heading_XXX.X"`` labels (one per bin centre).
        Values are lists of pings assigned to that direction.

    Examples
    --------
    Four cardinal directions::

        groups = by_heading(pings, num_directions=4, heading_offset=0)
        # keys: 'heading_000.0', 'heading_090.0',
        #        'heading_180.0', 'heading_270.0'

    Six directions with offset, mirrored::

        groups = by_heading(pings, num_directions=6,
                            heading_offset=10, mirror=True)
        # bins at 10°, 40°, 70°, 100°, 130°, 160°
        # a ping at 190° → assigned to 'heading_010.0'
    """
    if num_directions < 1:
        raise ValueError("num_directions must be >= 1")

    bin_centers = _make_bin_centers(num_directions, heading_offset, mirror)

    # Pre-build result dict with stable key order
    result: Dict[str, List[I_Ping]] = {
        _format_label(c): [] for c in bin_centers
    }

    if len(pings) == 0:
        return result

    it = get_progress_iterator(pings, progress, desc="Split pings by heading")

    for ping in it:
        geo = ping.get_geolocation()
        heading = geo.yaw % 360.0

        center = _nearest_bin(heading, bin_centers, mirror)
        result[_format_label(center)].append(ping)

    return result
