"""Split pings into *straight* and *turning* segments based on rate of turn.

The **Rate of Turn (ROT)** is the standard maritime measure for how fast
a vessel changes heading, expressed in degrees per minute.  Survey
vessels on a straight line typically have |ROT| < 2 °/min, while turns
are characterised by |ROT| > 5 °/min.

Two signals are available:

* **Heading ROT** – computed from consecutive ``yaw`` values.  Works
  reliably at any speed but can be noisy if the heading sensor drifts.
* **COG ROT** – computed from consecutive course-over-ground bearings
  (derived from lat/lon).  Requires some minimum speed to be
  meaningful.

A median-filter smoothing window suppresses sensor noise before the
threshold is applied.  Very short segments (fewer than
*min_segment_pings*) are merged into the preceding segment to avoid
chattering.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

from themachinethatgoesping.echosounders import filetemplates

I_Ping = filetemplates.I_Ping


# ──────────────────────────── helpers ────────────────────────────

def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed shortest angular difference *a − b* in degrees (−180, 180].

    Vectorised for numpy arrays.
    """
    return (a - b + 180.0) % 360.0 - 180.0


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Vectorised forward bearing in degrees [0, 360) from arrays."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_lambda = np.radians(lon2 - lon1)

    x = np.sin(d_lambda) * np.cos(phi2)
    y = (np.cos(phi1) * np.sin(phi2)
         - np.sin(phi1) * np.cos(phi2) * np.cos(d_lambda))

    return np.degrees(np.arctan2(x, y)) % 360.0


def _median_filter_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple 1-D median filter (no scipy dependency).

    Uses a centred window of size *window* (clipped at boundaries).
    For angular data this should be called on the **rate** (which is
    already a small linear value), not on the raw bearing.
    """
    if window <= 1:
        return arr
    out = np.empty_like(arr)
    half = window // 2
    n = len(arr)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(arr[lo:hi])
    return out


def _merge_short_segments(labels: np.ndarray, min_length: int) -> np.ndarray:
    """Merge segments shorter than *min_length* into neighbours.

    Short segments are relabelled to match the **preceding** segment
    (or the *following* one for the very first segment).
    """
    if min_length <= 1:
        return labels

    result = labels.copy()
    n = len(result)
    i = 0
    while i < n:
        # Find the end of the current run
        j = i + 1
        while j < n and result[j] == result[i]:
            j += 1
        run_len = j - i
        if run_len < min_length:
            # Merge into the preceding label (or following for the first run)
            merge_label = result[i - 1] if i > 0 else (result[j] if j < n else result[i])
            result[i:j] = merge_label
        i = j
    return result


# ──────────────────────────── public API ─────────────────────────

def by_turn(
    pings: List[I_Ping],
    max_rot_deg_per_min: float = 5.0,
    smoothing_window: int = 3,
    min_segment_pings: int = 5,
    heading_offset: float = 0.0,
    use_cog: bool = False,
    progress: bool = False,
) -> Dict[str, List[I_Ping]]:
    """Split pings into *straight* and *turning* groups.

    Parameters
    ----------
    pings : list of I_Ping
        Input ping sequence (must provide geolocation).
    max_rot_deg_per_min : float, optional
        Rate-of-turn threshold in **degrees per minute**.  Pings with
        |ROT| above this value are classified as *turning*.
        Default 5.0.
    smoothing_window : int, optional
        Width of the median filter applied to the ROT series before
        thresholding.  Must be odd and ≥ 1.  Default 3.
    min_segment_pings : int, optional
        Minimum number of consecutive pings for a segment to survive.
        Shorter segments are merged into their neighbour to suppress
        chattering.  Default 5.
    heading_offset : float, optional
        Constant offset in degrees added to ``yaw`` before computing
        the heading ROT.  Default 0.
    use_cog : bool, optional
        If *True*, compute ROT from the course-over-ground (bearing
        between consecutive lat/lon positions) instead of the heading
        sensor (``yaw``).  Default *False*.
    progress : bool, optional
        Show a progress bar.  Default *False*.

    Returns
    -------
    dict with keys ``"straight"`` and ``"turning"``
        Each value is a list of pings.
    """
    result: Dict[str, List[I_Ping]] = {"straight": [], "turning": []}

    n = len(pings)
    if n == 0:
        return result
    if n == 1:
        result["straight"].append(pings[0])
        return result

    # --- Extract geolocation arrays ---
    geolocations = [p.get_geolocation() for p in pings]
    timestamps = np.array([p.get_timestamp() for p in pings], dtype=float)
    dt = np.diff(timestamps)  # seconds between consecutive pings

    # Avoid division by zero for duplicate timestamps
    dt[dt == 0] = np.nan

    # --- Compute bearing series ---
    if use_cog:
        lats = np.array([g.latitude for g in geolocations])
        lons = np.array([g.longitude for g in geolocations])
        bearings = _bearing_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])
        # COG at index i is the bearing from ping i to ping i+1
        # Pad with last value so length == n
        bearings = np.concatenate((bearings, [bearings[-1]]))
    else:
        yaw = np.array([g.yaw for g in geolocations])
        bearings = (yaw + heading_offset) % 360.0

    # --- Rate of turn (°/min) ---
    # Angular difference between consecutive bearings
    d_bearing = _angular_diff(bearings[1:], bearings[:-1])  # signed, degrees
    rot_deg_per_s = d_bearing / dt                           # °/s
    rot_deg_per_min = rot_deg_per_s * 60.0                   # °/min

    # Pad to length n (first ping gets same ROT as second)
    rot = np.concatenate(([rot_deg_per_min[0]], rot_deg_per_min))

    # Replace NaN (from dt==0) with 0
    rot = np.where(np.isfinite(rot), rot, 0.0)

    # --- Smooth ---
    rot_abs = np.abs(rot)
    rot_smooth = _median_filter_1d(rot_abs, smoothing_window)

    # --- Threshold ---
    # True = turning, False = straight
    is_turning = rot_smooth > max_rot_deg_per_min
    labels = is_turning.astype(np.int8)  # 0 = straight, 1 = turning

    # --- Merge short segments ---
    labels = _merge_short_segments(labels, min_segment_pings)

    # --- Build output ---
    for i, ping in enumerate(pings):
        if labels[i] == 1:
            result["turning"].append(ping)
        else:
            result["straight"].append(ping)

    return result


def by_turn_segments(
    pings: List[I_Ping],
    max_rot_deg_per_min: float = 5.0,
    smoothing_window: int = 3,
    min_segment_pings: int = 5,
    heading_offset: float = 0.0,
    use_cog: bool = False,
    progress: bool = False,
) -> List[Dict]:
    """Split pings into an ordered list of straight/turning segments.

    Same algorithm as :func:`by_turn`, but returns the **temporal
    order** of segments so you can see *where* each turn happens.

    Parameters
    ----------
    (same as :func:`by_turn`)

    Returns
    -------
    list of dict
        Each dict has keys ``"type"`` (``"straight"`` or ``"turning"``)
        and ``"pings"`` (list of I_Ping).  Segments appear in temporal
        order.
    """
    n = len(pings)
    if n == 0:
        return []
    if n == 1:
        return [{"type": "straight", "pings": list(pings)}]

    # Re-use the same computation logic
    geolocations = [p.get_geolocation() for p in pings]
    timestamps = np.array([p.get_timestamp() for p in pings], dtype=float)
    dt = np.diff(timestamps)
    dt[dt == 0] = np.nan

    if use_cog:
        lats = np.array([g.latitude for g in geolocations])
        lons = np.array([g.longitude for g in geolocations])
        bearings = _bearing_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])
        bearings = np.concatenate((bearings, [bearings[-1]]))
    else:
        yaw = np.array([g.yaw for g in geolocations])
        bearings = (yaw + heading_offset) % 360.0

    d_bearing = _angular_diff(bearings[1:], bearings[:-1])
    rot_deg_per_s = d_bearing / dt
    rot_deg_per_min = rot_deg_per_s * 60.0
    rot = np.concatenate(([rot_deg_per_min[0]], rot_deg_per_min))
    rot = np.where(np.isfinite(rot), rot, 0.0)

    rot_abs = np.abs(rot)
    rot_smooth = _median_filter_1d(rot_abs, smoothing_window)
    labels = (rot_smooth > max_rot_deg_per_min).astype(np.int8)
    labels = _merge_short_segments(labels, min_segment_pings)

    # Build ordered segments
    segments: List[Dict] = []
    current_label = labels[0]
    current_pings = [pings[0]]

    for i in range(1, n):
        if labels[i] == current_label:
            current_pings.append(pings[i])
        else:
            segments.append({
                "type": "turning" if current_label == 1 else "straight",
                "pings": current_pings,
            })
            current_label = labels[i]
            current_pings = [pings[i]]

    segments.append({
        "type": "turning" if current_label == 1 else "straight",
        "pings": current_pings,
    })

    return segments
