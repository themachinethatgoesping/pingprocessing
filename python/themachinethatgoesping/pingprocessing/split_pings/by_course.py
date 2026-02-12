"""Split pings by comparing course-over-ground (COG) with true heading.

Use this to separate pings recorded while the vessel is **on course**
from those where it is drifting, manoeuvring, or sailing too slowly for
the heading to match the track direction.
"""

import math
from collections import defaultdict
from typing import Dict, List

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

from themachinethatgoesping.echosounders import filetemplates

I_Ping = filetemplates.I_Ping


# ──────────────────────────── helpers ────────────────────────────

def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial (forward) bearing from point 1 → point 2 in degrees [0, 360).

    Uses the standard spherical azimuth formula.
    North = 0°, East = 90°, South = 180°, West = 270°.

    Returns *NaN* when the two positions coincide.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)

    x = math.sin(d_lambda) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda))

    if x == 0.0 and y == 0.0:
        return float("nan")

    return math.degrees(math.atan2(x, y)) % 360.0


def _angular_distance(a: float, b: float) -> float:
    """Unsigned shortest angular distance between two bearings in degrees.

    Always returns a value in [0, 180].  Handles the 0°/360° wrap
    correctly via the identity ``|((a − b + 180) mod 360) − 180|``.
    """
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ──────────────────────────── public API ─────────────────────────

def by_course(
    pings: List[I_Ping],
    tolerance: float = 30.0,
    heading_offset: float = 0.0,
    min_distance_m: float = 0.0,
    progress: bool = False,
) -> Dict[str, List[I_Ping]]:
    """Split pings into *on_course* vs *off_course* groups.

    The **course-over-ground (COG)** is the bearing (north = 0°,
    east = 90°) from each ping's position to the next ping's position.
    It is compared with ``yaw + heading_offset``.  If the angular
    difference exceeds *tolerance* the ping is placed in the
    ``"off_course"`` group; otherwise it goes into ``"on_course"``.

    Parameters
    ----------
    pings : list of I_Ping
        Input ping sequence (must provide geolocation with lat/lon and
        yaw = true heading in the same 0°-north-clockwise convention).
    tolerance : float, optional
        Maximum allowed angular difference in **degrees** between COG
        and adjusted heading.  Default 30.
    heading_offset : float, optional
        Constant offset (degrees) added to the true heading (``yaw``)
        before comparison.  Useful for known mounting biases.
        Default 0.
    min_distance_m : float, optional
        Minimum great-circle distance (metres) between consecutive
        pings.  When two pings are closer than this the COG is
        considered unreliable and the ping is placed in
        ``"on_course"`` unconditionally.  Default 0 (always check).
    progress : bool, optional
        Show a progress bar.  Default *False*.

    Returns
    -------
    dict with keys ``"on_course"`` and ``"off_course"``
        Each value is a list of pings.
    """
    result: Dict[str, List[I_Ping]] = {"on_course": [], "off_course": []}

    if len(pings) == 0:
        return result

    if len(pings) == 1:
        result["on_course"].append(pings[0])
        return result

    it = get_progress_iterator(pings, progress, desc="Split pings by course")

    # Pre-fetch geolocations so we can look ahead
    geolocations = [p.get_geolocation() for p in pings]
    n = len(pings)

    use_min_dist = min_distance_m > 0.0
    if use_min_dist:
        from themachinethatgoesping.navigation.navtools import compute_latlon_distance_m

    for i, ping in enumerate(it):
        geo = geolocations[i]

        # For the last ping, reuse decision of the previous ping
        if i >= n - 1:
            if result["on_course"] and result["on_course"][-1] is pings[i - 1]:
                result["on_course"].append(ping)
            else:
                result["off_course"].append(ping)
            continue

        geo_next = geolocations[i + 1]

        # Optional minimum-distance gate
        if use_min_dist:
            dist = compute_latlon_distance_m(
                geo.latitude, geo.longitude,
                geo_next.latitude, geo_next.longitude,
            )
            if dist < min_distance_m:
                result["on_course"].append(ping)
                continue

        cog = _bearing_deg(
            geo.latitude, geo.longitude,
            geo_next.latitude, geo_next.longitude,
        )

        if math.isnan(cog):
            # Identical positions → keep as on-course
            result["on_course"].append(ping)
            continue

        adjusted_heading = (geo.yaw + heading_offset) % 360.0
        diff = _angular_distance(cog, adjusted_heading)

        if diff <= tolerance:
            result["on_course"].append(ping)
        else:
            result["off_course"].append(ping)

    return result
