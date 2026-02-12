
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.navigation.navtools import compute_latlon_distance_m
from themachinethatgoesping.echosounders import filetemplates

I_Ping = filetemplates.I_Ping


def by_region(
    pings: List[I_Ping],
    coordinates: Sequence[Tuple[float, float]],
    max_distance_m: Optional[Union[float, Sequence[float]]] = None,
    progress: bool = False,
) -> Dict[int, List[I_Ping]]:
    """Split pings by assigning each to the nearest reference coordinate.

    Each ping is assigned to the coordinate (by index) it is closest to.
    Pings that exceed *max_distance_m* from every coordinate are collected
    under key ``-1``.

    Parameters
    ----------
    pings : list of I_Ping
        Pings to split.
    coordinates : sequence of (lat, lon)
        Reference coordinates.  Each element is a ``(latitude, longitude)``
        tuple in degrees.
    max_distance_m : float or sequence of float, optional
        Maximum allowed distance in metres.

        - If a single float, the same limit applies to all coordinates.
        - If a sequence, must have the same length as *coordinates* and
          gives a per-coordinate limit.
        - If ``None`` (default), no distance limit is applied and every
          ping is assigned to its nearest coordinate.
    progress : bool, optional
        Show a progress bar, by default False.

    Returns
    -------
    dict[int, list[I_Ping]]
        Keys are coordinate indices (0 … len(coordinates)−1).
        Key ``-1`` holds pings that were too far from every coordinate
        (only present when *max_distance_m* is set and some pings are
        excluded).
    """
    if len(coordinates) == 0:
        raise ValueError("coordinates must not be empty.")

    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "coordinates must be a sequence of (lat, lon) pairs, "
            f"got shape {coords.shape}."
        )
    n_coords = len(coords)

    # Normalise max_distance_m to an array (one limit per coordinate).
    if max_distance_m is None:
        limits = None
    elif np.isscalar(max_distance_m):
        limits = np.full(n_coords, float(max_distance_m))
    else:
        limits = np.asarray(max_distance_m, dtype=float)
        if limits.shape != (n_coords,):
            raise ValueError(
                f"max_distance_m sequence length ({len(limits)}) must match "
                f"the number of coordinates ({n_coords})."
            )

    it = get_progress_iterator(pings, progress, desc="Split pings by region")

    split_pings: Dict[int, List[I_Ping]] = defaultdict(list)

    for ping in it:
        g = ping.get_geolocation()
        plat, plon = g.latitude, g.longitude

        best_idx = -1
        best_dist = np.inf

        for ci in range(n_coords):
            d = compute_latlon_distance_m(
                plat, plon, coords[ci, 0], coords[ci, 1]
            )
            if d < best_dist:
                best_dist = d
                best_idx = ci

        if limits is not None and best_dist > limits[best_idx]:
            split_pings[-1].append(ping)
        else:
            split_pings[best_idx].append(ping)

    return split_pings
