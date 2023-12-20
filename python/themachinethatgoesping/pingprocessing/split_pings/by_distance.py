
from collections import defaultdict
import numpy as np
import math
from typing import Dict, List, Union
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
import themachinethatgoesping.navigation as nav
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping

def by_distance_difference(
    pings: List[I_Ping], 
    meters: float,
    progress: bool = False) -> Dict[int, List[I_Ping]]:
    """
    Split pings into groups based on the distance between consecutive pings.

    Parameters
    ----------
    pings : List[I_Ping]
        List of pings to be split.
    meters : float
        Distance threshold in meters.
    progress : bool, optional
        If True, show progress bar, by default False.

    Returns
    -------
    Dict[int, List[I_Ping]]]
        A dictionary with keys as group numbers and values as lists of pings belonging to that group.
    """
    it = get_progress_iterator(pings, progress, desc = "Split pings by distance difference")

    split_pings = defaultdict(list)
    number = 0

    last_geolocation_utm = None
    for ping in it:
        if last_geolocation_utm is None:
            last_geolocation_utm = nav.datastructures.GeolocationUTM(ping.get_geolocation())
            split_pings[number].append(ping)
            continue

        g = ping.get_geolocation()
        g_utm = nav.datastructures.GeolocationUTM(g)
        if g_utm.utm_zone == last_geolocation_utm.utm_zone:
            g_utm_compare = g_utm
        else:
            g_utm_compare = nav.datastructures.GeolocationUTM(g, setzone=last_geolocation_utm.utm_zone)

        distance = math.dist(
            [last_geolocation_utm.easting, last_geolocation_utm.northing], 
            [g_utm_compare.easting, g_utm_compare.northing]
            )

        if distance > meters:
            number += 1

        split_pings[number].append(ping)
        last_geolocation_utm = g_utm

    return split_pings



