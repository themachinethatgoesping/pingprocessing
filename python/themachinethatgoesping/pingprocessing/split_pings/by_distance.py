
from collections import defaultdict
import numpy as np
import math

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
import themachinethatgoesping.navigation as nav

def by_distance_difference(
    pings, 
    meters,
    progress = False):

    it = get_progress_iterator(pings, progress, desc = "Split pings by distance difference")

    split_pings = defaultdict(list)
    number = 0

    last_geolocation_utm = None
    for ping in it:
        if last_geolocation_utm is None:
            last_geolocation_utm = nav.datastructures.GeoLocationUTM(ping.get_geolocation())
            split_pings[number].append(ping)
            continue

        g = ping.get_geolocation()
        g_utm = nav.datastructures.GeoLocationUTM(g)
        if g_utm.utm_zone == last_geolocation_utm.utm_zone:
            g_utm_compare = g_utm
        else:
            g_utm_compare = nav.datastructures.GeoLocationUTM(g, setzone=last_geolocation_utm.utm_zone)

        distance = math.dist(
            [last_geolocation_utm.easting, last_geolocation_utm.northing], 
            [g_utm_compare.easting, g_utm_compare.northing
            ])

        if distance > meters:
            number += 1

        split_pings[number].append(ping)
        last_geolocation_utm = g_utm

    return split_pings



