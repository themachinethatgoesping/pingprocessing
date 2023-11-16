
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

def by_latlon(
    pings, 
    min_lat = np.nan, 
    max_lat = np.nan,
    min_lon = np.nan,
    max_lon = np.nan,
    progress = False):

    if isinstance(pings, dict):
        for k, pingitems in pings.items():
            pings[k] = by_time(pingitems, min_timestamp, max_timestamp, progress)

        return pings

    it = get_progress_iterator(pings, progress, desc = "Filter pings by lat/lon region")

    filtered_pings = []

    for ping in it:
        g = ping.get_geolocation()
        if g.latitude < min_lat or g.latitude > max_lat:
            continue

        if g.longitude < min_lon or g.longitude > max_lon:
            continue

        filtered_pings.append(ping)

    return filtered_pings



