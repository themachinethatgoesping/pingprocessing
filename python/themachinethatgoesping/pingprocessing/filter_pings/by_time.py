
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

def by_time(
    pings, 
    min_timestamp = np.nan, 
    max_timestamp = np.nan,
    progress = False):

    if isinstance(pings, dict):
        for k, pingitems in pings.items():
            pings[k] = by_time(pingitems, min_timestamp, max_timestamp, progress)

        return pings

    it = get_progress_iterator(pings, progress, desc = "Filter pings by time")

    filtered_pings = []

    for ping in it:
        t = ping.get_timestamp()
        if t < min_timestamp or t > max_timestamp:
            continue

        filtered_pings.append(ping)

    return filtered_pings



