
from collections import defaultdict
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

def by_time_difference(
    pings, 
    seconds,
    progress = False):

    it = get_progress_iterator(pings, progress, desc = "Split pings by time difference")

    split_pings = defaultdict(list)
    number = 0

    last_timestamp = np.nan
    for ping in it:
        t = ping.get_timestamp()

        if t < last_timestamp + seconds:
            split_pings[number].append(ping)
        else:
            split_pings[number].append(ping)
            number += 1
            last_timestamp = t

    return split_pings



