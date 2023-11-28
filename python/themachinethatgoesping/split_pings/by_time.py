
from collections import defaultdict
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping

from typing import List, Dict
from collections import defaultdict
import numpy as np

def by_time_difference(
    pings: List[I_Ping], 
    seconds: float,
    progress: bool = False) -> Dict[int, List[I_Ping]]:
    """Split pings into groups based on the time difference between them.

    Parameters
    ----------
    pings : List[I_Ping]
        List of pings to be split.
    seconds : float
        Time difference in seconds to split the pings.
    progress : bool, optional
        Flag to show progress bar, by default False.

    Returns
    -------
    Dict[int, List[I_Ping]]
        Dictionary containing the split pings.
    """
    it = get_progress_iterator(pings, progress, desc = "Split pings by time difference")

    split_pings = defaultdict(list)
    number = -1

    last_timestamp = np.nan
    for ping in it:
        t = ping.get_timestamp()

        if t < last_timestamp + seconds:
            split_pings[number].append(ping)
            last_timestamp = t
        else:
            number += 1
            split_pings[number].append(ping)
            last_timestamp = t

    return split_pings



