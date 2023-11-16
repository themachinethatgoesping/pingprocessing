
import numpy as np

from typing import List
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping


def by_time(
    pings: List[I_Ping], 
    min_timestamp: float = np.nan, 
    max_timestamp: float = np.nan,
    progress: bool = False) -> List[I_Ping]:
    """
    Filter pings by time.

    Parameters
    ----------
    pings : List[I_Ping]
        List of ping objects to be filtered.
    min_timestamp : float, optional
        Minimum timestamp value to filter pings, by default np.nan.
    max_timestamp : float, optional
        Maximum timestamp value to filter pings, by default np.nan.
    progress : bool, optional
        If True, show a progress bar, by default False.

    Returns
    -------
    List[I_Ping]
        List of ping objects filtered by time.
    """
    # Get progress iterator
    it = get_progress_iterator(pings, progress, desc = "Filter pings by time")

    # Filter pings by time
    filtered_pings = []
    for ping in it:
        t = ping.get_timestamp()
        if t < min_timestamp or t > max_timestamp:
            continue
        filtered_pings.append(ping)

    return filtered_pings



