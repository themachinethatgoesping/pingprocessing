
import numpy as np

from typing import List
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping


def by_time_list(
    pings: List[I_Ping],
    times: List[float], 
    max_time_diff_seconds: float = 10,
    progress: bool = False) -> List[I_Ping]:
    """
    Filter pings by time.

    Parameters
    ----------
    pings : List[I_Ping]
        List of ping objects to be filtered.
    times : List[float]
        List of timestamps.
    max_time_diff_seconds : float
        Maximum difference in seconds between valid ping and closest timestamp.
    progress : bool, optional
        If True, show a progress bar, by default False.

    Returns
    -------
    List[I_Ping]
        List of ping objects filtered by time.
    """
    # Get progress iterator
    it = get_progress_iterator(pings, progress, desc = "Filter pings by time")

    # Filter pings by timestamps
    filtered_pings = []
    times_index = 0
    for ping in it:
        t = ping.get_timestamp()
        while times_index < len(times) and t - times[times_index] > max_time_diff_seconds:
            times_index += 1
        if times_index >= len(times):
            break
        if abs(t - times[times_index]) <= max_time_diff_seconds:
            filtered_pings.append(ping)
            times_index += 1

    return filtered_pings

def by_ping_times(
    pings: List[I_Ping],
    reference_pings: List[I_Ping], 
    max_time_diff_seconds: float = 10,
    progress: bool = False) -> List[I_Ping]:
    """
    Filter pings by time.

    Parameters
    ----------
    pings : List[I_Ping]
        List of ping objects to be filtered.
    reference_pings : List[I_Ping]
        List of reference ping objects.
    max_time_diff_seconds : float
        Maximum difference in seconds between valid ping and closest reference ping.
    progress : bool, optional
        If True, show a progress bar, by default False.

    Returns
    -------
    List[I_Ping]
        List of ping objects filtered by time.
    """
    # Get progress iterator
    it = get_progress_iterator(pings, progress, desc = "Filter pings by time")

    # Filter pings by timestamps
    filtered_pings = []
    reference_pings_index = 0
    for ping in it:
        t = ping.get_timestamp()
        while reference_pings_index < len(reference_pings) and t-reference_pings[reference_pings_index].get_timestamp() > max_time_diff_seconds:
            reference_pings_index += 1
        if reference_pings_index >= len(reference_pings):
            print(reference_pings_index,len(reference_pings))
            break
        if abs(t - reference_pings[reference_pings_index].get_timestamp()) <= max_time_diff_seconds:
            filtered_pings.append(ping)
            reference_pings_index += 1

    return filtered_pings