
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
import themachinethatgoesping.navigation as nav
from themachinethatgoesping.echosounders import filetemplates
from collections import defaultdict
from typing import Callable, List, Dict
import numpy as np
import math
I_Ping = filetemplates.I_Ping

def by_time_blocks(
    pings: List[I_Ping],
    times: List[float],
    progress: bool = True) -> Dict[float, List[I_Ping]]:
    """
    Split pings by time blocks.

    Args:
        pings: List of ping objects.
        times: List of time values to split the pings.
        progress: Flag to show progress bar (default: True).

    Returns:
        split_pings: Dictionary with time blocks as keys and list of pings as values.
    """

    it = get_progress_iterator(pings, progress, desc="Split pings by time blocks")

    split_pings = defaultdict(list)
    
    times = iter(sorted(times))
    next_time = next(times)
    current_time = pings[0].get_timestamp()

    for ping in it:
        ping_time = ping.get_timestamp()
        
        if ping_time >= next_time:
            current_time = next_time
            
            try:
                next_time = next(times)
            except:
                next_time = 99999999999999999999999999
            
        split_pings[current_time].append(ping)

    return split_pings