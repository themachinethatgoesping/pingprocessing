
from collections import defaultdict
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping

from typing import List, Dict, Union
from collections import defaultdict
import numpy as np
from pytimeparse2 import parse as timeparse
import dateutil
import datetime

def into_time_blocks(
    pings: List[I_Ping], 
    timeblock_size: Union[datetime.timedelta, str, float, int],
    overlap0: Union[datetime.timedelta, str, float, int] = 0,
    overlap1: Union[datetime.timedelta, str, float, int] = 0,
    full_hour_base: bool = True,
    progress: bool = False) -> Dict[int, List[I_Ping]]:    
    """
    Splits a list of pings into time blocks.
    Args:
        pings (List[I_Ping]): List of pings to be split.
        timeblock_size (Union[datetime.timedelta, str, float, int]): Size of each time block. Can be a timedelta, string, float, or int.
        overlap0 (Union[datetime.timedelta, str, float, int], optional): Overlap time before the start of each block. Defaults to 0.
        overlap1 (Union[datetime.timedelta, str, float, int], optional): Overlap time after the end of each block. Defaults to 0.
        full_hour_base (bool, optional): If True, the first time block will start at the beginning of the hour. Defaults to True.
        progress (bool, optional): If True, displays a progress bar. Defaults to False.
    Returns:
        Dict[int, List[I_Ping]]: Dictionary where keys are the start times of the blocks and values are lists of pings in those blocks.
    """

    if not isinstance(timeblock_size, datetime.timedelta) and not isinstance(timeblock_size,dateutil.relativedelta.relativedelta):
        timeblock_size = timeparse(timeblock_size, as_timedelta=True)
    if not isinstance(overlap0, datetime.timedelta) and not isinstance(overlap0,dateutil.relativedelta.relativedelta):
        overlap0 = timeparse(overlap0, as_timedelta=True)
    if not isinstance(overlap1, datetime.timedelta) and not isinstance(overlap1,dateutil.relativedelta.relativedelta):
        overlap1 = timeparse(overlap1, as_timedelta=True)

    it = get_progress_iterator(pings, progress, desc = "Spliting pings into ping time blocks")
    
    first_time = pings[0].get_datetime()
    if full_hour_base:
        first_time = first_time.replace(minute=0, second=0, microsecond=0)

    max_time = first_time + timeblock_size + overlap1
    first_time -= overlap0
    
    split_pings = defaultdict(list)
    
    last_timestamp = np.nan
    for ping in it:
        dt = ping.get_datetime()
    
        while dt > max_time:
            first_time += timeblock_size
            max_time += timeblock_size
            
        split_pings[first_time].append(ping)
    
    return split_pings


