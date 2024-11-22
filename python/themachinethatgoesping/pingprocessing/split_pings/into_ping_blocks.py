
from collections import defaultdict
import numpy as np

import themachinethatgoesping as theping
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping

from typing import List, Dict, Union
from collections import defaultdict
import numpy as np
from pytimeparse2 import parse as timeparse
import dateutil

def into_ping_blocks(
    pings: List[I_Ping], 
    block_size: int,
    max_ping_time_difference=None,
    overlap0: int = 0,
    overlap1: int = 0,
    progress: bool = False) -> Dict[int, List[I_Ping]]:
    """
    Splits a list of pings into blocks of a specified size, with optional overlaps and time difference constraints.
    Args:
        pings (List[I_Ping]): List of pings to be split into blocks.
        block_size (int): The size of each block.
        max_ping_time_difference (Optional[int]): Maximum allowed time difference between pings in a block. 
            If None, all pings are considered to be in the same time block.
        overlap0 (int, optional): Number of pings to overlap at the start of each block. Defaults to 0.
        overlap1 (int, optional): Number of pings to overlap at the end of each block. Defaults to 0.
        progress (bool, optional): If True, displays a progress bar. Defaults to False.
    Returns:
        Dict[int, List[I_Ping]]: A dictionary where keys are block numbers and values are lists of pings in each block.
    """
    
    if max_ping_time_difference is None:
       ping_time_blocks = {0 : pings} 
    else:
        ping_time_blocks = theping.pingprocessing.split_pings.by_time_difference(pings,max_ping_time_difference)

    it = get_progress_iterator(pings, progress, desc = "Split pings into ping blocks")
    
    split_pings = defaultdict(list)
    block_nr = 0

    for pingblock in ping_time_blocks.values():
        for i in range(0, len(pingblock), block_size):
            i0 = i - overlap0
            if i0 < 0:
                i0 = 0

            i1 = i + block_size + overlap1
            if i1 > len(pingblock):
                i1 = len(pingblock)

            if i1 == i0:
                break

            split_pings[block_nr] = pingblock[i0:i1]
            block_nr += 1

        # new block for next pingblock
        block_nr += 1
    
    return split_pings


