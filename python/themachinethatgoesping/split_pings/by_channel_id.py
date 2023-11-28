
from collections import defaultdict
import numpy as np
import math
from typing import Dict, List
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
import themachinethatgoesping.navigation as nav
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping


def by_channel_id(pings: List[I_Ping], progress: bool = False) -> Dict[int, List[I_Ping]]:
    """
    Splits a list of pings by channel id.

    Args:
        pings (List[I_Ping]): A list of pings to be split.
        progress (bool, optional): Whether to show a progress bar. Defaults to False.

    Returns:
        Dict[int, List[I_Ping]]: A dictionary where the keys are channel ids and the values are lists of pings.
    """
    it = get_progress_iterator(pings, progress, desc="Split pings by channel id")

    split_pings = defaultdict(list)

    for ping in it:
        split_pings[ping.get_channel_id()].append(ping)

    return split_pings
def by_channel_id(
    pings: List[I_Ping], 
    progress: bool = False) -> Dict[int, List[I_Ping]]:
    
    it = get_progress_iterator(pings, progress, desc = "Split pings by channel id")

    split_pings = defaultdict(list)

    for ping in it:
        split_pings[ping.get_channel_id()].append(ping)

    return split_pings



