from collections import defaultdict, OrderedDict
from typing import Dict, List
from themachinethatgoesping.echosounders import filetemplates 
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
import numpy as np

# Import necessary modules and types

I_Ping = filetemplates.I_Ping


def dual_head(pings: List[filetemplates.I_Ping], progress: bool = False) -> List[Dict[str, I_Ping]]:
    """
    Group dual head pings by file/ping number.

    Args:
        pings: List of I_Ping objects.
        progress: Flag to indicate whether to show progress.

    Returns:

        list of dicts, where each dict contains pings from a single dual head grouped by the receiver id.
    """

    it = get_progress_iterator(pings, progress, desc="Group dual head pings")
    
    ping_groups = []
    ping_group_map = {}
    for ping in it:
        
        key = (ping.file_data.get_primary_file_nr(), ping.file_data.get_file_ping_counter())

        if key not in ping_group_map.keys():
            ping_group_map[key] = len(ping_groups)
            ping_groups.append(OrderedDict())

        ping_groups[ping_group_map[key]][ping.get_channel_id()] = ping
        
    return ping_groups
    # get times for each group
    # TODO: this should not be necessary ... 
    # pg_times = []
    # for ping_group in ping_groups:
    #     times=[]
    #     for k,v in ping_group.items():
    #         times.append(v.get_timestamp())
    #     pg_times.append(np.nanmin(times))
              
    # return sorted by times
    # return np.array(ping_groups)[np.argsort(pg_times)]
