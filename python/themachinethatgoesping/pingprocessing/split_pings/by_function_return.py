
from collections import defaultdict
import numpy as np
import math
from typing import Dict, List
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
import themachinethatgoesping.navigation as nav
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping


def by_function_return(pings: List[I_Ping], function, min_length=1, required_features = None, progress: bool = True) -> Dict[int, List[I_Ping]]:

    if required_features is None:
        required_features = []

    it = get_progress_iterator(pings, progress, desc="Split pings by channel id")

    split_pings = defaultdict(list)

    for ping in it:
        if not ping.has_all_of_features(required_features):
            continue
        key = function(ping)
        split_pings[key].append(ping)


    del_keys = []
    for key, pings in split_pings.items():
        if len(pings) < min_length:
            del_keys.append(key)

    for key in del_keys:
        try:
            del split_pings[key]
        except Exception as e:
            #print(e)
            pass

    return split_pings
