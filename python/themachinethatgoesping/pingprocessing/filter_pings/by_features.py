
import numpy as np
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

from typing import List
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping

def by_features(
    pings: List[I_Ping], 
    features: List[str], 
    progress: bool = False) -> List[I_Ping]:
    """
    Filter pings by features.

    Parameters
    ----------
    pings : List[Ping]
        List of pings to filter.
    features : List[str]
        List of features to filter by.
    progress : bool, optional
        Whether to show progress bar, by default False.

    Returns
    -------
    List[Ping]
        List of filtered pings.
    """

    it = get_progress_iterator(pings, progress, desc = "Filter pings by features")

    base_features        = set()
    watercolumn_features = set()
    bottom_features      = set()
    for f in features:
        if f.startswith("watercolumn."):
            watercolumn_features.add(f.removeprefix("watercolumn."))
            base_features.add('watercolumn')
        elif f.startswith("bottom."):
            bottom_features.add(f.removeprefix("bottom."))
            base_features.add('bottom')
        else:
            base_features.add(f)

    filtered_pings = []

    for ping in it:
        
        skip = False
        for f in base_features:
            if not ping.has_feature(f):
                skip = True
                break
        else:
            for f in watercolumn_features:
                if not ping.watercolumn.has_feature(f):
                    skip = True
                    break
            else:
                for f in bottom_features:
                    if not ping.bottom.has_feature(f):
                        skip = True
                        break

        if skip:
            continue

        filtered_pings.append(ping)

    return filtered_pings



