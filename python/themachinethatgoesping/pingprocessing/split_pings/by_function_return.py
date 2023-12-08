
from collections import defaultdict
import numpy as np
import math

from typing import Callable, List, Dict

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
import themachinethatgoesping.navigation as nav
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping

def by_function_return(
    pings: List[I_Ping], 
    function: Callable[[I_Ping], int], 
    min_length: int = 1, 
    required_features: List[str] = None, 
    progress: bool = True) -> Dict[int, List[I_Ping]]:
    """
    Splits a list of pings into multiple lists based on the return value of a given function.

    Args:
        pings (List[I_Ping]): The list of pings to be split.
        function (Callable[[I_Ping], int]): The function used to determine the key for splitting.
        min_length (int, optional): The minimum length of each split list. Defaults to 1.
        required_features (List[str], optional): The list of required features that each ping must have. Defaults to None.
        progress (bool, optional): Whether to show progress bar during the splitting process. Defaults to True.

    Returns:
        Dict[int, List[I_Ping]]: A dictionary where the keys are the return values of the function and the values are the corresponding split lists of pings.
    """

    if required_features is None:
        required_features = []

    it = get_progress_iterator(pings, progress, desc="Split pings by function return")

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
