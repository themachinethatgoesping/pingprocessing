import numpy as np

from typing import List, Tuple
from collections import defaultdict
import datetime as dt

import matplotlib as mpl
import matplotlib.dates as mdates

from tqdm.auto import tqdm

import themachinethatgoesping.tools as ptools


def get_time_error(echogram_times, ping_time):
    if not echogram_times:
        return np.nan

    if len(echogram_times) < 3:
        return np.zeros((1))

    linpace_times = np.linspace(echogram_times[0], ping_time, len(echogram_times)+1)
    tdiff = linpace_times[1] - linpace_times[0]
    
    # the time error is the difference between the expected time and the actual time
    # it is always 0 for the first and last ping in the echogram 
    time_error = np.abs(echogram_times[1:]-linpace_times[1:-1])

    return time_error/tdiff


def create_echogram_groups(
    pings: List, 
    min_num_pings_per_group: int = 2,
    max_section_size: int = 10000,
    max_time_diff_error = 0.499,
    show_progress = True,
    pbar = None) -> List[Tuple]:

    echogram_groups = [EchogramGroup()]
    echogram_times = []

    if show_progress:
        if pbar is None:
            pbar = tqdm(total=len(pings), desc="Creating echogram groups")         

    for pnr,p in enumerate(pings):
        ping_time = p.get_timestamp()
        
        if len(echogram_groups[-1]) >= min_num_pings_per_group:
            if np.max(get_time_error(echogram_times, ping_time)) > max_time_diff_error or len(echogram_groups[-1]) > max_section_size:
                echogram_times = []
                echogram_groups.append(EchogramGroup())

        echogram_times.append(ping_time)
        echogram_groups[-1].add(p, pnr)

        if show_progress:
            pbar.update(1)

    return [eg.to_tuple() for eg in echogram_groups]

class EchogramGroup():
    def __init__(self):
        self.pings = []
        self.pnr = []
        self.resolution = None

    def add(self, ping, pnr):
        self.pings.append(ping)
        self.pnr.append(pnr)
        self.resolution = ping.watercolumn.get_sound_speed_at_transducer() * ping.watercolumn.get_sample_interval() * 0.5

    def to_tuple(self):
        return np.array(self.pnr), self.resolution, self.pings

    def __len__(self):
        return len(self.pings)

