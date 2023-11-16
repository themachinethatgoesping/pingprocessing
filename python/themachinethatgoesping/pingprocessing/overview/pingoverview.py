from collections import defaultdict

import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator


class PingOverview(object):

    def __init__(self, ping_list = None, progress = False):
        
        self.variables = defaultdict(list)
        self.stats = defaultdict(dict)

        if ping_list is not None:
            self.add_ping_list(ping_list, progress)

    def add_ping_list(self, ping_list, progress = False):
        it = get_progress_iterator(ping_list, progress, desc = "Ping statistics")

        for ping in it:
            self.add_ping(ping)

    def add_ping(self, ping):
        self.variables['timestamp'].append(ping.get_timestamp())

        geolocation = ping.get_geolocation()
        self.variables['latitude'].append(geolocation.latitude)
        self.variables['longitude'].append(geolocation.longitude)

        stats = defaultdict(dict)
        
    def get_stat_keys(self):
        return self.stats.keys()

    def get_min(self, key):
        if 'min' not in self.stats[key]:
            self.stats[key]['min'] = np.min(self.variables[key])

        return self.stats[key]['min']

    def get_max(self, key):
        if 'max' not in self.stats[key]:
            self.stats[key]['max'] = np.max(self.variables[key])

        return self.stats[key]['max']

    def get_minmax(self, key):
        return self.get_min(key), self.get_max(key)

    def get_mean(self, key):
        if 'mean' not in self.stats[key]:
            self.stats[key]['mean'] = np.mean(self.variables[key])

        return self.stats[key]['mean']

    def get_median(self, key):
        if 'median' not in self.stats[key]:
            self.stats[key]['median'] = np.median(self.variables[key])

        return self.stats[key]['median']


def get_ping_overview(ping_list, progress = False):

    if isinstance(ping_list, dict):
        statistics = {}

        for k, pingitems in ping_list.items():
            statistics[k] = get_ping_overview(pingitems, progress)

        return statistics

    return PingOverview(ping_list, progress)

