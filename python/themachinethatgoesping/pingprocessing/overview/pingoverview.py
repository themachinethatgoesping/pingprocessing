from collections import defaultdict

import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.navigation.navtools import cumulative_latlon_distances_m


from typing import List
from collections import defaultdict
import numpy as np

from . import nav_plot


class PingOverview:
    """
    A class to represent an overview of ping statistics.

    Attributes
    ----------
    variables : defaultdict
        A dictionary containing lists of variables.
    stats : defaultdict
        A dictionary containing statistics for each variable.

    Methods
    -------
    __init__(self, ping_list=None, progress=False)
        Constructs a PingOverview object.
    add_ping_list(self, ping_list, progress=False)
        Adds a list of pings to the overview.
    add_ping(self, ping)
        Adds a single ping to the overview.
    get_stat_keys(self)
        Returns the keys of the statistics dictionary.
    get_min(self, key)
        Returns the minimum value of a variable.
    get_max(self, key)
        Returns the maximum value of a variable.
    get_minmax(self, key)
        Returns the minimum and maximum values of a variable.
    get_mean(self, key)
        Returns the mean value of a variable.
    get_median(self, key)
        Returns the median value of a variable.
    """

    def __init__(self, ping_list: List = None, progress: bool = False) -> None:
        """
        Constructs a PingOverview object.

        Parameters
        ----------
        ping_list : List, optional
            A list of pings to add to the overview, by default None
        progress : bool, optional
            Whether to show a progress bar, by default False
        """
        self.variables = defaultdict(list)
        self.stats = defaultdict(dict)
        self._file_paths = []       # unique file paths (small list)
        self._file_path_map = {}    # path → index (for fast lookup during add)

        if ping_list is not None:
            self.add_ping_list(ping_list, progress)

    def plot_navigation(self, ax, label="survey", annotate=True, max_points=100000, **kwargs):
        """
        Plot latitude and longitude coordinates on a given axis.

        Parameters:
            ax (matplotlib.axes.Axes): The axis on which to plot the coordinates.
            label (str, optional): Name of the survey. Defaults to 'survey'.
            annotate (bool, optional): Whether to annotate the plot with the survey name. Defaults to True.
            max_points (int, optional): Maximum number of points to plot. Defaults to 100000.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            None
        """
        nav_plot.plot_latlon(
            self.variables["latitude"],
            self.variables["longitude"],
            ax=ax,
            label=label,
            annotate=annotate,
            max_points=max_points,
            **kwargs
        )
        
    def get_speed_in_knots(self):
        dt = np.array(self.variables['timestamp'])-self.variables['timestamp'][0]
        dd = cumulative_latlon_distances_m(self.variables['latitude'],self.variables['longitude'])
        speed_m_per_s = dd[1:] / dt[1:]
        speed_knots = speed_m_per_s * 1.94384
        return np.concatenate((speed_knots, [speed_knots[-1]]))
        
    def plot_speed_in_knots(self, ax, **kwargs):
        """
        Plot speed in knots over time on a given axis.

        Parameters:
            ax (matplotlib.axes.Axes): The axis on which to plot the speed.
            **kwargs: Additional keyword arguments to be passed to the plot function.
        Returns:
            None
        """
        speed_knots = self.get_speed_in_knots()
        ax.plot(self.variables['datetime'], speed_knots, **kwargs)
        ax.set_ylabel('Speed (knots)')
        ax.set_xlabel('DateTime')

    def get_ping_rate_hz(self):
        """
        Compute the ping rate in Hz (pings per second).

        Returns:
            np.ndarray: Array of ping rates in Hz for each ping interval.
        """
        timestamps = np.array(self.variables['timestamp'])
        dt = np.diff(timestamps)
        ping_rate_hz = 1.0 / dt
        return np.concatenate(([ping_rate_hz[0]], ping_rate_hz))

    def plot_ping_rate_hz(self, ax, **kwargs):
        """
        Plot ping rate in Hz over time on a given axis.

        Parameters:
            ax (matplotlib.axes.Axes): The axis on which to plot the ping rate.
            **kwargs: Additional keyword arguments to be passed to the plot function.
        Returns:
            None
        """
        ping_rate_hz = self.get_ping_rate_hz()
        ax.plot(self.variables['datetime'], ping_rate_hz, **kwargs)
        ax.set_ylabel('Ping Rate (Hz)')
        ax.set_xlabel('DateTime')

    def add_ping_list(self, ping_list: List, progress: bool = False) -> None:
        """
        Adds a list of pings to the overview.

        Parameters
        ----------
        ping_list : List
            A list of pings to add to the overview.
        progress : bool, optional
            Whether to show a progress bar, by default False
        """
        it = get_progress_iterator(ping_list, progress, desc="Ping statistics")

        for ping in it:
            self.add_ping(ping)

    def add_ping(self, ping) -> None:
        """
        Adds a single ping to the overview.

        Parameters
        ----------
        ping : Ping
            A ping to add to the overview.
        """
        self.variables["timestamp"].append(ping.get_timestamp())
        self.variables["datetime"].append(ping.get_datetime())

        geolocation = ping.get_geolocation()
        self.variables["latitude"].append(geolocation.latitude)
        self.variables["longitude"].append(geolocation.longitude)

        file_path = ping.file_data.get_primary_file_path()
        if file_path not in self._file_path_map:
            self._file_path_map[file_path] = len(self._file_paths)
            self._file_paths.append(file_path)
        self.variables["file_path_index"].append(self._file_path_map[file_path])

        stats = defaultdict(dict)

    def get_stat_keys(self) -> List:
        """
        Returns the keys of the statistics dictionary.

        Returns
        -------
        List
            The keys of the statistics dictionary.
        """
        return self.stats.keys()

    def get_min(self, key: str) -> float:
        """
        Returns the minimum value of a variable.

        Parameters
        ----------
        key : str
            The name of the variable.

        Returns
        -------
        float
            The minimum value of the variable.
        """
        if "min" not in self.stats[key]:
            self.stats[key]["min"] = np.min(self.variables[key])

        return self.stats[key]["min"]

    def get_max(self, key: str) -> float:
        """
        Returns the maximum value of a variable.

        Parameters
        ----------
        key : str
            The name of the variable.

        Returns
        -------
        float
            The maximum value of the variable.
        """
        if "max" not in self.stats[key]:
            self.stats[key]["max"] = np.max(self.variables[key])

        return self.stats[key]["max"]

    def get_minmax(self, key: str) -> tuple:
        """
        Returns the minimum and maximum values of a variable.

        Parameters
        ----------
        key : str
            The name of the variable.

        Returns
        -------
        tuple
            The minimum and maximum values of the variable.
        """
        return self.get_min(key), self.get_max(key)

    def get_mean(self, key: str) -> float:
        """
        Returns the mean value of a variable.

        Parameters
        ----------
        key : str
            The name of the variable.

        Returns
        -------
        float
            The mean value of the variable.
        """
        if "mean" not in self.stats[key]:
            self.stats[key]["mean"] = np.mean(self.variables[key])

        return self.stats[key]["mean"]

    def get_median(self, key: str) -> float:
        """
        Returns the median value of a variable.

        Parameters
        ----------
        key : str
            The name of the variable.

        Returns
        -------
        float
            The median value of the variable.
        """
        if "median" not in self.stats[key]:
            self.stats[key]["median"] = np.median(self.variables[key])

        return self.stats[key]["median"]

    def get_file_paths(self) -> List[str]:
        """
        Return the list of unique file paths.

        Returns
        -------
        List[str]
            Unique file paths referenced by pings in this overview.
        """
        return self._file_paths

    def get_file_path(self, ping_index: int) -> str:
        """
        Return the file path for a specific ping by its index.

        Parameters
        ----------
        ping_index : int
            Index of the ping in this overview.

        Returns
        -------
        str
            The file path of the ping.
        """
        return self._file_paths[self.variables["file_path_index"][ping_index]]

    def get_pings_per_file_path(self) -> dict:
        """
        Return a mapping from file path to list of ping indices.

        Returns
        -------
        dict
            Dictionary mapping file_path → list of ping indices.
        """
        result = defaultdict(list)
        for i, idx in enumerate(self.variables["file_path_index"]):
            result[self._file_paths[idx]].append(i)
        return dict(result)

    def _get_minmax_per_file(self, key: str) -> dict:
        """
        Return min and max of *key* grouped by file path.

        Parameters
        ----------
        key : str
            Variable name (e.g. 'timestamp', 'latitude').

        Returns
        -------
        dict
            ``{file_path: (min_val, max_val), …}``
        """
        vals = self.variables[key]
        result = {}
        for fp, indices in self.get_pings_per_file_path().items():
            file_vals = [vals[i] for i in indices]
            result[fp] = (min(file_vals), max(file_vals))
        return result

    def get_timestamp_range_per_file(self) -> dict:
        """
        Return min/max timestamp per file path.

        Returns
        -------
        dict
            ``{file_path: (min_timestamp, max_timestamp), …}``
        """
        return self._get_minmax_per_file("timestamp")

    def get_datetime_range_per_file(self) -> dict:
        """
        Return min/max datetime per file path.

        Returns
        -------
        dict
            ``{file_path: (min_datetime, max_datetime), …}``
        """
        return self._get_minmax_per_file("datetime")

    def get_latitude_range_per_file(self) -> dict:
        """
        Return min/max latitude per file path.

        Returns
        -------
        dict
            ``{file_path: (min_latitude, max_latitude), …}``
        """
        return self._get_minmax_per_file("latitude")

    def get_longitude_range_per_file(self) -> dict:
        """
        Return min/max longitude per file path.

        Returns
        -------
        dict
            ``{file_path: (min_longitude, max_longitude), …}``
        """
        return self._get_minmax_per_file("longitude")


from typing import Dict, List, Union


def get_ping_overview(
    ping_list: Union[Dict[str, List[float]], List[float]], progress: bool = False
) -> Union[Dict[str, Dict[str, Union[float, int]]], "PingOverview"]:
    """
    Returns a summary of ping statistics for a list of pings.

    If the input is a dictionary, the function will return a dictionary with the same keys and a summary of the pings for each key.
    If the input is a list, the function will return a PingOverview object with a summary of the pings.

    Parameters
    ----------
    ping_list : Union[Dict[str, List[float]], List[float]]
        A dictionary or list of pings to summarize.
    progress : bool, optional
        Whether to display a progress bar while processing, by default False.

    Returns
    -------
    Union[Dict[str, Dict[str, Union[float, int]]], 'PingOverview']
        A dictionary or PingOverview object with a summary of the pings.
    """

    if isinstance(ping_list, dict):
        statistics = {}

        for k, pingitems in ping_list.items():
            statistics[k] = get_ping_overview(pingitems, progress)

        return statistics

    return PingOverview(ping_list, progress)
