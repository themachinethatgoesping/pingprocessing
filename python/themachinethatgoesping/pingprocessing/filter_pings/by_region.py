
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping

def by_latlon(
    pings: list[I_Ping], 
    min_lat: float = np.nan, 
    max_lat: float = np.nan,
    min_lon: float = np.nan,
    max_lon: float = np.nan,
    progress: bool = False) -> list[I_Ping]:
    """
    Filter pings by latitude and longitude region.

    Parameters
    ----------
    pings : list
        List of pings to filter.
    min_lat : float, optional
        Minimum latitude value, by default np.nan.
    max_lat : float, optional
        Maximum latitude value, by default np.nan.
    min_lon : float, optional
        Minimum longitude value, by default np.nan.
    max_lon : float, optional
        Maximum longitude value, by default np.nan.
    progress : bool, optional
        Whether to show progress bar, by default False.

    Returns
    -------
    list
        List of filtered pings.
    """

    it = get_progress_iterator(pings, progress, desc="Filter pings by lat/lon region")

    filtered_pings = []

    for ping in it:
        g = ping.get_geolocation()
        if g.latitude < min_lat or g.latitude > max_lat:
            continue

        if g.longitude < min_lon or g.longitude > max_lon:
            continue

        filtered_pings.append(ping)

    return filtered_pings



