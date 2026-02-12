
import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.navigation.navtools import compute_latlon_distance_m

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


def by_distance(
    pings: list[I_Ping],
    center_lat: float,
    center_lon: float,
    max_distance_m: float,
    progress: bool = False,
) -> list[I_Ping]:
    """Filter pings by distance from a center coordinate.

    Keeps only pings whose position is within *max_distance_m* metres
    of the given (latitude, longitude) center.

    Parameters
    ----------
    pings : list
        List of pings to filter.
    center_lat : float
        Center latitude in degrees.
    center_lon : float
        Center longitude in degrees.
    max_distance_m : float
        Maximum distance from the center in metres.
    progress : bool, optional
        Whether to show a progress bar, by default False.

    Returns
    -------
    list
        Pings within *max_distance_m* of the center.
    """
    it = get_progress_iterator(
        pings, progress, desc="Filter pings by distance"
    )

    filtered_pings = []
    for ping in it:
        g = ping.get_geolocation()
        d = compute_latlon_distance_m(
            center_lat, center_lon, g.latitude, g.longitude
        )
        if d <= max_distance_m:
            filtered_pings.append(ping)

    return filtered_pings

