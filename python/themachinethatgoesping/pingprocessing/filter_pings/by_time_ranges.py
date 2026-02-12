
from datetime import datetime
from typing import List, Tuple, Union

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.echosounders import filetemplates
I_Ping = filetemplates.I_Ping

TimeValue = Union[float, datetime]
TimeRange = Tuple[TimeValue, TimeValue]


def _to_timestamp(value: TimeValue) -> float:
    """Convert a datetime or float timestamp to a unix timestamp (float)."""
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)


def _in_any_range(t: float, ranges: List[Tuple[float, float]]) -> bool:
    """Check if timestamp t falls within any of the given (start, end) ranges."""
    for start, end in ranges:
        if start <= t <= end:
            return True
    return False


def by_time_ranges(
    pings: List[I_Ping],
    time_ranges: List[TimeRange],
    exclude: bool = False,
    progress: bool = False) -> List[I_Ping]:
    """
    Filter pings by time ranges.

    By default, only pings whose timestamps fall within at least one of the
    given time ranges are returned. If ``exclude`` is True, pings inside the
    ranges are removed instead.

    Parameters
    ----------
    pings : List[I_Ping]
        List of ping objects to be filtered.
    time_ranges : List[Tuple[Union[float, datetime], Union[float, datetime]]]
        List of (start, end) tuples defining the time ranges.
        Each value may be a unix timestamp (float) or a datetime object.
    exclude : bool, optional
        If True, *exclude* pings that fall within the time ranges
        (i.e. return only pings outside all ranges). By default False.
    progress : bool, optional
        If True, show a progress bar, by default False.

    Returns
    -------
    List[I_Ping]
        Filtered list of ping objects.
    """
    # Normalise time ranges to float timestamps and sort by start time
    ranges = sorted(
        (_to_timestamp(start), _to_timestamp(end)) for start, end in time_ranges
    )

    it = get_progress_iterator(pings, progress, desc="Filter pings by time ranges")

    filtered_pings = []
    for ping in it:
        t = ping.get_timestamp()
        inside = _in_any_range(t, ranges)

        if exclude and not inside:
            filtered_pings.append(ping)
        elif not exclude and inside:
            filtered_pings.append(ping)

    return filtered_pings
