from collections import OrderedDict
from typing import Dict, List
from themachinethatgoesping.echosounders import filetemplates
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

# Import necessary modules and types

I_Ping = filetemplates.I_Ping


def sort_ping_groups(ping_groups: List[Dict[str, I_Ping]]) -> List[Dict[str, I_Ping]]:
    """
    Sort ping groups by their representative timestamp.

    The WCI / echogram / map viewers read a group's time from
    ``next(iter(group.values()))`` (the first-inserted / earliest head) and then
    use ``np.searchsorted`` for time matching, which requires a monotonic array.
    Sorting on that same value keeps the per-group timestamps ordered so the
    cross-viewer time synchronisation stays stable.

    Args:
        ping_groups: List of dicts (channel id -> ping), e.g. from :func:`dual_head`.

    Returns:
        A new list with the groups ordered by ascending timestamp.
    """
    return sorted(
        ping_groups,
        key=lambda group: next(iter(group.values())).get_timestamp(),
    )


def dual_head(
    pings: List[filetemplates.I_Ping],
    progress: bool = False,
    sort: bool = True,
) -> List[Dict[str, I_Ping]]:
    """
    Group dual head pings by file/ping number.

    Args:
        pings: List of I_Ping objects.
        progress: Flag to indicate whether to show progress.
        sort: Sort the resulting groups by timestamp (recommended). Keeps the
            viewers' binary-search time matching stable. Defaults to True.

    Returns:

        list of dicts, where each dict contains pings from a single dual head grouped by the receiver id.
    """

    it = get_progress_iterator(pings, progress, desc="Group dual head pings")

    ping_groups = []
    ping_group_map = {}
    for ping in it:

        # Key on the primary file PATH (globally unique) instead of the
        # per-file-handler file number, so pings coming from different file
        # handlers cannot collide on a shared (file_nr, ping_counter) key and
        # overwrite each other (which scrambles group timestamps).
        key = (ping.file_data.get_primary_file_path(), ping.file_data.get_file_ping_counter())

        if key not in ping_group_map:
            ping_group_map[key] = len(ping_groups)
            ping_groups.append(OrderedDict())

        ping_groups[ping_group_map[key]][ping.get_channel_id()] = ping

    if sort:
        ping_groups = sort_ping_groups(ping_groups)

    return ping_groups
