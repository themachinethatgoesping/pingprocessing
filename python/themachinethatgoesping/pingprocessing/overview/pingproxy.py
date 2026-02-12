# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

"""
Lightweight ping proxy objects that expose the same duck-typed interface
used by :mod:`~themachinethatgoesping.pingprocessing.filter_pings` and
:mod:`~themachinethatgoesping.pingprocessing.split_pings`, but are
backed by data stored in a :class:`PingOverview`.

This lets you **reuse every existing filter/split function** on overview
data without touching those functions at all.

Round-trip workflow::

    from themachinethatgoesping.pingprocessing.overview.pingproxy import (
        proxies_from_overview, overview_from_proxies,
    )
    from themachinethatgoesping.pingprocessing import filter_pings, split_pings

    # PingOverview → lightweight proxy list
    proxies = proxies_from_overview(overview)

    # Use any existing function unchanged
    filtered = filter_pings.by_time(proxies, min_timestamp=t0, max_timestamp=t1)
    groups   = split_pings.by_region(proxies, coordinates=coords)

    # Convert results back to PingOverview(s)
    overview_filtered = overview_from_proxies(filtered)
    overview_groups   = {k: overview_from_proxies(v) for k, v in groups.items()}

Supported ping methods
----------------------
The proxy implements exactly the methods that the filter/split functions
call on a ping:

- ``get_timestamp()``
- ``get_datetime()``
- ``get_geolocation()``  → returns a real ``GeolocationLatLon``
- ``file_data.get_primary_file_path()``
- ``file_data.get_file_paths()``

Functions that need methods not listed above (e.g. ``has_feature``,
``get_channel_id``, ``watercolumn``, ``bottom``) will not work — but
those don't make sense for overview-only data anyway.
"""

from collections import defaultdict
from typing import List

import themachinethatgoesping.navigation as nav
from .pingoverview import PingOverview


# ---------------------------------------------------------------------------
# File-data proxy  (duck-types ping.file_data)
# ---------------------------------------------------------------------------

class _FileDataProxy:
    """Mimics ``ping.file_data`` with pre-resolved path strings."""

    __slots__ = ("_primary_path", "_all_paths")

    def __init__(self, primary_path: str, all_paths: list):
        self._primary_path = primary_path
        self._all_paths = all_paths

    def get_primary_file_path(self) -> str:
        return self._primary_path

    def get_file_paths(self) -> list:
        return self._all_paths


# ---------------------------------------------------------------------------
# Ping proxy
# ---------------------------------------------------------------------------

class PingProxy:
    """Lightweight read-only proxy that quacks like ``I_Ping``.

    Constructed from a single row of :class:`PingOverview` data.  Holds
    only scalar/small values — no heavy sonar data.

    The *overview_index* attribute records which row in the source
    overview this proxy represents, so you can map results back.
    """

    __slots__ = (
        "_timestamp",
        "_datetime",
        "_latitude",
        "_longitude",
        "_yaw",
        "_pitch",
        "_roll",
        "_geolocation",
        "file_data",
        "overview_index",
    )

    def __init__(
        self,
        timestamp,
        datetime_val,
        latitude: float,
        longitude: float,
        primary_file_path: str,
        all_file_paths: list,
        overview_index: int,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ):
        self._timestamp = timestamp
        self._datetime = datetime_val
        self._latitude = latitude
        self._longitude = longitude
        self._yaw = yaw
        self._pitch = pitch
        self._roll = roll
        self._geolocation = None          # lazy
        self.file_data = _FileDataProxy(primary_file_path, all_file_paths)
        self.overview_index = overview_index

    # -- ping interface methods used by filter/split functions --

    def get_timestamp(self):
        return self._timestamp

    def get_datetime(self):
        return self._datetime

    def get_geolocation(self):
        if self._geolocation is None:
            g = nav.datastructures.GeolocationLatLon()
            g.latitude = self._latitude
            g.longitude = self._longitude
            g.yaw = self._yaw
            g.pitch = self._pitch
            g.roll = self._roll
            self._geolocation = g
        return self._geolocation

    def __repr__(self):
        return (
            f"PingProxy(t={self._timestamp}, "
            f"lat={self._latitude:.4f}, lon={self._longitude:.4f}, "
            f"idx={self.overview_index})"
        )


# ---------------------------------------------------------------------------
# Conversion: PingOverview → list[PingProxy]
# ---------------------------------------------------------------------------

def proxies_from_overview(overview: PingOverview) -> List[PingProxy]:
    """Create a list of :class:`PingProxy` objects from a PingOverview.

    Convenience wrapper around :meth:`PingOverview.to_ping_proxies`.

    Parameters
    ----------
    overview : PingOverview
        Source overview.

    Returns
    -------
    list of PingProxy
        One proxy per ping in the overview, in order.
    """
    return overview.to_ping_proxies()


# ---------------------------------------------------------------------------
# Conversion: list[PingProxy] → PingOverview
# ---------------------------------------------------------------------------

def overview_from_proxies(proxies: List[PingProxy]) -> PingOverview:
    """Reconstruct a :class:`PingOverview` from a list of :class:`PingProxy`.

    This is the inverse of :func:`proxies_from_overview`.  The returned
    overview has an ``original_indices`` attribute (list of int) mapping
    each position back to the source overview row.

    Parameters
    ----------
    proxies : list of PingProxy
        Proxy objects (typically the output of a filter/split function).

    Returns
    -------
    PingOverview
        A new overview built from the proxy data.
    """
    ov = PingOverview()

    primary_path_map: dict = {}
    primary_paths: list = []
    all_path_map: dict = {}
    all_paths: list = []

    for p in proxies:
        ov.variables["timestamp"].append(p._timestamp)
        ov.variables["datetime"].append(p._datetime)
        ov.variables["latitude"].append(p._latitude)
        ov.variables["longitude"].append(p._longitude)
        ov.variables["yaw"].append(p._yaw)
        ov.variables["pitch"].append(p._pitch)
        ov.variables["roll"].append(p._roll)

        # Primary file path (dedup)
        pfp = p.file_data.get_primary_file_path()
        if pfp not in primary_path_map:
            primary_path_map[pfp] = len(primary_paths)
            primary_paths.append(pfp)
        ov.variables["primary_file_path_index"].append(primary_path_map[pfp])

        # All file paths (dedup)
        idx_list = []
        for fp in p.file_data.get_file_paths():
            if fp not in all_path_map:
                all_path_map[fp] = len(all_paths)
                all_paths.append(fp)
            idx_list.append(all_path_map[fp])
        ov.variables["file_path_indices"].append(idx_list)

    ov._primary_file_paths = primary_paths
    ov._primary_file_path_map = primary_path_map
    ov._all_file_paths = all_paths
    ov._all_file_path_map = all_path_map

    # Store mapping back to source overview indices
    ov.original_indices = [p.overview_index for p in proxies]

    return ov
