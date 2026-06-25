"""Lightweight per-ping accessor used by :meth:`EchogramBuilder.iterate_ping_data`.

``PingData`` is a thin, transient view onto one ping of an echogram. Unlike the
old implementation it carries no coordinate state of its own -- it simply
delegates to the echogram, so it can never go out of sync with the echogram's
current axes or layers.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional, Tuple

import numpy as np


class PingData:
    """Accessor for a single ping's water-column data and layer slices."""

    __slots__ = ("echogram", "nr")

    def __init__(self, echogram, nr: int):
        self.echogram = echogram
        self.nr = int(nr)

    def get_wci(self) -> np.ndarray:
        """Full processed water-column column for this ping."""
        return self.echogram.get_column(self.nr)

    def get_wci_layers(self) -> Dict[str, np.ndarray]:
        """Water-column data split by named layer."""
        return self.echogram.get_wci_layers(self.nr)

    def get_extent_layers(self, axis_name: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
        """Per-layer outer extents in ``axis_name`` (defaults to current y-axis)."""
        return self.echogram.get_extent_layers(self.nr, axis_name=axis_name)

    def get_limits_layers(self, axis_name: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
        """Per-layer inner limits in ``axis_name`` (defaults to current y-axis)."""
        return self.echogram.get_limits_layers(self.nr, axis_name=axis_name)

    def get_ping_time(self) -> float:
        """Ping timestamp (Unix seconds)."""
        return float(self.echogram._coord_system.ping_times[self.nr])

    def get_datetime(self) -> dt.datetime:
        """Ping time as a timezone-aware datetime."""
        return dt.datetime.fromtimestamp(
            self.get_ping_time(), self.echogram._coord_system.time_zone)
