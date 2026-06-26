"""Echogram layer system: portable region-of-interest specs and aggregation.

Public API
----------
* :class:`Layer` / :class:`Boundary` -- portable, echogram-independent layer specs.
* :class:`LayerStore` / :class:`ResolvedBand` -- per-echogram resolution + caching.
* :class:`PingData` -- lightweight per-ping accessor.
* :func:`transfer_layer` / :func:`transfer_layers` -- move layers between echograms
  through a shared physical reference (depth by default).

For pooling layer samples into time blocks and cross-calibrating echograms, see
the sibling :mod:`..calibration` subpackage.
"""

from .layer import Boundary, Layer, REFERENCES
from .store import LayerStore, ResolvedBand
from .pingdata import PingData
from .transfer import transfer_layer, transfer_layers

__all__ = [
    "Boundary",
    "Layer",
    "REFERENCES",
    "LayerStore",
    "ResolvedBand",
    "PingData",
    "transfer_layer",
    "transfer_layers",
]
