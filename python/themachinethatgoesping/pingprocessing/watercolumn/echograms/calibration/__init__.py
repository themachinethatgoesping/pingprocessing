"""Cross-calibration of echograms: build, analyse, model.

This subpackage replaces the old ``LayerProcessor`` workflow with a clean,
fast, resumable pipeline:

* :class:`CalibrationBuilder` -- extract a base-vs-beam comparison from
  echograms into a compact, append-able on-disk dataset (Parquet).
* :class:`CalibrationData` -- open such a dataset and analyse it (per-range
  calibration with bootstrap CIs, cross-plot data, cheap splitting by station
  or environmental parameter).
* :class:`CalibrationPattern` -- fit a per-channel ``offset(angle, range)``
  surface and apply it back onto multibeam pings.
* :mod:`models` / :mod:`plotting` -- curve-fit models and plotting helpers.

Quick start
-----------
::

    from themachinethatgoesping.pingprocessing.watercolumn.echograms.calibration import (
        CalibrationBuilder, CalibrationData, CalibrationPattern)

    b = CalibrationBuilder('calib.dir', sbes_echo, base_name='EK80',
                           ranges=range(1, 31), deltaT='1min')
    b.add_beam('TRX-2004', 0.0, mbes_beam_echo)
    b.add_param('temperature', times, temps)

    data = CalibrationData.open('calib.dir')
    table = data.calibration_per_range('TRX-2004', 0.0)

    pattern = CalibrationPattern().fit(data)
"""

from .builder import CalibrationBuilder
from .data import CalibrationData, CalibrationStore
from .pattern import CalibrationPattern
from .models import CalibrationModel, PchipBlendChangePoint, LogisticSTR
from . import plotting

__all__ = [
    "CalibrationBuilder",
    "CalibrationData",
    "CalibrationStore",
    "CalibrationPattern",
    "CalibrationModel",
    "PchipBlendChangePoint",
    "LogisticSTR",
    "plotting",
]
