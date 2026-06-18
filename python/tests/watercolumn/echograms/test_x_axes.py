# SPDX-FileCopyrightText: 2024 Peter Urban, Ghent University
# SPDX-License-Identifier: MPL-2.0

"""Tests for the gap-compressed time axis and the new distance x-axis.

Covers:
- ``compress_axis_gaps`` / ``cumulative_haversine_distance`` helpers
- ``set_x_axis_ping_time(max_gap=...)`` / ``set_x_axis_date_time(max_gap=...)``
- ``EchogramBuilder.set_x_axis_distance`` (incl. ``max_gap``)
- performance / monotonicity of the vectorized tie-breaking on stationary data
"""

import time as _time
import datetime as dt

import numpy as np
import pytest

from themachinethatgoesping.pingprocessing.watercolumn.echograms.backends.image_backend import (
    ImageBackend,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.echogrambuilder import (
    EchogramBuilder,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.coordinate_system import (
    compress_axis_gaps,
    cumulative_haversine_distance,
)


T0 = 1_700_000_000.0


def _make_builder(times, lats=None, lons=None, n_samples=50,
                  depth_min=0.0, depth_max=50.0):
    """Build an EchogramBuilder from a uniform-fill ImageBackend."""
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    image = np.full((n, n_samples), -50.0, dtype=np.float32)
    backend = ImageBackend.from_image(
        image, times,
        y_min=depth_min, y_max=depth_max, y_axis="depth",
        latitudes=lats, longitudes=lons,
    )
    return EchogramBuilder.from_backend(backend)


# =========================================================================
# Helper functions
# =========================================================================

class TestCompressAxisGaps:
    def test_clamps_large_gaps_keeps_small(self):
        vals = np.array([0.0, 1.0, 2.0, 1000.0, 1001.0])
        out = compress_axis_gaps(vals, max_gap=5.0)
        d = np.diff(out)
        assert out[0] == 0.0
        assert np.all(d <= 5.0 + 1e-9)
        assert np.isclose(d[0], 1.0)
        assert np.isclose(d[1], 1.0)
        assert np.isclose(d[2], 5.0)   # the 998-wide gap is clamped
        assert np.isclose(d[3], 1.0)

    def test_none_returns_copy(self):
        vals = np.array([0.0, 3.0, 9.0])
        out = compress_axis_gaps(vals, None)
        assert np.array_equal(out, vals)
        assert out is not vals

    def test_short_input(self):
        assert compress_axis_gaps(np.array([5.0]), 1.0).tolist() == [5.0]


class TestHaversine:
    def test_one_degree_latitude(self):
        d = cumulative_haversine_distance([0.0, 1.0], [0.0, 0.0])
        assert d[0] == 0.0
        # 1 deg latitude ~= 111195 m for R = 6371 km
        assert abs(d[1] - 111194.9) < 50.0

    def test_monotonic_cumulative(self):
        lat = np.linspace(0.0, 0.01, 11)
        lon = np.zeros(11)
        d = cumulative_haversine_distance(lat, lon)
        assert np.all(np.diff(d) > 0)

    def test_nan_segments_are_zero(self):
        d = cumulative_haversine_distance([0.0, np.nan, 0.001], [0.0, 0.0, 0.0])
        assert np.all(np.isfinite(d))
        assert d[0] == 0.0
        assert d[1] == 0.0  # segment touching a NaN contributes nothing


# =========================================================================
# Time axis with max_gap
# =========================================================================

class TestPingTimeMaxGap:
    def _two_survey_times(self):
        s1 = T0 + np.arange(5)            # 5 pings, 1 s spacing
        s2 = T0 + 86400 + np.arange(5)    # next day
        return np.concatenate([s1, s2])

    def test_compresses_between_surveys(self):
        eb = _make_builder(self._two_survey_times())
        eb.set_y_axis_depth()

        eb.set_x_axis_ping_time()  # no compression
        cs = eb.coord_system
        span_full = float(cs.x_extent[1]) - float(cs.x_extent[0])

        eb.set_x_axis_ping_time(max_gap=5.0)
        span_comp = float(cs.x_extent[1]) - float(cs.x_extent[0])

        assert span_full > 80000.0       # ~1 day
        assert span_comp < 100.0         # 4 + 5 + 4 s + padding
        assert span_comp < span_full

    def test_none_unchanged_regression(self):
        times = self._two_survey_times()
        eb = _make_builder(times)
        eb.set_y_axis_depth()
        eb.set_x_axis_ping_time()
        cs = eb.coord_system
        span = float(cs.x_extent[1]) - float(cs.x_extent[0])
        # real span is ~ (max - min) plus one resolution of padding
        assert span >= (times[-1] - times[0])

    def test_image_builds_with_gap(self):
        eb = _make_builder(self._two_survey_times())
        eb.set_y_axis_depth()
        eb.set_x_axis_ping_time(max_gap=5.0)
        img, ext = eb.build_image()
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_date_time_max_gap_datetime_extent(self):
        eb = _make_builder(self._two_survey_times())
        eb.set_y_axis_depth()
        eb.set_x_axis_date_time(max_gap=5.0)
        cs = eb.coord_system
        assert cs.x_axis_name == "Date time"
        assert isinstance(cs.x_extent[0], dt.datetime)
        span = (cs.x_extent[1] - cs.x_extent[0]).total_seconds()
        assert span < 100.0
        img, _ = eb.build_image()
        assert img.shape[0] > 0

    def test_date_time_timedelta_max_gap(self):
        eb = _make_builder(self._two_survey_times())
        eb.set_y_axis_depth()
        eb.set_x_axis_date_time(max_gap=dt.timedelta(seconds=5))
        cs = eb.coord_system
        span = (cs.x_extent[1] - cs.x_extent[0]).total_seconds()
        assert span < 100.0

    def test_param_datetime_translation_with_gap(self):
        """Params referenced by real time still map onto the compressed axis."""
        times = self._two_survey_times()
        eb = _make_builder(times)
        eb.set_y_axis_depth()
        eb.set_x_axis_ping_time(max_gap=5.0)
        eb.add_ping_param("bottom", "Ping time", "Depth (m)",
                          times, np.full(len(times), 25.0))
        x, y = eb.coord_system.get_ping_param("bottom")
        assert len(x) == len(times)
        assert np.all(np.isfinite(y))
        # x lives in the compressed timeline -> small span, not a full day
        assert (np.nanmax(x) - np.nanmin(x)) < 100.0


# =========================================================================
# Distance axis
# =========================================================================

class TestDistanceAxis:
    def _line_builder(self, n=20):
        times = T0 + np.arange(n)
        lats = np.linspace(0.0, 0.001 * (n - 1), n)  # ~111 m / ping north
        lons = np.zeros(n)
        return _make_builder(times, lats, lons)

    def test_basic(self):
        eb = self._line_builder(20)
        eb.set_y_axis_depth()
        eb.set_x_axis_distance()
        cs = eb.coord_system
        assert cs.x_axis_name == "Distance"
        assert cs._custom_x_format == "distance"
        ppc = np.asarray(cs._custom_x_per_ping)
        assert np.all(np.diff(ppc) > 0)
        # 0.019 deg latitude ~ 2113 m
        assert 1500.0 < ppc[-1] < 2700.0
        img, ext = eb.build_image()
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_requires_latlon(self):
        eb = _make_builder(T0 + np.arange(5))  # no navigation
        eb.set_y_axis_depth()
        with pytest.raises(RuntimeError):
            eb.set_x_axis_distance()

    def test_max_gap_clamps_transit(self):
        n1 = n2 = 10
        times = T0 + np.arange(n1 + n2)
        lat1 = np.linspace(0.0, 0.001 * (n1 - 1), n1)        # short line
        lat2 = np.linspace(1.0, 1.0 + 0.001 * (n2 - 1), n2)  # ~110 km away
        lats = np.concatenate([lat1, lat2])
        lons = np.zeros(n1 + n2)
        eb = _make_builder(times, lats, lons)
        eb.set_y_axis_depth()

        eb.set_x_axis_distance()
        full = np.asarray(eb.coord_system._custom_x_per_ping)
        jump_full = full[n1] - full[n1 - 1]

        eb.set_x_axis_distance(max_gap=50.0)
        comp = np.asarray(eb.coord_system._custom_x_per_ping)
        jump_comp = comp[n1] - comp[n1 - 1]

        assert jump_full > 1000.0
        assert jump_comp <= 50.0 + 1.0
        assert comp[-1] < full[-1]

    def test_copy_xy_axis_preserves_distance(self):
        eb = self._line_builder(15)
        eb.set_y_axis_depth()
        eb.set_x_axis_distance()
        eb2 = self._line_builder(15)
        eb2.set_y_axis_depth()
        eb.copy_xy_axis(eb2)
        assert eb2.coord_system.x_axis_name == "Distance"
        assert eb2.coord_system._custom_x_format == "distance"

    def test_stationary_is_monotonic_and_fast(self):
        """Stationary navigation must not blow up (O(n) tie-breaking)."""
        n = 40000
        times = T0 + np.arange(n)
        lats = np.zeros(n)
        lons = np.zeros(n)
        eb = _make_builder(times, lats, lons, n_samples=4)
        eb.set_y_axis_depth()
        t_start = _time.perf_counter()
        eb.set_x_axis_distance()
        elapsed = _time.perf_counter() - t_start
        ppc = np.asarray(eb.coord_system._custom_x_per_ping)
        # strictly increasing despite zero movement (ties broken vectorized)
        assert np.all(np.diff(ppc) > 0)
        # the old O(n^2) nudge would take minutes here
        assert elapsed < 20.0
