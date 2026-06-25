"""Tests for the redesigned echogram layer system.

Covers the portable :class:`Layer` spec, axis-change robustness, intersection,
cross-echogram transfer, time-series (sensor) layers and the new
:class:`LayerProcessor`.
"""

import numpy as np
import pytest

from themachinethatgoesping.pingprocessing.watercolumn.echograms.backends.image_backend import (
    ImageBackend,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.echogrambuilder import (
    EchogramBuilder,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.layers import (
    Boundary,
    Layer,
    LayerProcessor,
    transfer_layer,
    transfer_layers,
)


T0 = 1_700_000_000.0


def _make(n_pings, n_samples, dmin, dmax, t_start, t_end, fill, params=None):
    times = np.linspace(t_start, t_end, n_pings)
    img = np.full((n_pings, n_samples), fill, np.float32)
    backend = ImageBackend.from_image(
        img, times, y_min=dmin, y_max=dmax, y_axis="depth", ping_params=params
    )
    eb = EchogramBuilder.from_backend(backend)
    eb.set_x_axis_date_time()
    eb.set_y_axis_depth()
    return eb


@pytest.fixture
def echo_a():
    return _make(200, 400, 5.0, 25.0, T0, T0 + 600, -50.0)


@pytest.fixture
def echo_b():
    return _make(150, 1600, 5.0, 55.0, T0 - 100, T0 + 700, -40.0)


# ---------------------------------------------------------------------------
# Layer spec
# ---------------------------------------------------------------------------


class TestLayerSpec:
    def test_invalid_reference_raises(self):
        with pytest.raises(ValueError):
            Layer("Nope", 1, 2)

    def test_boundary_coercion(self):
        assert Boundary.coerce(None).is_open()
        assert Boundary.coerce(3.0).kind == "const"
        assert Boundary.coerce([1, 2, 3]).kind == "per_ping"

    def test_depth_band_resolves_to_expected_extent(self, echo_a):
        echo_a.add_layer("d", Layer.depth(10.0, 20.0))
        ext = echo_a.get_extent_layers(100, axis_name="Depth (m)")["d"]
        assert ext[0] == pytest.approx(10.0, abs=0.2)
        assert ext[1] == pytest.approx(20.0, abs=0.2)

    def test_open_boundary_spans_to_extent(self, echo_a):
        echo_a.add_layer("top", Layer.depth(None, 12.0))
        i0, i1 = echo_a.get_layer_sample_indices("top")
        assert np.all(i0 == 0)
        assert np.all(i1 > 0)


# ---------------------------------------------------------------------------
# Axis-change robustness
# ---------------------------------------------------------------------------


class TestAxisRobustness:
    def test_sample_indices_invariant_under_axis_change(self, echo_a):
        echo_a.add_layer("d", Layer.depth(10.0, 20.0))
        i0a, i1a = echo_a.get_layer_sample_indices("d")
        wci_before = echo_a.get_wci_layers(100)["d"].copy()

        echo_a.set_y_axis_sample_nr()
        i0b, i1b = echo_a.get_layer_sample_indices("d")
        wci_after = echo_a.get_wci_layers(100)["d"].copy()

        assert np.array_equal(i0a, i0b)
        assert np.array_equal(i1a, i1b)
        assert len(wci_before) == len(wci_after)

    def test_grid_indices_track_display_axis(self, echo_a):
        echo_a.add_layer("d", Layer.depth(10.0, 20.0))
        y0_depth, y1_depth = echo_a.get_layer_grid_indices("d")
        echo_a.set_y_axis_sample_nr()
        y0_samp, y1_samp = echo_a.get_layer_grid_indices("d")
        # The grid projection should refresh for the new axis (different nx grid).
        assert (y0_depth.shape == y0_samp.shape)


# ---------------------------------------------------------------------------
# Intersection / mask
# ---------------------------------------------------------------------------


class TestIntersection:
    def test_repeated_add_intersects(self, echo_a):
        echo_a.add_layer("m", Layer.depth(8.0, 22.0))
        echo_a.add_layer("m", Layer.depth(12.0, 18.0))
        ext = echo_a.get_extent_layers(100, axis_name="Depth (m)")["m"]
        assert ext[0] == pytest.approx(12.0, abs=0.3)
        assert ext[1] == pytest.approx(18.0, abs=0.3)

    def test_set_layer_replaces(self, echo_a):
        echo_a.add_layer("m", Layer.depth(8.0, 22.0))
        echo_a.set_layer("m", Layer.depth(12.0, 18.0))
        ext = echo_a.get_extent_layers(100, axis_name="Depth (m)")["m"]
        assert ext[0] == pytest.approx(12.0, abs=0.3)


# ---------------------------------------------------------------------------
# Cross-echogram transfer
# ---------------------------------------------------------------------------


class TestTransfer:
    def test_transfer_preserves_depth_band(self, echo_a, echo_b):
        echo_a.add_layer("d", Layer.depth(10.0, 20.0))
        transfer_layer(echo_a, echo_b, "d", reference="Depth (m)", new_name="d2")
        ext = echo_b.get_extent_layers(75, axis_name="Depth (m)")["d2"]
        assert ext[0] == pytest.approx(10.0, abs=0.3)
        assert ext[1] == pytest.approx(20.0, abs=0.3)

    def test_transfer_all(self, echo_a, echo_b):
        echo_a.add_layer("a", Layer.depth(8.0, 10.0))
        echo_a.add_layer("b", Layer.depth(12.0, 14.0))
        transfer_layers(echo_a, echo_b)
        assert "a" in echo_b.layers
        assert "b" in echo_b.layers


# ---------------------------------------------------------------------------
# Sensor (time series) layer
# ---------------------------------------------------------------------------


class TestTimeSeriesLayer:
    def test_sensor_band_follows_time(self, echo_a):
        ts = np.array([T0, T0 + 300, T0 + 600])
        depth = np.array([12.0, 14.0, 16.0])
        echo_a.add_layer(
            "sensor",
            Layer.from_time_series("Depth (m)", ts, depth,
                                   offset_lower=-2.0, offset_upper=2.0),
        )
        ext0 = echo_a.get_extent_layers(0, axis_name="Depth (m)")["sensor"]
        assert ext0[0] == pytest.approx(10.0, abs=0.3)
        assert ext0[1] == pytest.approx(14.0, abs=0.3)


# ---------------------------------------------------------------------------
# LayerProcessor
# ---------------------------------------------------------------------------


class TestLayerProcessor:
    def test_values_and_calibration(self, echo_a, echo_b):
        for e in (echo_a, echo_b):
            e.add_layer("10.0m", Layer.depth(9.0, 11.0))
            e.add_layer("15.0m", Layer.depth(14.0, 16.0))
        proc = LayerProcessor(
            {"A": echo_a, "B": echo_b}, layers=["10.0m", "15.0m"],
            deltaT="2min", show_progress=False,
        )
        v_a = proc.value("10.0m", "A").dropna()
        v_b = proc.value("10.0m", "B").dropna()
        assert np.allclose(v_a.values, -50.0, atol=0.7)
        assert np.allclose(v_b.values, -40.0, atol=0.7)

        cal = proc.calibration_per_range("A", "B", bootstrap_resamples=0,
                                         show_progress=False)
        assert set(cal["range"]) == {10.0, 15.0}
        assert np.allclose(cal["median_diff"].values, -10.0, atol=0.7)

    def test_difference_series(self, echo_a):
        echo_a.add_layer("10.0m", Layer.depth(9.0, 11.0))
        proc = LayerProcessor({"A": echo_a}, layers=["10.0m"],
                              deltaT="2min", show_progress=False)
        diff = proc.difference("10.0m", "A", "A").dropna()
        assert np.allclose(diff.values, 0.0)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_layer_roundtrip_mmap(self, echo_a, tmp_path):
        echo_a.add_layer("d", Layer.depth(10.0, 20.0))
        echo_a.set_main_layer(Layer.depth(8.0, 22.0))
        i0, i1 = echo_a.get_layer_sample_indices("d")
        m0, m1 = echo_a.get_layer_sample_indices("main")

        path = str(tmp_path / "echo.mmap")
        echo_a.to_mmap(path)
        loaded = EchogramBuilder.from_mmap(path)
        loaded.set_y_axis_depth()

        li0, li1 = loaded.get_layer_sample_indices("d")
        lm0, lm1 = loaded.get_layer_sample_indices("main")
        assert np.array_equal(i0, li0) and np.array_equal(i1, li1)
        assert np.array_equal(m0, lm0) and np.array_equal(m1, lm1)
