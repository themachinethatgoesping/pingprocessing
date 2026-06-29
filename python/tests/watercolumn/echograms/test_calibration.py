"""Tests for the cross-calibration subpackage.

End-to-end coverage on synthetic echograms: pooling core, the builder + on-disk
dataset, the analysis API (per-range calibration, splitting, cross data),
resumability, the curve-fit models and (if the bivariate interpolator is
available) the beam pattern.
"""

import numpy as np
import pytest

from themachinethatgoesping.pingprocessing.watercolumn.echograms.backends.image_backend import (
    ImageBackend,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.echogrambuilder import (
    EchogramBuilder,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.layers import Layer
from themachinethatgoesping.pingprocessing.watercolumn.echograms.calibration import (
    CalibrationBuilder,
    CalibrationData,
    CalibrationPattern,
    PchipBlendChangePoint,
    LogisticSTR,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.calibration import _pooling


T0 = 1_700_000_000.0


def _echo(fill, n_pings=120, n_samples=300, dmin=0.0, dmax=30.0,
          t0=T0, t1=T0 + 600, valid=(2.0, 28.0)):
    times = np.linspace(t0, t1, n_pings)
    img = np.full((n_pings, n_samples), fill, np.float32)
    backend = ImageBackend.from_image(img, times, y_min=dmin, y_max=dmax, y_axis="depth")
    eb = EchogramBuilder.from_backend(backend)
    eb.set_x_axis_date_time()
    eb.set_y_axis_depth()
    if valid is not None:
        eb.add_layer("valid", Layer.depth(valid[0], valid[1]))
    return eb


# ---------------------------------------------------------------------------
# pooling core
# ---------------------------------------------------------------------------


class TestPooling:
    def test_blocks_centered_on_minute(self):
        edges, centers = _pooling.make_time_blocks(T0, T0 + 600, 60.0)
        assert len(edges) == len(centers) + 1
        # centers sit on whole-minute ticks
        assert np.allclose(centers % 60, 0.0)
        # each center is the midpoint of its edges
        assert np.allclose(centers, 0.5 * (edges[:-1] + edges[1:]))

    def test_pool_values_bins_by_block(self):
        edges, centers = _pooling.make_time_blocks(T0, T0 + 180, 60.0)
        times = [T0 + 1, T0 + 2, T0 + 65]
        values = [10.0, 20.0, 99.0]
        pooled = _pooling.pool_values(times, values, edges)
        # first populated block holds median(10,20)=15, another holds 99
        finite = pooled[np.isfinite(pooled)]
        assert 15.0 in finite and 99.0 in finite

    def test_pool_values_length_mismatch_raises(self):
        edges, _ = _pooling.make_time_blocks(T0, T0 + 60, 60.0)
        with pytest.raises(ValueError):
            _pooling.pool_values([T0], [1.0, 2.0], edges)

    def test_pool_values_interpolates_missing_blocks(self):
        edges, centers = _pooling.make_time_blocks(T0, T0 + 180, 60.0)
        times = [centers[1], centers[3]]
        values = [10.0, 20.0]
        pooled = _pooling.pool_values(times, values, edges, centers=centers, interpolate=True)
        assert len(pooled) == len(edges) - 1
        assert np.isfinite(pooled).all()
        assert pooled[1] == pytest.approx(10.0)
        assert pooled[2] == pytest.approx(15.0)
        assert pooled[3] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# builder + data end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path):
    base = _echo(-50.0)
    beam = _echo(-60.0)
    b = CalibrationBuilder(
        tmp_path / "calib", base, base_name="BASE",
        ranges=[5, 10, 15, 20], layer_size=2.0, deltaT="1min",
        layer_reference="Depth (m)", show_progress=False,
    )
    b.add_beam("CH", 0.0, beam, show_progress=False)
    return b, tmp_path / "calib"


class TestBuilderData:
    def test_calibration_offset_is_constant(self, dataset):
        _, path = dataset
        data = CalibrationData.open(path)
        table = data.calibration_per_range("CH", 0.0, bootstrap=0)
        assert list(table["layer"]) == ["5m", "10m", "15m", "20m"]
        # base(-50) - beam(-60) = +10 dB everywhere
        assert np.allclose(table["csv"].to_numpy(), 10.0, atol=0.3)
        assert (table["n"] > 0).all()

    def test_geometry_columns(self, dataset):
        _, path = dataset
        data = CalibrationData.open(path)
        table = data.calibration_per_range("CH", 0.0, bootstrap=0)
        # layers defined in depth -> depth matches nominal range centre
        assert np.allclose(table["depth"].to_numpy(),
                           [5.0, 10.0, 15.0, 20.0], atol=0.3)

    def test_counts_positive(self, dataset):
        _, path = dataset
        data = CalibrationData.open(path)
        s = data.series("CH", 0.0, "10m")
        assert (s["base_count"] > 0).any()
        assert (s["beam_count"] > 0).any()

    def test_introspection(self, dataset):
        _, path = dataset
        data = CalibrationData.open(path)
        assert data.channels == ["CH"]
        assert data.angles("CH").tolist() == [0.0]
        assert data.base_name == "BASE"
        assert set(data.layers) == {"5m", "10m", "15m", "20m"}


# ---------------------------------------------------------------------------
# resumability
# ---------------------------------------------------------------------------


class TestResumable:
    def test_existing_beam_skipped(self, tmp_path):
        base = _echo(-50.0)
        b = CalibrationBuilder(tmp_path / "c", base, ranges=[10], layer_size=2.0,
                               layer_reference="Depth (m)", show_progress=False)
        b.add_beam("CH", 0.0, _echo(-60.0), show_progress=False)
        assert b.has_beam("CH", 0.0)
        # a second add with different data must be skipped (no overwrite)
        b.add_beam("CH", 0.0, _echo(-30.0), show_progress=False)
        data = b.result()
        table = data.calibration_per_range("CH", 0.0, bootstrap=0)
        assert np.allclose(table["csv"].to_numpy(), 10.0, atol=0.3)

    def test_reopen_after_restart(self, tmp_path):
        base = _echo(-50.0)
        b1 = CalibrationBuilder(tmp_path / "c", base, ranges=[10], layer_size=2.0,
                                layer_reference="Depth (m)", show_progress=False)
        b1.add_beam("CH", 0.0, _echo(-60.0), show_progress=False)
        # new builder on same dir resumes; existing beam present
        b2 = CalibrationBuilder(tmp_path / "c", base, ranges=[10], layer_size=2.0,
                                layer_reference="Depth (m)", show_progress=False)
        assert b2.has_beam("CH", 0.0)
        b2.add_beam("CH", 10.0, _echo(-55.0), show_progress=False)
        data = CalibrationData.open(tmp_path / "c")
        assert sorted(data.angles("CH").tolist()) == [0.0, 10.0]

    def test_reopen_keeps_param_grid_for_add_param(self, tmp_path):
        short = _echo(-50.0, t1=T0 + 120.0)
        b1 = CalibrationBuilder(tmp_path / "c", short, ranges=[10], layer_size=2.0,
                                layer_reference="Depth (m)", show_progress=False)
        b1.add_beam("CH", 0.0, _echo(-60.0, t1=T0 + 120.0), show_progress=False)

        # Reopen with a much larger extent; persisted params grid must win.
        long = _echo(-50.0, t1=T0 + 3600.0 * 4.0)
        b2 = CalibrationBuilder(tmp_path / "c", long, ranges=[10], layer_size=2.0,
                                layer_reference="Depth (m)", show_progress=False)
        times = np.linspace(T0, T0 + 3600.0 * 4.0, 40)
        values = np.linspace(1.0, 5.0, 40)
        b2.add_param("focus_range", times, values)

        data = b2.result()
        assert len(data.params["focus_range"]) == len(data.params)


# ---------------------------------------------------------------------------
# params + splitting
# ---------------------------------------------------------------------------


class TestParamsSplit:
    def test_param_and_split(self, tmp_path):
        base = _echo(-50.0)
        b = CalibrationBuilder(tmp_path / "c", base, ranges=[10], layer_size=2.0,
                               layer_reference="Depth (m)", show_progress=False)
        b.add_beam("CH", 0.0, _echo(-60.0), show_progress=False)
        times = np.linspace(T0, T0 + 600, 50)
        temp = np.linspace(0.0, 10.0, 50)
        b.add_param("temperature", times, temp)
        data = b.result()
        assert "temperature" in data.params.columns

        parts = data.split_by_param("temperature", [("lo", -1, 5), ("hi", 5, 20)])
        assert set(parts) == {"lo", "hi"}
        # each split still yields the same +10 offset
        for part in parts.values():
            t = part.calibration_per_range("CH", 0.0, bootstrap=0)
            if len(t):
                assert np.allclose(t["csv"].to_numpy(), 10.0, atol=0.5)

    def test_intervals_label(self, tmp_path):
        base = _echo(-50.0)
        b = CalibrationBuilder(tmp_path / "c", base, ranges=[10], layer_size=2.0,
                               layer_reference="Depth (m)", show_progress=False)
        b.add_beam("CH", 0.0, _echo(-60.0), show_progress=False)
        b.add_intervals("station", [("S1", T0, T0 + 300), ("S2", T0 + 300, T0 + 600)])
        data = b.result()
        parts = data.split_by_label("station")
        assert set(parts) >= {"S1", "S2"}


# ---------------------------------------------------------------------------
# cross data
# ---------------------------------------------------------------------------


class TestCrossData:
    def test_cross_data_columns(self, dataset):
        _, path = dataset
        data = CalibrationData.open(path)
        cross = data.cross_data("CH", 0.0, "10m")
        assert set(["base", "beam", "diff", "inlier"]).issubset(cross.columns)
        good = cross[cross["inlier"]]
        assert np.allclose(good["diff"].to_numpy(), 10.0, atol=0.3)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class TestModels:
    def test_pchip_constant_tail(self):
        x = np.linspace(1, 30, 30)
        y = np.where(x < 10, 20 - (10 - x), 20.0)  # ramps up then constant 20
        m = PchipBlendChangePoint(blend_width=0.5, window_frac=0.5,
                                  mode="constant").fit(x, y)
        assert m(25.0) == pytest.approx(20.0, abs=1.0)
        xg, yg = m.get_fit(0, 30, 31)
        assert len(xg) == len(yg) == 31

    def test_logistic_str_runs(self):
        x = np.linspace(1, 30, 40)
        y = np.where(x < 12, 10 + 0.5 * x, 16.0)
        m = LogisticSTR(tail="constant").fit(x, y)
        assert np.isfinite(m(20.0))


# ---------------------------------------------------------------------------
# pattern (needs the bivariate interpolator from tools)
# ---------------------------------------------------------------------------


class TestPattern:
    def _multi_angle_dataset(self, tmp_path):
        base = _echo(-50.0)
        b = CalibrationBuilder(tmp_path / "c", base, ranges=[5, 10, 15, 20, 25],
                               layer_size=2.0, layer_reference="Depth (m)",
                               show_progress=False)
        for angle, fill in [(0.0, -60.0), (10.0, -59.0), (20.0, -58.0)]:
            b.add_beam("CH", angle, _echo(fill), show_progress=False)
        return b.result()

    def test_pattern_fit_and_eval(self, tmp_path):
        pytest.importorskip(
            "themachinethatgoesping.tools.vectorinterpolators.bivectorinterpolators")
        data = self._multi_angle_dataset(tmp_path)
        pattern = CalibrationPattern(x="depth", min_n=1, min_points=4,
                                     range_grid=(0.0, 30.0, 31)).fit(
            data, show_progress=False, bootstrap=0)
        assert "CH" in pattern.channels()
        surf = pattern.evaluate("CH", [0.0, 10.0], [10.0, 20.0])
        assert np.asarray(surf).shape == (2, 2)
        angles, offsets = pattern.far_field("CH", far_range=25.0)
        assert len(angles) == len(offsets) >= 3
