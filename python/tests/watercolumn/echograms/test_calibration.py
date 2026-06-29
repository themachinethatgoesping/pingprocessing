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
# pool_layer_values: per-ping aggregation + the no-contribution skip
# ---------------------------------------------------------------------------


def _ramp_echo(n_pings=12, n_samples=300, dmin=0.0, dmax=30.0,
               t0=T0, t1=T0 + 660):
    """Echogram whose every sample has a distinct value (per ping & sample).

    A non-constant column makes per-block medians sensitive to exactly which
    pings/samples were pooled, so a brute-force comparison is meaningful.
    """
    times = np.linspace(t0, t1, n_pings)
    img = (np.arange(n_pings)[:, None] * 1000.0
           + np.arange(n_samples)[None, :]).astype(np.float32)
    backend = ImageBackend.from_image(img, times, y_min=dmin, y_max=dmax, y_axis="depth")
    eb = EchogramBuilder.from_backend(backend)
    eb.set_x_axis_date_time()
    eb.set_y_axis_depth()
    return eb, times


def _bruteforce_pool(eb, layer_names, block_edges, reduce=np.nanmedian):
    """Reference pooler that reads *every* in-grid ping (no skip optimisation)."""
    cs = eb._coord_system
    times = np.asarray(cs.ping_times, dtype=np.float64)
    n_blocks = len(block_edges) - 1
    block_idx = np.searchsorted(block_edges, times, side="right") - 1
    bands = [eb.get_layer_sample_indices(n) for n in layer_names]
    pools = {n: {} for n in layer_names}
    for nr in range(cs.n_pings):
        b = int(block_idx[nr])
        if 0 <= b < n_blocks:
            col = eb.get_column(nr)
            ncol = col.shape[0]
            for name, (i0a, i1a) in zip(layer_names, bands):
                i0 = int(i0a[nr])
                i1 = min(int(i1a[nr]), ncol)
                if i1 > i0:
                    pools[name].setdefault(b, []).append(col[i0:i1])
    out = {}
    for name in layer_names:
        vals = np.full(n_blocks, np.nan)
        cnts = np.zeros(n_blocks, dtype=np.int64)
        for b, ch in pools[name].items():
            p = np.concatenate(ch)
            f = np.isfinite(p)
            cnts[b] = int(f.sum())
            if f.any():
                vals[b] = reduce(p[f])
        out[name] = (vals, cnts)
    return out


def _count_reads(eb):
    """Wrap ``eb.get_column`` to count calls; returns the mutable counter list."""
    orig = eb.get_column
    counter = [0]

    def wrapper(nr):
        counter[0] += 1
        return orig(nr)

    eb.get_column = wrapper
    return counter


def _assert_pool_equal(a, b):
    assert set(a) == set(b)
    for name in a:
        va, ca = a[name]
        vb, cb = b[name]
        np.testing.assert_array_equal(ca, cb)
        np.testing.assert_allclose(va, vb, equal_nan=True)


class TestPoolLayerValuesSkip:
    def test_matches_bruteforce(self):
        eb, times = _ramp_echo()
        eb.add_layer("band", Layer.depth(5.0, 10.0))
        edges, _ = _pooling.make_time_blocks(times[0], times[-1], 120.0)
        got = eb.pool_layer_values(["band"], edges)
        ref = _bruteforce_pool(eb, ["band"], edges)
        _assert_pool_equal(got, ref)
        # sanity: at least one block actually pooled samples
        assert got["band"][1].sum() > 0

    def test_skips_out_of_grid_pings(self):
        eb, times = _ramp_echo()
        eb.add_layer("band", Layer.depth(5.0, 10.0))  # non-empty for every ping
        # Grid that only spans pings 3..8 -> pings 0..2 and 9..11 are out of grid.
        edges = np.array([times[3] - 1.0, times[6] + 1.0, times[8] + 1.0])
        counter = _count_reads(eb)
        got = eb.pool_layer_values(["band"], edges)
        in_grid = ((times >= edges[0]) & (times < edges[-1])).sum()
        assert counter[0] == in_grid
        assert counter[0] < len(times)  # genuinely skipped some reads
        _assert_pool_equal(got, _bruteforce_pool(eb, ["band"], edges))

    def test_skips_empty_band_pings(self):
        eb, times = _ramp_echo()
        n = len(times)
        # Even pings: a real depth band; odd pings: a band far beyond the data
        # (resolves to an empty sample range) -> only even pings contribute.
        lows = np.where(np.arange(n) % 2 == 0, 5.0, 500.0)
        highs = np.where(np.arange(n) % 2 == 0, 10.0, 510.0)
        eb.add_layer("band", Layer.depth(lows, highs))
        edges, _ = _pooling.make_time_blocks(times[0], times[-1], 120.0)
        counter = _count_reads(eb)
        got = eb.pool_layer_values(["band"], edges)
        assert counter[0] == (np.arange(n) % 2 == 0).sum()
        _assert_pool_equal(got, _bruteforce_pool(eb, ["band"], edges))

    def test_all_empty_reads_nothing(self):
        eb, times = _ramp_echo()
        eb.add_layer("band", Layer.depth(500.0, 510.0))  # empty for every ping
        edges, _ = _pooling.make_time_blocks(times[0], times[-1], 120.0)
        counter = _count_reads(eb)
        values, counts = eb.pool_layer_values(["band"], edges)["band"]
        assert counter[0] == 0
        assert counts.sum() == 0
        assert np.isnan(values).all()

    def test_multi_layer_reads_once_per_contributing_ping(self):
        eb, times = _ramp_echo()
        # Two non-empty bands: each contributing ping must be read exactly once.
        eb.add_layer("shallow", Layer.depth(3.0, 8.0))
        eb.add_layer("deep", Layer.depth(15.0, 22.0))
        edges, _ = _pooling.make_time_blocks(times[0], times[-1], 120.0)
        counter = _count_reads(eb)
        got = eb.pool_layer_values(["shallow", "deep"], edges)
        assert counter[0] == len(times)  # all pings contribute, read once each
        _assert_pool_equal(got, _bruteforce_pool(eb, ["shallow", "deep"], edges))


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
        # range_beam must equal the nominal layer centre (1m, 2m, ...)
        assert np.allclose(table["range_beam"].to_numpy(),
                           [5.0, 10.0, 15.0, 20.0], atol=1e-6)
        # layers defined in depth -> depth matches nominal range centre
        assert np.allclose(table["depth"].to_numpy(),
                           [5.0, 10.0, 15.0, 20.0], atol=0.3)

    def test_counts_positive(self, dataset):
        _, path = dataset
        data = CalibrationData.open(path)
        s = data.series("CH", 0.0, "10m")
        assert (s["base_count"] > 0).any()
        assert (s["beam_count"] > 0).any()


class TestGeometry:
    def test_range_to_depth_uses_beam_angle(self, tmp_path):
        # SBES base: depth==range (vertical). MBES beam: slant range, depth=R*cos30.
        cos30 = np.cos(np.deg2rad(30.0))
        base = _echo(-50.0)
        base._coord_system.set_range_extent(np.zeros(120), np.full(120, 30.0))
        beam = _echo(-60.0)
        beam._coord_system.set_range_extent(np.zeros(120), np.full(120, 30.0))
        beam._coord_system.set_depth_extent(np.zeros(120), np.full(120, 30.0 * cos30))

        b = CalibrationBuilder(tmp_path / "c", base, ranges=[10, 20], layer_size=2.0,
                               layer_reference="Range (m)", reference="Depth (m)",
                               show_progress=False)
        b.add_beam("CH", 30.0, beam, show_progress=False)
        table = b.result().calibration_per_range("CH", 30.0, bootstrap=0)
        # range_beam stays nominal; depth = R*cos30; range_base ~= depth (vertical sbes)
        assert np.allclose(table["range_beam"].to_numpy(), [10.0, 20.0], atol=1e-6)
        assert np.allclose(table["depth"].to_numpy(), [10.0 * cos30, 20.0 * cos30], atol=0.3)
        assert np.allclose(table["range_base"].to_numpy(), [10.0 * cos30, 20.0 * cos30], atol=0.5)

    def test_preview_layers_match_pooled(self, tmp_path):
        base = _echo(-50.0)
        beam = _echo(-60.0)
        b = CalibrationBuilder(tmp_path / "c", base, ranges=[5, 10], layer_size=2.0,
                               layer_reference="Depth (m)", show_progress=False)
        beam2, base2 = b.add_calibration_layers(beam)
        # the exact bands add_beam pools are now present on both echograms
        assert "5m" in beam2.layer_names() and "10m" in beam2.layer_names()
        assert "5m" in base2.layer_names() and "10m" in base2.layer_names()
        # geometry dict reports nominal range_beam centres (ref, range_beam, range_base, depth)
        assert b.last_layer_geometry["5m"][1] == pytest.approx(5.0)
        assert b.last_layer_geometry["10m"][1] == pytest.approx(10.0)

    def test_layer_values_match_calibration(self, tmp_path):
        # The calibration medians equal manual medians from get_wci_layers,
        # proving the stored values come from exactly the visible layers.
        base = _echo(-50.0)
        beam = _echo(-60.0)
        b = CalibrationBuilder(tmp_path / "c", base, ranges=[10], layer_size=2.0,
                               layer_reference="Depth (m)", show_progress=False)
        b.add_beam("CH", 0.0, beam, show_progress=False)
        beam2, base2 = b.add_calibration_layers(beam)
        samples = []
        for nr in range(beam2._coord_system.n_pings):
            samples.extend(beam2.get_wci_layers(nr)["10m"])
        manual = np.nanmedian(samples)
        table = b.result().calibration_per_range("CH", 0.0, bootstrap=0)
        # base(-50) - manual(-60) = +10; stored beam median == manual median
        assert manual == pytest.approx(-60.0, abs=0.1)
        assert np.allclose(table["csv"].to_numpy(), 10.0, atol=0.3)



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
