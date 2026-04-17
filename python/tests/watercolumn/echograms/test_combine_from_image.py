"""Tests for EchogramBuilder.combine using from_image (ImageBackend).

Tests cover:
- Basic combine with nanmean/nandiff
- Different time ranges (partial overlap, no overlap, SBES starts later)
- Different sample counts and depth ranges
- Layer propagation through combine
- Time alignment with different ping rates
- NaN handling when one backend has no data in a time region
"""

import numpy as np
import pytest

from themachinethatgoesping.pingprocessing.watercolumn.echograms.backends.image_backend import (
    ImageBackend,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.echogrambuilder_new import (
    EchogramBuilder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(
    n_pings,
    n_samples,
    t_start,
    t_end,
    depth_min,
    depth_max,
    fill_value=-50.0,
    y_axis="depth",
):
    """Create a simple ImageBackend with uniform fill."""
    times = np.linspace(t_start, t_end, n_pings)
    image = np.full((n_pings, n_samples), fill_value, dtype=np.float32)
    return ImageBackend.from_image(
        image,
        times,
        y_min=depth_min,
        y_max=depth_max,
        y_axis=y_axis,
    )


def _make_builder(backend, max_steps=1024):
    """Wrap a backend into an EchogramBuilder with date-time x and depth y."""
    eb = EchogramBuilder.from_backend(backend)
    eb.set_x_axis_date_time(max_steps=max_steps)
    eb.set_y_axis_depth()
    return eb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def t0():
    """Common base time."""
    return 1_679_241_600.0


@pytest.fixture
def mbes_builder(t0):
    """MBES-like builder: 500 pings, 834 samples, depth 5-25 m."""
    backend = _make_backend(
        n_pings=500,
        n_samples=834,
        t_start=t0,
        t_end=t0 + 600,  # 10 min
        depth_min=5.0,
        depth_max=25.0,
        fill_value=-50.0,
    )
    return _make_builder(backend)


@pytest.fixture
def sbes_builder(t0):
    """SBES-like builder: 300 pings, 5505 samples, depth 5-55 m, wider time."""
    backend = _make_backend(
        n_pings=300,
        n_samples=5505,
        t_start=t0 - 200,
        t_end=t0 + 800,  # wider than MBES
        depth_min=5.0,
        depth_max=55.0,
        fill_value=-40.0,
    )
    return _make_builder(backend)


# =========================================================================
# Basic combine tests
# =========================================================================


class TestCombineBasic:
    """Basic combine operations with overlapping backends."""

    def test_combine_nanmean_shape(self, mbes_builder, sbes_builder):
        """Combined image has same x-size as first builder, y-size matches max_steps."""
        ec = EchogramBuilder.combine(
            [mbes_builder, sbes_builder], combine_func="nanmean", linear=False
        )
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()
        img_c, _ = ec.build_image()
        img_m, _ = mbes_builder.build_image()
        # x-size (pings) should match since combined uses first backend's pings
        assert img_c.shape[0] == img_m.shape[0]
        # y-size may differ since combined depth range is the union of all backends
        assert img_c.shape[1] > 0

    def test_combine_nanmean_overlap_values(self, t0):
        """In the overlap region, nanmean(-50, -40) == -45."""
        # Both cover the same time range
        b1 = _make_backend(100, 200, t0, t0 + 100, 5.0, 25.0, fill_value=-50.0)
        b2 = _make_backend(100, 200, t0, t0 + 100, 5.0, 25.0, fill_value=-40.0)
        eb1 = _make_builder(b1)
        eb2 = _make_builder(b2)

        ec = EchogramBuilder.combine([eb1, eb2], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()
        img, _ = ec.build_image()

        # All valid pixels should be close to -45
        valid = np.isfinite(img)
        assert valid.sum() > 0
        np.testing.assert_allclose(img[valid], -45.0, atol=0.5)

    def test_combine_nandiff_overlap_values(self, t0):
        """In the overlap region, nandiff = first - second."""
        b1 = _make_backend(100, 200, t0, t0 + 100, 5.0, 25.0, fill_value=-50.0)
        b2 = _make_backend(100, 200, t0, t0 + 100, 5.0, 25.0, fill_value=-40.0)
        eb1 = _make_builder(b1)
        eb2 = _make_builder(b2)

        ec = EchogramBuilder.combine([eb1, eb2], combine_func="nandiff", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()
        img, _ = ec.build_image()

        valid = np.isfinite(img)
        assert valid.sum() > 0
        np.testing.assert_allclose(img[valid], -10.0, atol=0.5)

    def test_combine_uses_first_backend_as_reference(self, mbes_builder, sbes_builder):
        """Combined backend should use first backend's ping times and count."""
        ec = EchogramBuilder.combine(
            [mbes_builder, sbes_builder], combine_func="nanmean", linear=False
        )
        assert ec.backend.n_pings == mbes_builder.backend.n_pings
        np.testing.assert_array_equal(
            ec.backend.ping_times, mbes_builder.backend.ping_times
        )


# =========================================================================
# Time alignment tests
# =========================================================================


class TestCombineTimeAlignment:
    """Ensure correct time alignment between backends with different ping rates."""

    def test_sbes_nan_before_start_combined_matches_mbes(self, t0):
        """When SBES has NaN before T1, combined == MBES before T1."""
        n_mbes, n_sbes = 500, 500
        t_mbes = np.linspace(t0, t0 + 600, n_mbes)
        t_sbes = np.linspace(t0 - 200, t0 + 800, n_sbes)

        T1 = t0 + 300  # half-way through MBES

        img_mbes = np.full((n_mbes, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 200), np.nan, dtype=np.float32)
        img_sbes[t_sbes >= T1, :] = -40.0  # SBES valid only after T1

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=5.0, y_max=25.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_m, _ = eb_m.build_image()
        img_c, _ = ec.build_image()

        # Before T1: combined must match MBES
        t1_idx = np.argmin(np.abs(t_mbes - T1))
        before = img_c[:t1_idx, :]
        mbes_before = img_m[:t1_idx, :]
        valid = np.isfinite(before) & np.isfinite(mbes_before)
        assert valid.sum() > 0, "No valid pixels before T1"
        np.testing.assert_allclose(
            before[valid], mbes_before[valid], atol=0.5,
            err_msg="Before T1 combined should match MBES (SBES is NaN)"
        )

    def test_sbes_nan_before_start_combined_after_t1(self, t0):
        """After T1, combined should reflect both backends."""
        n_mbes, n_sbes = 500, 500
        t_mbes = np.linspace(t0, t0 + 600, n_mbes)
        t_sbes = np.linspace(t0 - 200, t0 + 800, n_sbes)

        T1 = t0 + 300

        img_mbes = np.full((n_mbes, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 200), np.nan, dtype=np.float32)
        img_sbes[t_sbes >= T1, :] = -40.0

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=5.0, y_max=25.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_c, _ = ec.build_image()

        # After T1: combined should be ~-45 (nanmean of -50 and -40)
        t1_idx = np.argmin(np.abs(t_mbes - T1))
        # Skip first few columns after T1 for transition
        after = img_c[t1_idx + 5:, :]
        valid = np.isfinite(after)
        assert valid.sum() > 0, "No valid pixels after T1"
        # Most should be -45.0 (some edge effects possible)
        assert np.sum(np.abs(after[valid] - (-45.0)) < 1.0) > 0.8 * valid.sum(), \
            "After T1 most values should be nanmean(-50,-40)=-45"

    def test_different_ping_rates_all_matched(self, t0):
        """With adaptive tolerance, different ping rates still match."""
        # MBES: fast ping rate (0.5 sec/ping)
        n_mbes = 1000
        t_mbes = np.linspace(t0, t0 + 500, n_mbes)

        # SBES: slow ping rate (2 sec/ping)
        n_sbes = 500
        t_sbes = np.linspace(t0, t0 + 500, n_sbes)  # same time range, fewer pings

        img_mbes = np.full((n_mbes, 100), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 100), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=5.0, y_max=25.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_c, _ = ec.build_image()
        valid = np.isfinite(img_c)
        assert valid.sum() > 0

        # With adaptive tolerance, most columns should be combined (-45)
        # and very few should be MBES-only (-50)
        values = img_c[valid]
        n_combined = np.sum(np.abs(values - (-45.0)) < 1.0)
        n_mbes_only = np.sum(np.abs(values - (-50.0)) < 1.0)
        assert n_combined > 0.9 * len(values), \
            f"Expected >90% combined, got {n_combined}/{len(values)} ({100*n_combined/len(values):.0f}%)"

    def test_no_overlap_combined_is_first_backend(self, t0):
        """When backends don't overlap in time, combined == first backend."""
        b_m = _make_backend(200, 100, t0, t0 + 200, 5.0, 25.0, -50.0)
        b_s = _make_backend(200, 100, t0 + 500, t0 + 700, 5.0, 25.0, -40.0)  # no overlap
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_m, _ = eb_m.build_image()
        img_c, _ = ec.build_image()

        valid = np.isfinite(img_c) & np.isfinite(img_m)
        assert valid.sum() > 0
        np.testing.assert_allclose(
            img_c[valid], img_m[valid], atol=0.5,
            err_msg="No overlap: combined should equal first backend"
        )


# =========================================================================
# Depth alignment tests
# =========================================================================


class TestCombineDepthAlignment:
    """Verify depth-to-sample mapping works correctly when combining."""

    def test_different_depth_ranges_gradient(self, t0):
        """Two backends with different depth ranges and gradient images align by depth."""
        n = 200
        times = np.linspace(t0, t0 + 200, n)

        # MBES: 500 samples, depth 5-25m, gradient -50 (top) to -30 (bottom)
        img_m = np.linspace(-50, -30, 500)[np.newaxis, :].repeat(n, axis=0).astype(np.float32)
        # SBES: 2000 samples, depth 5-55m, gradient -45 (top) to -15 (bottom)
        img_s = np.linspace(-45, -15, 2000)[np.newaxis, :].repeat(n, axis=0).astype(np.float32)

        b_m = ImageBackend.from_image(img_m, times, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_s, times, y_min=5.0, y_max=55.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_c, _ = ec.build_image()
        y_grid = ec.coord_system.y_coordinates

        # At depth 15m (both have data): check the combined value
        d15_idx = np.argmin(np.abs(y_grid - 15.0))
        mid_x = img_c.shape[0] // 2

        # MBES at 15m: frac = (15-5)/(25-5) = 0.5, val = -50 + 20*0.5 = -40
        # SBES at 15m: frac = (15-5)/(55-5) = 0.2, val = -45 + 30*0.2 = -39
        # nanmean = (-40 + -39) / 2 = -39.5
        expected = (-40.0 + -39.0) / 2
        actual = img_c[mid_x, d15_idx]
        assert abs(actual - expected) < 1.0, \
            f"At depth 15m: expected ~{expected:.1f}, got {actual:.1f}"

    def test_sbes_only_deep_region(self, t0):
        """Below MBES depth range, only SBES data should appear."""
        n = 200
        times = np.linspace(t0, t0 + 200, n)

        img_m = np.full((n, 100), -50.0, dtype=np.float32)
        img_s = np.full((n, 500), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_m, times, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_s, times, y_min=5.0, y_max=55.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_c, _ = ec.build_image()
        y_grid = ec.coord_system.y_coordinates

        # At depth 40m (SBES-only): should be -40
        d40_idx = np.argmin(np.abs(y_grid - 40.0))
        mid_x = img_c.shape[0] // 2
        val = img_c[mid_x, d40_idx]
        assert abs(val - (-40.0)) < 1.0, \
            f"At depth 40m (SBES only): expected ~-40, got {val:.1f}"

    def test_very_different_sample_counts(self, t0):
        """Backends with very different sample counts (833 vs 5504) combine correctly."""
        n = 200
        times = np.linspace(t0, t0 + 200, n)

        # Match user's exact parameters
        img_m = np.full((n, 834), -50.0, dtype=np.float32)
        img_s = np.full((n, 5505), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_m, times, y_min=4.6, y_max=25.6, y_axis="depth")
        b_s = ImageBackend.from_image(img_s, times, y_min=4.8, y_max=53.8, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_c, _ = ec.build_image()
        y_grid = ec.coord_system.y_coordinates

        mid_x = img_c.shape[0] // 2

        # At depth 15m (overlap): nanmean(-50, -40) = -45
        d15_idx = np.argmin(np.abs(y_grid - 15.0))
        assert abs(img_c[mid_x, d15_idx] - (-45.0)) < 0.5

        # At depth 40m (SBES only): -40
        d40_idx = np.argmin(np.abs(y_grid - 40.0))
        assert abs(img_c[mid_x, d40_idx] - (-40.0)) < 0.5


# =========================================================================
# Layer tests
# =========================================================================


class TestCombineWithLayers:
    """Test layer management when combining echograms."""

    def test_combine_starts_with_no_layers(self, t0):
        """Combined echogram should start with empty layers."""
        b1 = _make_backend(200, 100, t0, t0 + 200, 5.0, 25.0, -50.0)
        b2 = _make_backend(200, 100, t0, t0 + 200, 5.0, 25.0, -40.0)
        eb1 = _make_builder(b1)
        eb2 = _make_builder(b2)

        # Add a layer to the first builder
        eb1.add_layer_from_static_layer("bottom", 20.0, 25.0)

        ec = EchogramBuilder.combine(
            [eb1, eb2], combine_func="nanmean", linear=False
        )
        # Layers are NOT copied during combine
        assert len(ec.layers) == 0
        assert ec.main_layer is None

    def test_can_add_layers_after_combine(self, t0):
        """Layers can be added to the combined echogram after creation."""
        b1 = _make_backend(200, 100, t0, t0 + 200, 5.0, 25.0, -50.0)
        b2 = _make_backend(200, 100, t0, t0 + 200, 5.0, 25.0, -40.0)
        eb1 = _make_builder(b1)
        eb2 = _make_builder(b2)

        ec = EchogramBuilder.combine([eb1, eb2], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        # Add a static layer
        ec.add_layer_from_static_layer("wc", 8.0, 20.0)
        assert "wc" in ec.layers

    def test_main_layer_restricts_combined_image(self, t0):
        """Main layer should restrict which samples appear in the combined image."""
        n = 200
        times = np.linspace(t0, t0 + 200, n)

        # Gradient images so we can verify layer boundaries
        img1 = np.linspace(-60, -30, 200)[np.newaxis, :].repeat(n, axis=0).astype(np.float32)
        img2 = np.linspace(-55, -25, 200)[np.newaxis, :].repeat(n, axis=0).astype(np.float32)

        b1 = ImageBackend.from_image(img1, times, y_min=5.0, y_max=25.0, y_axis="depth")
        b2 = ImageBackend.from_image(img2, times, y_min=5.0, y_max=25.0, y_axis="depth")
        eb1 = _make_builder(b1)
        eb2 = _make_builder(b2)

        ec = EchogramBuilder.combine([eb1, eb2], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        # Set main layer: only depths 10-20m
        ec.add_layer_from_static_layer("main", 10.0, 20.0)

        img_full, _ = ec.build_image()
        img_layer, layer_img, _ = ec.build_image_and_layer_image()

        y_grid = ec.coord_system.y_coordinates

        # Outside layer (e.g., depth 7m): main-layer image should have NaN
        d7_idx = np.argmin(np.abs(y_grid - 7.0))
        d15_idx = np.argmin(np.abs(y_grid - 15.0))
        mid_x = img_layer.shape[0] // 2

        if d7_idx < img_layer.shape[1]:
            assert np.isnan(img_layer[mid_x, d7_idx]) or img_layer[mid_x, d7_idx] == np.nan, \
                "Outside main layer should be NaN"

        # Inside layer (depth 15m): should have valid data
        if d15_idx < img_layer.shape[1]:
            assert np.isfinite(img_layer[mid_x, d15_idx]), \
                "Inside main layer should have valid data"

    def test_named_layer_image(self, t0):
        """Named layers produce correct layer images."""
        n = 200
        times = np.linspace(t0, t0 + 200, n)

        img1 = np.full((n, 200), -50.0, dtype=np.float32)
        img2 = np.full((n, 200), -40.0, dtype=np.float32)

        b1 = ImageBackend.from_image(img1, times, y_min=5.0, y_max=25.0, y_axis="depth")
        b2 = ImageBackend.from_image(img2, times, y_min=5.0, y_max=25.0, y_axis="depth")
        eb1 = _make_builder(b1)
        eb2 = _make_builder(b2)

        ec = EchogramBuilder.combine([eb1, eb2], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        # Add a named layer for the bottom region
        ec.add_layer_from_static_layer("bottom_layer", 18.0, 25.0)

        img_main, layer_images, _ = ec.build_image_and_layer_images()

        assert "bottom_layer" in layer_images
        layer_img = layer_images["bottom_layer"]

        y_grid = ec.coord_system.y_coordinates
        d20_idx = np.argmin(np.abs(y_grid - 20.0))
        d8_idx = np.argmin(np.abs(y_grid - 8.0))
        mid_x = img_main.shape[0] // 2

        # Inside the bottom layer: should have valid data
        if d20_idx < layer_img.shape[1]:
            assert np.isfinite(layer_img[mid_x, d20_idx]), \
                "Inside bottom_layer should have valid data"

        # Outside the bottom layer: should be NaN
        if d8_idx < layer_img.shape[1]:
            assert np.isnan(layer_img[mid_x, d8_idx]), \
                "Outside bottom_layer should be NaN"


# =========================================================================
# Edge cases
# =========================================================================


class TestCombineEdgeCases:
    """Edge cases and regression tests."""

    def test_combine_single_backend(self, t0):
        """Combining a single backend should be identity."""
        b = _make_backend(200, 100, t0, t0 + 200, 5.0, 25.0, -50.0)
        eb = _make_builder(b)

        ec = EchogramBuilder.combine([eb], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_orig, _ = eb.build_image()
        img_comb, _ = ec.build_image()

        valid = np.isfinite(img_orig) & np.isfinite(img_comb)
        assert valid.sum() > 0
        np.testing.assert_allclose(img_comb[valid], img_orig[valid], atol=0.5)

    def test_combine_dict_input(self, t0):
        """Combine accepts dict of builders."""
        b1 = _make_backend(200, 100, t0, t0 + 200, 5.0, 25.0, -50.0)
        b2 = _make_backend(200, 100, t0, t0 + 200, 5.0, 25.0, -40.0)
        eb1 = _make_builder(b1)
        eb2 = _make_builder(b2)

        ec = EchogramBuilder.combine(
            {"mbes": eb1, "sbes": eb2}, combine_func="nanmean", linear=False
        )
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()
        img, _ = ec.build_image()
        assert img.shape[0] > 0

    def test_combine_with_varying_depth_per_ping(self, t0):
        """Backends with per-ping varying depth extents combine correctly."""
        n = 200
        times = np.linspace(t0, t0 + 200, n)

        # MBES: depth varies from 5-25m to 8-30m across pings
        min_d_m = np.linspace(5.0, 8.0, n)
        max_d_m = np.linspace(25.0, 30.0, n)
        img_m = np.full((n, 200), -50.0, dtype=np.float32)

        # SBES: constant depth 5-55m
        img_s = np.full((n, 500), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_m, times, y_min=min_d_m, y_max=max_d_m, y_axis="depth")
        b_s = ImageBackend.from_image(img_s, times, y_min=5.0, y_max=55.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_c, _ = ec.build_image()

        # Should not crash and produce valid data
        assert np.isfinite(img_c).sum() > 0

    def test_sbes_starts_later_deep_only_after_sbes(self, t0):
        """SBES starts later than MBES: deep data only where SBES has pings."""
        n_mbes, n_sbes = 500, 300

        # MBES: t0 to t0+600
        t_mbes = np.linspace(t0, t0 + 600, n_mbes)
        img_mbes = np.full((n_mbes, 200), -50.0, dtype=np.float32)

        # SBES: starts 300s later, t0+300 to t0+900
        t_sbes = np.linspace(t0 + 300, t0 + 900, n_sbes)
        img_sbes = np.full((n_sbes, 500), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=1024)
        ec.set_y_axis_depth()

        img_m, _ = eb_m.build_image()
        img_c, _ = ec.build_image()
        y_grid = ec.coord_system.y_coordinates

        # Before SBES starts (first half of MBES range):
        # Combined should match MBES at shallow depth
        sbes_start_idx = np.argmin(np.abs(t_mbes - (t0 + 300)))
        d15_idx = np.argmin(np.abs(y_grid - 15.0))
        d40_idx = np.argmin(np.abs(y_grid - 40.0))

        # Check early columns (before SBES)
        early_combined = img_c[:sbes_start_idx - 5, d15_idx]
        early_mbes = img_m[:sbes_start_idx - 5, d15_idx]
        valid = np.isfinite(early_combined) & np.isfinite(early_mbes)
        if valid.sum() > 0:
            np.testing.assert_allclose(
                early_combined[valid], early_mbes[valid], atol=0.5,
                err_msg="Before SBES starts, combined should match MBES in shallow"
            )

        # Deep region before SBES: should be NaN (no data)
        early_deep = img_c[:sbes_start_idx - 5, d40_idx]
        assert np.all(np.isnan(early_deep)), \
            "Deep region should be NaN before SBES starts"

        # After SBES starts: shallow should be combined (-45)
        late_combined = img_c[sbes_start_idx + 5:, d15_idx]
        valid_late = np.isfinite(late_combined)
        if valid_late.sum() > 0:
            n_combined = np.sum(np.abs(late_combined[valid_late] - (-45.0)) < 1.0)
            assert n_combined > 0.8 * valid_late.sum(), \
                "After SBES starts, shallow should be ~-45 (combined)"

        # After SBES starts: deep should be SBES (-40)
        late_deep = img_c[sbes_start_idx + 5:, d40_idx]
        valid_deep = np.isfinite(late_deep)
        if valid_deep.sum() > 0:
            n_sbes = np.sum(np.abs(late_deep[valid_deep] - (-40.0)) < 1.0)
            assert n_sbes > 0.8 * valid_deep.sum(), \
                "After SBES starts, deep should be ~-40 (SBES only)"


# =========================================================================
# Layer + time-alignment regression tests
# =========================================================================


class TestCombineLayerTimeAlignment:
    """Regression tests: layers on combined echograms must respect time alignment.

    The layer code path uses get_column() (per-column iteration), which must
    apply the same time-based ping mapping that get_image() uses.  These tests
    verify that SBES data does NOT leak into columns before the SBES time
    range, even when layers are present.
    """

    def test_layer_image_matches_main_before_sbes(self, t0):
        """Layer image should equal main image before SBES starts."""
        n = 200
        t_mbes = np.linspace(t0, t0 + 600, n)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n)

        img_mbes = np.full((n, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n, 500), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=200)
        ec.set_y_axis_depth()

        ec.add_layer_from_static_layer("wc", 8.0, 20.0)
        img_main, layer_img, _ = ec.build_image_and_layer_image()

        y_grid = ec.coord_system.y_coordinates
        d15_idx = np.argmin(np.abs(y_grid - 15.0))

        sbes_start_col = np.argmin(np.abs(t_mbes - (t0 + 300)))
        # Before SBES: both main and layer must be MBES-only (-50)
        before_main = img_main[:sbes_start_col - 2, d15_idx]
        before_layer = layer_img[:sbes_start_col - 2, d15_idx]
        valid = np.isfinite(before_main) & np.isfinite(before_layer)
        assert valid.sum() > 0
        np.testing.assert_allclose(before_layer[valid], before_main[valid], atol=0.5,
            err_msg="Layer should match main image before SBES starts")
        np.testing.assert_allclose(before_main[valid], -50.0, atol=0.5,
            err_msg="Before SBES, values should be MBES-only (-50)")

    def test_layer_image_combined_after_sbes(self, t0):
        """Layer image should show combined values after SBES starts."""
        n = 200
        t_mbes = np.linspace(t0, t0 + 600, n)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n)

        img_mbes = np.full((n, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n, 500), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=200)
        ec.set_y_axis_depth()

        ec.add_layer_from_static_layer("wc", 8.0, 20.0)
        img_main, layer_img, _ = ec.build_image_and_layer_image()

        y_grid = ec.coord_system.y_coordinates
        d15_idx = np.argmin(np.abs(y_grid - 15.0))
        sbes_start_col = np.argmin(np.abs(t_mbes - (t0 + 300)))

        # After SBES: layer should be combined (-45)
        after_layer = layer_img[sbes_start_col + 5:, d15_idx]
        valid = np.isfinite(after_layer)
        assert valid.sum() > 0
        n_combined = np.sum(np.abs(after_layer[valid] - (-45.0)) < 1.0)
        assert n_combined > 0.8 * valid.sum(), \
            "After SBES starts, layer should show combined (-45)"

    def test_get_column_time_aligned_before_sbes(self, t0):
        """get_column must return MBES-only for pings before SBES time range."""
        n = 200
        t_mbes = np.linspace(t0, t0 + 600, n)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n)

        img_mbes = np.full((n, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n, 500), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)

        # Early pings (MBES-only): get_column should return -50
        for ping_idx in [0, 10, 50]:
            col = ec.backend.get_column(ping_idx)
            assert len(col) > 0
            valid = np.isfinite(col)
            assert np.all(np.abs(col[valid] - (-50.0)) < 0.5), \
                f"Ping {ping_idx}: get_column should return MBES-only (-50)"

    def test_get_column_time_aligned_after_sbes(self, t0):
        """get_column must return combined values after SBES starts."""
        n = 200
        t_mbes = np.linspace(t0, t0 + 600, n)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n)

        img_mbes = np.full((n, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n, 500), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth")
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)

        # Late pings (overlap): get_column should have some combined values (-45)
        # and some SBES-only (-40) for samples beyond MBES depth range
        for ping_idx in [150, 170, 190]:
            col = ec.backend.get_column(ping_idx)
            valid = np.isfinite(col)
            n_combined = np.sum(np.abs(col[valid] - (-45.0)) < 1.0)
            n_sbes_only = np.sum(np.abs(col[valid] - (-40.0)) < 1.0)
            assert (n_combined + n_sbes_only) > 0.8 * valid.sum(), \
                f"Ping {ping_idx}: should have combined (-45) and/or SBES-only (-40)"
            assert n_combined > 0, \
                f"Ping {ping_idx}: should have at least some combined (-45) samples"

    def test_layer_with_ping_param_from_mbes_sbes_starts_later(self, t0):
        """Layer defined from MBES ping param must not leak SBES data early.

        Scenario: MBES bottom detection covers t0..t0+600. SBES starts at
        t0+300.  A "bottom" layer is built from the MBES param on the combined
        echogram.  Before t0+300 the layer should show MBES-only values.
        """
        n_mbes, n_sbes = 500, 300
        t_mbes = np.linspace(t0, t0 + 600, n_mbes)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n_sbes)

        img_mbes = np.full((n_mbes, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 500), -40.0, dtype=np.float32)

        # Bottom depth varies 20-22m across MBES pings
        bottom_depths = np.linspace(20.0, 22.0, n_mbes)
        mbes_params = {"bottom": ("Depth (m)", (t_mbes.copy(), bottom_depths.copy()))}

        b_m = ImageBackend.from_image(
            img_mbes, t_mbes, y_min=5.0, y_max=25.0,
            y_axis="depth", ping_params=mbes_params,
        )
        b_s = ImageBackend.from_image(
            img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth",
        )
        eb_m = _make_builder(b_m, max_steps=500)
        eb_s = _make_builder(b_s, max_steps=500)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=500)
        ec.set_y_axis_depth()

        # The combined backend should have the "bottom" param
        assert "bottom" in ec.backend.get_ping_params()

        # Add layer from the bottom param (-5m above, +2m below)
        ec.add_layer_from_ping_param_offsets_absolute("wc", "bottom", -5.0, +2.0)

        img_main, layer_img, _ = ec.build_image_and_layer_image()
        y_grid = ec.coord_system.y_coordinates
        d21_idx = np.argmin(np.abs(y_grid - 21.0))

        sbes_start_col = np.argmin(np.abs(t_mbes - (t0 + 300)))

        # Before SBES: layer at 21m (inside bottom layer) must be MBES-only
        before_layer = layer_img[:sbes_start_col - 5, d21_idx]
        valid = np.isfinite(before_layer)
        if valid.sum() > 0:
            np.testing.assert_allclose(
                before_layer[valid], -50.0, atol=0.5,
                err_msg="Before SBES starts, bottom layer should show MBES-only (-50)"
            )

        # After SBES: should show combined
        after_layer = layer_img[sbes_start_col + 5:, d21_idx]
        valid_after = np.isfinite(after_layer)
        if valid_after.sum() > 0:
            n_combined = np.sum(np.abs(after_layer[valid_after] - (-45.0)) < 1.0)
            assert n_combined > 0.5 * valid_after.sum(), \
                "After SBES starts, bottom layer should show combined (~-45)"

    def test_layer_different_resolution_and_depth(self, t0):
        """Layer on combined with very different sample counts and depth ranges.

        MBES: 834 samples, 4.6-25.6m depth, t0 to t0+1680
        SBES: 5505 samples, 4.8-53.8m depth, t0-900 to t0+2580 (wider)
        Layer from MBES bottom param that has data ONLY during MBES time.
        """
        n_mbes, n_sbes = 2000, 2000
        t_mbes = np.linspace(t0, t0 + 1680, n_mbes)
        t_sbes = np.linspace(t0 - 900, t0 + 2580, n_sbes)

        img_mbes = np.full((n_mbes, 834), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 5505), -40.0, dtype=np.float32)

        bottom_depths = np.linspace(20.0, 23.0, n_mbes)
        mbes_params = {"bottom": ("Depth (m)", (t_mbes.copy(), bottom_depths.copy()))}

        b_m = ImageBackend.from_image(
            img_mbes, t_mbes, y_min=4.6, y_max=25.6,
            y_axis="depth", ping_params=mbes_params,
        )
        b_s = ImageBackend.from_image(
            img_sbes, t_sbes, y_min=4.8, y_max=53.8, y_axis="depth",
        )
        eb_m = _make_builder(b_m, max_steps=2000)
        eb_s = _make_builder(b_s, max_steps=2000)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=2000)
        ec.set_y_axis_depth()

        # Add layer from bottom param
        ec.add_layer_from_ping_param_offsets_absolute("wc", "bottom", -3.0, +2.0)

        img_main, layer_img, _ = ec.build_image_and_layer_image()
        y_grid = ec.coord_system.y_coordinates
        d21_idx = np.argmin(np.abs(y_grid - 21.0))

        # SBES data starts earlier than MBES in time – find where MBES starts
        # in the combined image (combined uses MBES times, so col 0 == MBES start).
        # At early pings, the SBES covers a wider time range, so the combined
        # should still be valid.  Main thing: layer values should be finite and
        # consistent with the main image.
        for x in [0, 500, 1000, 1500, 1999]:
            if x < img_main.shape[0]:
                main_v = img_main[x, d21_idx]
                layer_v = layer_img[x, d21_idx]
                if np.isfinite(main_v) and np.isfinite(layer_v):
                    # layer must agree with main within quantization tolerance
                    assert abs(layer_v - main_v) < 1.0, \
                        f"x={x}: layer ({layer_v:.1f}) != main ({main_v:.1f})"

    def test_main_layer_with_sbes_late_start(self, t0):
        """Main layer on combined: SBES starts later, different depth/resolution."""
        n = 300
        t_mbes = np.linspace(t0, t0 + 600, n)
        t_sbes = np.linspace(t0 + 200, t0 + 800, n)

        img_mbes = np.full((n, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n, 800), -40.0, dtype=np.float32)

        b_m = ImageBackend.from_image(img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth")
        b_s = ImageBackend.from_image(img_sbes, t_sbes, y_min=3.0, y_max=60.0, y_axis="depth")
        eb_m = _make_builder(b_m, max_steps=300)
        eb_s = _make_builder(b_s, max_steps=300)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=300)
        ec.set_y_axis_depth()

        # Set main layer to restrict to 10-22m
        ec.add_layer_from_static_layer("main", 10.0, 22.0)
        img_main, layer_img, _ = ec.build_image_and_layer_image()

        y_grid = ec.coord_system.y_coordinates
        d15_idx = np.argmin(np.abs(y_grid - 15.0))
        d7_idx = np.argmin(np.abs(y_grid - 7.0))
        sbes_start_col = np.argmin(np.abs(t_mbes - (t0 + 200)))

        # Outside main layer (7m): should be NaN
        mid_x = img_main.shape[0] // 2
        if d7_idx < img_main.shape[1]:
            assert np.isnan(img_main[mid_x, d7_idx]), \
                "Outside main layer should be NaN"

        # Before SBES inside main layer (15m): should be MBES-only
        before = img_main[:sbes_start_col - 5, d15_idx]
        valid = np.isfinite(before)
        if valid.sum() > 0:
            np.testing.assert_allclose(before[valid], -50.0, atol=0.5,
                err_msg="Before SBES, main-layer image should be MBES-only")

        # After SBES inside main layer: should be combined
        after = img_main[sbes_start_col + 5:, d15_idx]
        valid_after = np.isfinite(after)
        if valid_after.sum() > 0:
            n_combined = np.sum(np.abs(after[valid_after] - (-45.0)) < 1.0)
            assert n_combined > 0.5 * valid_after.sum(), \
                "After SBES, main-layer image should show combined (-45)"


# =========================================================================
# Ping param extrapolation regression tests
# =========================================================================


class TestCombinePingParamNoExtrapolation:
    """Regression: combined ping params must NOT extrapolate outside each backend's time range.

    When MBES bottom is at 20m and SBES bottom is at 25m, and SBES starts later,
    the combined bottom before SBES starts must be 20m (MBES-only), not 22.5m
    (mean of MBES 20 and SBES 25 extrapolated).  Using 'nearest' extrapolation
    mode on the interpolator caused SBES values to leak into times before SBES
    existed.  Fix: use 'nan' extrapolation mode so nanmean ignores absent backends.
    """

    def test_combined_bottom_mbes_only_before_sbes(self, t0):
        """Combined bottom param should be MBES-only before SBES time range."""
        n_mbes, n_sbes = 500, 300
        t_mbes = np.linspace(t0, t0 + 600, n_mbes)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n_sbes)

        img_mbes = np.full((n_mbes, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 500), -40.0, dtype=np.float32)

        # Deliberately different bottom depths
        mbes_bottom = np.full(n_mbes, 20.0)
        sbes_bottom = np.full(n_sbes, 25.0)

        b_m = ImageBackend.from_image(
            img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth",
            ping_params={"bottom": ("Depth (m)", (t_mbes.copy(), mbes_bottom.copy()))},
        )
        b_s = ImageBackend.from_image(
            img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth",
            ping_params={"bottom": ("Depth (m)", (t_sbes.copy(), sbes_bottom.copy()))},
        )
        eb_m = _make_builder(b_m, max_steps=500)
        eb_s = _make_builder(b_s, max_steps=500)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)

        params = ec.backend.get_ping_params()
        assert "bottom" in params
        y_ref, (times, values) = params["bottom"]

        # Before SBES starts: combined bottom must be MBES-only (20.0)
        early_mask = (times >= t0) & (times < (t0 + 300))
        early_values = values[early_mask]
        valid_early = np.isfinite(early_values)
        assert valid_early.sum() > 0, "Should have valid bottom values before SBES"
        np.testing.assert_allclose(
            early_values[valid_early], 20.0, atol=0.1,
            err_msg="Before SBES, combined bottom should be MBES-only (20m)"
        )

    def test_combined_bottom_mean_during_overlap(self, t0):
        """Combined bottom should be mean of MBES and SBES during overlap."""
        n_mbes, n_sbes = 500, 300
        t_mbes = np.linspace(t0, t0 + 600, n_mbes)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n_sbes)

        img_mbes = np.full((n_mbes, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 500), -40.0, dtype=np.float32)

        mbes_bottom = np.full(n_mbes, 20.0)
        sbes_bottom = np.full(n_sbes, 25.0)

        b_m = ImageBackend.from_image(
            img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth",
            ping_params={"bottom": ("Depth (m)", (t_mbes.copy(), mbes_bottom.copy()))},
        )
        b_s = ImageBackend.from_image(
            img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth",
            ping_params={"bottom": ("Depth (m)", (t_sbes.copy(), sbes_bottom.copy()))},
        )
        eb_m = _make_builder(b_m, max_steps=500)
        eb_s = _make_builder(b_s, max_steps=500)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)

        params = ec.backend.get_ping_params()
        y_ref, (times, values) = params["bottom"]

        # During overlap (both have data): should be nanmean(20, 25) = 22.5
        overlap_mask = (times >= (t0 + 310)) & (times <= (t0 + 590))
        overlap_values = values[overlap_mask]
        valid_overlap = np.isfinite(overlap_values)
        assert valid_overlap.sum() > 0, "Should have valid bottom values during overlap"
        np.testing.assert_allclose(
            overlap_values[valid_overlap], 22.5, atol=0.5,
            err_msg="During overlap, combined bottom should be mean of MBES and SBES"
        )

    def test_combined_bottom_sbes_only_after_mbes(self, t0):
        """Combined bottom should be SBES-only after MBES time range ends."""
        n_mbes, n_sbes = 500, 300
        t_mbes = np.linspace(t0, t0 + 600, n_mbes)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n_sbes)

        img_mbes = np.full((n_mbes, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 500), -40.0, dtype=np.float32)

        mbes_bottom = np.full(n_mbes, 20.0)
        sbes_bottom = np.full(n_sbes, 25.0)

        b_m = ImageBackend.from_image(
            img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth",
            ping_params={"bottom": ("Depth (m)", (t_mbes.copy(), mbes_bottom.copy()))},
        )
        b_s = ImageBackend.from_image(
            img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth",
            ping_params={"bottom": ("Depth (m)", (t_sbes.copy(), sbes_bottom.copy()))},
        )
        eb_m = _make_builder(b_m, max_steps=500)
        eb_s = _make_builder(b_s, max_steps=500)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)

        params = ec.backend.get_ping_params()
        y_ref, (times, values) = params["bottom"]

        # After MBES ends: should be SBES-only (25.0)
        late_mask = times > (t0 + 610)
        late_values = values[late_mask]
        valid_late = np.isfinite(late_values)
        assert valid_late.sum() > 0, "Should have valid bottom values after MBES ends"
        np.testing.assert_allclose(
            late_values[valid_late], 25.0, atol=0.1,
            err_msg="After MBES ends, combined bottom should be SBES-only (25m)"
        )

    def test_layer_depth_correct_before_sbes(self, t0):
        """Layer built from combined bottom must show correct depth before SBES.

        Regression: with 'nearest' extrapolation, the SBES bottom (25m)
        leaked into MBES-only times, shifting the layer boundary deeper.
        After fix (extrapolation_mode='nan'), the layer boundary before SBES
        should follow the MBES bottom (20m), not the mean (22.5m).
        """
        n_mbes, n_sbes = 500, 300
        t_mbes = np.linspace(t0, t0 + 600, n_mbes)
        t_sbes = np.linspace(t0 + 300, t0 + 900, n_sbes)

        img_mbes = np.full((n_mbes, 200), -50.0, dtype=np.float32)
        img_sbes = np.full((n_sbes, 500), -40.0, dtype=np.float32)

        mbes_bottom = np.full(n_mbes, 20.0)
        sbes_bottom = np.full(n_sbes, 25.0)

        b_m = ImageBackend.from_image(
            img_mbes, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth",
            ping_params={"bottom": ("Depth (m)", (t_mbes.copy(), mbes_bottom.copy()))},
        )
        b_s = ImageBackend.from_image(
            img_sbes, t_sbes, y_min=5.0, y_max=55.0, y_axis="depth",
            ping_params={"bottom": ("Depth (m)", (t_sbes.copy(), sbes_bottom.copy()))},
        )
        eb_m = _make_builder(b_m, max_steps=500)
        eb_s = _make_builder(b_s, max_steps=500)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=500)
        ec.set_y_axis_depth()

        # Add layer from bottom param: data from 15m to bottom+2m
        ec.add_layer_from_ping_param_offsets_absolute("wc", "bottom", -5.0, +2.0)

        img_main, layer_img, _ = ec.build_image_and_layer_image()
        y_grid = ec.coord_system.y_coordinates

        # Check that at 21m depth (inside MBES layer: 15-22m), before SBES,
        # the layer has data (MBES values)
        d21_idx = np.argmin(np.abs(y_grid - 21.0))
        sbes_start_col = np.argmin(np.abs(t_mbes - (t0 + 300)))
        before_layer = layer_img[:sbes_start_col - 5, d21_idx]
        valid = np.isfinite(before_layer)
        if valid.sum() > 0:
            np.testing.assert_allclose(
                before_layer[valid], -50.0, atol=0.5,
                err_msg="Before SBES, layer at 21m should be MBES-only (-50)"
            )

        # Check that at 24m depth (deeper than MBES bottom+2=22m), before SBES,
        # the layer should NOT have data (should be NaN because layer boundary
        # is at 20+2=22m for MBES-only). If the bug existed, the layer boundary
        # would be at 22.5+2=24.5m and 24m would have data.
        d24_idx = np.argmin(np.abs(y_grid - 24.0))
        before_deep = layer_img[:sbes_start_col - 5, d24_idx]
        valid_deep = np.isfinite(before_deep)
        # Most of these should be NaN (outside MBES bottom+2=22m layer)
        # Allow some tolerance for edge effects
        assert valid_deep.sum() < 0.3 * len(before_deep), \
            f"Before SBES, 24m should be mostly outside layer (20+2=22m), " \
            f"but {valid_deep.sum()}/{len(before_deep)} are finite"


# ===========================================================================
# Test depth extent time alignment
# ===========================================================================


class TestCombineDepthExtentTimeAlignment:
    """Test that depth extents are time-aligned, not naively length-padded.

    When SBES starts later than MBES, the SBES depth extents should only
    contribute to combined pings at the correct time positions.  Before
    SBES starts, the combined depth extents should equal MBES-only depth
    extents.  The fix addresses the bug where SBES depth_max leaked into
    early (MBES-only) pings, widening the depth range and corrupting the
    depth-to-sample resolution.
    """

    def test_depth_extents_mbes_only_region(self):
        """Before SBES starts, combined depth_max == MBES depth_max."""
        t0 = 1_000_000_000.0
        n_mbes, n_sbes = 200, 50
        # MBES: 0-25m, SBES: 0-55m, SBES starts at t0+300
        b_m = _make_backend(n_mbes, 833, t0, t0 + 400, 5.0, 25.0)
        b_s = _make_backend(n_sbes, 5504, t0 + 300, t0 + 400, 5.0, 55.0)
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        cb = ec.backend  # CombineBackend

        # Find the combined ping just before SBES starts (with margin for
        # time alignment boundary effects - nearest-neighbour can pull the
        # first SBES ping one combined ping earlier)
        sbes_start_idx = np.searchsorted(cb.ping_times, t0 + 300)
        safe_end = max(0, sbes_start_idx - 2)
        # All pings well before SBES should have MBES-only depth extents
        depth_max_before = cb.depth_extents[1][:safe_end]
        assert np.all(depth_max_before <= 25.0 + 0.1), (
            f"Before SBES, depth_max should be <=25m (MBES only), "
            f"got max={np.nanmax(depth_max_before):.1f}"
        )

    def test_depth_extents_overlap_region(self):
        """During overlap, combined depth_max == max(MBES, SBES) depth_max."""
        t0 = 1_000_000_000.0
        n_mbes, n_sbes = 200, 50
        b_m = _make_backend(n_mbes, 833, t0, t0 + 400, 5.0, 25.0)
        b_s = _make_backend(n_sbes, 5504, t0 + 300, t0 + 400, 5.0, 55.0)
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        cb = ec.backend

        # Find overlap region
        sbes_start_idx = np.searchsorted(cb.ping_times, t0 + 300)
        # Pings in the overlap should have wider depth range (from SBES)
        depth_max_overlap = cb.depth_extents[1][sbes_start_idx:]
        valid = np.isfinite(depth_max_overlap)
        if valid.sum() > 0:
            assert np.nanmax(depth_max_overlap) >= 50.0, (
                f"During overlap, depth_max should be >=50m (SBES contribution), "
                f"got max={np.nanmax(depth_max_overlap):.1f}"
            )

    def test_resolution_mbes_only_matches_standalone(self):
        """Resolution in MBES-only region should match standalone MBES."""
        t0 = 1_000_000_000.0
        n_mbes, n_sbes = 200, 50
        b_m = _make_backend(n_mbes, 833, t0, t0 + 400, 5.0, 25.0)
        b_s = _make_backend(n_sbes, 5504, t0 + 300, t0 + 400, 5.0, 55.0)
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        cb = ec.backend

        # MBES standalone resolution
        mbes_res = (25.0 - 5.0) / 833  # ~0.024

        # Combined resolution before SBES starts (with safe margin)
        sbes_start_idx = np.searchsorted(cb.ping_times, t0 + 300)
        safe_end = max(0, sbes_start_idx - 2)
        for i in [0, safe_end // 2, max(0, safe_end - 1)]:
            d_min = cb.depth_extents[0][i]
            d_max = cb.depth_extents[1][i]
            samples = cb.max_sample_counts[i]
            if samples > 0:
                comb_res = (d_max - d_min) / samples
                np.testing.assert_allclose(
                    comb_res, mbes_res, atol=0.002,
                    err_msg=f"Ping {i}: combined resolution {comb_res:.4f} != MBES {mbes_res:.4f}"
                )

    def test_max_sample_counts_time_aligned(self):
        """max_sample_counts uses first backend; depth extents are time-aligned."""
        t0 = 1_000_000_000.0
        n_mbes, n_sbes = 200, 50
        b_m = _make_backend(n_mbes, 833, t0, t0 + 400, 5.0, 25.0)
        b_s = _make_backend(n_sbes, 5504, t0 + 300, t0 + 400, 5.0, 55.0)
        eb_m = _make_builder(b_m)
        eb_s = _make_builder(b_s)

        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        cb = ec.backend

        # max_sample_counts stays from first backend (MBES) everywhere
        # (kept for get_column/layer compatibility)
        assert np.all(cb.max_sample_counts <= 833), (
            f"max_sample_counts should be from first backend (<=833), "
            f"got max={np.max(cb.max_sample_counts)}"
        )

        # But depth extents should be time-aligned
        sbes_start_idx = np.searchsorted(cb.ping_times, t0 + 300)
        safe_end = max(0, sbes_start_idx - 2)
        # Before SBES: depth_max <= MBES depth_max
        assert np.all(cb.depth_extents[1][:safe_end] <= 25.1)
        # During overlap: depth_max >= SBES depth_max
        overlap_max = cb.depth_extents[1][sbes_start_idx:]
        valid = np.isfinite(overlap_max)
        if valid.sum() > 0:
            assert np.nanmax(overlap_max) >= 50.0

    def test_combined_image_matches_mbes_before_sbes(self):
        """In the MBES-only time region, combined image should match MBES image."""
        t0 = 1_000_000_000.0
        n_mbes, n_sbes = 200, 50
        # Use a gradient so we can check the depth mapping is correct
        t_mbes = np.linspace(t0, t0 + 400, n_mbes)
        mbes_image = np.zeros((n_mbes, 833), dtype=np.float32)
        # Create a depth marker: strong signal at sample 400 (~14.6m at 0.024m/sample)
        mbes_image[:, 390:410] = -30.0  # everything else is -50

        b_m = ImageBackend.from_image(
            mbes_image, t_mbes, y_min=5.0, y_max=25.0, y_axis="depth"
        )
        b_s = _make_backend(n_sbes, 5504, t0 + 300, t0 + 400, 5.0, 55.0, fill_value=-60.0)
        eb_m = _make_builder(b_m, max_steps=200)
        eb_s = _make_builder(b_s, max_steps=200)

        # Build standalone MBES image
        img_m, ext_m = eb_m.build_image()
        y_m = eb_m.coord_system.y_coordinates

        # Build combined
        ec = EchogramBuilder.combine([eb_m, eb_s], combine_func="nanmean", linear=False)
        ec.set_x_axis_date_time(max_steps=200)
        ec.set_y_axis_depth()
        img_c, ext_c = ec.build_image()
        y_c = ec.coord_system.y_coordinates

        # Find the depth of the marker in MBES (~14.6m)
        marker_depth = 5.0 + 400 * (25.0 - 5.0) / 833  # ~14.6m

        # In MBES image, find row closest to marker_depth
        m_row = np.argmin(np.abs(y_m - marker_depth))
        # In combined image, same depth
        c_row = np.argmin(np.abs(y_c - marker_depth))

        # Before SBES starts, the marker should appear at the same depth
        # in both images (check first few columns which are MBES-only)
        n_cols_before = min(10, img_m.shape[0], img_c.shape[0])
        for col in range(n_cols_before):
            m_val = img_m[col, m_row]
            c_val = img_c[col, c_row]
            if np.isfinite(m_val) and np.isfinite(c_val):
                # Both should show the marker (strong signal near -30dB)
                assert c_val > -45.0, (
                    f"Col {col}: combined value at marker depth {marker_depth:.1f}m "
                    f"is {c_val:.1f}dB, expected > -45dB (marker). "
                    f"MBES value is {m_val:.1f}dB"
                )
