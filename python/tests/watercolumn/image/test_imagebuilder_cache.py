"""Tests for ImageBuilder's per-ping stacking cache."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder import ImageBuilder


# ---------------------------------------------------------------------------
# Helpers: stub pings that produce deterministic WCI sub-images
# ---------------------------------------------------------------------------

def _make_stub_pings(n=20):
    """Return a list of *n* lightweight stub pings.

    The stubs are not used directly – ``make_wci_dual_head`` is patched below
    so that each ping is identified purely by its list-index.
    """
    return [f"ping_{i}" for i in range(n)]


def _fake_make_wci_dual_head(ping, horizontal_pixels, y_coordinates=None,
                              z_coordinates=None, **kwargs):
    """Deterministic replacement for ``make_wci_dual_head``.

    Returns a 4×6 image whose pixel values are set to the ping index
    (encoded in the ping string ``"ping_<i>"``).
    """
    idx = int(ping.split("_")[1])
    ny = len(y_coordinates) if y_coordinates is not None else 4
    nz = len(z_coordinates) if z_coordinates is not None else 6
    img = np.full((ny, nz), float(idx), dtype=np.float32)

    y_res = y_coordinates[1] - y_coordinates[0] if y_coordinates is not None else 1.0
    z_res = z_coordinates[1] - z_coordinates[0] if z_coordinates is not None else 1.0
    extent = (
        y_coordinates[0] - y_res * 0.5,
        y_coordinates[-1] + y_res * 0.5,
        z_coordinates[-1] + z_res * 0.5,
        z_coordinates[0] - z_res * 0.5,
    )
    return img, extent


@pytest.fixture()
def builder():
    """ImageBuilder with patched ``make_wci_dual_head``."""
    pings = _make_stub_pings(20)
    ib = ImageBuilder(
        pings,
        horizontal_pixels=4,
        max_cache_images=50,
        # Fixed-view bounds – required for caching
        hmin=-10.0, hmax=10.0, vmin=0.0, vmax=30.0,
    )
    return ib


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCacheKeyAndCoordinates:
    def test_compute_fixed_coordinates_returns_correct_shape(self):
        y, z, ext = ImageBuilder._compute_fixed_coordinates(
            hp=100, hmin=-10, hmax=10, vmin=0, vmax=30,
        )
        # v_range (30) > h_range (20) → z gets 100 pixels, y gets ceil(100*20/30)=67
        assert len(z) == 100
        assert len(y) == 67
        assert len(ext) == 4

    def test_compute_fixed_coordinates_square(self):
        y, z, ext = ImageBuilder._compute_fixed_coordinates(
            hp=50, hmin=-5, hmax=5, vmin=0, vmax=10,
        )
        assert len(y) == len(z) == 50

    def test_cache_key_changes_with_hmin(self):
        k1 = ImageBuilder._make_cache_key({"horizontal_pixels": 4, "hmin": -10, "hmax": 10, "vmin": 0, "vmax": 30})
        k2 = ImageBuilder._make_cache_key({"horizontal_pixels": 4, "hmin": -5,  "hmax": 10, "vmin": 0, "vmax": 30})
        assert k1 != k2

    def test_cache_key_stable_for_same_args(self):
        args = {"horizontal_pixels": 4, "hmin": -10, "hmax": 10, "vmin": 0, "vmax": 30, "wci_value": "sv"}
        assert ImageBuilder._make_cache_key(args) == ImageBuilder._make_cache_key(args)


class TestCachedBuild:
    @patch(
        "themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder.make_wci_dual_head",
        side_effect=_fake_make_wci_dual_head,
    )
    def test_cache_avoids_recompute(self, mock_dual):
        """Sliding window by 1 should only compute the new ping."""
        pings = _make_stub_pings(20)
        ib = ImageBuilder(pings, horizontal_pixels=4, max_cache_images=50,
                          hmin=-10.0, hmax=10.0, vmin=0.0, vmax=30.0)

        # First build: stack=5, index=0 → needs pings 0..4
        ib.build(index=0, stack=5, stack_step=1)
        assert mock_dual.call_count == 5

        # Second build: stack=5, index=1 → needs pings 1..5
        # Pings 1..4 are cached, only ping 5 is new
        mock_dual.reset_mock()
        ib.build(index=1, stack=5, stack_step=1)
        assert mock_dual.call_count == 1  # only ping 5

    @patch(
        "themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder.make_wci_dual_head",
        side_effect=_fake_make_wci_dual_head,
    )
    def test_cache_cleared_on_args_change(self, mock_dual):
        """Changing wci_value should invalidate the cache."""
        pings = _make_stub_pings(10)
        ib = ImageBuilder(pings, horizontal_pixels=4, max_cache_images=50,
                          hmin=-10.0, hmax=10.0, vmin=0.0, vmax=30.0)

        ib.build(index=0, stack=3, stack_step=1)
        assert mock_dual.call_count == 3

        # Change a render param
        ib.update_args(wci_value="rv")
        mock_dual.reset_mock()
        ib.build(index=0, stack=3, stack_step=1)
        # All 3 must be rebuilt because cache key changed
        assert mock_dual.call_count == 3

    @patch(
        "themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder.make_wci_dual_head",
        side_effect=_fake_make_wci_dual_head,
    )
    def test_cache_eviction(self, mock_dual):
        """Cache should evict oldest entries when exceeding max_cache_images."""
        pings = _make_stub_pings(20)
        ib = ImageBuilder(pings, horizontal_pixels=4, max_cache_images=5,
                          hmin=-10.0, hmax=10.0, vmin=0.0, vmax=30.0)

        # Build with 3 pings at index 0
        ib.build(index=0, stack=3, stack_step=1)
        assert len(ib._cache) == 3

        # Build with 3 pings at index 3 → 6 total, exceeds cap of 5
        ib.build(index=3, stack=3, stack_step=1)
        assert len(ib._cache) <= 5

        # Earliest cached entry (index 0) should have been evicted
        assert 0 not in ib._cache

    @patch(
        "themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder.make_wci_dual_head",
        side_effect=_fake_make_wci_dual_head,
    )
    def test_cached_result_matches_aggregation(self, mock_dual):
        """Cached build should produce the same result as manual aggregation."""
        pings = _make_stub_pings(10)
        ib = ImageBuilder(pings, horizontal_pixels=4, max_cache_images=50,
                          hmin=-10.0, hmax=10.0, vmin=0.0, vmax=30.0,
                          linear_mean=True)

        wci, extent = ib.build(index=2, stack=3, stack_step=1)

        # Manual aggregation: pings 2, 3, 4
        # Each ping's image is filled with its index value (dB).
        # linear_mean: 10*log10(mean(10^(v*0.1) for v in [2,3,4]))
        vals = np.array([2.0, 3.0, 4.0])
        linear_vals = np.power(10, vals * 0.1)
        expected_db = 10 * np.log10(np.mean(linear_vals))

        np.testing.assert_allclose(wci.flat[0], expected_db, rtol=1e-5)

    @patch(
        "themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder.make_wci_dual_head",
        side_effect=_fake_make_wci_dual_head,
    )
    def test_no_cache_when_view_not_fixed(self, mock_dual):
        """Without fixed bounds, every build should recompute all pings."""
        pings = _make_stub_pings(10)
        ib = ImageBuilder(pings, horizontal_pixels=4, max_cache_images=50)
        # hmin/hmax/vmin/vmax are all None → not fixed

        # Need to patch make_wci_stack too for the non-cached path
        with patch(
            "themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder.make_wci_stack",
        ) as mock_stack:
            mock_stack.return_value = (np.zeros((4, 6)), (0, 1, 1, 0))
            ib.build(index=0, stack=3, stack_step=1)
            assert mock_stack.call_count == 1
            assert mock_dual.call_count == 0  # dual_head not called; stack called instead

    @patch(
        "themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder.make_wci_dual_head",
        side_effect=_fake_make_wci_dual_head,
    )
    def test_cache_disabled_when_max_zero(self, mock_dual):
        """max_cache_images=0 should disable the cache."""
        pings = _make_stub_pings(10)
        ib = ImageBuilder(pings, horizontal_pixels=4, max_cache_images=0,
                          hmin=-10.0, hmax=10.0, vmin=0.0, vmax=30.0)

        with patch(
            "themachinethatgoesping.pingprocessing.watercolumn.image.imagebuilder.make_wci_stack",
        ) as mock_stack:
            mock_stack.return_value = (np.zeros((4, 6)), (0, 1, 1, 0))
            ib.build(index=0, stack=3, stack_step=1)
            assert mock_stack.call_count == 1
