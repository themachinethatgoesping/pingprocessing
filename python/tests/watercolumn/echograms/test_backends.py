# SPDX-FileCopyrightText: 2022 - 2024 Peter Urban, Ghent University
# SPDX-License-Identifier: MPL-2.0

"""
Tests for echogram backends: StorageAxisMode, ConcatBackend, CombineBackend
"""

import pytest
import numpy as np
from typing import Dict, Tuple, Optional

# Import the modules we want to test
from themachinethatgoesping.pingprocessing.watercolumn.echograms.backends import (
    StorageAxisMode,
    XAxisType,
    YAxisType,
    ResolutionStrategy,
    ConcatBackend,
    CombineBackend,
    COMBINE_FUNCTIONS,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.backends.base import EchogramDataBackend
from themachinethatgoesping.pingprocessing.watercolumn.echograms.indexers import EchogramImageRequest


# =============================================================================
# Mock Backend for Testing
# =============================================================================

class MockBackend(EchogramDataBackend):
    """A simple mock backend for testing ConcatBackend and CombineBackend."""

    def __init__(
        self,
        ping_times: np.ndarray,
        sample_data: np.ndarray,
        min_depths: Optional[np.ndarray] = None,
        max_depths: Optional[np.ndarray] = None,
        ping_params: Optional[Dict] = None,
    ):
        """
        Create a mock backend.

        Parameters
        ----------
        ping_times : np.ndarray
            Unix timestamps for each ping (shape: (num_pings,))
        sample_data : np.ndarray
            Pre-computed sample data (shape: (num_pings, num_samples))
        min_depths : np.ndarray, optional
            Min depth per ping
        max_depths : np.ndarray, optional
            Max depth per ping
        ping_params : dict, optional
            Ping parameters dict
        """
        self._ping_times = np.asarray(ping_times, dtype=np.float64)
        self._sample_data = np.asarray(sample_data, dtype=np.float32)
        self._num_pings = len(ping_times)
        self._num_samples = sample_data.shape[1] if sample_data.ndim == 2 else 1
        
        self._min_depths = min_depths if min_depths is not None else np.zeros(self._num_pings, dtype=np.float32)
        self._max_depths = max_depths if max_depths is not None else np.full(self._num_pings, 100.0, dtype=np.float32)
        self._ping_params = ping_params or {}

    # Required abstract property implementations
    @property
    def n_pings(self) -> int:
        return self._num_pings

    @property
    def ping_times(self) -> np.ndarray:
        return self._ping_times

    @property
    def max_sample_counts(self) -> np.ndarray:
        return np.full(self._num_pings, self._num_samples, dtype=np.int32)

    @property
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(self._num_pings, dtype=np.float32),
            np.full(self._num_pings, self._num_samples, dtype=np.float32),
        )

    @property
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return (self._min_depths, self._max_depths)

    @property
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return (self._min_depths, self._max_depths)

    @property
    def has_navigation(self) -> bool:
        return True

    @property
    def wci_value(self) -> str:
        return "sv"

    @property
    def linear_mean(self) -> bool:
        return True

    def get_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        return self._ping_params

    def get_column(self, ping_index: int) -> np.ndarray:
        return self._sample_data[ping_index, :].copy()

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        return self.get_column(ping_index)

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        """Return pre-computed data for the requested pings."""
        # Create output array
        image = np.full((request.nx, request.ny), np.nan, dtype=np.float32)
        
        # Fill with data from valid pings
        for x_idx in range(request.nx):
            ping_idx = request.ping_indexer[x_idx]
            if ping_idx >= 0 and ping_idx < self._num_pings:
                sample_indices = request.compute_sample_indices(ping_idx)
                valid_mask = sample_indices >= 0
                if np.any(valid_mask):
                    valid_samples = sample_indices[valid_mask]
                    valid_samples = np.clip(valid_samples, 0, self._num_samples - 1)
                    image[x_idx, valid_mask] = self._sample_data[ping_idx, valid_samples]
        
        return image


# =============================================================================
# StorageAxisMode Tests
# =============================================================================

class TestStorageAxisMode:
    """Tests for StorageAxisMode dataclass."""

    def test_default_mode(self):
        """Test default storage mode creation."""
        mode = StorageAxisMode.default()
        assert mode.x_axis == "ping_index"  # Stored as string
        assert mode.y_axis == "sample_index"
        assert mode.y_resolution == 1.0  # Default is 1.0, not None
        assert mode.x_resolution is None
        assert mode.y_origin == 0.0  # Default is 0.0, not None

    def test_ping_time_depth_mode(self):
        """Test ping_time_depth factory method."""
        mode = StorageAxisMode.ping_time_depth(y_resolution=0.1, y_origin=0.5)
        assert mode.x_axis == "ping_time"
        assert mode.y_axis == "depth"
        assert mode.y_resolution == 0.1
        assert mode.y_origin == 0.5

    def test_ping_time_range_mode(self):
        """Test ping_time_range factory method."""
        mode = StorageAxisMode.ping_time_range(y_resolution=0.05)
        assert mode.x_axis == "ping_time"
        assert mode.y_axis == "range"
        assert mode.y_resolution == 0.05

    def test_to_dict_and_back(self):
        """Test serialization and deserialization."""
        original = StorageAxisMode.ping_time_depth(y_resolution=0.25, y_origin=1.0)
        data = original.to_dict()
        
        assert data["x_axis"] == "ping_time"
        assert data["y_axis"] == "depth"
        assert data["y_resolution"] == 0.25
        assert data["y_origin"] == 1.0
        
        restored = StorageAxisMode.from_dict(data)
        assert restored == original

    def test_from_dict_handles_missing_keys(self):
        """Test from_dict with minimal dict."""
        data = {"x_axis": "ping_index", "y_axis": "sample_nr"}
        mode = StorageAxisMode.from_dict(data)
        assert mode.x_axis == "ping_index"
        assert mode.y_axis == "sample_nr"
        assert mode.x_resolution is None


# =============================================================================
# ConcatBackend Tests
# =============================================================================

class TestConcatBackend:
    """Tests for ConcatBackend - sequential concatenation."""

    @pytest.fixture
    def two_backends(self):
        """Create two simple backends for testing."""
        # Backend 1: times 0-9, values 0-99
        times1 = np.arange(10, dtype=float)
        data1 = np.arange(100).reshape(10, 10).astype(float)
        backend1 = MockBackend(times1, data1)

        # Backend 2: times 15-24, values 100-199
        times2 = np.arange(15, 25, dtype=float)
        data2 = np.arange(100, 200).reshape(10, 10).astype(float)
        backend2 = MockBackend(times2, data2)

        return backend1, backend2

    def test_concat_num_pings(self, two_backends):
        """Test concatenated backend has correct total ping count."""
        b1, b2 = two_backends
        concat = ConcatBackend([b1, b2])
        assert concat.n_pings == 20

    def test_concat_ping_times_preserve_gaps(self, two_backends):
        """Test ping times with gap preservation."""
        b1, b2 = two_backends
        concat = ConcatBackend([b1, b2], gap_handling="preserve")
        
        times = concat.ping_times
        assert len(times) == 20
        # First backend: 0-9
        np.testing.assert_array_equal(times[:10], np.arange(10))
        # Second backend: 15-24
        np.testing.assert_array_equal(times[10:], np.arange(15, 25))

    def test_concat_ping_times_continuous(self, two_backends):
        """Test ping times with continuous gap handling - preserves original times."""
        b1, b2 = two_backends
        concat = ConcatBackend([b1, b2], gap_handling="continuous")
        
        times = concat.ping_times
        assert len(times) == 20
        # Currently both preserve and continuous keep original times
        # The difference is in how they're interpreted for display
        np.testing.assert_array_equal(times[:10], np.arange(10))

    def test_concat_get_column(self, two_backends):
        """Test get_column delegates to correct backend."""
        b1, b2 = two_backends
        concat = ConcatBackend([b1, b2])
        
        # Ping 0 should return first row of backend1 data
        col0 = concat.get_column(0)
        np.testing.assert_array_equal(col0, np.arange(10).astype(float))
        
        # Ping 10 should return first row of backend2 data
        col10 = concat.get_column(10)
        np.testing.assert_array_equal(col10, np.arange(100, 110).astype(float))

    def test_concat_extents(self, two_backends):
        """Test extents are concatenated properly."""
        b1, b2 = two_backends
        concat = ConcatBackend([b1, b2])
        
        sample_min, sample_max = concat.sample_nr_extents
        assert len(sample_min) == 20
        assert len(sample_max) == 20


# =============================================================================
# CombineBackend Tests
# =============================================================================

class TestCombineBackend:
    """Tests for CombineBackend - mathematical combination."""

    @pytest.fixture
    def overlapping_backends(self):
        """Create backends with overlapping time ranges."""
        # Backend 1: values 1.0
        times1 = np.arange(10, dtype=float)
        data1 = np.ones((10, 10), dtype=float)
        backend1 = MockBackend(times1, data1)

        # Backend 2: values 3.0 (same times)
        times2 = np.arange(10, dtype=float)
        data2 = np.ones((10, 10), dtype=float) * 3.0
        backend2 = MockBackend(times2, data2)

        return backend1, backend2

    def test_combine_num_pings_same_times(self, overlapping_backends):
        """Test combined backend with identical time grids."""
        b1, b2 = overlapping_backends
        combine = CombineBackend([b1, b2], combine_func="nanmean")
        
        # Should have 10 pings (same times in both)
        assert combine.n_pings == 10

    def test_combine_get_column_nanmean(self, overlapping_backends):
        """Test nanmean combination via get_column."""
        b1, b2 = overlapping_backends
        combine = CombineBackend([b1, b2], combine_func="nanmean", linear=False)
        
        col = combine.get_column(0)
        # Mean of 1.0 and 3.0 = 2.0
        np.testing.assert_array_almost_equal(col, np.ones(10) * 2.0)

    def test_combine_custom_function(self, overlapping_backends):
        """Test custom combine function."""
        b1, b2 = overlapping_backends
        
        def custom_combine(data, axis=0):
            """Return maximum value."""
            return np.nanmax(data, axis=axis)
        
        combine = CombineBackend([b1, b2], combine_func=custom_combine, linear=False)
        
        col = combine.get_column(0)
        # Max of 1.0 and 3.0 = 3.0
        np.testing.assert_array_almost_equal(col, np.ones(10) * 3.0)

    def test_combine_with_nan_values(self):
        """Test combination handles NaN values correctly."""
        times = np.arange(10, dtype=float)
        
        # Backend 1: some NaNs
        data1 = np.ones((10, 10), dtype=float)
        data1[0:5, :] = np.nan
        backend1 = MockBackend(times, data1)

        # Backend 2: complementary NaNs
        data2 = np.ones((10, 10), dtype=float) * 2.0
        data2[5:10, :] = np.nan
        backend2 = MockBackend(times, data2)

        combine = CombineBackend([backend1, backend2], combine_func="nanmean", linear=False)
        
        # First 5 pings: only backend2 valid (2.0)
        col0 = combine.get_column(0)
        np.testing.assert_array_almost_equal(col0, 2.0)
        
        # Last 5 pings: only backend1 valid (1.0)
        col5 = combine.get_column(5)
        np.testing.assert_array_almost_equal(col5, 1.0)

    def test_builtin_combine_functions_registered(self):
        """Test that built-in combine functions are available."""
        expected_funcs = ["nanmean", "nanmedian", "nansum", "nanmax", "nanmin", "nanstd"]
        for func_name in expected_funcs:
            assert func_name in COMBINE_FUNCTIONS, f"Missing combine function: {func_name}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_combine_of_concats(self):
        """Test CombineBackend wrapping ConcatBackends (channel combination)."""
        # Create data for channel 1: file1 (times 0-4), file2 (times 10-14)
        ch1_file1 = MockBackend(np.arange(5, dtype=float), np.ones((5, 10)))
        ch1_file2 = MockBackend(np.arange(10, 15, dtype=float), np.ones((5, 10)))
        ch1_concat = ConcatBackend([ch1_file1, ch1_file2])
        
        # Create data for channel 2: same files but values 3.0
        ch2_file1 = MockBackend(np.arange(5, dtype=float), np.ones((5, 10)) * 3.0)
        ch2_file2 = MockBackend(np.arange(10, 15, dtype=float), np.ones((5, 10)) * 3.0)
        ch2_concat = ConcatBackend([ch2_file1, ch2_file2])
        
        # Combine the two channels
        combined = CombineBackend([ch1_concat, ch2_concat], combine_func="nanmean", linear=False)
        
        assert combined.n_pings == 10  # 5 + 5 from each concat
        
        # Mean of 1.0 and 3.0 = 2.0
        col = combined.get_column(0)
        np.testing.assert_array_almost_equal(col, 2.0)

    def test_combine_backends_different_sizes(self):
        """Test CombineBackend with backends that have different ping counts.
        
        This tests the case where one echogram is incomplete (e.g., one channel
        has fewer pings than another).
        """
        # Backend 1: 20 pings (full coverage)
        times1 = np.arange(20, dtype=float)
        data1 = np.ones((20, 10), dtype=float) * 1.0
        backend1 = MockBackend(times1, data1)
        
        # Backend 2: Only 10 pings (partial coverage)
        times2 = np.arange(10, dtype=float)
        data2 = np.ones((10, 10), dtype=float) * 3.0
        backend2 = MockBackend(times2, data2)
        
        # Combine - should not crash even though backend2 is shorter
        combined = CombineBackend([backend1, backend2], combine_func="nanmean", linear=False)
        
        # Combined should have 20 pings (from the longer backend)
        assert combined.n_pings == 20
        
        # First 10 pings: mean of 1.0 and 3.0 = 2.0
        col0 = combined.get_column(0)
        np.testing.assert_array_almost_equal(col0, 2.0)
        
        # Pings 10-19: only backend1 has data, so value is 1.0
        col15 = combined.get_column(15)
        np.testing.assert_array_almost_equal(col15, 1.0)

    def test_storage_mode_on_backends(self):
        """Test that backends have storage_mode property."""
        backend = MockBackend(np.arange(10, dtype=float), np.ones((10, 10)))
        mode = backend.storage_mode
        
        # Default storage mode
        assert mode.x_axis == "ping_index"
        assert mode.y_axis == "sample_index"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
