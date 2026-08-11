import numpy as np

from themachinethatgoesping.pingprocessing.watercolumn.echograms.backends.image_backend import (
    ImageBackend,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.echogrambuilder import (
    EchogramBuilder,
)


def _make_backend(times, fill=-50.0, ping_params=None):
    times = np.asarray(times, dtype=np.float64)
    n_pings = len(times)
    image = np.full((n_pings, 8), fill, dtype=np.float32)
    return ImageBackend.from_image(
        image,
        times,
        y_min=0.0,
        y_max=20.0,
        y_axis="depth",
        ping_params=ping_params,
    )


def _param(name, times, values, y_ref="Depth (m)"):
    return {name: (y_ref, (np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)))}


def test_concat_sort_by_time_orders_inputs_by_start_time():
    b_late = _make_backend([200.0, 201.0, 202.0])
    b_early = _make_backend([100.0, 101.0, 102.0])

    combined = EchogramBuilder.concat([b_late, b_early], sort_by_time=True)

    ping_times = np.asarray(combined.backend.ping_times, dtype=np.float64)
    assert ping_times[0] <= ping_times[-1]
    assert np.isclose(ping_times[0], 100.0)


def test_combine_keeps_suffixed_params_when_time_ranges_overlap():
    p0 = _param("minslant", [100.0, 101.0, 102.0], [10.0, 10.1, 10.2])
    p1 = _param("minslant", [101.5, 102.5, 103.5], [20.0, 20.1, 20.2])

    b0 = _make_backend([100.0, 101.0, 102.0], ping_params=p0)
    b1 = _make_backend([101.5, 102.5, 103.5], ping_params=p1)

    combined = EchogramBuilder.combine([b0, b1])
    names = set(combined.backend.get_ping_params().keys())

    assert "minslant" in names
    assert "minslant_0" in names
    assert "minslant_1" in names


def test_combine_skips_suffixed_params_when_time_ranges_do_not_overlap():
    p0 = _param("minslant", [100.0, 101.0, 102.0], [10.0, 10.1, 10.2])
    p1 = _param("minslant", [200.0, 201.0, 202.0], [20.0, 20.1, 20.2])

    b0 = _make_backend([100.0, 101.0, 102.0], ping_params=p0)
    b1 = _make_backend([200.0, 201.0, 202.0], ping_params=p1)

    combined = EchogramBuilder.combine([b0, b1])
    names = set(combined.backend.get_ping_params().keys())

    assert "minslant" in names
    assert "minslant_0" not in names
    assert "minslant_1" not in names


# ---------------------------------------------------------------------------
# Coordinate-system params (added via add_ping_param after construction, e.g.
# by detect_bottom) must survive concat/combine even when they use a non-time
# x-reference such as 'Ping index'. These live only in the builder coordinate
# system, not in the backend, so they exercise the param re-injection path.
# ---------------------------------------------------------------------------


def _make_range_backend(times, n_samples=8, fill=-50.0, ping_params=None):
    times = np.asarray(times, dtype=np.float64)
    n_pings = len(times)
    image = np.full((n_pings, n_samples), fill, dtype=np.float32)
    return ImageBackend.from_image(
        image,
        times,
        y_min=0.0,
        y_max=20.0,
        y_axis="range",
        ping_params=ping_params,
    )


def test_concat_preserves_coordinate_param_with_ping_index_reference():
    b0 = EchogramBuilder.from_backend(_make_backend([100.0, 101.0, 102.0]))
    b1 = EchogramBuilder.from_backend(_make_backend([200.0, 201.0, 202.0]))

    # 'Ping index' reference -> param lives only in the coordinate system.
    b0.add_ping_param("Bottom", "Ping index", "Sample number", [0, 1, 2], [3.0, 3.0, 3.0])
    b1.add_ping_param("Bottom", "Ping index", "Sample number", [0, 1, 2], [5.0, 5.0, 5.0])

    combined = EchogramBuilder.concat([b0, b1])

    assert "Bottom" in combined.get_param_names()
    y_ref, dense = combined.coord_system.param["Bottom"]
    assert y_ref == "Sample number"
    assert len(dense) == 6
    assert np.isclose(dense[0], 3.0)
    assert np.isclose(dense[-1], 5.0)


def test_concat_sort_by_time_with_builders_orders_and_keeps_param():
    b_late = EchogramBuilder.from_backend(_make_backend([200.0, 201.0, 202.0]))
    b_early = EchogramBuilder.from_backend(_make_backend([100.0, 101.0, 102.0]))
    b_late.add_ping_param("Bottom", "Ping index", "Sample number", [0, 1, 2], [5.0, 5.0, 5.0])
    b_early.add_ping_param("Bottom", "Ping index", "Sample number", [0, 1, 2], [3.0, 3.0, 3.0])

    combined = EchogramBuilder.concat([b_late, b_early], sort_by_time=True)

    ping_times = np.asarray(combined.backend.ping_times, dtype=np.float64)
    assert np.isclose(ping_times[0], 100.0)

    assert "Bottom" in combined.get_param_names()
    _, dense = combined.coord_system.param["Bottom"]
    # Earliest survey (value 3) must come first after time-sorting.
    assert np.isclose(dense[0], 3.0)
    assert np.isclose(dense[-1], 5.0)


def test_combine_preserves_coordinate_param():
    b0 = EchogramBuilder.from_backend(_make_backend([100.0, 101.0, 102.0]))
    b1 = EchogramBuilder.from_backend(_make_backend([100.0, 101.0, 102.0]))
    b0.add_ping_param("Bottom", "Ping index", "Sample number", [0, 1, 2], [4.0, 4.0, 4.0])

    combined = EchogramBuilder.combine([b0, b1])

    assert "Bottom" in combined.get_param_names()
    _, dense = combined.coord_system.param["Bottom"]
    assert np.allclose(dense, 4.0)


def test_combined_range_builder_exposes_res_ranges():
    b0 = EchogramBuilder.from_backend(_make_range_backend([100.0, 101.0, 102.0]))
    b1 = EchogramBuilder.from_backend(_make_range_backend([100.0, 101.0, 102.0]))

    combined = EchogramBuilder.combine([b0, b1])

    assert combined.coord_system.res_ranges is not None
    assert np.all(np.isfinite(combined.coord_system.res_ranges))


def test_depth_only_builder_has_res_ranges_none_not_attribute_error():
    # Depth-axis echogram has no range extents; res_ranges must be a defined
    # attribute (None) rather than raising AttributeError.
    b = EchogramBuilder.from_backend(_make_backend([100.0, 101.0, 102.0]))
    assert b.coord_system.res_ranges is None


# ---------------------------------------------------------------------------
# concat(sort_by_time=True) must produce a strictly increasing timeline even
# when backends overlap in time. Sorting inputs by start time alone leaves the
# concatenated ping times non-monotonic, which the coordinate system's time
# feature (a strictly-increasing interpolator) rejects. These reproduce that
# crash and verify the per-ping time ordering keeps data access consistent.
# ---------------------------------------------------------------------------


def _make_valued_backend(times, row_values, n_samples=8):
    """Backend whose every sample in ping i equals row_values[i] (so a column
    can be traced back to the ping it came from)."""
    times = np.asarray(times, dtype=np.float64)
    rows = np.asarray(row_values, dtype=np.float32).reshape(len(times), 1)
    image = np.repeat(rows, n_samples, axis=1)
    return ImageBackend.from_image(
        image, times, y_min=0.0, y_max=20.0, y_axis="depth"
    )


def test_concat_sort_by_time_interleaves_overlapping_backends():
    # Even vs odd timestamps -> the two backends must interleave ping-by-ping.
    a = _make_valued_backend([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
    b = _make_valued_backend([1.0, 3.0, 5.0], [1.0, 3.0, 5.0])

    combined = EchogramBuilder.concat([a, b], sort_by_time=True)
    backend = combined.backend

    times = np.asarray(backend.ping_times, dtype=np.float64)
    assert np.all(np.diff(times) > 0)  # strictly increasing
    np.testing.assert_array_equal(times, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Each public column must carry the value of the ping at that time.
    for g in range(6):
        assert backend.get_column(g)[0] == times[g]


def test_concat_sort_by_time_handles_fully_duplicate_backends():
    # Real-world trigger: the same recording exported under two paths, so both
    # backends carry identical timestamps. Previously raised
    # "X list is not sorted in ascending order!" from set_ping_times.
    times = [100.0, 101.0, 102.0]
    a = _make_valued_backend(times, [10.0, 11.0, 12.0])
    b = _make_valued_backend(times, [10.0, 11.0, 12.0])

    combined = EchogramBuilder.concat([a, b], sort_by_time=True)
    pt = np.asarray(combined.backend.ping_times, dtype=np.float64)

    assert len(pt) == 6  # no pings dropped
    assert np.all(np.diff(pt) > 0)  # duplicates nudged to strictly increasing

    # The full display path (time feature + image) must not raise.
    combined.set_x_axis_date_time(max_steps=64)
    combined.set_y_axis_depth(max_steps=64)
    image, _ = combined.build_image()
    assert np.isfinite(image).any()


def test_concat_sort_by_time_leaves_sorted_timeline_untouched():
    # Non-overlapping, already-ordered inputs must not be permuted or nudged.
    a = _make_valued_backend([100.0, 101.0, 102.0], [1.0, 2.0, 3.0])
    b = _make_valued_backend([200.0, 201.0, 202.0], [4.0, 5.0, 6.0])

    combined = EchogramBuilder.concat([a, b], sort_by_time=True)
    backend = combined.backend

    assert backend._order is None  # fast path, no permutation
    np.testing.assert_array_equal(
        np.asarray(backend.ping_times, dtype=np.float64),
        [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
    )
