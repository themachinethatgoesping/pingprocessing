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
