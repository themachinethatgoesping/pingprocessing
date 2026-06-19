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
