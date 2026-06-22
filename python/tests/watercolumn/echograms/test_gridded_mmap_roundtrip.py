import numpy as np

from themachinethatgoesping.pingprocessing.watercolumn.echograms.echogrambuilder import (
    EchogramBuilder,
)


def _make_range_builder(n_pings=6, n_samples=201):
    image = np.full((n_pings, n_samples), -50.0, dtype=np.float32)
    ping_times = np.arange(1000.0, 1000.0 + n_pings, dtype=np.float64)
    builder = EchogramBuilder.from_image(
        image,
        ping_times,
        y_min=0.0,
        y_max=20.0,
        y_axis="range",
    )
    return builder


def test_gridded_roundtrip_preserves_range_info_for_sample_axis_export(tmp_path):
    builder = _make_range_builder()

    # Keep default y-axis ("Y indice"), i.e. gridding in sample-index space.
    out = tmp_path / "grid_sample_axis"
    builder.to_gridded_mmap(str(out), x_step=1.0, y_step=1.0, progress=False)

    reloaded = EchogramBuilder.from_gridded_mmap(str(out))

    # Roundtrip must still expose range extents/resolution for detect_bottom.
    assert reloaded.coord_system.has_ranges
    assert reloaded.coord_system.res_ranges is not None
    assert np.all(np.isfinite(reloaded.coord_system.res_ranges))


def test_gridded_roundtrip_preserves_sample_number_param_conversion(tmp_path):
    builder = _make_range_builder()

    # Bottom in sample-number units: 20 / 200 * 20 m = 2 m.
    pn = np.arange(builder.backend.n_pings, dtype=np.float64)
    builder.add_ping_param("Bottom", "Ping index", "Sample number", pn, np.full_like(pn, 20.0))

    out = tmp_path / "grid_with_bottom"
    builder.to_gridded_mmap(str(out), x_step=1.0, y_step=1.0, progress=False)

    reloaded = EchogramBuilder.from_gridded_mmap(str(out))
    reloaded.set_y_axis_range()
    _, bottom_range = reloaded.get_ping_param("Bottom")

    assert np.isfinite(bottom_range).any()
    assert np.isclose(np.nanmedian(bottom_range), 2.0, atol=0.25)
