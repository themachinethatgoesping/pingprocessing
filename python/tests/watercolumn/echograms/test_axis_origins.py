"""Regression tests for y-axis origins (range vs depth).

A range axis must start at the stored range origin (typically 0 m at the
transducer) and a depth axis at the stored depth origin (transducer depth),
independent of which axis the data was gridded/stored on or which backend wraps
it. This guards against range collapsing onto depth (range starting at
~depth_min instead of 0) -- in particular the concat/combine backends, which
recomputed a depth affine for every axis.
"""

import json
from pathlib import Path

import numpy as np

from themachinethatgoesping.pingprocessing.watercolumn.echograms.coordinate_system import (
    EchogramCoordinateSystem,
)
from themachinethatgoesping.pingprocessing.watercolumn.echograms.echogrambuilder import (
    EchogramBuilder,
)


def _cs(n_pings=6, n_samples=200):
    return EchogramCoordinateSystem(
        n_pings, np.full(n_pings, n_samples, dtype=np.float32),
        np.arange(1000.0, 1000.0 + n_pings))


# ---------------------------------------------------------------------------
# Dual-axis synthetic mmap: range 0..SPAN, depth DEPTH_MIN..DEPTH_MIN+SPAN,
# with a single bright marker row at sample MARKER. The marker therefore sits
# at range = SPAN*MARKER/(n_samples-1) and depth = DEPTH_MIN + that range.
# ---------------------------------------------------------------------------

N_PINGS = 8
N_SAMPLES = 100
MARKER = 50
SPAN = 20.0
DEPTH_MIN = 5.0
MARKER_RANGE = SPAN * MARKER / (N_SAMPLES - 1)          # ~10.1 m
MARKER_DEPTH = DEPTH_MIN + MARKER_RANGE                 # ~15.1 m


def _write_dual_axis_mmap(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    img = np.full((N_PINGS, N_SAMPLES), -50.0, dtype=np.float32)
    img[:, MARKER] = 0.0  # bright marker row (brightest -> argmax)
    img.tofile(path / "wci_data.bin")

    np.save(path / "ping_times.npy", np.arange(1000.0, 1000.0 + N_PINGS))
    np.save(path / "max_sample_counts.npy",
            np.full(N_PINGS, N_SAMPLES - 1, dtype=np.int32))
    np.save(path / "sample_nr_min.npy", np.zeros(N_PINGS, dtype=np.float32))
    np.save(path / "sample_nr_max.npy",
            np.full(N_PINGS, N_SAMPLES - 1, dtype=np.float32))
    np.save(path / "range_min.npy", np.zeros(N_PINGS, dtype=np.float32))
    np.save(path / "range_max.npy", np.full(N_PINGS, SPAN, dtype=np.float32))
    np.save(path / "depth_min.npy", np.full(N_PINGS, DEPTH_MIN, dtype=np.float32))
    np.save(path / "depth_max.npy",
            np.full(N_PINGS, DEPTH_MIN + SPAN, dtype=np.float32))

    meta = {
        "format_version": "3.0",
        "n_pings": N_PINGS,
        "n_samples": N_SAMPLES,
        "wci_value": "sv",
        "linear_mean": False,
        "has_navigation": False,
        "ping_param_names": [],
        "ping_params_meta": {},
        "ping_metainfo_names": [],
        "ping_metainfo_meta": {},
        "storage_mode": {
            "x_axis": "ping_index", "y_axis": "sample_index",
            "x_resolution": None, "x_origin": None,
            "y_resolution": 1.0, "y_origin": 0.0,
        },
    }
    (path / "metadata.json").write_text(json.dumps(meta))
    return path


def _probe(eb, axis: str):
    """Return ``(top_y, marker_y)`` of the built image on ``axis``.

    ``top_y`` is the y-coordinate of the first finite row (the axis origin);
    ``marker_y`` is the y-coordinate of the brightest row (the marker).
    """
    eb.set_x_axis_ping_index()
    getattr(eb, "set_y_axis_" + axis)(max_steps=200)
    img, ext = eb.build_image()
    ny = img.shape[1]
    ys = np.linspace(ext[3], ext[2], ny)  # top -> bottom
    col = img[img.shape[0] // 2]
    finite = np.where(np.isfinite(col))[0]
    top_y = float(ys[finite[0]])
    marker_y = float(ys[int(np.nanargmax(col))])
    return top_y, marker_y


class TestBackendAxisOrigins:
    """Range vs depth origin + marker position must be correct for every backend."""

    def _check(self, eb, *, tol=0.6):
        # range: origin ~0, marker at ~MARKER_RANGE (NOT at depth offset)
        top_r, mark_r = _probe(eb, "range")
        assert abs(top_r) <= tol, f"range origin {top_r} != 0"
        assert abs(mark_r - MARKER_RANGE) <= tol, f"range marker {mark_r} != {MARKER_RANGE}"
        # depth: origin ~DEPTH_MIN, marker at ~MARKER_DEPTH
        top_d, mark_d = _probe(eb, "depth")
        assert abs(top_d - DEPTH_MIN) <= tol, f"depth origin {top_d} != {DEPTH_MIN}"
        assert abs(mark_d - MARKER_DEPTH) <= tol, f"depth marker {mark_d} != {MARKER_DEPTH}"

    def test_mmap(self, tmp_path):
        store = _write_dual_axis_mmap(tmp_path / "m")
        self._check(EchogramBuilder.from_mmap(str(store)))

    def test_concat(self, tmp_path):
        store = _write_dual_axis_mmap(tmp_path / "m")
        eb = EchogramBuilder.from_mmap(str(store))
        self._check(EchogramBuilder.concat([eb]))

    def test_combine(self, tmp_path):
        store = _write_dual_axis_mmap(tmp_path / "m")
        eb = EchogramBuilder.from_mmap(str(store))
        eb.set_x_axis_ping_index()
        eb.set_y_axis_range()
        self._check(EchogramBuilder.combine([eb, eb]))

    def test_gridded_mmap(self, tmp_path):
        store = _write_dual_axis_mmap(tmp_path / "m")
        eb = EchogramBuilder.from_mmap(str(store))
        eb.set_x_axis_ping_index()
        eb.set_y_axis_depth()  # grid on depth; range must still reconstruct
        out = tmp_path / "grid"
        eb.to_gridded_mmap(str(out), x_step=1.0, y_step=0.2, progress=False)
        # gridding quantises rows -> looser tolerance (one y_step)
        self._check(EchogramBuilder.from_gridded_mmap(str(out)), tol=0.8)


def test_range_and_depth_origins_are_independent():
    cs = _cs()
    cs.set_range_extent(np.zeros(6), np.full(6, 20.0))   # range 0..20
    cs.set_depth_extent(np.full(6, 5.0), np.full(6, 25.0))  # depth 5..25
    cs.set_y_axis_range(max_steps=200)
    assert cs.sample_index_to_value("Range (m)", np.zeros(6))[0] == 0.0
    cs.set_y_axis_depth(max_steps=200)
    assert cs.sample_index_to_value("Depth (m)", np.zeros(6))[0] == 5.0


def test_imagebackend_range_origin_zero():
    img = np.full((6, 200), -50.0, np.float32)
    eb = EchogramBuilder.from_image(img, np.arange(1000.0, 1006.0),
                                    y_min=0.0, y_max=20.0, y_axis="range")
    eb.set_x_axis_ping_index(); eb.set_y_axis_range(max_steps=100)
    _, ext = eb.build_image()
    assert min(ext[2:]) <= 0.1  # top of range extent reaches ~0 m


def test_gridded_range_roundtrip_origin_zero(tmp_path):
    img = np.full((6, 200), -50.0, np.float32)
    eb = EchogramBuilder.from_image(img, np.arange(1000.0, 1006.0),
                                    y_min=0.0, y_max=20.0, y_axis="range")
    eb.set_y_axis_range()
    out = tmp_path / "grid"
    eb.to_gridded_mmap(str(out), x_step=1.0, y_step=0.5, progress=False)
    r = EchogramBuilder.from_gridded_mmap(str(out))
    assert r.coord_system.has_ranges
    assert float(np.nanmin(r.coord_system.min_ranges)) <= 0.25


# ---------------------------------------------------------------------------
# Combine backend: x/y alignment must follow the current display axes, not the
# axis active at combine() time (otherwise depth affines leak into a range view,
# or ping-index alignment leaks into a time view).
# ---------------------------------------------------------------------------


def _combine_two_range():
    img = np.full((N_PINGS, N_SAMPLES), -50.0, dtype=np.float32)
    img[:, MARKER] = 0.0
    times = np.arange(1000.0, 1000.0 + N_PINGS)
    a = EchogramBuilder.from_image(img, times, y_min=0.0, y_max=SPAN, y_axis="range")
    b = EchogramBuilder.from_image(img, times, y_min=0.0, y_max=SPAN, y_axis="range")
    # combine while the first builder is on ping-index / sample axis so the
    # backend's construction-time alignment differs from what we set later.
    a.set_x_axis_ping_index()
    a.set_y_axis_y_indice()
    return EchogramBuilder.combine([a, b])


class TestCombineAlignmentTracksAxes:
    def test_x_align_follows_axis(self):
        cmb = _combine_two_range()
        cmb.set_x_axis_ping_index(); cmb.build_image()
        assert cmb._backend.x_align == "ping_index"
        cmb.set_x_axis_date_time(); cmb.build_image()
        assert cmb._backend.x_align == "time"
        cmb.set_x_axis_ping_index(); cmb.build_image()
        assert cmb._backend.x_align == "ping_index"

    def test_y_align_follows_axis(self):
        cmb = _combine_two_range()
        cmb.set_y_axis_range(); cmb.build_image()
        assert cmb._backend.y_align == "range"
        cmb.set_y_axis_y_indice(); cmb.build_image()
        assert cmb._backend.y_align == "sample_index"


def test_combine_time_alignment_changes_output():
    """Two echograms with offset timing must combine by TIME when the view is
    time-based: temporally non-overlapping pings keep their own value, whereas
    ping-index alignment would average everything to the same value.
    """
    n, ns = 6, 20
    a_times = np.arange(100.0, 100.0 + n)          # 100..105
    b_times = np.arange(103.0, 103.0 + n)          # 103..108 (3-ping overlap)
    a = EchogramBuilder.from_image(np.full((n, ns), -60.0, np.float32), a_times,
                                   y_min=0.0, y_max=20.0, y_axis="range")
    b = EchogramBuilder.from_image(np.full((n, ns), -40.0, np.float32), b_times,
                                   y_min=0.0, y_max=20.0, y_axis="range")

    cmb = EchogramBuilder.combine([a, b], combine_func="nanmean", linear=False)
    cmb.set_x_axis_date_time()
    cmb.set_y_axis_range()
    img, _ = cmb.build_image()

    finite = img[np.isfinite(img)]
    # A-only region keeps ~-60, B-only region keeps ~-40, overlap averages ~-50.
    assert finite.min() < -55.0, "A-only (time) region missing -> not time-aligned"
    assert finite.max() > -45.0, "B-only (time) region missing -> not time-aligned"

    # In ping-index mode the same pings pair up 1:1 -> everything averages ~-50.
    cmb.set_x_axis_ping_index()
    img2, _ = cmb.build_image()
    finite2 = img2[np.isfinite(img2)]
    assert np.all(np.abs(finite2 - (-50.0)) < 1.0), "ping-index mode should average to -50"

