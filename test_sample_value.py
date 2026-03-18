"""Determine exactly how pyqtgraph maps data coordinates to pixel indices.

Creates known images, sets them up exactly as the viewers do (transpose +
setRect with y-inversion), then computes the inverse of ImageItem's transform
to learn the correct pixel for every data coordinate.
"""
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def setup_like_viewer(plot_item, image, extent):
    """Set up an ImageItem exactly as both viewers do."""
    image_item = pg.ImageItem()
    plot_item.addItem(image_item)

    array = image.transpose()  # shape (ncols, nrows)
    image_item.setImage(array, autoLevels=False)

    x0, x1, y0, y1 = extent
    vb = plot_item.getViewBox()
    if vb.yInverted():
        y0, y1 = y1, y0
    width = x1 - x0
    height = y1 - y0
    rect = QtCore.QRectF(x0, y0, width, height)
    image_item.setRect(rect)
    return image_item, rect


def pixel_at_data_coords(image_item, rect, x, y, transposed_shape):
    """Given data (view) coordinates, compute the pixel index in the
    stored (transposed) array using the rect transform.

    ImageItem maps its pixel array [0..M) x [0..N) to the rect.
    Pixel [i, j] occupies the area:
      x: rect.x + i/M * rect.w  to  rect.x + (i+1)/M * rect.w
      y: rect.y + j/N * rect.h  to  rect.y + (j+1)/N * rect.h

    Data coords (x, y) map to:
      fi = (x - rect.x) / rect.w * M
      fj = (y - rect.y) / rect.h * N
    Then pixel index = floor(fi), floor(fj), clamped.
    """
    M, N = transposed_shape
    rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()

    fi = (x - rx) / rw * M if rw != 0 else 0
    fj = (y - ry) / rh * N if rh != 0 else 0

    # Clamp to valid range
    ix = int(np.clip(int(np.floor(fi)), 0, M - 1))
    iy = int(np.clip(int(np.floor(fj)), 0, N - 1))
    return ix, iy


def discover_mapping(image, extent, y_inverted=True):
    """Discover the actual pixel displayed at each corner and centre."""
    nrows, ncols = image.shape
    x0, x1, y0, y1 = extent

    win = pg.GraphicsLayoutWidget()
    plot = win.addPlot()
    if y_inverted:
        plot.getViewBox().invertY(True)
    image_item, rect = setup_like_viewer(plot, image, extent)

    stored = image_item.image  # transposed array, shape (ncols, nrows)
    trans_shape = stored.shape

    print(f"  Image shape: {image.shape} (rows, cols)")
    print(f"  Transposed shape: {trans_shape}")
    print(f"  Extent: x=[{x0}, {x1}], y=[{y0}, {y1}]")
    print(f"  Rect: x={rect.x()}, y={rect.y()}, w={rect.width()}, h={rect.height()}")
    print(f"  yInverted: {y_inverted}")
    print()

    # Test at pixel centres (to avoid edge ambiguity)
    # Pixel [i, j] centre in data coords:
    #   cx = rect.x + (i + 0.5) / M * rect.w
    #   cy = rect.y + (j + 0.5) / N * rect.h
    M, N = trans_shape
    rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()

    print("  --- Pixel-centre verification (should all match) ---")
    mismatches = 0
    for i in range(M):
        for j in range(N):
            cx = rx + (i + 0.5) / M * rw
            cy = ry + (j + 0.5) / N * rh
            ix, iy = pixel_at_data_coords(image_item, rect, cx, cy, trans_shape)
            if ix != i or iy != j:
                print(f"    MISMATCH at pixel [{i},{j}]: "
                      f"data=({cx:.3f},{cy:.3f}) -> pixel [{ix},{iy}]")
                mismatches += 1
    if mismatches == 0:
        print("    All pixel centres map back correctly")
    print()

    # Now check corners and centre in data-extent coordinates
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    test_points = [
        ("(x0,y0)",     x0, y0),
        ("(x1,y0)",     x1, y0),
        ("(x0,y1)",     x0, y1),
        ("(x1,y1)",     x1, y1),
        ("centre",      mid_x, mid_y),
    ]

    results = []
    for label, tx, ty in test_points:
        ix, iy = pixel_at_data_coords(image_item, rect, tx, ty, trans_shape)
        # stored[ix, iy] = image.T[ix, iy] = image[iy, ix]
        stored_val = float(stored[ix, iy])
        orig_row, orig_col = iy, ix
        orig_val = float(image[orig_row, orig_col])
        print(f"  {label:14s}  data=({tx:7.2f}, {ty:7.2f})  "
              f"trans_px=[{ix},{iy}]  orig[{orig_row},{orig_col}]={orig_val}")
        results.append((label, tx, ty, orig_row, orig_col, orig_val))

    win.close()
    return results


def test_sample_value_formula(image, extent, mapping_results, y_inverted=True):
    """Test formula variants against pyqtgraph ground truth."""
    x0, x1, y0, y1 = extent
    nrows, ncols = image.shape

    # Reconstruct rect as the viewer does
    rx, rw = x0, x1 - x0
    if y_inverted:
        ry, rh = y1, y0 - y1   # swapped
    else:
        ry, rh = y0, y1 - y0

    print("\n  --- Testing formula: rect-based (floor+clamp) ---")
    all_pass = True

    for label, tx, ty, exp_row, exp_col, exp_val in mapping_results:
        # Map data coords → transposed-array pixel via rect inverse
        fi = (tx - rx) / rw * ncols if rw != 0 else 0
        fj = (ty - ry) / rh * nrows if rh != 0 else 0
        col = min(max(int(np.floor(fi)), 0), ncols - 1)
        row = min(max(int(np.floor(fj)), 0), nrows - 1)
        val = float(image[row, col])

        ok = (val == exp_val)
        if not ok:
            all_pass = False

        print(f"  {label:14s}  expected={exp_val:6.0f}  "
              f"got[{row},{col}]={val:6.0f} {'OK' if ok else 'WRONG'}")

    print(f"\n  Rect-based formula: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return all_pass


def run_test(name, image, extent, y_inverted=True):
    print(f"\n{'=' * 60}")
    print(f"=== {name} ===")
    print(f"{'=' * 60}")
    results = discover_mapping(image, extent, y_inverted)
    ok = test_sample_value_formula(image, extent, results, y_inverted)
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    img5x7 = np.array([[r * 100 + c for c in range(7)] for r in range(5)],
                       dtype=float)
    img4x6 = np.array([[r * 100 + c for c in range(6)] for r in range(4)],
                       dtype=float)
    img3x4 = np.array([[r * 100 + c for c in range(4)] for r in range(3)],
                       dtype=float)

    results = []
    results.append(run_test(
        "WCI-style (yInverted, depth extent)",
        img5x7, (-3.0, 3.0, 0.0, 50.0), y_inverted=True))
    results.append(run_test(
        "Echogram-style (yInverted)",
        img4x6, (0.0, 100.0, 0.0, 30.0), y_inverted=True))
    results.append(run_test(
        "No y-inversion",
        img3x4, (-10.0, 10.0, -20.0, 20.0), y_inverted=False))

    print("\n" + "=" * 60)
    all_ok = all(results)
    for (name, ok) in zip(
        ["WCI-style", "Echogram-style", "No y-inversion"], results
    ):
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\n{'All tests passed!' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)
