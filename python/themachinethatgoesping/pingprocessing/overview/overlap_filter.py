# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

"""
Functions for filtering and matching pings across multiple PingOverview objects
based on spatial or temporal proximity.

Uses a sparse-grid approach: pings are hashed into grid cells, a bitmask
tracks which overviews occupy each cell, and a 3×3 dilation finds cells
where *all* overviews are present in the local neighbourhood.

Complexity: O(N) time and memory (N = total pings across all overviews).

Typical usage::

    from themachinethatgoesping.pingprocessing.overview import overlap_filter

    filtered = overlap_filter.filter_by_spatial_overlap(
        [overview_a, overview_b, overview_c],
        max_distance_m=100.0,
    )
    # filtered[i].original_indices  → indices into the original overview
"""

from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm
from typing import List

from .pingoverview import PingOverview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_utm_epsg(latitudes, longitudes) -> int:
    """Determine the appropriate UTM EPSG code from representative coords."""
    mean_lon = float(np.mean(longitudes))
    mean_lat = float(np.mean(latitudes))
    utm_zone = int((mean_lon + 180) / 6) + 1
    return 32600 + utm_zone if mean_lat >= 0 else 32700 + utm_zone


# Fast int64-key packing for grid cells (handles negative cell indices).
_OFFSET = np.int64(1 << 31)


def _pack_cells(ix, iy):
    """Vectorised: pack int arrays → int64 keys."""
    return ((ix.astype(np.int64) + _OFFSET) << np.int64(32)) | (
        iy.astype(np.int64) + _OFFSET
    )


def _pack_cell(ix, iy):
    """Scalar: pack one (ix, iy) → int64 key."""
    return (np.int64(int(ix) + int(_OFFSET)) << np.int64(32)) | np.int64(
        int(iy) + int(_OFFSET)
    )


def subset_overview(overview: PingOverview, indices) -> PingOverview:
    """
    Create a new PingOverview containing only the specified ping indices.

    Parameters
    ----------
    overview : PingOverview
        Source overview.
    indices : array-like of int
        Ping indices to keep (0-based, must be sorted and unique).

    Returns
    -------
    PingOverview
        New overview with ``original_indices`` attribute mapping each
        new ping position back to its index in the source overview.
    """
    subset = PingOverview()
    indices = np.asarray(indices, dtype=int)

    for key, values in overview.variables.items():
        subset.variables[key] = [values[i] for i in indices]

    # Rebuild file-path lookup with only the paths actually referenced
    # by the kept pings, and remap file_path_index to new indices.
    old_indices = subset.variables.get("file_path_index", [])
    used_old = sorted(set(old_indices))
    old_to_new = {old: new for new, old in enumerate(used_old)}

    subset._file_paths = [overview._file_paths[i] for i in used_old]
    subset._file_path_map = {fp: i for i, fp in enumerate(subset._file_paths)}
    subset.variables["file_path_index"] = [old_to_new[i] for i in old_indices]

    subset.original_indices = indices
    return subset


# ---------------------------------------------------------------------------
# Sparse-grid overlap engine
# ---------------------------------------------------------------------------

def _grid_overlap_mask(
    coord_arrays,
    cell_size: float,
    progress: bool = True,
):
    """
    Core grid-based overlap computation.

    Parameters
    ----------
    coord_arrays : list of np.ndarray
        One array per overview.  Shape (N_i, D) — 2-D for spatial
        (easting, northing) or 1-D reshaped to (N_i, 1) for temporal.
    cell_size : float
        Grid cell size (metres or seconds).
    progress : bool
        Show tqdm progress bars.

    Returns
    -------
    list of np.ndarray[bool]
        One boolean mask per overview (True = keep).
    """
    n_overviews = len(coord_arrays)
    full_mask_val = (1 << n_overviews) - 1   # all bits set
    ndim = coord_arrays[0].shape[1]

    # --- 1. For each overview, find unique occupied cells and set bitmask ---
    cell_presence = defaultdict(int)          # int64-key → bitmask
    # Also store the per-ping cell keys for step 3 (reuse, not recompute)
    ping_keys_per_ov = []

    for ov_idx, coords in enumerate(
        tqdm(coord_arrays, desc="Gridding overviews", disable=not progress)
    ):
        bit = 1 << ov_idx
        # Compute cell indices per dimension
        cell_idx = np.floor(coords / cell_size).astype(np.int64)

        if ndim == 1:
            keys = cell_idx[:, 0].astype(np.int64) + _OFFSET
        else:
            keys = _pack_cells(cell_idx[:, 0], cell_idx[:, 1])

        ping_keys_per_ov.append(keys)

        # Only iterate unique cells (typically 10-100k for 1M pings)
        for k in np.unique(keys):
            cell_presence[int(k)] |= bit

    # --- 2. Dilate: find overlap cells (neighbourhood check) ---
    # Build the list of neighbour offsets (3^ndim kernel).
    if ndim == 1:
        deltas = [np.int64(-1), np.int64(0), np.int64(1)]
    else:
        deltas = [
            _pack_cell(di, dj) - _pack_cell(0, 0)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
        ]

    overlap_keys = set()
    for cell_key in tqdm(
        cell_presence, desc="Finding overlap cells", disable=not progress
    ):
        combined = 0
        for d in deltas:
            combined |= cell_presence.get(int(np.int64(cell_key) + d), 0)
            if combined == full_mask_val:       # early exit
                break
        if combined == full_mask_val:
            overlap_keys.add(cell_key)

    # --- 3. Filter pings whose cell is in the overlap set ---
    # Convert to sorted int64 array for np.isin (binary-search fast path).
    overlap_arr = np.array(sorted(overlap_keys), dtype=np.int64)

    masks = []
    for keys in tqdm(
        ping_keys_per_ov, desc="Filtering pings", disable=not progress
    ):
        masks.append(np.isin(keys, overlap_arr))

    return masks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_by_spatial_overlap(
    overviews: List[PingOverview],
    max_distance_m: float = 100.0,
    progress: bool = True,
) -> List[PingOverview]:
    """
    Keep only pings in areas where **all** overviews have coverage.

    Algorithm
    ---------
    1. Project lat/lon to a common UTM coordinate system (metres).
    2. Hash every ping into a square grid cell of size *max_distance_m*.
    3. For each cell, record which overviews are present (bitmask).
    4. A cell is an **overlap cell** if, considering itself and its 8
       immediate neighbours (3 × 3 block), every overview is represented.
    5. Retain only pings whose cell is an overlap cell.

    This is an *area-based* filter: it guarantees that for every kept
    ping there are pings from all other overviews within roughly
    *max_distance_m* (up to ~1.5 × due to cell-boundary effects).
    Use a smaller *max_distance_m* if tighter control is needed.

    Parameters
    ----------
    overviews : list of PingOverview
        Two or more overviews to compare.
    max_distance_m : float
        Grid cell size in metres (default 100).
    progress : bool
        Show tqdm progress bars (default True).

    Returns
    -------
    list of PingOverview
        One filtered overview per input.  Each has an attribute
        ``original_indices`` (np.ndarray of int) mapping new-ping-index
        → source-ping-index::

            matched_pings = [pings_a[i] for i in filtered[0].original_indices]
    """
    from pyproj import Transformer

    n = len(overviews)
    if n < 2:
        raise ValueError("Need at least 2 overviews.")

    # 1. Common UTM projection
    all_lats = np.concatenate([np.asarray(ov.variables["latitude"]) for ov in overviews])
    all_lons = np.concatenate([np.asarray(ov.variables["longitude"]) for ov in overviews])
    epsg = _get_utm_epsg(all_lats, all_lons)
    del all_lats, all_lons

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    # 2. Project each overview
    coord_arrays = []
    for ov in tqdm(overviews, desc="Projecting to UTM", disable=not progress):
        x, y = transformer.transform(
            np.asarray(ov.variables["longitude"], dtype=float),
            np.asarray(ov.variables["latitude"], dtype=float),
        )
        coord_arrays.append(np.column_stack([x, y]))

    # 3. Grid overlap
    masks = _grid_overlap_mask(coord_arrays, cell_size=max_distance_m, progress=progress)

    # 4. Build subset overviews
    return [
        subset_overview(ov, np.where(mask)[0])
        for ov, mask in zip(overviews, masks)
    ]


def filter_by_temporal_overlap(
    overviews: List[PingOverview],
    max_time_s: float = 1.0,
    progress: bool = True,
) -> List[PingOverview]:
    """
    Keep only pings in time windows where **all** overviews have coverage.

    Same grid approach as :func:`filter_by_spatial_overlap` but in 1-D
    (timestamps in seconds).

    Parameters
    ----------
    overviews : list of PingOverview
    max_time_s : float
        Grid cell size in seconds (default 1).
    progress : bool
        Show tqdm progress bars (default True).

    Returns
    -------
    list of PingOverview
        Filtered overviews with ``original_indices`` attribute.
    """
    n = len(overviews)
    if n < 2:
        raise ValueError("Need at least 2 overviews.")

    coord_arrays = [
        np.asarray(ov.variables["timestamp"], dtype=float).reshape(-1, 1)
        for ov in overviews
    ]

    masks = _grid_overlap_mask(coord_arrays, cell_size=max_time_s, progress=progress)

    return [
        subset_overview(ov, np.where(mask)[0])
        for ov, mask in zip(overviews, masks)
    ]
