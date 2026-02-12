# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

"""
Spatial clustering of PingOverview objects.

Provides two clustering approaches:

- :func:`cluster_by_region` — Grid-based connected components.
  **O(n)** time and memory, fastest option.  Best for splitting survey
  data into separate geographic areas or lines.

- :func:`cluster_by_kmeans` — Mini-Batch K-Means on downsampled UTM
  coordinates.  Supports automatic *k* selection via a maximum cluster
  radius.  Requires ``scikit-learn``.

Both return a :class:`ClusterResult` containing the sub-overviews,
cluster-center coordinates, and per-ping labels.

Typical usage::

    from themachinethatgoesping.pingprocessing.overview import cluster

    # Split survey into contiguous geographic regions (O(n), no extra deps)
    result = cluster.cluster_by_region(overview, max_distance_m=500)

    # Or use K-Means with automatic cluster count
    result = cluster.cluster_by_kmeans(
        overview, max_cluster_radius_m=2000, downsample_factor=10,
    )

    for i, ov in enumerate(result):
        lat, lon = result.centers_latlon[i]
        print(f"Cluster {i}: {len(ov.variables['latitude'])} pings "
              f"at ({lat:.4f}, {lon:.4f})")
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm

from .pingoverview import PingOverview
from .overlap_filter import _get_utm_epsg, subset_overview, _pack_cells, _OFFSET


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Result of spatial clustering.

    Attributes
    ----------
    overviews : list of PingOverview
        One PingOverview per cluster, sorted largest-first.
        Each has an ``original_indices`` attribute mapping back to
        the source overview.
    centers_latlon : list of (float, float)
        (latitude, longitude) of each cluster centroid.
    labels : np.ndarray
        Cluster label (0 .. k−1) for every ping in the original overview.
        Labels are ordered so 0 = largest cluster.
        Pings dropped by *min_cluster_size* have label −1.
    """

    overviews: List[PingOverview]
    centers_latlon: List[Tuple[float, float]]
    labels: np.ndarray

    def __len__(self):
        return len(self.overviews)

    def __getitem__(self, idx):
        return self.overviews[idx]

    def __iter__(self):
        return iter(self.overviews)

    def __repr__(self):
        sizes = [len(ov.variables.get("latitude", [])) for ov in self.overviews]
        return f"ClusterResult(n_clusters={len(self)}, sizes={sizes})"


# ---------------------------------------------------------------------------
# Union-Find for connected components
# ---------------------------------------------------------------------------

class _UnionFind:
    """Path-compressed union-find (disjoint-set) on int64 keys."""

    __slots__ = ("parent", "rank")

    def __init__(self):
        self.parent: dict = {}
        self.rank: dict = {}

    def find(self, x: int) -> int:
        p = self.parent
        if x not in p:
            p[x] = x
            self.rank[x] = 0
            return x
        root = x
        while p[root] != root:
            root = p[root]
        # path compression
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        r = self.rank
        if r[rx] < r[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if r[rx] == r[ry]:
            r[rx] += 1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _project_overview_to_utm(overview, progress=True):
    """Project overview lat/lon to UTM.

    Returns (x, y, epsg, transformer, inv_transformer).
    """
    from pyproj import Transformer

    lats = np.asarray(overview.variables["latitude"], dtype=float)
    lons = np.asarray(overview.variables["longitude"], dtype=float)

    epsg = _get_utm_epsg(lats, lons)
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    inv_transformer = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )

    if progress:
        tqdm.write(f"Projecting {len(lats):,} pings to UTM (EPSG:{epsg})")

    x, y = transformer.transform(lons, lats)
    return np.asarray(x), np.asarray(y), epsg, transformer, inv_transformer


def _assign_nearest_batched(coords, centers, batch_size=200_000):
    """Assign each point to its nearest center, in batches (memory-safe).

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
    centers : np.ndarray, shape (k, 2)
    batch_size : int

    Returns
    -------
    np.ndarray of int, shape (n,)
    """
    n = coords.shape[0]
    labels = np.empty(n, dtype=np.intp)
    # (a-b)² = a² - 2ab + b²  — avoids materialising (n, k, 2)
    c_sq = np.sum(centers ** 2, axis=1)  # (k,)
    for i in range(0, n, batch_size):
        chunk = coords[i : i + batch_size]
        dists_sq = np.sum(chunk ** 2, axis=1, keepdims=True) - 2 * (chunk @ centers.T) + c_sq
        labels[i : i + batch_size] = np.argmin(dists_sq, axis=1)
    return labels


def _relabel_by_size(labels, min_cluster_size=1):
    """Relabel clusters 0..k−1 sorted by descending size.

    Clusters smaller than *min_cluster_size* get label −1.

    Returns (new_labels, kept_original_ids).
    """
    comp_ids, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    size_order = np.argsort(-counts)  # largest first

    new_labels = np.full(len(labels), -1, dtype=int)
    kept = []
    for old_pos in size_order:
        if counts[old_pos] < min_cluster_size:
            continue
        new_labels[inverse == old_pos] = len(kept)
        kept.append(comp_ids[old_pos])

    return new_labels, kept


def _build_cluster_result(overview, labels, x, y, inv_transformer, progress):
    """Build a ClusterResult from per-ping labels and UTM coordinates."""
    n_clusters = int(labels.max() + 1) if len(labels) > 0 and labels.max() >= 0 else 0

    overviews = []
    centers_latlon = []

    for ci in tqdm(range(n_clusters), desc="Building clusters", disable=not progress):
        indices = np.where(labels == ci)[0]
        ov = subset_overview(overview, indices)
        overviews.append(ov)

        cx, cy = float(x[indices].mean()), float(y[indices].mean())
        clon, clat = inv_transformer.transform(cx, cy)
        centers_latlon.append((float(clat), float(clon)))

    return ClusterResult(
        overviews=overviews,
        centers_latlon=centers_latlon,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Grid-connected-components clustering
# ---------------------------------------------------------------------------

def cluster_by_region(
    overview: PingOverview,
    max_distance_m: float = 1000.0,
    min_cluster_size: int = 10,
    progress: bool = True,
) -> ClusterResult:
    """Cluster pings into contiguous geographic regions.

    Pings are hashed into square grid cells and adjacent occupied cells
    (8-connected) are merged via union-find.  Each connected component
    becomes one cluster.

    **Complexity**: O(n) time and memory — typically < 1 s for 1 M pings.

    Parameters
    ----------
    overview : PingOverview
        The overview to cluster.
    max_distance_m : float
        Grid cell size in metres (default 1 000).  Pings up to ~1.4×
        this distance apart (cell diagonal) can end up in the same cluster.
    min_cluster_size : int
        Drop clusters with fewer pings (default 10).
    progress : bool
        Show tqdm progress bars (default True).

    Returns
    -------
    ClusterResult
        Clusters sorted largest-first.  Each sub-overview has an
        ``original_indices`` attribute.

    Examples
    --------
    >>> result = cluster_by_region(overview, max_distance_m=500)
    >>> for i, ov in enumerate(result):
    ...     lat, lon = result.centers_latlon[i]
    ...     print(f"Cluster {i}: {len(ov.variables['latitude'])} pings")
    """
    if "latitude" not in overview.variables or "longitude" not in overview.variables:
        raise ValueError("Overview must contain 'latitude' and 'longitude' variables.")

    lats = np.asarray(overview.variables["latitude"], dtype=float)
    if len(lats) == 0:
        return ClusterResult([], [], np.array([], dtype=int))

    x, y, _epsg, _tf, inv_transformer = _project_overview_to_utm(overview, progress)

    # --- 1. Hash pings into grid cells (vectorised) ---
    cell_size = float(max_distance_m)
    ix = np.floor(x / cell_size).astype(np.int64)
    iy = np.floor(y / cell_size).astype(np.int64)
    keys = _pack_cells(ix, iy)

    unique_keys = np.unique(keys)                     # sorted
    occupied = set(unique_keys.tolist())               # O(1) lookup
    n_cells = len(occupied)

    if progress:
        tqdm.write(f"Grid: {n_cells:,} cells from {len(lats):,} pings")

    # --- 2. Pre-compute 8-connected neighbour deltas ---
    _zero = int(_pack_cells(np.zeros(1, np.int64), np.zeros(1, np.int64))[0])
    deltas = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            d = int(
                _pack_cells(
                    np.array([di], np.int64), np.array([dj], np.int64)
                )[0]
            ) - _zero
            deltas.append(d)

    # --- 3. Union-Find on occupied cells ---
    uf = _UnionFind()
    for ck in tqdm(unique_keys.tolist(), desc="Connecting cells", disable=not progress):
        uf.find(ck)
        for d in deltas:
            nb = ck + d
            if nb in occupied:
                uf.union(ck, nb)

    # --- 4. Map cells → component roots (loop over unique cells, not pings) ---
    comp_roots = np.array(
        [uf.find(int(k)) for k in unique_keys], dtype=np.int64
    )

    # --- 5. Vectorised: map each ping's cell → component via searchsorted ---
    cell_idx = np.searchsorted(unique_keys, keys)
    ping_components = comp_roots[cell_idx]

    # --- 6. Relabel by size, filter small clusters ---
    labels, _ = _relabel_by_size(ping_components, min_cluster_size)

    return _build_cluster_result(overview, labels, x, y, inv_transformer, progress)


# ---------------------------------------------------------------------------
# Mini-Batch K-Means clustering
# ---------------------------------------------------------------------------

def cluster_by_kmeans(
    overview: PingOverview,
    n_clusters: int = None,
    max_cluster_radius_m: float = None,
    downsample_factor: int = 10,
    min_cluster_size: int = 10,
    batch_size: int = 1024,
    max_k: int = 50,
    progress: bool = True,
) -> ClusterResult:
    """Cluster pings using Mini-Batch K-Means on UTM coordinates.

    Specify **either** *n_clusters* (fixed) **or** *max_cluster_radius_m*
    (automatic: *k* is increased from 2 until every cluster fits within
    the radius, up to *max_k*).

    Fitting is done on downsampled coordinates (every
    *downsample_factor*-th ping); **all** pings are then assigned to the
    nearest center in a batched, memory-efficient pass.

    **Complexity**: O(n) assignment + O(n/ds × k × iters) fitting.
    For 2 M pings with downsample=10 and k≈10, runs in ~1–2 s.

    Requires ``scikit-learn`` (``pip install scikit-learn``).

    Parameters
    ----------
    overview : PingOverview
        The overview to cluster.
    n_clusters : int, optional
        Fixed number of clusters.
    max_cluster_radius_m : float, optional
        Maximum allowed cluster radius in metres.  *k* is increased
        from 2 until satisfied (up to *max_k*).
    downsample_factor : int
        Subsample every n-th ping for fitting (default 10).
    min_cluster_size : int
        Drop clusters with fewer pings (default 10).
    batch_size : int
        Mini-batch size for K-Means (default 1 024).
    max_k : int
        Upper bound on *k* for auto-selection (default 50).
    progress : bool
        Show tqdm progress bars (default True).

    Returns
    -------
    ClusterResult
        Clusters sorted largest-first.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If neither *n_clusters* nor *max_cluster_radius_m* is given.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        raise ImportError(
            "cluster_by_kmeans requires scikit-learn.  "
            "Install it with:  pip install scikit-learn"
        )

    if n_clusters is None and max_cluster_radius_m is None:
        raise ValueError("Specify either n_clusters or max_cluster_radius_m.")

    if "latitude" not in overview.variables or "longitude" not in overview.variables:
        raise ValueError("Overview must contain 'latitude' and 'longitude' variables.")

    lats = np.asarray(overview.variables["latitude"], dtype=float)
    if len(lats) == 0:
        return ClusterResult([], [], np.array([], dtype=int))

    x, y, _epsg, _tf, inv_transformer = _project_overview_to_utm(overview, progress)
    coords_all = np.column_stack([x, y])

    # --- Downsample for fitting ---
    step = max(1, int(downsample_factor))
    sample = coords_all[::step]

    if progress:
        tqdm.write(
            f"K-Means fitting on {len(sample):,} points "
            f"(downsampled {step}× from {len(lats):,})"
        )

    if n_clusters is not None:
        # ---- Fixed k ----
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=42,
            n_init=3,
        ).fit(sample)
    else:
        # ---- Auto k: grow until max_cluster_radius_m is satisfied ----
        km = None
        for k in tqdm(
            range(2, max_k + 1),
            desc="Searching for optimal k",
            disable=not progress,
        ):
            candidate = MiniBatchKMeans(
                n_clusters=k,
                batch_size=batch_size,
                random_state=42,
                n_init=3,
            ).fit(sample)

            # Measure the maximum cluster radius on the sample
            max_r = 0.0
            for j in range(k):
                mask = candidate.labels_ == j
                if mask.any():
                    dists = np.linalg.norm(
                        sample[mask] - candidate.cluster_centers_[j], axis=1
                    )
                    r = float(dists.max())
                    if r > max_r:
                        max_r = r

            if progress:
                tqdm.write(f"  k={k}: max radius = {max_r:,.0f} m")

            if max_r <= max_cluster_radius_m:
                km = candidate
                break

        if km is None:
            if progress:
                tqdm.write(
                    f"Warning: reached max_k={max_k} without satisfying "
                    f"max_cluster_radius_m={max_cluster_radius_m:,.0f} m "
                    f"(last radius={max_r:,.0f} m).  Using k={max_k}."
                )
            km = candidate  # use the last candidate

    centers = km.cluster_centers_

    if progress:
        tqdm.write(
            f"Assigning {len(coords_all):,} pings to {len(centers)} clusters"
        )

    # --- Assign ALL pings (batched for memory) ---
    raw_labels = _assign_nearest_batched(coords_all, centers)

    # --- Relabel by size, filter small clusters ---
    labels, _ = _relabel_by_size(raw_labels, min_cluster_size)

    return _build_cluster_result(overview, labels, x, y, inv_transformer, progress)
