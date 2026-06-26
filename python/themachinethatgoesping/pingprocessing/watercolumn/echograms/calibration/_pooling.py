"""Fast time-block pooling of echogram layer samples.

This is the performance-critical core shared by the calibration builder. It is
a generalised, echogram-agnostic version of the old ``LayerProcessor`` inner
loop: per-ping sample bands are resolved vectorially, each ping is assigned to
a time block with a single :func:`numpy.searchsorted`, and the samples that fall
inside a layer are pooled per block and reduced (median by default).

The only unavoidable Python-level loop is the per-ping column read (data is
read one column at a time from the backend); everything else is vectorised.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Callable, Dict, Sequence, Tuple

import numpy as np


def make_time_blocks(t0: float, t1: float, delta_seconds: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(edges, centers)`` for fixed-length time blocks.

    Block *centers* sit on exact ``delta_seconds`` ticks anchored to the whole
    hour at/just before ``t0`` (e.g. whole minutes for ``deltaT='1min'``), and
    *edges* sit half a step around the centers. A ping at time ``t`` belongs to
    block ``searchsorted(edges, t, 'right') - 1``.
    """
    start = dt.datetime.fromtimestamp(t0, dt.timezone.utc).replace(
        minute=0, second=0, microsecond=0).timestamp()
    end_dt = dt.datetime.fromtimestamp(t1, dt.timezone.utc).replace(
        minute=0, second=0, microsecond=0)
    end = (end_dt + dt.timedelta(hours=1)).timestamp()
    centers = np.arange(start, end, delta_seconds)
    edges = np.arange(start - 0.5 * delta_seconds,
                      end + 0.5 * delta_seconds, delta_seconds)
    # Guarantee edges has exactly len(centers)+1 entries.
    edges = edges[: len(centers) + 1]
    return edges, centers


def pool_layers(
    echogram,
    layer_names: Sequence[str],
    block_edges: np.ndarray,
    *,
    step: int = 1,
    reduce: Callable[[np.ndarray], float] = np.nanmedian,
    progress=None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Pool each named layer's samples into time blocks for one echogram.

    Returns ``{layer_name: (values, counts)}`` where ``values`` is the reduced
    value per block (NaN where empty) and ``counts`` the number of finite
    samples per block. Both arrays have length ``len(block_edges) - 1``.
    """
    cs = echogram._coord_system
    times = np.asarray(cs.ping_times, dtype=np.float64)
    n_blocks = len(block_edges) - 1
    block_idx = np.searchsorted(block_edges, times, side="right") - 1

    layer_names = list(layer_names)
    bands = {name: echogram.get_layer_sample_indices(name) for name in layer_names}
    pools: Dict[str, Dict[int, list]] = {name: defaultdict(list) for name in layer_names}

    n_pings = cs.n_pings
    for nr in range(0, n_pings, step):
        b = int(block_idx[nr])
        if 0 <= b < n_blocks:
            column = echogram.get_column(nr)
            ncol = len(column)
            for name in layer_names:
                i0 = int(bands[name][0][nr])
                i1 = min(int(bands[name][1][nr]), ncol)
                if i1 > i0:
                    pools[name][b].append(column[i0:i1])
        if progress is not None:
            progress.update(1)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in layer_names:
        values = np.full(n_blocks, np.nan, dtype=np.float64)
        counts = np.zeros(n_blocks, dtype=np.int64)
        for b, chunks in pools[name].items():
            pooled = np.concatenate(chunks)
            finite = np.isfinite(pooled)
            n_finite = int(finite.sum())
            counts[b] = n_finite
            if n_finite > 0:
                values[b] = reduce(pooled[finite])
        out[name] = (values, counts)
    return out


def pool_values(
    times,
    values,
    block_edges: np.ndarray,
    *,
    reduce: Callable[[np.ndarray], float] = np.nanmedian,
) -> np.ndarray:
    """Pool an external ``(times, values)`` series into the same time blocks.

    ``times`` may be POSIX seconds, ``datetime`` objects or ``datetime64``.
    Returns one reduced value per block (NaN where empty).
    """
    t = _to_unix(times)
    v = np.asarray(values, dtype=np.float64)
    if len(t) != len(v):
        raise ValueError(f"len(times)={len(t)} != len(values)={len(v)}")

    n_blocks = len(block_edges) - 1
    block_idx = np.searchsorted(block_edges, t, side="right") - 1
    valid = (block_idx >= 0) & (block_idx < n_blocks)
    block_idx = block_idx[valid]
    v = v[valid]

    result = np.full(n_blocks, np.nan, dtype=np.float64)
    if len(block_idx) == 0:
        return result

    order = np.argsort(block_idx, kind="stable")
    sorted_blocks = block_idx[order]
    sorted_vals = v[order]
    unique_blocks, starts = np.unique(sorted_blocks, return_index=True)
    ends = np.append(starts[1:], len(sorted_blocks))
    for b, i0, i1 in zip(unique_blocks, starts, ends):
        chunk = sorted_vals[i0:i1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) > 0:
            result[int(b)] = reduce(finite)
    return result


def assign_blocks(times, block_edges: np.ndarray) -> np.ndarray:
    """Per-time block index (``-1`` outside the grid)."""
    t = _to_unix(times)
    n_blocks = len(block_edges) - 1
    idx = np.searchsorted(block_edges, t, side="right") - 1
    idx[(idx < 0) | (idx >= n_blocks)] = -1
    return idx


def _to_unix(times) -> np.ndarray:
    """Coerce POSIX seconds / datetimes / datetime64 to float POSIX seconds."""
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.float64) / 1e9
    if arr.dtype == object and arr.size and isinstance(arr.flat[0], dt.datetime):
        return np.array([x.timestamp() for x in arr.flat], dtype=np.float64)
    return arr.astype(np.float64)
