"""Aggregate layer acoustic values over time blocks across echograms.

This is a clean redesign of the old ``LayerProcessor``. Given one or more
echograms that share named layers, it pools the acoustic samples that fall
inside each layer into fixed-length time blocks and reduces them (median by
default) to a tidy :class:`pandas.DataFrame`. Typical use is inter-echosounder
comparison / calibration, where one echogram is the reference and others are
compared against it per layer (e.g. per range band).

Design notes
------------
* Echograms are passed as ``{name: echogram}``; nothing is mutated on them.
* Layers are referenced by name and resolved per echogram (so the same layer
  name may cover a different physical band on each echogram -- which is exactly
  what you want after transferring a layer between echograms).
* The per-ping sample extraction loop is unavoidable (data is read per ping),
  but layer sample-index resolution and block assignment are vectorized.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytimeparse2
from tqdm.auto import tqdm


def _resolve_reduce(reduce: Union[str, Callable]) -> Callable:
    if callable(reduce):
        return reduce
    table = {
        "median": np.nanmedian,
        "nanmedian": np.nanmedian,
        "mean": np.nanmean,
        "nanmean": np.nanmean,
        "sum": np.nansum,
        "nansum": np.nansum,
        "max": np.nanmax,
        "nanmax": np.nanmax,
        "min": np.nanmin,
        "nanmin": np.nanmin,
    }
    if reduce not in table:
        raise ValueError(f"Unknown reduce '{reduce}'. Use one of {list(table)} or a callable.")
    return table[reduce]


class LayerProcessor:
    """Pool layer samples into time blocks and reduce them per echogram.

    Parameters
    ----------
    echograms:
        ``{name: echogram}`` mapping (or a single echogram). All echograms must
        expose the requested layer names.
    layers:
        Layer names to process. ``None`` uses the layers common to every
        echogram.
    deltaT:
        Time-block size (e.g. ``'1min'``, ``'30s'``) or seconds as a number.
    step:
        Ping stride for extraction (``1`` = every ping).
    reduce:
        Block reducer applied to the pooled samples: a name
        (``'median'``/``'mean'``/...) or a callable ``f(values)->scalar``.
    min_count:
        Blocks with fewer than this many finite samples are set to NaN.
    show_progress:
        ``True``/``False`` or an existing tqdm instance.
    """

    def __init__(
        self,
        echograms: Union[Dict[str, object], Sequence, object],
        *,
        layers: Optional[Sequence[str]] = None,
        deltaT: Union[str, float] = "1min",
        step: int = 1,
        reduce: Union[str, Callable] = "median",
        min_count: int = 1,
        show_progress: Union[bool, object] = True,
    ):
        self._echograms = self._normalise_echograms(echograms)
        if not self._echograms:
            raise ValueError("LayerProcessor: no echograms provided")
        self._names = list(self._echograms.keys())
        self._deltaT = deltaT
        self._delta_seconds = float(pytimeparse2.parse(deltaT, as_timedelta=False)
                                    if isinstance(deltaT, str) else deltaT)
        self._step = max(1, int(step))
        self._reduce = _resolve_reduce(reduce)
        self._min_count = int(min_count)

        self._layers = self._resolve_layer_names(layers)
        if not self._layers:
            raise ValueError("LayerProcessor: no shared layers to process")

        self._block_edges, self._block_centers = self._make_time_blocks()
        self._data = self._build_dataframe(show_progress)
        # Drop references to the (potentially heavy) echograms once the data has
        # been extracted, so the processor and its results pickle cheaply.
        self._echograms = None

    # -- setup helpers ---------------------------------------------------
    @staticmethod
    def _normalise_echograms(echograms) -> Dict[str, object]:
        if isinstance(echograms, dict):
            return dict(echograms)
        if hasattr(echograms, "get_column"):  # a single echogram
            return {"echogram": echograms}
        return {f"echogram_{i}": e for i, e in enumerate(echograms)}

    def _resolve_layer_names(self, layers: Optional[Sequence[str]]) -> List[str]:
        if layers is not None:
            return list(layers)
        common: Optional[set] = None
        for echo in self._echograms.values():
            names = set(echo.layer_names())
            common = names if common is None else (common & names)
        return sorted(common) if common else []

    def _make_time_blocks(self):
        t0 = min(np.min(echo.ping_times) for echo in self._echograms.values())
        t1 = max(np.max(echo.ping_times) for echo in self._echograms.values())
        # Snap to whole hours around the data for stable, readable blocks.
        start = dt.datetime.fromtimestamp(t0, dt.timezone.utc).replace(
            minute=0, second=0, microsecond=0).timestamp()
        end_dt = dt.datetime.fromtimestamp(t1, dt.timezone.utc).replace(
            minute=0, second=0, microsecond=0)
        end = (end_dt + dt.timedelta(hours=1)).timestamp()
        edges = np.arange(start, end + self._delta_seconds, self._delta_seconds)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return edges, centers

    # -- core extraction -------------------------------------------------
    def _build_dataframe(self, show_progress) -> pd.DataFrame:
        n_blocks = len(self._block_centers)
        index = pd.to_datetime(self._block_centers, unit="s", utc=True)

        total = sum(len(range(0, echo._coord_system.n_pings, self._step))
                    for echo in self._echograms.values())
        progress = self._make_progress(show_progress, total)

        # Collect all columns first, then build the DataFrame in one shot to
        # avoid pandas' per-insert fragmentation penalty.
        all_cols: Dict[str, np.ndarray] = {
            "unixtime": self._block_centers,
        }
        for echo_name, echo in self._echograms.items():
            if progress is not None:
                progress.set_description(f"LayerProcessor [{echo_name}]")
            self._process_echogram(echo_name, echo, n_blocks, all_cols, progress)

        self._close_progress(show_progress, progress)
        return pd.DataFrame(all_cols, index=index)

    def _process_echogram(self, echo_name, echo, n_blocks, cols, progress):
        cs = echo._coord_system
        times = np.asarray(cs.ping_times, dtype=np.float64)
        block_idx = np.searchsorted(self._block_edges, times, side="right") - 1

        bands = {layer: echo.get_layer_sample_indices(layer) for layer in self._layers}
        pools: Dict[str, Dict[int, list]] = {layer: defaultdict(list) for layer in self._layers}

        n_pings = cs.n_pings
        for nr in range(0, n_pings, self._step):
            b = int(block_idx[nr])
            if b < 0 or b >= n_blocks:
                if progress is not None:
                    progress.update(1)
                continue
            column = echo.get_column(nr)
            ncol = len(column)
            for layer in self._layers:
                i0 = int(bands[layer][0][nr])
                i1 = min(int(bands[layer][1][nr]), ncol)
                if i1 > i0:
                    pools[layer][b].append(column[i0:i1])
            if progress is not None:
                progress.update(1)

        for layer in self._layers:
            values = np.full(n_blocks, np.nan, dtype=np.float64)
            counts = np.zeros(n_blocks, dtype=np.int64)
            for b, chunks in pools[layer].items():
                pooled = np.concatenate(chunks)
                finite = np.isfinite(pooled)
                n_finite = int(finite.sum())
                counts[b] = n_finite
                if n_finite >= self._min_count:
                    values[b] = self._reduce(pooled[finite])
            cols[self._value_key(layer, echo_name)] = values
            cols[self._count_key(layer, echo_name)] = counts

    @staticmethod
    def _make_progress(show_progress, total):
        if show_progress is True:
            return tqdm(total=total, desc="LayerProcessor")
        # Use identity checks so we never trigger tqdm's __eq__ override,
        # which raises AttributeError when the other operand is bool/None.
        if show_progress is False or show_progress is None:
            return None
        show_progress.reset(total=total)
        return show_progress

    @staticmethod
    def _close_progress(show_progress, progress):
        if progress is None:
            return
        if show_progress is True:
            progress.close()
        else:
            progress.refresh()

    # -- column naming ---------------------------------------------------
    @staticmethod
    def _value_key(layer: str, echo_name: str) -> str:
        return f"{layer}|{echo_name}|value"

    @staticmethod
    def _count_key(layer: str, echo_name: str) -> str:
        return f"{layer}|{echo_name}|count"

    # -- accessors -------------------------------------------------------
    @property
    def data(self) -> pd.DataFrame:
        """The tidy result frame (one row per time block)."""
        return self._data

    @property
    def layers(self) -> List[str]:
        return list(self._layers)

    @property
    def names(self) -> List[str]:
        return list(self._names)

    @property
    def times(self) -> pd.DatetimeIndex:
        return self._data.index

    def value(self, layer: str, name: str) -> pd.Series:
        """Reduced layer value series for ``layer`` on echogram ``name``."""
        return self._data[self._value_key(layer, name)]

    def count(self, layer: str, name: str) -> pd.Series:
        """Sample-count series for ``layer`` on echogram ``name``."""
        return self._data[self._count_key(layer, name)]

    def difference(self, layer: str, name_a: str, name_b: str) -> pd.Series:
        """``value(layer, name_a) - value(layer, name_b)`` (e.g. calibration)."""
        return self.value(layer, name_a) - self.value(layer, name_b)

    # -- calibration -----------------------------------------------------
    @staticmethod
    def _layer_range(layer: str) -> Optional[float]:
        """Parse a numeric range from a layer name like ``'12.0m'`` -> 12.0."""
        token = layer.split("m")[0]
        try:
            return float(token)
        except ValueError:
            return None

    def calibration_per_range(
        self,
        name_a: str,
        name_b: str,
        *,
        layers: Optional[Iterable[str]] = None,
        bootstrap_resamples: int = 100,
        confidence_level: float = 0.95,
        min_count: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Median difference (``name_a - name_b``) per range-band layer.

        Layers are interpreted by parsing a leading number from their name
        (``'12.0m'`` -> range 12.0 m). Returns a frame indexed by range with
        the median difference and a bootstrap confidence interval.
        """
        from scipy import stats

        layer_list = list(layers) if layers is not None else self._layers
        rows = []
        iterator = tqdm(layer_list) if show_progress else layer_list
        for layer in iterator:
            rng = self._layer_range(layer)
            if rng is None:
                continue
            diff = self.difference(layer, name_a, name_b).values
            diff = diff[np.isfinite(diff)]
            if min_count is not None and len(diff) < min_count:
                continue
            if len(diff) == 0:
                continue
            median = float(np.nanmedian(diff))
            low, high = median, median
            if len(diff) > 1 and bootstrap_resamples > 0:
                res = stats.bootstrap(
                    (diff,), np.nanmedian, n_resamples=bootstrap_resamples,
                    confidence_level=confidence_level, method="percentile")
                low = float(res.confidence_interval.low)
                high = float(res.confidence_interval.high)
            rows.append({
                "range": rng,
                "median_diff": median,
                "ci_low": low,
                "ci_high": high,
                "n": int(len(diff)),
            })
        out = pd.DataFrame(rows).sort_values("range").reset_index(drop=True)
        return out

    def __repr__(self) -> str:
        return (f"LayerProcessor(echograms={self._names}, "
                f"layers={self._layers}, deltaT={self._deltaT!r})")
