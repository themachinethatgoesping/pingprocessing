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
from copy import copy
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
        # Keep labels on exact delta ticks (e.g. whole minutes) and use
        # half-step edges for bin assignment, matching legacy LayerProcessor.
        centers = np.arange(start, end, self._delta_seconds)
        edges = np.arange(
            start - 0.5 * self._delta_seconds,
            end + 0.5 * self._delta_seconds,
            self._delta_seconds,
        )
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
    def layers_sorted(self) -> List[str]:
        """Layers sorted numerically by the leading range value (e.g. ``'9m'`` before ``'10m'``)."""
        def _key(layer):
            r = self._layer_range(layer)
            return r if r is not None else float("inf")
        return sorted(self._layers, key=_key)

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

    # -- filtering -------------------------------------------------------
    def filtered_data(
        self,
        *,
        count_filter: Optional[float] = None,
        value_min_quantile: Optional[float] = None,
        value_min_offset: float = 0.0,
    ) -> pd.DataFrame:
        """Return a filtered copy of :attr:`data`.

        Count/layer-size filtering is applied to ``|count`` columns first
        (low counts are set to ``0``), then ``|value`` columns are masked from
        those filtered counts.

        Parameters
        ----------
        count_filter:
            Count/layer-size threshold:
            - ``None``: auto threshold per echogram name from positive counts
              using ``median *0.66``.
            - numeric: fixed minimum sample count.
        value_min_quantile:
            If set, per-column value floor at
            ``nanquantile(value, value_min_quantile) + value_min_offset``.
        value_min_offset:
            Additive offset for the value quantile floor.
        """
        out = self._data.copy()

        if count_filter is None:
            threshold_per_name = {}
            for name in self._names:
                counts = []
                for layer in self._layers:
                    count_key = self._count_key(layer, name)
                    if count_key in out:
                        c = out[count_key].to_numpy(dtype=np.float64, copy=False)
                        counts.append(c[c > 0])
                if counts:
                    n = np.concatenate(counts)
                    if len(n) > 0:
                        median = np.nanmedian(n)
                        threshold_per_name[name] = 0.66 * median
                    else:
                        threshold_per_name[name] = 0
                else:
                    threshold_per_name[name] = 0
        else:
            threshold = int(count_filter)
            threshold_per_name = {name: threshold for name in self._names}

        if threshold_per_name:
            for layer in self._layers:
                for name in self._names:
                    value_key = self._value_key(layer, name)
                    count_key = self._count_key(layer, name)
                    if value_key in out and count_key in out:
                        low_count = out[count_key] < threshold_per_name[name]
                        out.loc[low_count, count_key] = 0
                        out.loc[out[count_key] <= 0, value_key] = np.nan

        if value_min_quantile is not None:
            value_min_quantile = float(value_min_quantile)
            for layer in self._layers:
                for name in self._names:
                    value_key = self._value_key(layer, name)
                    if value_key not in out:
                        continue
                    v = out[value_key].to_numpy(dtype=np.float64, copy=False)
                    if not np.isfinite(v).any():
                        continue
                    vmin = float(np.nanquantile(v, value_min_quantile) + value_min_offset)
                    out.loc[out[value_key] < vmin, value_key] = np.nan

        return out

    def filtered(
        self,
        *,
        count_filter: Optional[float] = None,
        value_min_quantile: Optional[float] = None,
        value_min_offset: float = 0.0,
    ) -> "LayerProcessor":
        """Return a lightweight copy with filtered :attr:`data`.

        The original processor stays unchanged, so you can compare raw vs
        filtered behavior side-by-side.
        """
        proc = copy(self)
        proc._data = self.filtered_data(
            count_filter=count_filter,
            value_min_quantile=value_min_quantile,
            value_min_offset=value_min_offset,
        )
        return proc

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

    def add_param(
        self,
        name: str,
        times,
        values,
        *,
        reduce: Union[str, Callable, None] = None,
    ) -> None:
        """Bin external (times, values) into the same time blocks and add as a column.

        Parameters
        ----------
        name:
            Column name to add to :attr:`data`.
        times:
            Unix timestamps (seconds) of the observations.
        values:
            Corresponding scalar values (same length as *times*).
        reduce:
            Reducer for observations that fall in the same block. Defaults to
            the processor's own reduce function. Accepts the same names or
            callables as the constructor's *reduce* parameter.
        """
        reducer = _resolve_reduce(reduce) if reduce is not None else self._reduce
        t_arr = np.asarray(times)
        if np.issubdtype(t_arr.dtype, np.datetime64):
            times = t_arr.astype("datetime64[ns]").astype(np.float64) / 1e9
        else:
            times = np.asarray(
                [t.timestamp() if isinstance(t, dt.datetime) else float(t) for t in t_arr],
                dtype=np.float64,
            )
        values = np.asarray(values, dtype=np.float64)
        if len(times) != len(values):
            raise ValueError(
                f"add_param: len(times)={len(times)} != len(values)={len(values)}"
            )

        n_blocks = len(self._block_centers)
        block_idx = np.searchsorted(self._block_edges, times, side="right") - 1

        valid = (block_idx >= 0) & (block_idx < n_blocks)
        block_idx = block_idx[valid]
        values = values[valid]

        result = np.full(n_blocks, np.nan, dtype=np.float64)

        if len(block_idx) > 0:
            order = np.argsort(block_idx, kind="stable")
            sorted_blocks = block_idx[order]
            sorted_vals = values[order]

            unique_blocks, starts = np.unique(sorted_blocks, return_index=True)
            ends = np.append(starts[1:], len(sorted_blocks))

            for b, i0, i1 in zip(unique_blocks, starts, ends):
                chunk = sorted_vals[i0:i1]
                finite = chunk[np.isfinite(chunk)]
                if len(finite) > 0:
                    result[b] = reducer(finite)

        self._data = self._data.copy()
        self._data[name] = result

    def split_per_param_range(self, param_name, param_ranges):
        processor_per_param = {}
        
        for name, r0, r1 in param_ranges:
            param_data = self._data[self._data[param_name] >= r0]
            param_data = param_data[param_data[param_name]<= r1]
            processor_per_param[name] = deepcopy(self)
            processor_per_param[name].__data = param_data.copy()
            
        return processor_per_param

    def __repr__(self) -> str:
        return (f"LayerProcessor(echograms={self._names}, "
                f"layers={self._layers}, deltaT={self._deltaT!r})")
