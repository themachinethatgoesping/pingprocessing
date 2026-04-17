"""Backend for pre-built echogram images with extent metadata.

Wraps a 2D numpy array (in-memory or memory-mapped) together with
matplotlib-style extent metadata so it can be used as an EchogramBuilder
backend.  This replaces the old ``EchoData`` helper.

Typical usage::

    # In-memory
    backend = ImageBackend.from_image(image, ping_times, y_min, y_max,
                                      y_axis="depth")

    # Memory-mapped file
    backend = ImageBackend.from_mmap(path, ping_times, y_min, y_max,
                                     y_axis="range")

    builder = EchogramBuilder.from_backend(backend)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .base import EchogramDataBackend
from .storage_mode import StorageAxisMode
from ..indexers import EchogramImageRequest


class ImageBackend(EchogramDataBackend):
    """Backend for a pre-built 2D echogram image with uniform y-axis.

    The image has shape ``(n_pings, n_samples)`` and a regular y-axis
    grid defined by per-ping ``(y_min, y_max)`` arrays (range or depth
    in metres, or plain sample numbers).

    Supports both in-memory ``ndarray`` and on-disk ``memmap`` arrays.
    """

    def __init__(
        self,
        image: np.ndarray,
        ping_times: np.ndarray,
        y_min: np.ndarray,
        y_max: np.ndarray,
        *,
        y_axis: str = "range",
        wci_value: str = "sv",
        linear_mean: bool = True,
        latitudes: Optional[np.ndarray] = None,
        longitudes: Optional[np.ndarray] = None,
        ping_params: Optional[Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    ):
        """Initialise the backend.

        Prefer the factory methods ``from_image`` / ``from_mmap``.

        Parameters
        ----------
        image : ndarray or memmap, shape (n_pings, n_samples)
            Echogram data.  May be float32/float64.
        ping_times : ndarray, shape (n_pings,)
            Unix timestamps per ping.
        y_min, y_max : ndarray, shape (n_pings,)
            Y-axis (range/depth) extents per ping.  Scalars are broadcast.
        y_axis : str
            One of ``"range"``, ``"depth"``, ``"sample_index"``.
        wci_value : str
            Label for the stored quantity (e.g. ``"sv"``).
        linear_mean : bool
            Whether beam averaging was done in linear domain.
        latitudes, longitudes : ndarray, optional
            Per-ping coordinates.
        ping_params : dict, optional
            Pre-computed ping parameters, same format as other backends:
            ``{name: (y_reference, (timestamps, values))}``.
        """
        if image.ndim != 2:
            raise ValueError(f"image must be 2-D, got shape {image.shape}")

        n_pings, n_samples = image.shape
        ping_times = np.asarray(ping_times, dtype=np.float64).ravel()
        if ping_times.shape[0] != n_pings:
            raise ValueError(
                f"ping_times length ({ping_times.shape[0]}) != n_pings ({n_pings})"
            )

        # Accept scalar or per-ping y extents
        y_min = np.broadcast_to(np.asarray(y_min, dtype=np.float64), (n_pings,)).copy()
        y_max = np.broadcast_to(np.asarray(y_max, dtype=np.float64), (n_pings,)).copy()

        self._image = image
        self._ping_times = ping_times
        self._y_min = y_min
        self._y_max = y_max
        self._n_pings = n_pings
        self._n_samples = n_samples
        self._y_axis = y_axis
        self._wci_value = wci_value
        self._linear_mean = linear_mean
        self._latitudes = latitudes
        self._longitudes = longitudes
        self._ping_params = ping_params or {}

        # Pre-compute sample counts (constant unless masked)
        # Use n_samples - 1 (max valid index) to match PingDataBackend convention.
        # This ensures the affine mapping: sample 0 → y_min, sample (n-1) → y_max.
        self._max_sample_counts = np.full(n_pings, n_samples - 1, dtype=np.int64)

        # Pre-compute sample-number extents (0..n_samples-1 for every ping)
        self._sample_nr_min = np.zeros(n_pings, dtype=np.int64)
        self._sample_nr_max = np.full(n_pings, n_samples - 1, dtype=np.int64)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        ping_times: np.ndarray,
        y_min: Union[float, np.ndarray],
        y_max: Union[float, np.ndarray],
        *,
        y_axis: str = "range",
        wci_value: str = "sv",
        linear_mean: bool = True,
        latitudes: Optional[np.ndarray] = None,
        longitudes: Optional[np.ndarray] = None,
        ping_params: Optional[Dict] = None,
    ) -> "ImageBackend":
        """Create a backend from an in-memory numpy array.

        Parameters
        ----------
        image : ndarray (n_pings, n_samples)
            Echogram image data.
        ping_times : ndarray (n_pings,)
            Unix timestamps.
        y_min, y_max : float or ndarray (n_pings,)
            Per-ping y-axis extents.  Scalars are broadcast.
        y_axis : str
            ``"range"``, ``"depth"`` or ``"sample_index"``.
        wci_value, linear_mean, latitudes, longitudes, ping_params
            See ``__init__``.
        """
        image = np.asarray(image, dtype=np.float32)
        return cls(
            image, ping_times, y_min, y_max,
            y_axis=y_axis,
            wci_value=wci_value,
            linear_mean=linear_mean,
            latitudes=latitudes,
            longitudes=longitudes,
            ping_params=ping_params,
        )

    @classmethod
    def from_mmap(
        cls,
        path: Union[str, Path],
        ping_times: np.ndarray,
        y_min: Union[float, np.ndarray],
        y_max: Union[float, np.ndarray],
        *,
        shape: Optional[Tuple[int, int]] = None,
        dtype: np.dtype = np.float32,
        y_axis: str = "range",
        wci_value: str = "sv",
        linear_mean: bool = True,
        latitudes: Optional[np.ndarray] = None,
        longitudes: Optional[np.ndarray] = None,
        ping_params: Optional[Dict] = None,
    ) -> "ImageBackend":
        """Create a backend from a memory-mapped binary file.

        Parameters
        ----------
        path : str or Path
            Path to the raw binary file (flat float32 by default).
        ping_times : ndarray (n_pings,)
            Unix timestamps.
        y_min, y_max : float or ndarray
            Per-ping y-axis extents.
        shape : (n_pings, n_samples), optional
            If *None*, inferred from ``len(ping_times)`` and file size.
        dtype : numpy dtype
            Element type of the binary file (default ``float32``).
        y_axis, wci_value, linear_mean, latitudes, longitudes, ping_params
            See ``__init__``.
        """
        path = Path(path)
        ping_times = np.asarray(ping_times, dtype=np.float64).ravel()

        if shape is None:
            file_bytes = path.stat().st_size
            n_pings = len(ping_times)
            n_samples = file_bytes // (n_pings * np.dtype(dtype).itemsize)
            shape = (n_pings, n_samples)

        mmap = np.memmap(path, dtype=dtype, mode="r", shape=shape)

        return cls(
            mmap, ping_times, y_min, y_max,
            y_axis=y_axis,
            wci_value=wci_value,
            linear_mean=linear_mean,
            latitudes=latitudes,
            longitudes=longitudes,
            ping_params=ping_params,
        )

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def n_pings(self) -> int:
        return self._n_pings

    @property
    def ping_times(self) -> np.ndarray:
        return self._ping_times

    @property
    def max_sample_counts(self) -> np.ndarray:
        return self._max_sample_counts

    @property
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._sample_nr_min, self._sample_nr_max

    @property
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._y_axis == "range":
            return self._y_min, self._y_max
        return None

    @property
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._y_axis == "depth":
            return self._y_min, self._y_max
        return None

    @property
    def has_navigation(self) -> bool:
        return self._y_axis == "depth"

    @property
    def latitudes(self) -> Optional[np.ndarray]:
        return self._latitudes

    @property
    def longitudes(self) -> Optional[np.ndarray]:
        return self._longitudes

    @property
    def wci_value(self) -> str:
        return self._wci_value

    @property
    def linear_mean(self) -> bool:
        return self._linear_mean

    @property
    def storage_mode(self) -> StorageAxisMode:
        y_map = {"range": "range", "depth": "depth", "sample_index": "sample_index"}
        y_type = y_map.get(self._y_axis, "sample_index")

        y_resolution = 1.0
        y_origin = 0.0
        if self._y_axis in ("range", "depth") and self._n_samples > 1:
            # Uniform y grid: y = y_min + i * resolution
            y_resolution = float(
                (self._y_max[0] - self._y_min[0]) / (self._n_samples - 1)
            )
            y_origin = float(self._y_min[0])

        return StorageAxisMode(
            x_axis="ping_time",
            y_axis=y_type,
            y_resolution=y_resolution,
            y_origin=y_origin,
        )

    # ------------------------------------------------------------------
    # Ping parameters
    # ------------------------------------------------------------------

    def get_ping_params(self) -> Dict[str, Tuple[str, Tuple[np.ndarray, np.ndarray]]]:
        return dict(self._ping_params)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_column(self, ping_index: int) -> np.ndarray:
        return np.asarray(self._image[int(ping_index), :], dtype=np.float32).copy()

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        return self.get_column(ping_index)

    def get_chunk(self, start_ping: int, end_ping: int) -> np.ndarray:
        return np.asarray(self._image[start_ping:end_ping, :], dtype=np.float32)

    # ------------------------------------------------------------------
    # Image generation (vectorised, mirrors MmapDataBackend approach)
    # ------------------------------------------------------------------

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        image = np.full(
            (request.nx, request.ny), request.fill_value, dtype=np.float32
        )

        valid_x_mask = request.ping_indexer >= 0
        valid_x_indices = np.where(valid_x_mask)[0]
        if len(valid_x_indices) == 0:
            return image

        unique_pings, inverse_indices = np.unique(
            request.ping_indexer[valid_x_mask], return_inverse=True
        )

        n_unique = len(unique_pings)
        ny = request.ny
        y_coords = request.y_coordinates

        # Affine: sample_idx = round(a + b * y)
        a = request.affine_a[unique_pings, np.newaxis]   # (n_unique, 1)
        b = request.affine_b[unique_pings, np.newaxis]
        max_s = request.max_sample_indices[unique_pings, np.newaxis]

        sample_idx_f = a + b * y_coords[np.newaxis, :]
        nan_mask = np.isnan(sample_idx_f)
        sample_idx = np.where(nan_mask, -1, np.rint(sample_idx_f)).astype(np.int32)
        valid = (sample_idx >= 0) & (sample_idx < max_s)
        sample_idx_clipped = np.clip(sample_idx, 0, self._n_samples - 1)

        # Gather via advanced indexing
        ping_flat = np.repeat(unique_pings, ny)
        sample_flat = sample_idx_clipped.ravel()
        values = np.asarray(
            self._image[ping_flat, sample_flat], dtype=np.float32
        ).reshape(n_unique, ny)
        values = np.where(valid, values, request.fill_value)

        image[valid_x_indices] = values[inverse_indices]
        return image

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mem = "memmap" if isinstance(self._image, np.memmap) else "ndarray"
        return (
            f"ImageBackend(n_pings={self._n_pings}, n_samples={self._n_samples}, "
            f"y_axis='{self._y_axis}', wci_value='{self._wci_value}', "
            f"storage={mem})"
        )
