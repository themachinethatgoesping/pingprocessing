from collections import OrderedDict

import numpy as np

#themachinethatgoesping.pingprocessing imports
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

#subpackage imports
from .make_wci import make_wci, make_wci_dual_head, make_wci_stack, make_beam_sample_image, downsample_wci

class ImageBuilder:

    def __init__(
        self, 
        pings,
        horizontal_pixels,
        wci_render = 'linear',
        progress = False,
        oversampling = 1,
        oversampling_mode = 'linear_mean',
        max_cache_images = 200,
        **kwargs):

        self.pings = pings
        self.default_args = {
            "horizontal_pixels" : horizontal_pixels,
        }
        self.default_args.update(kwargs)
        if wci_render == 'beamsample':
            self.beam_sample_view = True
        else:            
            self.beam_sample_view = False
        self.progress = progress
        self.oversampling = max(1, int(oversampling))
        self.oversampling_mode = oversampling_mode

        # Per-ping image cache for sliding-window stack builds.
        # Only active when the view is fixed (hmin/hmax/vmin/vmax all set).
        # Stores dB-domain sub-images keyed by ping index.
        self.max_cache_images = int(max_cache_images)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_key: tuple = ()
        self._cache_coords: tuple = ()  # (y_coordinates, z_coordinates, extent)

    def clear_cache(self):
        """Drop all cached sub-images."""
        self._cache.clear()
        self._cache_key = ()
        self._cache_coords = ()

    def update_args(self, wci_render = 'linear', oversampling = None,
                    oversampling_mode = None, max_cache_images = None, **kwargs):
        if wci_render == 'beamsample':
            self.beam_sample_view = True
        else:            
            self.beam_sample_view = False
        if oversampling is not None:
            self.oversampling = max(1, int(oversampling))
        if oversampling_mode is not None:
            self.oversampling_mode = oversampling_mode
        if max_cache_images is not None:
            self.max_cache_images = int(max_cache_images)
        self.default_args.update(kwargs)

    # -----------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _make_cache_key(kwargs):
        """Hashable fingerprint of args that affect per-ping sub-images."""
        return (
            kwargs.get("horizontal_pixels"),
            kwargs.get("hmin"),
            kwargs.get("hmax"),
            kwargs.get("vmin"),
            kwargs.get("vmax"),
            kwargs.get("wci_value"),
            str(kwargs.get("ping_sample_selector", "")),
            kwargs.get("apply_pss_to_bottom"),
            kwargs.get("from_bottom_xyz"),
            kwargs.get("linear_mean"),
        )

    @staticmethod
    def _compute_fixed_coordinates(hp, hmin, hmax, vmin, vmax):
        """Compute the deterministic pixel grid for a fixed view.

        Mirrors the logic in __WCI_scaling_infos.from_pings_and_limits
        when all four bounds are locked.
        """
        h_range = hmax - hmin
        v_range = vmax - vmin

        if h_range >= v_range:
            y_pixels = hp
            z_pixels = max(1, int(np.ceil(hp * v_range / h_range)))
        else:
            z_pixels = hp
            y_pixels = max(1, int(np.ceil(hp * h_range / v_range)))

        y_coords = np.linspace(hmin, hmax, y_pixels)
        z_coords = np.linspace(vmin, vmax, z_pixels)

        y_res = y_coords[1] - y_coords[0] if len(y_coords) > 1 else h_range
        z_res = z_coords[1] - z_coords[0] if len(z_coords) > 1 else v_range

        extent = (
            hmin - y_res * 0.5,
            hmax + y_res * 0.5,
            vmax + z_res * 0.5,
            vmin - z_res * 0.5,
        )
        return y_coords, z_coords, extent

    def _build_stack_cached(self, index, stack, stack_step, _kwargs):
        """Build a stacked WCI image using the per-ping cache.

        Only called when the view is fixed (all spatial bounds set) and
        ``max_cache_images > 0``.
        """
        # Check / refresh cache validity
        key = self._make_cache_key(_kwargs)
        if key != self._cache_key:
            self._cache.clear()
            self._cache_key = key
            hp = _kwargs["horizontal_pixels"]
            self._cache_coords = self._compute_fixed_coordinates(
                hp, _kwargs["hmin"], _kwargs["hmax"],
                _kwargs["vmin"], _kwargs["vmax"],
            )

        y_coords, z_coords, extent = self._cache_coords

        # Determine which ping indices form the window
        max_index = min(index + stack, len(self.pings))
        needed = list(range(index, max_index, stack_step))

        # Build only uncached pings
        linear_mean = _kwargs.get("linear_mean", True)
        to_build = [i for i in needed if i not in self._cache]

        if to_build:
            it = get_progress_iterator(
                to_build, self.progress,
                desc=f"Caching pings ({len(to_build)}/{len(needed)})",
                total=len(to_build),
            )
            for i in it:
                wci_db, _ = make_wci_dual_head(
                    self.pings[i],
                    horizontal_pixels=_kwargs["horizontal_pixels"],
                    y_coordinates=y_coords,
                    z_coordinates=z_coords,
                    from_bottom_xyz=_kwargs.get("from_bottom_xyz", False),
                    wci_value=_kwargs.get("wci_value", "sv/av/pv/rv"),
                    ping_sample_selector=_kwargs.get("ping_sample_selector"),
                    apply_pss_to_bottom=_kwargs.get("apply_pss_to_bottom", False),
                    mp_cores=_kwargs.get("mp_cores", 1),
                )
                # Store in linear domain when linear_mean is active so that
                # aggregation only needs to sum — no repeated dB→linear conversion.
                if linear_mean:
                    cached = np.empty_like(wci_db, dtype=np.float64)
                    cached.fill(np.nan)
                    use = np.isfinite(wci_db)
                    cached[use] = np.power(10, wci_db[use].astype(np.float64) * 0.1)
                    self._cache[i] = cached
                else:
                    self._cache[i] = wci_db

        # Aggregate cached sub-images
        WCI = None
        NUM = None
        for i in needed:
            img = self._cache[i]
            use = np.isfinite(img)
            if WCI is None:
                WCI = np.full(img.shape, np.nan, dtype=np.float64)
                NUM = np.zeros(img.shape, dtype=np.float64)
            WCI[use] = np.nansum(np.stack([WCI[use], img[use].astype(np.float64)]), axis=0)
            NUM[use] += 1

        if WCI is None:
            WCI = np.empty((len(y_coords), len(z_coords)), dtype=np.float32)
            WCI.fill(np.nan)
        else:
            WCI = WCI / NUM
            if linear_mean:
                WCI = 10 * np.log10(WCI)

        # Evict oldest entries when cache exceeds limit
        while len(self._cache) > self.max_cache_images:
            self._cache.popitem(last=False)

        return WCI.astype(np.float32), extent

    # -----------------------------------------------------------------
    # Main build
    # -----------------------------------------------------------------

    def build(self, index, stack = 1, stack_step = 1, **kwargs):

        _kwargs = self.default_args.copy()
        _kwargs.update(kwargs)
        
        # Apply oversampling: multiply horizontal_pixels
        effective_oversampling = self.oversampling
        if effective_oversampling > 1 and not self.beam_sample_view:
            _kwargs["horizontal_pixels"] = _kwargs["horizontal_pixels"] * effective_oversampling

        if stack > 1:
            # Use cache when view is fixed and caching is enabled
            can_cache = (
                not self.beam_sample_view
                and self.max_cache_images > 0
                and _kwargs.get("hmin") is not None
                and _kwargs.get("hmax") is not None
                and _kwargs.get("vmin") is not None
                and _kwargs.get("vmax") is not None
            )

            if can_cache:
                wci, extent = self._build_stack_cached(
                    index, stack, stack_step, _kwargs)
            else:
                max_index = index + stack
                if max_index > len(self.pings):
                    max_index = len(self.pings)

                stack_pings = self.pings[index:max_index:stack_step]

                wci, extent = make_wci_stack(
                    stack_pings,
                    progress=self.progress,
                    **_kwargs)
        elif self.beam_sample_view:
            wci, extent = make_beam_sample_image(
                self.pings[index],
                **_kwargs)
        else:
            wci, extent = make_wci_dual_head(
                self.pings[index],
                **_kwargs)
        
        # Downsample if oversampling was applied
        if effective_oversampling > 1 and not self.beam_sample_view:
            wci, extent = downsample_wci(wci, extent, effective_oversampling, mode=self.oversampling_mode)
        
        return wci, extent
