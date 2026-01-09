"""Backend for reading echogram data from ping objects."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm
import warnings

from themachinethatgoesping import echosounders
from themachinethatgoesping.algorithms.geoprocessing.functions import to_raypoints
from themachinethatgoesping.algorithms.gridding import ForwardGridder1D

from .base import EchogramDataBackend
from ..indexers import EchogramImageRequest
from ...helper import select_get_wci_image, apply_pss
from ...helper import make_image_helper


class PingDataBackend(EchogramDataBackend):
    """Backend that reads data from echosounders ping objects.
    
    This backend wraps ping objects and provides access to water column data
    through the EchogramDataBackend interface.
    """

    def __init__(
        self,
        pings: List,
        beam_sample_selections: List,
        ping_times: np.ndarray,
        wci_value: str,
        linear_mean: bool,
        max_sample_counts: np.ndarray,
        sample_nr_extents: Tuple[np.ndarray, np.ndarray],
        range_extents: Optional[Tuple[np.ndarray, np.ndarray]],
        depth_extents: Optional[Tuple[np.ndarray, np.ndarray]],
        ping_params: Dict[str, Tuple[str, np.ndarray]],
        depth_stack: bool = False,
        mp_cores: int = 1,
    ):
        """Initialize PingDataBackend.
        
        Prefer using the `from_pings` factory method instead of this constructor.
        
        Args:
            pings: List of ping objects.
            beam_sample_selections: List of BeamSampleSelection objects, one per ping.
            ping_times: Array of ping timestamps.
            wci_value: Water column image value type (e.g., 'sv', 'av').
            linear_mean: Whether to use linear mean for beam averaging.
            max_sample_counts: Maximum sample count per ping.
            sample_nr_extents: Tuple of (min_sample_nrs, max_sample_nrs) arrays.
            range_extents: Tuple of (min_ranges, max_ranges) arrays, or None.
            depth_extents: Tuple of (min_depths, max_depths) arrays, or None.
            ping_params: Dictionary of pre-computed ping parameters.
            depth_stack: If True, use depth stacking mode (requires navigation).
            mp_cores: Number of cores for parallel processing.
        """
        self._pings = pings
        self._beam_sample_selections = beam_sample_selections
        self._ping_times = ping_times
        self._wci_value = wci_value
        self._linear_mean = linear_mean
        self._max_sample_counts = max_sample_counts
        self._sample_nr_extents = sample_nr_extents
        self._range_extents = range_extents
        self._depth_extents = depth_extents
        self._ping_params = ping_params
        self._depth_stack = depth_stack
        self._mp_cores = mp_cores
        
        if depth_stack and depth_extents is None:
            raise ValueError("depth_stack=True requires navigation data (depth_extents)")

    # =========================================================================
    # Factory method
    # =========================================================================

    @classmethod
    def from_pings(
        cls,
        pings,
        pss=None,
        wci_value: str = "sv/av/pv/rv",
        linear_mean: bool = True,
        no_navigation: bool = False,
        apply_pss_to_bottom: bool = False,
        force_angle: Optional[float] = None,
        depth_stack: bool = False,
        verbose: bool = True,
        mp_cores: int = 1,
    ) -> "PingDataBackend":
        """Create a PingDataBackend from a list of pings.
        
        Args:
            pings: List of ping objects.
            pss: PingSampleSelector for beam/sample selection. If None, uses default.
            wci_value: Water column image value type (e.g., 'sv', 'av', 'sv/av/pv/rv').
            linear_mean: Whether to use linear mean for beam averaging.
            no_navigation: If True, skip depth calculations (useful for data without nav).
            apply_pss_to_bottom: Whether to apply PSS to bottom detection.
            force_angle: Force a specific angle for depth projection (radians).
            depth_stack: If True, use depth stacking mode (requires navigation).
            verbose: Whether to show progress bar.
            mp_cores: Number of cores for parallel processing.
            
        Returns:
            PingDataBackend instance.
        """
        if pss is None:
            pss = echosounders.pingtools.PingSampleSelector()

        # Pre-allocate arrays
        n_pings = len(pings)
        min_s = np.empty(n_pings, dtype=np.float32)
        max_s = np.empty(n_pings, dtype=np.float32)
        min_r = np.empty(n_pings, dtype=np.float32)
        max_r = np.empty(n_pings, dtype=np.float32)
        min_d = np.empty(n_pings, dtype=np.float32)
        max_d = np.empty(n_pings, dtype=np.float32)
        times = np.empty(n_pings, dtype=np.float64)

        # Ping parameter accumulators
        bottom_d_times = []
        bottom_d = []
        minslant_d = []
        echosounder_d_times = []
        echosounder_d = []

        for arr in [min_s, max_s, min_r, max_r, min_d, max_d, times]:
            arr.fill(np.nan)

        beam_sample_selections = []

        for nr, ping in enumerate(tqdm(pings, disable=(not verbose), delay=1, desc="Extracting ping metadata")):
            sel = apply_pss(ping, pss, apply_pss_to_bottom)
            beam_sample_selections.append(sel)

            times[nr] = ping.get_timestamp()
            if len(sel.get_beam_numbers()) == 0:
                continue

            c = ping.watercolumn.get_sound_speed_at_transducer()
            range_res = ping.watercolumn.get_sample_interval() * c * 0.5

            if force_angle is None:
                angle_factor = np.cos(
                    np.radians(np.mean(ping.watercolumn.get_beam_crosstrack_angles()[sel.get_beam_numbers()]))
                )
            else:
                angle_factor = np.cos(np.radians(force_angle))

            min_s[nr] = sel.get_first_sample_number_ensemble()
            max_s[nr] = sel.get_last_sample_number_ensemble()
            min_r[nr] = min_s[nr] * range_res
            max_r[nr] = max_s[nr] * range_res
            echosounder_d_times.append(times[nr])

            if not no_navigation:
                if not ping.has_geolocation():
                    raise RuntimeError(
                        f"ERROR: ping {nr} has no geolocation. Either filter pings based on "
                        "geolocation feature or set no_navigation to True"
                    )

                z = ping.get_geolocation().z
                min_d[nr] = z + min_r[nr] * angle_factor
                max_d[nr] = z + max_r[nr] * angle_factor
                echosounder_d.append(z)

            if max_d[nr] > 6000:
                print(
                    f"WARNING [{nr}], s0={min_s[nr]}, s1={max_s[nr]}, "
                    f"r0={min_r[nr]}, r1={max_r[nr]}, d0={min_d[nr]}, d1={max_d[nr]}, z={z}"
                )

            # Extract bottom parameters if available
            if ping.has_bottom():
                if ping.bottom.has_xyz():
                    try:
                        br = ping.bottom.get_bottom_z(sel)
                        mr = ping.watercolumn.get_minslant_sample_nr() * range_res * angle_factor
                        bottom_d_times.append(times[nr])

                        if not no_navigation:
                            bd = br + z
                            md = mr + z
                            bottom_d.append(bd)
                            minslant_d.append(md)
                    except Exception:
                        pass  # TODO: this should create a warning in the log

        # Compute max sample counts
        max_sample_counts = np.array(
            [sel.get_number_of_samples_ensemble() - 1 for sel in beam_sample_selections]
        )

        # Build ping params dictionary
        ping_params = {}
        if len(bottom_d) > 0:
            ping_params["bottom"] = ("Depth (m)", (bottom_d_times, bottom_d))
            ping_params["minslant"] = ("Depth (m)", (bottom_d_times, minslant_d))
        if len(echosounder_d) > 0:
            ping_params["echosounder"] = ("Depth (m)", (echosounder_d_times, echosounder_d))

        # Build extents
        sample_nr_extents = (min_s, max_s)
        range_extents = (min_r, max_r) if np.any(np.isfinite(min_r)) else None
        depth_extents = (min_d, max_d) if not no_navigation and np.any(np.isfinite(min_d)) else None

        return cls(
            pings=pings,
            beam_sample_selections=beam_sample_selections,
            ping_times=times,
            wci_value=wci_value,
            linear_mean=linear_mean,
            max_sample_counts=max_sample_counts,
            sample_nr_extents=sample_nr_extents,
            range_extents=range_extents,
            depth_extents=depth_extents,
            ping_params=ping_params,
            depth_stack=depth_stack,
            mp_cores=mp_cores,
        )

    # =========================================================================
    # Metadata properties
    # =========================================================================

    @property
    def n_pings(self) -> int:
        return len(self._pings)

    @property
    def ping_times(self) -> np.ndarray:
        return self._ping_times

    @property
    def max_sample_counts(self) -> np.ndarray:
        return self._max_sample_counts

    @property
    def sample_nr_extents(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._sample_nr_extents

    @property
    def range_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._range_extents

    @property
    def depth_extents(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._depth_extents

    @property
    def has_navigation(self) -> bool:
        return self._depth_extents is not None

    @property
    def wci_value(self) -> str:
        return self._wci_value

    @property
    def linear_mean(self) -> bool:
        return self._linear_mean

    # =========================================================================
    # Ping parameters
    # =========================================================================

    def get_ping_params(self) -> Dict[str, Tuple[str, Tuple]]:
        """Return pre-computed ping parameters.
        
        Returns:
            Dictionary mapping parameter names to (y_reference, (times, values)) tuples.
        """
        return self._ping_params

    # =========================================================================
    # Data access methods
    # =========================================================================

    def get_column(self, ping_index: int) -> np.ndarray:
        """Get column data for a ping.
        
        Returns beam-averaged water column data. If depth_stack mode is enabled,
        the data is transformed via raypoints and re-gridded to depth coordinates
        with the same number of samples.
        
        Args:
            ping_index: Index of the ping to retrieve.
            
        Returns:
            1D array of shape (n_samples,) with processed values.
        """
        if self._depth_stack:
            return self._get_depth_stack_column(ping_index)
        else:
            return self._get_range_stack_column(ping_index)

    def _get_range_stack_column(self, ping_index: int) -> np.ndarray:
        """Get beam-averaged column data for a ping (range-stacked)."""
        sel = self._beam_sample_selections[ping_index]
        ping = self._pings[ping_index]

        wci = select_get_wci_image(ping, sel, self._wci_value, self._mp_cores)

        if wci.shape[0] == 1:
            return wci[0]
        else:
            if self._linear_mean:
                wci = np.power(10, wci * 0.1)

            wci = np.nanmean(wci, axis=0)

            if self._linear_mean:
                wci = 10 * np.log10(wci)

            return wci

    def _get_depth_stack_column(self, ping_index: int, from_bottom_xyz: bool = False) -> np.ndarray:
        """Get depth-gridded column data for a ping.
        
        Uses internal gridder based on depth_extents, returns array of same
        size as range_stack (n_samples).
        """
        sel = self._beam_sample_selections[ping_index]
        n_samples = int(self._max_sample_counts[ping_index]) + 1
        
        if sel.empty() or n_samples <= 1:
            column = np.empty(n_samples)
            column.fill(np.nan)
            return column

        min_d = self._depth_extents[0][ping_index]
        max_d = self._depth_extents[1][ping_index]
        
        if not (np.isfinite(min_d) and np.isfinite(max_d) and min_d < max_d):
            column = np.empty(n_samples)
            column.fill(np.nan)
            return column
        
        # Create internal gridder: depth range mapped to n_samples bins
        y_gridder = ForwardGridder1D.from_res((max_d-min_d)/n_samples,min_d, max_d)

        ping = self._pings[ping_index]
        wci = select_get_wci_image(ping, sel, self._wci_value, self._mp_cores)

        if from_bottom_xyz:
            xyz, bd, bdsn = make_image_helper.get_bottom_directions_bottom(ping, selection=sel)
        else:
            xyz, bd, bdsn = make_image_helper.get_bottom_directions_wci(ping, selection=sel)

        geolocation = ping.get_geolocation()

        z = to_raypoints(
            geolocation.z,
            np.array(xyz.z).astype(np.float32),
            0.5,
            np.array(bdsn + 0.5).astype(np.float32),
            np.array(range(wci.shape[1])).astype(np.float32),
        )

        arg = np.where(np.isfinite(wci.flatten()))

        if self._linear_mean:
            wci_flat = np.power(10, 0.1 * wci.flatten()[arg])
        else:
            wci_flat = wci.flatten()[arg]
        z = z.flatten()[arg]

        v, w = y_gridder.interpolate_block_mean(z, wci_flat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self._linear_mean:
                column = 10 * np.log10(v / w)
            else:
                column = v / w

        return column

    def get_raw_column(self, ping_index: int) -> np.ndarray:
        """Get full-resolution beam-averaged column data for a ping.
        
        Always returns range-stacked data regardless of depth_stack mode.
        """
        return self._get_range_stack_column(ping_index)

    def get_beam_sample_selection(self, ping_index: int):
        """Get the beam sample selection for a ping."""
        return self._beam_sample_selections[ping_index]

    # =========================================================================
    # Direct ping access (for advanced use cases)
    # =========================================================================

    def get_ping(self, ping_index: int):
        """Get the raw ping object at the given index.
        
        This provides direct access to the underlying ping for advanced use cases
        that need functionality not exposed through the backend interface.
        """
        return self._pings[ping_index]

    @property
    def pings(self):
        """Direct access to the ping list (for backward compatibility)."""
        return self._pings

    @property
    def beam_sample_selections(self):
        """Direct access to beam sample selections (for backward compatibility)."""
        return self._beam_sample_selections

    # =========================================================================
    # Image generation
    # =========================================================================

    def get_image(self, request: EchogramImageRequest) -> np.ndarray:
        """Build a complete echogram image from a request.
        
        Loops over x columns using get_column() to build the image.
        
        Args:
            request: Image request with ping mapping and affine parameters.
            
        Returns:
            2D array of shape (nx, ny) with echogram data (ping, sample).
        """
        image = np.full((request.nx, request.ny), request.fill_value, dtype=np.float32)
        
        for x_idx in range(request.nx):
            ping_idx = request.ping_indexer[x_idx]
            if ping_idx < 0:
                continue
                
            # Get sample indices for this ping
            sample_indices = request.compute_sample_indices(ping_idx)
            
            # Get column data for this ping
            column_data = self.get_column(ping_idx)
            
            # Map sample indices to values
            valid_mask = sample_indices >= 0
            if np.any(valid_mask):
                image[x_idx, valid_mask] = column_data[sample_indices[valid_mask]]
        
        return image
