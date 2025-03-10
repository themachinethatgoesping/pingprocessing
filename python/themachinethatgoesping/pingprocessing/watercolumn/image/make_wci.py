import numpy as np

from typing import Tuple

# themachinethatgoesping imports
import themachinethatgoesping.echosounders as echosounders
import themachinethatgoesping.algorithms.geoprocessing as geoprocessing

# themachinethatgoesping.pingtools imports
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

# themachinethatgoesping.pingtools/watercolumn imports
import themachinethatgoesping.pingprocessing.watercolumn.helper as wchelper

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class __WCI_scaling_infos:
    def __init__(
        self,
        xyz,
        bottom_directions,
        bottom_direction_sample_numbers,
        geolocation,
        y_coordinates,
        z_coordinates,
        extent,
        ping_offsets,
        ping_sensor_configurations,
    ):
        self.xyz = xyz
        self.bottom_directions = bottom_directions
        self.bottom_direction_sample_numbers = bottom_direction_sample_numbers
        self.y_coordinates = y_coordinates
        self.z_coordinates = z_coordinates
        self.extent = extent
        self.geolocation = geolocation
        self.ping_offsets = ping_offsets
        self.ping_sensor_configurations = ping_sensor_configurations

    @classmethod
    def from_pings_and_coordinates(
        cls,
        pings,
        y_coordinates: float = None,
        z_coordinates: float = None,
        from_bottom_xyz=False,  # this does not yet work,
        ping_sample_selector=echosounders.pingtools.PingSampleSelector(),
        apply_pss_to_bottom=False,
    ):
        if not is_iterable(pings):
            iterator = [pings]
        else:
            iterator = pings

        xyzs = []
        bottom_directions = []
        bottom_direction_sample_numbers = []
        geolocations = []
        ping_offsets = []
        ping_sensor_configurations = []

        valid_pings = False
        for ping_dict in iterator:
            # dual head case
            if not isinstance(ping_dict, dict):
                ping_dict = {"ping": ping_dict}

            for ping in ping_dict.values():
                selection = wchelper.apply_pss(ping, ping_sample_selector, apply_pss_to_bottom)

                if selection.empty():
                    continue
                valid_pings = True

                if from_bottom_xyz:
                    xyz, bd, bdsn = wchelper.make_image_helper.get_bottom_directions_bottom(ping, selection=selection)
                else:
                    xyz, bd, bdsn = wchelper.make_image_helper.get_bottom_directions_wci(ping, selection=selection)

                xyzs.append(xyz)
                bottom_directions.append(bd)
                bottom_direction_sample_numbers.append(bdsn)
                geolocations.append(ping.get_geolocation())
                ping_sensor_configurations.append(ping.get_sensor_configuration())
                ping_offsets.append(ping_sensor_configurations[-1].get_target("Transducer"))

        if not valid_pings:
            raise ValueError("No valid pings found. (BeamSampleSelection empty for all pings)")

        y_res = y_coordinates[1] - y_coordinates[0]
        z_res = z_coordinates[1] - z_coordinates[0]

        # compute the extent
        extent = [
            y_coordinates[0] - y_res * 0.5,
            y_coordinates[-1] + y_res * 0.5,
            z_coordinates[-1] + z_res * 0.5,
            z_coordinates[0] - z_res * 0.5,
        ]

        # single ping case
        if not is_iterable(pings):
            return cls(
                xyz=xyzs[0],
                bottom_directions=bottom_directions[0],
                bottom_direction_sample_numbers=bottom_direction_sample_numbers[0],
                geolocation=geolocations[0],
                y_coordinates=y_coordinates,
                z_coordinates=z_coordinates,
                ping_offsets=ping_offsets[0],
                ping_sensor_configurations=ping_sensor_configurations[0],
                extent=extent,
            )
        return cls(
            xyz=xyzs,
            bottom_directions=bottom_directions,
            bottom_direction_sample_numbers=bottom_direction_sample_numbers,
            geolocation=geolocations,
            y_coordinates=y_coordinates,
            z_coordinates=z_coordinates,
            ping_offsets=ping_offsets,
            ping_sensor_configurations=ping_sensor_configurations,
            extent=extent,
        )

    @classmethod
    def from_pings_and_limits(
        cls,
        pings,
        horizontal_pixels,
        hmin: float = None,
        hmax: float = None,
        vmin: float = None,
        vmax: float = None,
        from_bottom_xyz=False,  # this does not yet work,
        ping_sample_selector=echosounders.pingtools.PingSampleSelector(),
        apply_pss_to_bottom: bool = False,
    ):
        if not is_iterable(pings):
            iterator = [pings]
        else:
            iterator = pings

        _hmin = np.nan
        _hmax = np.nan
        _vmin = np.nan
        _vmax = np.nan

        xyzs = []
        bottom_directions = []
        bottom_direction_sample_numbers = []
        geolocations = []
        ping_offsets = []
        ping_sensor_configurations = []

        valid_pings = False
        for ping_dict in iterator:
            # dual head case
            if not isinstance(ping_dict, dict):
                ping_dict = {"ping": ping_dict}

            for ping in ping_dict.values():
                selection = wchelper.apply_pss(ping, ping_sample_selector, apply_pss_to_bottom)

                if selection.empty():
                    continue
                valid_pings = True

                if from_bottom_xyz:
                    xyz, bd, bdsn = wchelper.make_image_helper.get_bottom_directions_bottom(ping, selection=selection)
                else:
                    xyz, bd, bdsn = wchelper.make_image_helper.get_bottom_directions_wci(ping, selection=selection)

                xyzs.append(xyz)
                bottom_directions.append(bd)
                bottom_direction_sample_numbers.append(bdsn)
                geolocations.append(ping.get_geolocation())
                ping_sensor_configurations.append(ping.get_sensor_configuration())
                ping_offsets.append(ping_sensor_configurations[-1].get_target("Transducer"))

                # compute limits of the create image
                tr_y = ping_offsets[-1].y
                if hmin is None:
                    _hmin = np.nanmin([_hmin, np.nanmin(xyz.y), tr_y])
                if hmax is None:
                    _hmax = np.nanmax([_hmax, np.nanmax(xyz.y), tr_y])
                if vmax is None:
                    _vmax = np.nanmax([_vmax, np.nanmax(xyz.z)])

        if not valid_pings:
            raise ValueError("No valid pings found. (BeamSampleSelection empty for all pings)")

        if hmin is None:
            hmin = _hmin * 1.02
        if hmax is None:
            hmax = _hmax * 1.02
        if vmin is None:
            vmin = np.nanmin([g.z for g in geolocations])
        if vmax is None:
            vmax = _vmax + (_vmax - vmin) * 0.01

        # build array with backtraced positions (beam angle, range from transducer)
        y_coordinates = np.linspace(hmin, hmax, horizontal_pixels)
        res = y_coordinates[1] - y_coordinates[0]
        z_coordinates = np.arange(vmin, vmax + res, res)

        # compute the extent
        extent = [hmin - res * 0.5, hmax + res * 0.5, vmax + res * 0.5, vmin - res * 0.5]

        # single ping case
        if not is_iterable(pings):
            return cls(
                xyz=xyzs[0],
                bottom_directions=bottom_directions[0],
                bottom_direction_sample_numbers=bottom_direction_sample_numbers[0],
                geolocation=geolocations[0],
                y_coordinates=y_coordinates,
                z_coordinates=z_coordinates,
                ping_offsets=ping_offsets[0],
                ping_sensor_configurations=ping_sensor_configurations[0],
                extent=extent,
            )
        return cls(
            xyz=xyzs,
            bottom_directions=bottom_directions,
            bottom_direction_sample_numbers=bottom_direction_sample_numbers,
            geolocation=geolocations,
            y_coordinates=y_coordinates,
            z_coordinates=z_coordinates,
            ping_offsets=ping_offsets,
            ping_sensor_configurations=ping_sensor_configurations,
            extent=extent,
        )


def make_beam_sample_image(
    ping: echosounders.filetemplates.I_Ping,
    hmin: float = None,
    hmax: float = None,
    vmin: float = None,
    vmax: float = None,
    wci_value: str = "sv/av/pv/rv",
    ping_sample_selector=echosounders.pingtools.PingSampleSelector(),
    apply_pss_to_bottom: bool = False,
    **kwargs,
):

    # dual head case
    if isinstance(ping, dict):
        W = []
        nbeams = 0
        nsamples = 0
        for p in reversed(sorted(ping.values(), key=lambda p: np.mean(p.watercolumn.get_beam_crosstrack_angles()))):
            w, e = make_beam_sample_image(
                p,
                hmin=hmin,
                hmax=hmax,
                vmin=vmin,
                wci_value=wci_value,
                ping_sample_selector=ping_sample_selector,
                apply_pss_to_bottom=apply_pss_to_bottom,
                **kwargs,
            )
            nbeams += w.shape[0]
            nsamples = max([nsamples, w.shape[1]])
            W.append(w)

        wci = np.empty((nbeams, nsamples))
        wci.fill(np.nan)
        b = 0
        for w in W:
            wci[b : b + w.shape[0], : w.shape[1]] = w
            b += w.shape[0]

        return wci, [-0.5, nbeams + 0.5, nsamples + 0.5, -0.5]

    sel = wchelper.apply_pss(ping, ping_sample_selector, apply_pss_to_bottom)

    # select which ping.watercolumn.get_ function to call based on wci_value
    wci = wchelper.select_get_wci_image(ping, sel, wci_value)

    return wci, [-0.5, wci.shape[0] + 0.5, wci.shape[1] + 0.5, -0.5]


def make_wci(
    ping: echosounders.filetemplates.I_Ping,
    horizontal_pixels: int,
    hmin: float = None,
    hmax: float = None,
    vmin: float = None,
    vmax: float = None,
    y_coordinates: float = None,
    z_coordinates: float = None,
    from_bottom_xyz: bool = False,
    wci_value: str = "sv/av/pv/rv",
    ping_sample_selector=echosounders.pingtools.PingSampleSelector(),
    apply_pss_to_bottom: bool = False,
    mp_cores: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:

    if (y_coordinates is None and z_coordinates is not None) or (y_coordinates is not None and z_coordinates is None):
        raise ValueError("if y_coordinates or z_coordinates is specified, both must be specified.")

    if y_coordinates is not None and z_coordinates is not None:
        try:
            scaling_infos = __WCI_scaling_infos.from_pings_and_coordinates(
                pings=ping,
                y_coordinates=y_coordinates,
                z_coordinates=z_coordinates,
                from_bottom_xyz=from_bottom_xyz,
                ping_sample_selector=ping_sample_selector,
                apply_pss_to_bottom=apply_pss_to_bottom,
            )
        except ValueError:
            y_res = y_coordinates[1] - y_coordinates[0]
            z_res = z_coordinates[1] - z_coordinates[0]

            # compute the extent
            extent = [
                y_coordinates[0] - y_res * 0.5,
                y_coordinates[-1] + y_res * 0.5,
                z_coordinates[-1] + z_res * 0.5,
                z_coordinates[0] - z_res * 0.5,
            ]

            wci = np.empty((len(y_coordinates), len(z_coordinates)), dtype=np.float32)
            wci.fill(np.nan)
            return wci, tuple(extent)

    else:
        try:
            scaling_infos = __WCI_scaling_infos.from_pings_and_limits(
                pings=ping,
                horizontal_pixels=horizontal_pixels,
                hmin=hmin,
                hmax=hmax,
                vmin=vmin,
                vmax=vmax,
                from_bottom_xyz=from_bottom_xyz,
                ping_sample_selector=ping_sample_selector,
                apply_pss_to_bottom=apply_pss_to_bottom,
            )
        except ValueError:
            return np.empty((0, 0), dtype=np.float32), (0, 0, 0, 0)

    # t.append(time()) # 4
    # bt = geoprocessing.backtracers.BTConstantSVP(
    #     scaling_infos.geolocation, scaling_infos.ping_offsets.x, scaling_infos.ping_offsets.y
    # )
    # geolocation x is x location in image as well for now
    bt = geoprocessing.backtracers.BTConstantSVP(scaling_infos.geolocation, 0, scaling_infos.ping_offsets.y)

    # t.append(time()) # 5
    sd_grid = bt.backtrace_image(scaling_infos.y_coordinates, scaling_infos.z_coordinates, mp_cores=mp_cores)

    sel = wchelper.apply_pss(ping, ping_sample_selector, apply_pss_to_bottom)

    if sel.empty():
        wci = np.empty(sd_grid.shape(), dtype=np.float32)
        wci.fill(np.nan)

    else:
        # t.append(time()) # 6

        # select which ping.watercolumn.get_ function to call based on wci_value
        wci = wchelper.select_get_wci_image(ping, sel, wci_value)

        # t.append(time()) # 7
        # lookup beam/sample numbers for each pixel
        wci = bt.lookup(
            wci,
            scaling_infos.bottom_directions,
            scaling_infos.bottom_direction_sample_numbers,
            sd_grid,
            wci_first_sample_number=sel.get_first_sample_number_ensemble(),
            wci_sample_number_step=sel.get_sample_step_ensemble(),
            mp_cores=mp_cores,
        )

        # t.append(time()) # 8

    # for i in range(1,len(t)):
    #    print(f"Time {i}: {t[i]-t[i-1]}, {t[i]-t[0]}")

    # return the resulting water column image and the extent of the image
    return wci, tuple(scaling_infos.extent)


def make_wci_dual_head(
    ping_group: echosounders.filetemplates.I_Ping,
    horizontal_pixels: int,
    hmin: float = None,
    hmax: float = None,
    vmin: float = None,
    vmax: float = None,
    y_coordinates=None,
    z_coordinates=None,
    from_bottom_xyz: bool = False,
    wci_value: str = "sv/av/pv/rv",
    ping_sample_selector=echosounders.pingtools.PingSampleSelector(),
    apply_pss_to_bottom: bool = False,
    mp_cores: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:

    if not isinstance(ping_group, dict):
        return make_wci(
            ping_group,
            horizontal_pixels,
            hmin,
            hmax,
            vmin,
            vmax,
            y_coordinates,
            z_coordinates,
            from_bottom_xyz,
            wci_value,
            ping_sample_selector,
            apply_pss_to_bottom,
            mp_cores,
        )

    pings = list(ping_group.values())
    if len(pings) == 1:
        return make_wci(
            pings[0],
            horizontal_pixels,
            hmin,
            hmax,
            vmin,
            vmax,
            y_coordinates,
            z_coordinates,
            from_bottom_xyz,
            wci_value,
            ping_sample_selector,
            apply_pss_to_bottom,
            mp_cores,
        )
    if len(pings) != 2:
        raise ValueError("ping_group must contain exactly one or two pings.")

    ping1, ping2 = pings
    if np.nanmedian(ping1.watercolumn.get_beam_crosstrack_angles()) < np.nanmedian(
        ping2.watercolumn.get_beam_crosstrack_angles()
    ):
        ping1, ping2 = ping2, ping1

    if y_coordinates is not None and z_coordinates is not None:
        try:
            scaling_infos = __WCI_scaling_infos.from_pings_and_coordinates(
                pings=pings,
                y_coordinates=y_coordinates,
                z_coordinates=z_coordinates,
                from_bottom_xyz=from_bottom_xyz,
                ping_sample_selector=ping_sample_selector,
                apply_pss_to_bottom=apply_pss_to_bottom,
            )
        except ValueError:
            y_res = y_coordinates[1] - y_coordinates[0]
            z_res = z_coordinates[1] - z_coordinates[0]

            # compute the extent
            extent = [
                y_coordinates[0] - y_res * 0.5,
                y_coordinates[-1] + y_res * 0.5,
                z_coordinates[-1] + z_res * 0.5,
                z_coordinates[0] - z_res * 0.5,
            ]

            wci = np.empty((len(y_coordinates), len(z_coordinates)), dtype=np.float32)
            wci.fill(np.nan)
            return wci, tuple(extent)

    else:
        try:
            scaling_infos = __WCI_scaling_infos.from_pings_and_limits(
                pings=pings,
                horizontal_pixels=horizontal_pixels,
                hmin=hmin,
                hmax=hmax,
                vmin=vmin,
                vmax=vmax,
                from_bottom_xyz=from_bottom_xyz,
                ping_sample_selector=ping_sample_selector,
                apply_pss_to_bottom=apply_pss_to_bottom,
            )
        except ValueError:
            return np.empty((0, 0), dtype=np.float32), (0, 0, 0, 0)

    y_coordinates1 = scaling_infos.y_coordinates[scaling_infos.y_coordinates <= 0]
    y_coordinates2 = scaling_infos.y_coordinates[scaling_infos.y_coordinates > 0]

    try:
        wci1, extent1 = make_wci(
            ping1,
            horizontal_pixels,
            y_coordinates=y_coordinates1,
            z_coordinates=scaling_infos.z_coordinates,
            from_bottom_xyz=from_bottom_xyz,
            wci_value=wci_value,
            ping_sample_selector=ping_sample_selector,
            apply_pss_to_bottom=apply_pss_to_bottom,
            mp_cores=mp_cores,
        )
    except Exception as e:
        return make_wci(
            ping2,
            horizontal_pixels,
            hmin,
            hmax,
            vmin,
            vmax,
            y_coordinates,
            z_coordinates,
            from_bottom_xyz,
            wci_value,
            ping_sample_selector,
            apply_pss_to_bottom,
            mp_cores,
        )

    try:
        wci2, extent2 = make_wci(
            ping2,
            horizontal_pixels,
            y_coordinates=y_coordinates2,
            z_coordinates=scaling_infos.z_coordinates,
            from_bottom_xyz=from_bottom_xyz,
            wci_value=wci_value,
            ping_sample_selector=ping_sample_selector,
            apply_pss_to_bottom=apply_pss_to_bottom,
            mp_cores=mp_cores,
        )
    except Exception as e:
        return make_wci(
            ping1,
            horizontal_pixels,
            hmin,
            hmax,
            vmin,
            vmax,
            y_coordinates,
            z_coordinates,
            from_bottom_xyz,
            wci_value,
            ping_sample_selector,
            apply_pss_to_bottom,
            mp_cores,
        )

    # Return
    return np.append(wci1, wci2, axis=0), tuple(scaling_infos.extent)


def make_wci_stack(
    pings: list,
    horizontal_pixels: int,
    linear_mean: bool = True,
    hmin: float = None,
    hmax: float = None,
    vmin: float = None,
    vmax: float = None,
    from_bottom_xyz: bool = False,
    wci_value: str = "sv/av/pv/rv",
    ping_sample_selector=echosounders.pingtools.PingSampleSelector(),
    apply_pss_to_bottom: bool = False,
    progress=None,
    mp_cores: int = 1,
    **kwargs,
):
    try:
        scaling_infos = __WCI_scaling_infos.from_pings_and_limits(
            pings=pings,
            horizontal_pixels=horizontal_pixels,
            hmin=hmin,
            hmax=hmax,
            vmin=vmin,
            vmax=vmax,
            from_bottom_xyz=from_bottom_xyz,
            ping_sample_selector=ping_sample_selector,
            apply_pss_to_bottom=apply_pss_to_bottom,
        )
    except ValueError:
        return np.empty((0, 0), dtype=np.float32), (0, 0, 0, 0)

    WCI = None
    NUM = None

    it = get_progress_iterator(pings, progress, desc="Stacking pings")

    # loop through each ping
    for pn, ping in enumerate(it):
        # create backtracer object
        wci, extent = make_wci_dual_head(
            ping,
            horizontal_pixels=horizontal_pixels,
            y_coordinates=scaling_infos.y_coordinates,
            z_coordinates=scaling_infos.z_coordinates,
            from_bottom_xyz=from_bottom_xyz,
            wci_value=wci_value,
            ping_sample_selector=ping_sample_selector,
            apply_pss_to_bottom=apply_pss_to_bottom,
            mp_cores=mp_cores,
        )

        use = np.isfinite(wci).astype(bool)

        # apply linear mean if specified
        if linear_mean:
            wci[use] = np.power(10, wci[use] * 0.1)

        # initialize WCI and NUM arrays
        if WCI is None:
            WCI = np.empty_like(wci)
            WCI.fill(np.nan)
            NUM = np.zeros_like(wci)

        # accumulate WCI and NUM arrays
        WCI[use] = np.nansum([WCI[use], wci[use]], axis=0)
        NUM[use] += 1

    # compute the final WCI array
    WCI = WCI / NUM

    # apply logarithmic scaling if specified
    if linear_mean:
        WCI = 10 * np.log10(WCI)

    # return the WCI array and extent
    return WCI, tuple(scaling_infos.extent)
