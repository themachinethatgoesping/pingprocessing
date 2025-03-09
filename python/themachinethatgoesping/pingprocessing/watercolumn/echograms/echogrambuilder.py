import numpy as np

from typing import Tuple
from collections import defaultdict
from copy import deepcopy
import datetime as dt
from tqdm.auto import tqdm

import matplotlib as mpl
import matplotlib.dates as mdates

# external Ping packages
from themachinethatgoesping import echosounders
from themachinethatgoesping import tools

# internal Ping.pingprocessing packages
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.pingprocessing.core.asserts import assert_length, assert_valid_argument

from themachinethatgoesping.pingprocessing.watercolumn import helper as wchelper

# subpackages
from .layers.echolayer import EchoLayer, PingData


class EchogramBuilder:
    def __init__(self, pings, times, beam_sample_selections, wci_value, linear_mean):
        assert_length("EchoData", pings, [times])
        if len(pings) == 0:
            raise RuntimeError("ERROR[EchoData]: trying to initialize empty data (no valid pings)")

        assert len(pings) == len(
            beam_sample_selections
        ), "ERROR[EchoData]: pings and beam_sample_selections must have the same length"
        self.beam_sample_selections = beam_sample_selections

        self.layers = {}
        self.main_layer = None
        self.linear_mean = linear_mean
        self.wci_value = wci_value
        self.pings = pings
        # self.wc_data = wc_data
        self.x_axis_name = None
        self.y_axis_name = None
        self.max_number_of_samples = np.array(
            [sel.get_number_of_samples_ensemble() - 1 for sel in self.beam_sample_selections]
        )
        self.set_ping_times(times)
        self.ping_numbers = np.arange(len(self.pings))

        self.param = {}

        self.has_ranges = False
        self.has_depths = False
        self.has_sample_nrs = False
        self.set_y_axis_y_indice()
        self.set_x_axis_ping_nr()
        self.initialized = True

    def set_linear_mean(self, linear_mean):
        self.linear_mean = linear_mean
        self.initialized = False

    def set_ping_numbers(self, ping_numbers):
        assert_length("set_ping_numbers", self.pings, [ping_numbers])
        self.ping_numbers = ping_numbers
        self.initialized = False

    def set_ping_times(self, ping_times, time_zone=dt.timezone.utc):
        assert_length("set_ping_times", self.pings, [ping_times])
        self.ping_times = ping_times
        self.time_zone = time_zone
        self.initialized = False

    def set_range_extent(self, min_ranges, max_ranges):
        assert_length("set_range_extent", self.pings, [min_ranges, max_ranges])
        self.min_ranges = np.array(min_ranges)
        self.max_ranges = np.array(max_ranges)
        self.res_ranges = (self.max_ranges - self.min_ranges) / self.max_number_of_samples
        self.has_ranges = True
        self.initialized = False

    def set_depth_extent(self, min_depths, max_depths):
        assert_length("set_depth_extent", self.pings, [min_depths, max_depths])
        self.min_depths = np.array(min_depths)
        self.max_depths = np.array(max_depths)
        self.res_depths = (self.max_depths - self.min_depths) / self.max_number_of_samples
        self.has_depths = True
        self.initialized = False

    def set_sample_nr_extent(self, min_sample_nrs, max_sample_nrs):
        assert_length(
            "set_sample_nr_extent", self.pings, [min_sample_nrs, max_sample_nrs]
        )
        self.min_sample_nrs = np.array(min_sample_nrs)
        self.max_sample_nrs = np.array(max_sample_nrs)
        self.res_sample_nrs = (self.max_sample_nrs - self.min_sample_nrs) / self.max_number_of_samples
        self.has_sample_nrs = True
        self.initialized = False

    def add_ping_param(self, name, x_reference, y_reference, vec_x_val, vec_y_val):
        assert_valid_argument(
            "add_ping_param", x_reference, ["Ping number", "Ping time", "Date time"]
        )
        assert_valid_argument(
            "add_ping_param", y_reference, ["Y indice", "Sample number", "Depth (m)", "Range (m)"]
        )

        # convert datetimes to timestamps
        if isinstance(vec_x_val[0], dt.datetime):
            vec_x_val = [x.timestamp() for x in vec_x_val]

        # convert to numpy arrays
        vec_x_val = np.array(vec_x_val)
        vec_y_val = np.array(vec_y_val)

        # filter nans and infs
        arg = np.where(np.isfinite(vec_x_val))[0]
        vec_x_val = vec_x_val[arg]
        vec_y_val = vec_y_val[arg]
        arg = np.where(np.isfinite(vec_y_val))[0]
        vec_x_val = vec_x_val[arg]
        vec_y_val = vec_y_val[arg]

        match x_reference:
            case "Ping number":
                comp_vec_x_val = self.ping_numbers
            case "Ping time":
                comp_vec_x_val = self.ping_times
            case "Date time":
                comp_vec_x_val = self.ping_times
                # comp_vec_x_val = [dt.datetime.fromtimestamp(t, self.time_zone) for t in self.ping_times]

        # average vec_y_val for all double vec_x_vals
        unique_x_vals, indices = np.unique(vec_x_val, return_inverse=True)
        averaged_y_vals = np.zeros(len(unique_x_vals))
        counts = np.bincount(indices)
        for i in range(len(unique_x_vals)):
            start_index = np.where(indices == i)[0][0]
            end_index = start_index + counts[i]
            averaged_y_vals[i] = np.mean(vec_y_val[start_index:end_index])

        vec_x_val = unique_x_vals
        vec_y_val = averaged_y_vals

        # convert to to represent indices
        vec_y_val = tools.vectorinterpolators.LinearInterpolator(
            vec_x_val, vec_y_val, extrapolation_mode="nearest"
        )(comp_vec_x_val)

        self.param[name] = y_reference, vec_y_val

    def get_ping_param(self, name, use_x_coordinates=False):
        self.reinit()
        assert name in self.param.keys(), f"ERROR[get_ping_param]: name '{name}' not registered"
        # x_coordinates = self.indice_to_x_coordinate_interpolator(np.arange(len(self.pings)))
        if use_x_coordinates:
            x_coordinates = self.x_coordinates
            x_indices = np.array(self.x_coordinate_indice_interpolator(x_coordinates))
        else:
            x_indices = np.arange(len(self.pings))
            x_coordinates = self.vec_x_val[x_indices]

        reference, param = self.param[name]
        param = np.array(param)[x_indices]

        return_param = np.empty(len(param))
        return_param.fill(np.nan)

        match reference:
            case "Y indice":
                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.y_indice_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case "Sample number":
                assert (
                    self.has_sample_nrs
                ), "ERROR: Sample nr values not initialized for ech data, call set_sample_nr_extent method"

                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.sample_nr_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case "Depth (m)":
                assert (
                    self.has_depths
                ), "ERROR: Depths values not initialized for ech data, call set_depth_extent method"

                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.depth_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case "Range (m)":
                assert (
                    self.has_ranges
                ), "ERROR: Ranges values not initialized for ech data, call set_range_extent method"

                for nr, (indice, p) in enumerate(zip(x_indices, param)):
                    if np.isfinite(p):
                        I = self.range_to_y_coordinate_interpolator[indice]
                        if I is not None:
                            return_param[nr] = I(p)

            case _:
                raise RuntimeError(f"Invalid reference '{reference}'. This should not happen, please report")

        if self.x_axis_name == "Date time":
            x_coordinates = [dt.datetime.fromtimestamp(t, self.time_zone) for t in x_coordinates]

        return x_coordinates, return_param

    def get_x_kwargs(self):
        return deepcopy(self.__x_kwargs)

    def get_y_kwargs(self):
        return deepcopy(self.__y_kwargs)

    @classmethod
    def from_pings(
        cls,
        pings,
        pss=echosounders.pingtools.PingSampleSelector(),
        wci_value: str = "sv/av/pv/rv",
        linear_mean=True,
        no_navigation=False,
        apply_pss_to_bottom=False,
        verbose=True,
    ):

        # wc_data = [np.empty(0) for _ in pings]
        min_s = np.empty(len(pings), dtype=np.float32)
        max_s = np.empty(len(pings), dtype=np.float32)
        min_r = np.empty(len(pings), dtype=np.float32)
        max_r = np.empty(len(pings), dtype=np.float32)
        min_d = np.empty(len(pings), dtype=np.float32)
        max_d = np.empty(len(pings), dtype=np.float32)
        times = np.empty(len(pings), dtype=np.float64)

        bottom_d_times = []
        bottom_d = []
        minslant_d = []
        echosounder_d_times = []
        echosounder_d = []

        for arg in [min_s, max_s, min_r, max_r, min_d, max_d, times]:
            arg.fill(np.nan)

        beam_sample_selections = []

        for nr, ping in enumerate(tqdm(pings, disable=(not verbose), delay=1)):
            sel = wchelper.apply_pss(ping, pss, apply_pss_to_bottom)

            beam_sample_selections.append(sel)

            times[nr] = ping.get_timestamp()
            if len(sel.get_beam_numbers()) == 0:
                continue

            c = ping.watercolumn.get_sound_speed_at_transducer()
            range_res = ping.watercolumn.get_sample_interval() * c * 0.5
            angle_factor = np.cos(
                np.radians(np.mean(ping.watercolumn.get_beam_crosstrack_angles()[sel.get_beam_numbers()]))
            )
            min_s[nr] = sel.get_first_sample_number_ensemble()
            max_s[nr] = sel.get_last_sample_number_ensemble()
            min_r[nr] = min_s[nr] * range_res
            max_r[nr] = max_s[nr] * range_res
            echosounder_d_times.append(times[nr])

            if not no_navigation:
                if not ping.has_geolocation():
                    raise RuntimeError(
                        f"ERROR: ping {nr} has no geolocation. Either filter pings based on geolocation feature or set no_navigation to True"
                    )

                z = ping.get_geolocation().z
                min_d[nr] = z + min_r[nr] * angle_factor
                max_d[nr] = z + max_r[nr] * angle_factor
                echosounder_d.append(z)

            if max_d[nr] > 6000:
                print(
                    f"ERROR [{nr}], s0{min_s[nr]}, s1{max_s[nr]}, r0{min_r[nr]}, r1{max_r[nr]}, d0{min_d[nr]}, d1{max_d[nr]}",
                    z,
                    angle_factor,
                )

            if ping.has_bottom():
                if ping.bottom.has_xyz():
                    # sel_bottom = pss.apply_selection(ping.bottom)
                    # bd = np.nanmin(p.bottom.get_xyz(sel_bottom).z) + p.get_geolocation().z
                    # this is incorrect

                    # br = np.nanquantile(ping.bottom.get_xyz(sel).z, 0.05)
                    try:
                        br = ping.bottom.get_bottom_z(sel)
                        # mr = np.nanquantile(ping.watercolumn.get_bottom_range_samples(), 0.05) * range_res * angle_factor
                        mr = ping.watercolumn.get_minslant_sample_nr() * range_res * angle_factor
                        bottom_d_times.append(times[nr])

                        if not no_navigation:
                            bd = br + z
                            md = mr + z
                            bottom_d.append(bd)
                            minslant_d.append(md)
                    except Exception as e:
                        pass
                        #TODO: this should create a warning in the log

        data = cls(pings, times, beam_sample_selections, wci_value, linear_mean=linear_mean)
        data.set_sample_nr_extent(min_s, max_s)
        data.set_range_extent(min_r, max_r)
        if not no_navigation:
            data.set_depth_extent(min_d, max_d)

        if len(bottom_d) > 0:
            data.add_ping_param("bottom", "Ping time", "Depth (m)", bottom_d_times, bottom_d)
            data.add_ping_param("minslant", "Ping time", "Depth (m)", bottom_d_times, minslant_d)
        if len(echosounder_d) > 0:
            data.add_ping_param("echosounder", "Ping time", "Depth (m)", echosounder_d_times, echosounder_d)

        data.verbose = verbose
        return data

    def get_wci(self, nr):
        sel = self.beam_sample_selections[nr]
        ping = self.pings[nr]

        # select which ping.watercolumn.get_ function to call based on wci_value
        wci = wchelper.select_get_wci_image(ping, sel, self.wci_value)

        if wci.shape[0] == 1:
            return wci[0]
        else:
            if self.linear_mean:
                wci = np.power(10, wci * 0.1)

            wci = np.nanmean(wci, axis=0)

            if self.linear_mean:
                wci = 10 * np.log10(wci)

            return wci

    @staticmethod
    def sample_y_coordinates(vec_min_y, vec_max_y, vec_res_y, min_y, max_y, max_steps=1024):
        vec_min_y = np.array(vec_min_y)
        vec_max_y = np.array(vec_max_y)

        # vec_min_y = vec_min_y[vec_min_y >= 0]
        # vec_max_y = vec_max_y[vec_max_y > 0]
        vec_min_y = vec_min_y[np.isfinite(vec_min_y)]
        vec_max_y = vec_max_y[np.isfinite(vec_max_y)]

        # filter
        if not np.isfinite(min_y):
            iqr = np.nanquantile(vec_min_y, 0.90) - np.nanquantile(vec_min_y, 0.10)
            med = np.nanmedian(vec_min_y)
            min_y = med - iqr * 1.5
            # print(f"min_y: {min_y}, med: {med}, iqr: {iqr}")
        if not np.isfinite(max_y):
            iqr = np.nanquantile(vec_max_y, 0.90) - np.nanquantile(vec_max_y, 0.10)
            med = np.nanmedian(vec_max_y)
            max_y = med + iqr * 1.5
            # print(f"max_y: {max_y}, med: {med}, iqr: {iqr}")

        iqr = np.nanquantile(vec_res_y, 0.75) - np.nanquantile(vec_res_y, 0.25)
        med = np.nanmedian(vec_res_y)
        min_resolution = med - iqr * 1.5

        res = np.nanmax([np.nanmin(vec_res_y), min_resolution])
        y_min = np.nanmax([np.nanmin(vec_min_y), min_y])
        y_max = np.nanmin([np.nanmax(vec_max_y), max_y])

        y_coordinates = np.arange(y_min, y_max + res, res)

        if len(y_coordinates) > max_steps:
            y_coordinates = np.linspace(y_min, y_max, max_steps)

        return y_coordinates, res

    def reinit(self):
        if self.initialized:
            return

        self.y_axis_function(**self.__y_kwargs)
        self.x_axis_function(**self.__x_kwargs)

    def set_y_coordinates(self, name, y_coordinates, y_resolution, vec_min_y, vec_max_y):
        assert (
            len(vec_min_y) == len(vec_max_y) == len(self.pings)
        ), f"ERROR min/max y vectors must have the same length as internal pings vector"
        self.y_axis_name = name
        self.y_coordinates = y_coordinates
        self.y_resolution = y_resolution
        self.y_extent = [
            self.y_coordinates[-1] + self.y_resolution / 2,
            self.y_coordinates[0] - self.y_resolution / 2,
        ]
        self.vec_min_y = vec_min_y
        self.vec_max_y = vec_max_y

        self.y_coordinate_indice_interpolator = [None for _ in self.pings]
        self.y_indice_to_y_coordinate_interpolator = [None for _ in self.pings]
        self.depth_to_y_coordinate_interpolator = [None for _ in self.pings]
        self.range_to_y_coordinate_interpolator = [None for _ in self.pings]
        self.sample_nr_to_y_coordinate_interpolator = [None for _ in self.pings]
        self.y_indice_to_depth_interpolator = [None for _ in self.pings]
        self.y_indice_to_range_interpolator = [None for _ in self.pings]
        self.y_indice_to_sample_nr_interpolator = [None for _ in self.pings]

        for nr, (y1, y2, sel) in enumerate(zip(vec_min_y, vec_max_y, self.beam_sample_selections)):
            try:
                n_samples = sel.get_number_of_samples_ensemble()

                if n_samples > 1:
                    I = tools.vectorinterpolators.LinearInterpolatorF([y1, y2], [0, n_samples - 1])
                    self.y_coordinate_indice_interpolator[nr] = I

                    I = tools.vectorinterpolators.LinearInterpolatorF([0, n_samples - 1], [y1, y2])
                    self.y_indice_to_y_coordinate_interpolator[nr] = I

                    if self.has_depths:
                        I = tools.vectorinterpolators.LinearInterpolatorF(
                            [self.min_depths[nr], self.max_depths[nr]], [y1, y2]
                        )
                        self.depth_to_y_coordinate_interpolator[nr] = I
                        I = tools.vectorinterpolators.LinearInterpolatorF(
                            [0, n_samples - 1], [self.min_depths[nr], self.max_depths[nr]]
                        )
                        self.y_indice_to_depth_interpolator[nr] = I

                    if self.has_ranges:
                        I = tools.vectorinterpolators.LinearInterpolatorF(
                            [self.min_ranges[nr], self.max_ranges[nr]], [y1, y2]
                        )
                        self.range_to_y_coordinate_interpolator[nr] = I
                        I = tools.vectorinterpolators.LinearInterpolatorF(
                            [0, n_samples - 1], [self.min_ranges[nr], self.max_ranges[nr]]
                        )
                        self.y_indice_to_range_interpolator[nr] = I

                    if self.has_sample_nrs:
                        I = tools.vectorinterpolators.LinearInterpolatorF(
                            [self.min_sample_nrs[nr], self.max_sample_nrs[nr]], [y1, y2]
                        )
                        self.sample_nr_to_y_coordinate_interpolator[nr] = I
                        I = tools.vectorinterpolators.LinearInterpolatorF(
                            [0, n_samples - 1], [self.min_sample_nrs[nr], self.max_sample_nrs[nr]]
                        )
                        self.y_indice_to_sample_nr_interpolator[nr] = I

            except Exception as e:
                message = f"{e}\n- nr {nr}\n- y1 {y1}\n -y2 {y2}\n -n_samples {n_samples}"
                raise RuntimeError(message)

    def get_filtered_by_y_extent(self, vec_x_val, vec_min_y, vec_max_y):
        assert_length("get_filtered_by_y_extent", vec_x_val, [vec_min_y, vec_max_y])

        # convert datetimes to timestamps
        if isinstance(vec_x_val[0], dt.datetime):
            vec_x_val = [x.timestamp() for x in vec_x_val]

        # convert to numpy arrays
        vec_x_val = np.array(vec_x_val)
        vec_min_y = np.array(vec_min_y)
        vec_max_y = np.array(vec_max_y)

        # filter nans and infs
        arg = np.where(np.isfinite(vec_x_val))[0]
        vec_min_y = vec_min_y[arg]
        vec_max_y = vec_max_y[arg]
        vec_x_val = vec_x_val[arg]
        arg = np.where(np.isfinite(vec_min_y))[0]
        vec_min_y = vec_min_y[arg]
        vec_max_y = vec_max_y[arg]
        vec_x_val = vec_x_val[arg]
        arg = np.where(np.isfinite(vec_max_y))[0]
        vec_min_y = vec_min_y[arg]
        vec_max_y = vec_max_y[arg]
        vec_x_val = vec_x_val[arg]

        # convert to to represent indices
        vec_min_y = tools.vectorinterpolators.LinearInterpolator(
            vec_x_val, vec_min_y, extrapolation_mode="nearest"
        )(self.vec_x_val)
        vec_max_y = tools.vectorinterpolators.LinearInterpolator(
            vec_x_val, vec_max_y, extrapolation_mode="nearest"
        )(self.vec_x_val)

        # wc_data = [np.empty(0) for _ in self.wc_data]
        beam_sample_selections = deepcopy(self.beam_sample_selections)
        min_s = np.empty(len(self.pings), dtype=np.float32)
        max_s = np.empty(len(self.pings), dtype=np.float32)
        min_r = np.empty(len(self.pings), dtype=np.float32)
        max_r = np.empty(len(self.pings), dtype=np.float32)
        min_d = np.empty(len(self.pings), dtype=np.float32)
        max_d = np.empty(len(self.pings), dtype=np.float32)

        for arg in [min_s, max_s, min_r, max_r, min_d, max_d]:
            arg.fill(np.nan)
        global nr, bss, i1, i2
        for nr, (y1, y2, bss) in enumerate(zip(vec_min_y, vec_max_y, beam_sample_selections)):
            nsamples = bss.get_number_of_samples_ensemble()
            if nsamples > 0:
                i1 = int(self.y_coordinate_indice_interpolator[nr](y1) + 0.5)
                i2 = int(self.y_coordinate_indice_interpolator[nr](y2) + 0.5)
                iy1 = self.y_indice_to_y_coordinate_interpolator[nr](i1)
                iy2 = self.y_indice_to_y_coordinate_interpolator[nr](i2)
                isn1 = self.y_indice_to_sample_nr_interpolator[nr](i1)
                isn2 = self.y_indice_to_sample_nr_interpolator[nr](i2)

                if iy1 < y1:
                    iy1 += self.y_resolution
                    i1 += 1

                if iy2 > y2:
                    iy2 -= self.y_resolution
                    i2 -= 1

                if i1 < 0:
                    i1 = 0
                if i2 < 0:
                    i2 = 0
                isn1 = int(self.y_indice_to_sample_nr_interpolator[nr](i1) + 0.5)
                isn2 = int(self.y_indice_to_sample_nr_interpolator[nr](i2) + 0.5)

                # #self.wc_data[nr = self.wc_data[nr][i1 : i2 + 1]
                bss.set_first_sample_number_ensemble(isn1)
                bss.set_last_sample_number_ensemble(isn2)
                if bss.get_number_of_samples_ensemble() > 1:
                    if self.has_depths:
                        min_d[nr] = self.y_indice_to_depth_interpolator[nr](i1)
                        max_d[nr] = self.y_indice_to_depth_interpolator[nr](i2)
                    if self.has_ranges:
                        min_r[nr] = self.y_indice_to_range_interpolator[nr](i1)
                        max_r[nr] = self.y_indice_to_range_interpolator[nr](i2)
                    if self.has_sample_nrs:
                        min_s[nr] = self.y_indice_to_sample_nr_interpolator[nr](i1)
                        max_s[nr] = self.y_indice_to_sample_nr_interpolator[nr](i2)

        out = EchogramBuilder(
            pings=self.pings,
            times=self.ping_times,
            beam_sample_selections=beam_sample_selections,
            wci_value=self.wci_value,
            linear_mean=self.linear_mean,
        )
        if self.has_depths:
            out.set_depth_extent(min_d, max_d)
        if self.has_ranges:
            out.set_range_extent(min_r, max_r)
        if self.has_sample_nrs:
            out.set_sample_nr_extent(min_s, max_s)

        out.set_ping_times(self.ping_times, self.time_zone)
        out.set_ping_numbers(self.ping_numbers.copy())
        out.param = self.param.copy()

        self.copy_xy_axis(out)

        return out

    def copy_xy_axis(self, other):
        match self.x_axis_name:
            case "Date time":
                other.set_x_axis_date_time(**self.__x_kwargs)
            case "Ping time":
                other.set_x_axis_ping_time(**self.__x_kwargs)
            case "Ping number":
                other.set_x_axis_ping_nr(**self.__x_kwargs)
            case _:
                raise RuntimeError(f"ERROR: unknown x axis name '{self.x_axis_name}'")

        match self.y_axis_name:
            case "Depth (m)":
                other.set_y_axis_depth(**self.__y_kwargs)
            case "Range (m)":
                other.set_y_axis_range(**self.__y_kwargs)
            case "Sample number":
                other.set_y_axis_sample_nr(**self.__y_kwargs)
            case "Y indice":
                other.set_y_axis_y_indice(**self.__y_kwargs)
            case _:
                raise RuntimeError(f"ERROR: unknown y axis name '{self.y_axis_name}'")

    def set_x_coordinates(self, name, x_coordinates, x_resolution, x_interpolation_limit, vec_x_val):
        self.x_axis_name = name
        self.vec_x_val = vec_x_val
        self.x_coordinates = x_coordinates
        self.x_resolution = x_resolution
        self.x_interpolation_limit = x_interpolation_limit
        self.x_extent = [
            self.x_coordinates[0] - self.x_resolution / 2,
            self.x_coordinates[-1] + self.x_resolution / 2,
        ]

        self.x_coordinate_indice_interpolator = tools.vectorinterpolators.NearestInterpolatorDI(
            vec_x_val, np.arange(len(self.pings))
        )
        self.indice_to_x_coordinate_interpolator = tools.vectorinterpolators.NearestInterpolator(
            np.arange(len(self.pings)), vec_x_val
        )

    def set_y_axis_y_indice(
        self,
        min_sample_nr=0,
        max_sample_nr=np.nan,
        max_steps=1024,
        **kwargs,  # to catch [and ignore] any additional arguments
    ):
        vec_min_y = np.zeros((len(self.pings)))
        vec_max_y = self.max_number_of_samples

        y_kwargs = {"min_sample_nr": min_sample_nr, "max_sample_nr": max_sample_nr, "max_steps": max_steps}
        self.y_axis_function = self.set_y_axis_y_indice
        if self.y_axis_name == "Y indice" and self.__y_kwargs == y_kwargs:
            return
        self.__y_kwargs = y_kwargs

        y_coordinates, y_res = self.sample_y_coordinates(
            vec_min_y=vec_min_y,
            vec_max_y=vec_max_y,
            vec_res_y=np.ones((len(self.pings))),
            min_y=min_sample_nr,
            max_y=max_sample_nr,
            max_steps=max_steps,
        )

        self.set_y_coordinates("Y indice", y_coordinates, y_res, vec_min_y, vec_max_y)

    def set_y_axis_depth(
        self,
        min_depth=np.nan,
        max_depth=np.nan,
        max_steps=1024,
        **kwargs,  # to catch [and ignore] any additional arguments
    ):
        assert self.has_depths, "ERROR: Depths values not initialized for ech data, call set_depth_extent method"

        y_kwargs = {"min_depth": min_depth, "max_depth": max_depth, "max_steps": max_steps}
        self.y_axis_function = self.set_y_axis_depth
        if self.y_axis_name == "Depth (m)" and self.__y_kwargs == y_kwargs:
            return
        self.__y_kwargs = y_kwargs

        y_coordinates, y_res = self.sample_y_coordinates(
            vec_min_y=self.min_depths,
            vec_max_y=self.max_depths,
            vec_res_y=self.res_depths,
            min_y=min_depth,
            max_y=max_depth,
            max_steps=max_steps,
        )

        self.set_y_coordinates("Depth (m)", y_coordinates, y_res, self.min_depths, self.max_depths)

    def set_y_axis_range(
        self,
        min_range=np.nan,
        max_range=np.nan,
        max_steps=1024,
        **kwargs,  # to catch [and ignore] any additional arguments
    ):
        assert self.has_ranges, "ERROR: Range values not initialized for ech data, call set_angess method"

        y_kwargs = {"min_range": min_range, "max_range": max_range, "max_steps": max_steps}
        self.y_axis_function = self.set_y_axis_range
        if self.y_axis_name == "Range (m)" and self.__y_kwargs == y_kwargs:
            return
        self.__y_kwargs = y_kwargs

        y_coordinates, y_res = self.sample_y_coordinates(
            vec_min_y=self.min_ranges,
            vec_max_y=self.max_ranges,
            vec_res_y=self.res_ranges,
            min_y=min_range,
            max_y=max_range,
            max_steps=max_steps,
        )

        self.set_y_coordinates("Range (m)", y_coordinates, y_res, self.min_ranges, self.max_ranges)

    def set_y_axis_sample_nr(
        self,
        min_sample_nr=0,
        max_sample_nr=np.nan,
        max_steps=1024,
        **kwargs,  # to catch [and ignore] any additional arguments
    ):
        assert (
            self.has_sample_nrs
        ), "ERROR: Sample nr values not initialized for ech data, call set_sample_nr_extent method"

        y_kwargs = {"min_sample_nr": min_sample_nr, "max_sample_nr": max_sample_nr, "max_steps": max_steps}
        self.y_axis_function = self.set_y_axis_sample_nr
        if self.y_axis_name == "Sample number" and self.__y_kwargs == y_kwargs:
            return
        self.__y_kwargs = y_kwargs

        y_coordinates, y_res = self.sample_y_coordinates(
            vec_min_y=self.min_sample_nrs,
            vec_max_y=self.max_sample_nrs,
            vec_res_y=self.res_sample_nrs,
            min_y=min_sample_nr,
            max_y=max_sample_nr,
            max_steps=max_steps,
        )

        self.set_y_coordinates("Sample number", y_coordinates, y_res, self.min_sample_nrs, self.max_sample_nrs)

    def set_x_axis_ping_nr(
        self,
        min_ping_nr=0,
        max_ping_nr=np.nan,
        max_steps=4096,
        **kwargs,  # to catch [and ignore] any additional arguments
    ):

        x_kwargs = {
            "min_ping_nr": min_ping_nr,
            "max_ping_nr": max_ping_nr,
            "max_steps": max_steps,
        }

        self.x_axis_function = self.set_x_axis_ping_nr
        if self.x_axis_name == "Ping number" and self.__x_kwargs == x_kwargs:
            return

        self.__x_kwargs = x_kwargs

        if not np.isfinite(max_ping_nr):
            max_ping_nr = np.max(self.ping_numbers)

        if not np.isfinite(max_ping_nr):
            max_ping_nr = np.max(self.ping_numbers)

        npings = int(max_ping_nr - min_ping_nr) + 1

        if npings > len(self.pings):
            npings = len(self.pings)

        if npings > max_steps:
            npings = max_steps

        x_coordinates = np.linspace(min_ping_nr, max_ping_nr, npings)
        if npings > 1:
            x_resolution = x_coordinates[1] - x_coordinates[0]
        else:
            x_resolution = 1

        self.set_x_coordinates("Ping number", x_coordinates, x_resolution, 1, self.ping_numbers)

    def set_x_axis_ping_time(
        self,
        min_timestamp=np.nan,
        max_timestamp=np.nan,
        time_resolution=np.nan,
        time_interpolation_limit=np.nan,
        max_steps=4096,
        **kwargs,  # to catch [and ignore] any additional arguments
    ):

        x_kwargs = {
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
            "time_resolution": time_resolution,
            "time_interpolation_limit": time_interpolation_limit,
            "max_steps": max_steps,
        }

        self.x_axis_function = self.set_x_axis_ping_time
        if self.x_axis_name == "Ping time" and self.__x_kwargs == x_kwargs:
            return

        self.__x_kwargs = x_kwargs

        if not np.isfinite(min_timestamp):
            min_timestamp = np.min(self.ping_times)

        if not np.isfinite(max_timestamp):
            max_timestamp = np.max(self.ping_times)

        ping_delta_t = np.array(self.ping_times[1:] - self.ping_times[:-1])
        if len(ping_delta_t[ping_delta_t < 0]) > 0:
            raise RuntimeError("ERROR: ping times are not sorted in ascending order!")

        zero_time_diff = np.where(abs(ping_delta_t) < 0.000001)[0]
        while len(zero_time_diff) > 0:
            self.ping_times[zero_time_diff + 1] += 0.0001
            ping_delta_t = np.array(self.ping_times[1:] - self.ping_times[:-1])
            zero_time_diff = np.where(abs(ping_delta_t) < 0.0001)[0]

        if not np.isfinite(time_resolution):
            time_resolution = np.nanquantile(ping_delta_t, 0.05)

        if not np.isfinite(time_interpolation_limit):
            time_interpolation_limit = np.nanquantile(ping_delta_t, 0.95)

        try:
            arange = False
            if (max_timestamp + time_resolution - min_timestamp) / time_resolution + 1 <= max_steps:
                x_coordinates = np.arange(min_timestamp, max_timestamp + time_resolution, time_resolution)
            else:
                arange = True

            if arange or len(x_coordinates) > max_steps:
                x_coordinates = np.linspace(min_timestamp, max_timestamp, max_steps)
                if max_steps > 1:
                    time_resolution = x_coordinates[1] - x_coordinates[0]
                else:
                    time_resolution = 1
        except Exception as e:
            message = f"{e}\n -min_timestamp: {min_timestamp}\n -max_timestamp: {max_timestamp}\n -time_resolution: {time_resolution}\n -max_steps: {max_steps}"

            raise RuntimeError(message)

        self.set_x_coordinates("Ping time", x_coordinates, time_resolution, time_interpolation_limit, self.ping_times)

    def set_x_axis_date_time(
        self,
        min_ping_time=np.nan,
        max_ping_time=np.nan,
        time_resolution=np.nan,
        time_interpolation_limit=np.nan,
        max_steps=4096,
        **kwargs,  # to catch [and ignore] any additional arguments
    ):
        x_kwargs = {
            "min_ping_time": min_ping_time,
            "max_ping_time": max_ping_time,
            "time_resolution": time_resolution,
            "time_interpolation_limit": time_interpolation_limit,
            "max_steps": max_steps,
        }

        self.x_axis_function = self.set_x_axis_date_time
        if self.x_axis_name == "Date time" and self.__x_kwargs == x_kwargs:
            return

        if isinstance(min_ping_time, dt.datetime):
            min_ping_time = min_ping_time.timestamp()

        if isinstance(max_ping_time, dt.datetime):
            max_ping_time = max_ping_time.timestamp()

        if isinstance(time_resolution, dt.timedelta):
            time_resolution = time_resolution.total_seconds()

        if isinstance(time_interpolation_limit, dt.timedelta):
            time_interpolation_limit = time_interpolation_limit.total_seconds()

        self.set_x_axis_ping_time(
            min_timestamp=min_ping_time,
            max_timestamp=max_ping_time,
            time_resolution=time_resolution,
            time_interpolation_limit=time_interpolation_limit,
            max_steps=max_steps,
        )

        self.x_extent[0] = dt.datetime.fromtimestamp(self.x_extent[0], self.time_zone)
        self.x_extent[1] = dt.datetime.fromtimestamp(self.x_extent[1], self.time_zone)
        self.x_axis_name = "Date time"
        self.x_axis_function = self.set_x_axis_date_time
        self.__x_kwargs = x_kwargs

    def get_y_indices(self, wci_nr):
        n_samples = self.beam_sample_selections[wci_nr].get_number_of_samples_ensemble()
        y_indices_image = np.arange(len(self.y_coordinates))
        y_indices_wci = np.round(self.y_coordinate_indice_interpolator[wci_nr](self.y_coordinates)).astype(int)

        valid_coordinates = np.where(np.logical_and(y_indices_wci >= 0, y_indices_wci < n_samples))[0]

        return y_indices_image[valid_coordinates], y_indices_wci[valid_coordinates]

    def get_x_indices(self):
        image_index, wci_index = np.arange(len(self.x_coordinates)), np.array(
            self.x_coordinate_indice_interpolator(self.x_coordinates)
        )
        delta_x = np.abs(self.vec_x_val[wci_index] - self.x_coordinates)
        valid = np.where(delta_x < self.x_interpolation_limit)[0]
        return image_index[valid], wci_index[valid]

    def build_image(self, progress=None):
        self.reinit()
        ny = len(self.y_coordinates)
        nx = len(self.x_coordinates)

        image = np.empty((nx, ny), dtype=np.float32)
        image.fill(np.nan)

        image_indices, wci_indices = self.get_x_indices()
        image_indices = get_progress_iterator(image_indices, progress, desc="Building echogram image")

        for image_index, wci_index in zip(image_indices, wci_indices):
            wci = self.get_wci(wci_index)
            if len(wci) > 1:
                y1, y2 = self.get_y_indices(wci_index)
                image[image_index, y1] = wci[y2]

        extent = deepcopy(self.x_extent)
        extent.extend(self.y_extent)

        return image, extent

    # --- layer functions ---
    def build_image_and_layer_image(self, progress=None):
        self.reinit()
        ny = len(self.y_coordinates)
        nx = len(self.x_coordinates)

        image = np.empty((nx, ny), dtype=np.float32)
        image.fill(np.nan)
        layer_image = image.copy()

        image_indices, wci_indices = self.get_x_indices()
        image_indices = get_progress_iterator(image_indices, progress, desc="Building echogram image")

        for image_index, wci_index in zip(image_indices, wci_indices):
            wci = self.get_wci(wci_index)
            if len(wci) > 1:
                if self.main_layer is None:
                    y1, y2 = self.get_y_indices(wci_index)
                    image[image_index, y1] = wci[y2]
                else:
                    y1, y2 = self.main_layer.get_y_indices(wci_index)
                    if y1 is not None:
                        image[image_index, y1] = wci[y2]

                for k, layer in self.layers.items():
                    y1_layer, y2_layer = layer.get_y_indices(wci_index)
                    if y1_layer is not None:
                        layer_image[image_index, y1_layer] = wci[y2_layer]

        extent = deepcopy(self.x_extent)
        extent.extend(self.y_extent)

        return image, layer_image, extent

    def build_image_and_layer_images(self, progress=None):
        self.reinit()
        ny = len(self.y_coordinates)
        nx = len(self.x_coordinates)

        image = np.empty((nx, ny), dtype=np.float32)
        image.fill(np.nan)

        layer_images = {}
        for key in self.layers.keys():
            layer_images[key] = image.copy()

        image_indices, wci_indices = self.get_x_indices()
        image_indices = get_progress_iterator(image_indices, progress, desc="Building echogram image")

        for image_index, wci_index in zip(image_indices, wci_indices):
            wci = self.get_wci(wci_index)
            if len(wci) > 1:
                if self.main_layer is None:
                    y1, y2 = self.get_y_indices(wci_index)
                    image[image_index, y1] = wci[y2]
                else:
                    y1, y2 = self.main_layer.get_y_indices(wci_index)
                    if y1 is not None:
                        image[image_index, y1] = wci[y2]

                for key, layer in self.layers.items():
                    y1_layer, y2_layer = layer.get_y_indices(wci_index)
                    if y1_layer is not None:
                        layer_images[key][image_index, y1_layer] = wci[y2_layer]

        extent = deepcopy(self.x_extent)
        extent.extend(self.y_extent)

        return image, layer_images, extent

    def get_wci_layers(self, nr):
        wci = self.get_wci(nr)

        wci_layers = {}
        for key, layer in self.layers.items():
            wci_layers[key] = wci[layer.i0[nr] : layer.i1[nr]]

        return wci_layers

    def get_extent_layers(self, nr, axis_name=None):
        if axis_name is None:
            axis_name = self.y_axis_name
        extents = {}

        for key, layer in self.layers.items():
            match axis_name:
                case "Y indice":
                    extents[key] = layer.i0[nr] - 0.5, layer.i1[nr] - 0.5

                case "Sample number":
                    assert (
                        self.has_sample_nrs
                    ), "ERROR: Sample nr values not initialized for ech data, call set_sample_nr_extent method"

                    extents[key] = self.y_indice_to_sample_nr_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case "Depth (m)":
                    assert (
                        self.has_depths
                    ), "ERROR: Depths values not initialized for ech data, call set_depth_extent method"

                    extents[key] = self.y_indice_to_depth_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case "Range (m)":
                    assert (
                        self.has_ranges
                    ), "ERROR: Ranges values not initialized for ech data, call set_range_extent method"

                    extents[key] = self.y_indice_to_range_interpolator[nr]([layer.i0[nr] - 0.5, layer.i1[nr] - 0.5])

                case _:
                    raise RuntimeError(f"Invalid reference '{reference}'. This should not happen, please report")

        return extents

    def get_limits_layers(self, nr, axis_name=None):
        if axis_name is None:
            axis_name = self.y_axis_name
        extents = {}

        for key, layer in self.layers.items():
            match axis_name:
                case "Y indice":
                    extents[key] = layer.i0[nr] - 0.5, layer.i1[nr] - 0.5

                case "Sample number":
                    assert (
                        self.has_sample_nrs
                    ), "ERROR: Sample nr values not initialized for ech data, call set_sample_nr_extent method"

                    extents[key] = self.y_indice_to_sample_nr_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case "Depth (m)":
                    assert (
                        self.has_depths
                    ), "ERROR: Depths values not initialized for ech data, call set_depth_extent method"

                    extents[key] = self.y_indice_to_depth_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case "Range (m)":
                    assert (
                        self.has_ranges
                    ), "ERROR: Ranges values not initialized for ech data, call set_range_extent method"

                    extents[key] = self.y_indice_to_range_interpolator[nr]([layer.i0[nr], layer.i1[nr] - 1])

                case _:
                    raise RuntimeError(f"Invalid reference '{reference}'. This should not happen, please report")

        return extents

    def __set_layer__(self, name, layer):
        if name == "main":
            if self.main_layer is not None:
                self.main_layer.combine(layer)
            else:
                self.main_layer = layer
        else:
            if name in self.layers.keys():
                self.layers[name].combine(layer)
            else:
                self.layers[name] = layer

    def add_layer(self, name, vec_x_val, vec_min_y, vec_max_y):
        layer = EchoLayer(self, vec_x_val, vec_min_y, vec_max_y)
        self.__set_layer__(name, layer)

    def add_layer_from_static_layer(self, name, min_y, max_y):
        layer = EchoLayer.from_static_layer(self, min_y, max_y)
        self.__set_layer__(name, layer)

    def add_layer_from_ping_param_offsets_absolute(self, name, ping_param_name, offset_0, offset_1):
        layer = EchoLayer.from_ping_param_offsets_absolute(self, ping_param_name, offset_0, offset_1)
        self.__set_layer__(name, layer)

    def add_layer_from_ping_param_offsets_relative(self, name, ping_param_name, offset_0, offset_1):
        layer = EchoLayer.from_ping_param_offsets_relative(self, ping_param_name, offset_0, offset_1)
        self.__set_layer__(name, layer)

    def remove_layer(self, name):
        if name == "main":
            self.main_layer = None
        elif name in self.layers.keys():
            self.layers.pop(name)

    def clear_layers(self):
        self.layers = {}

    def clear_main_layer(self):
        self.main_layer = None

    def iterate_ping_data(self, keep_to_xlimits=True):
        if keep_to_xlimits:
            xcoord = self.get_x_indices()[1]
            nrs = np.arange(xcoord[0], xcoord[-1] + 1)
        else:
            nrs = range(len(self.ping_times))

        return [PingData(self, nr) for nr in nrs]
