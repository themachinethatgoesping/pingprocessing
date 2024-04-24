import numpy as np

from typing import Tuple
from collections import defaultdict
import datetime as dt
from tqdm.auto import tqdm

import matplotlib as mpl
import matplotlib.dates as mdates

from themachinethatgoesping import echosounders, pingprocessing
import themachinethatgoesping as Ping


class TimeImageBuilder:
    def __init__(self,
                 pings,
                 max_pings   = 10000,
                 max_samples = None,
                 max_image_size = 10000 * 3000,
                 min_time = np.nan,
                 max_time = np.nan,
                 min_depth = np.nan,
                 max_depth = np.nan,
                 min_delta_t = np.nan,
                 min_delta_t_quantile = 0.05,
                 pss = echosounders.pingtools.PingSampleSelector(),
                 linear_mean = True,
                 apply_pss_to_bottom = False,
                 verbose=True):

        self.verbose = verbose

        echopingtimes, echopingnumbers = self.sample_ping_times(
            pings = pings,
            max_pings = max_pings,
            min_time = min_time,
            max_time = max_time,
            min_delta_t = min_delta_t,
            min_delta_t_quantile = min_delta_t_quantile,
            verbose = verbose)

        if max_samples is None:
            max_samples = int(max_image_size/len(echopingnumbers))

        AV = []
        min_r = []
        max_r = []
        res_r = []
        min_d = []
        max_d = []
        bottom_d = []
        minslant_d = []
        nrs = {}
        times = []

        nroff=0

        for NR,nr in enumerate(tqdm(np.unique(echopingnumbers), disable=(not verbose), delay=1)):
            if nr == -1:
                nroff += 1
                continue

            ping = pings[nr]

            # change angle selection
            if apply_pss_to_bottom:
                aw = ping.watercolumn.get_beam_crosstrack_angles()
                ab = ping.bottom.get_beam_crosstrack_angles()
                ad = np.median(aw-ab)
                pss_ = pss.copy()
                pss_.select_beam_range_by_angles(pss.get_min_beam_angle()+ad,pss.get_max_beam_angle()+ad)

                sel = pss_.apply_selection(ping.watercolumn)
            else:
                sel = pss.apply_selection(ping.watercolumn)


            if len(sel.get_beam_numbers()) == 0:
                nroff += 1
                continue

            nrs[nr] = NR - nroff


            c = ping.watercolumn.get_sound_speed_at_transducer()
            z = ping.get_geolocation().z
            angle_factor = np.cos(np.radians(np.mean(ping.watercolumn.get_beam_crosstrack_angles()[sel.get_beam_numbers()])))
            res_r.append(ping.watercolumn.get_sample_interval()*c*0.5)
            min_r.append(np.max(ping.watercolumn.get_first_sample_offset_per_beam()[sel.get_beam_numbers()])*res_r[-1])
            max_r.append(np.max(ping.watercolumn.get_number_of_samples_per_beam(sel))*res_r[-1] + min_r[-1])
            min_d.append(z + min_r[-1] * angle_factor)
            max_d.append(z + max_r[-1] * angle_factor)
            times.append(ping.get_timestamp())

            if ping.has_bottom():
                if ping.bottom.has_xyz():
                    #sel_bottom = pss.apply_selection(ping.bottom)
                    #bd = np.nanmin(p.bottom.get_xyz(sel_bottom).z) + p.get_geolocation().z
                    # this is incorrect
                    bd = np.nanmin(ping.bottom.get_xyz(sel).z) + ping.get_geolocation().z
                    minslant_d = np.nanquantile(ping.watercolumn.get_bottom_range_samples(),0.01)*res_r[-1]*angle_factor + ping.get_geolocation().z
                    bd = minslant_d

                bottom_d.append(bd)
            else:
                bottom_d.append(np.nan)

            av = ping.watercolumn.get_av(sel)
            if av.shape[0] == 1:
                av = av[0]
            else:
                if linear_mean:
                    av = np.power(10,av*0.1)

                av = np.nanmean(av,axis=0)

                if linear_mean:
                    av = 10*np.log10(av)

            AV.append(av)

        self.AV    = AV
        self.min_d = np.array(min_d)
        self.max_d = np.array(max_d)
        self.res_r = np.array(res_r)
        self.min_r = np.array(min_r)
        self.max_r = np.array(max_r)
        self.bottom_d = np.array(bottom_d)
        self.times=np.array(times)
        self.echopingtimes = np.array(echopingtimes)
        self.echopingtimestep = self.echopingtimes[1] - self.echopingtimes[0]
        self.echopingnumbers = np.array(echopingnumbers)
        self.nrs = nrs
        self.depths = self.sample_image_depths(self.min_d, self.max_d, self.res_r, min_depth, max_depth, max_samples)
        self.ranges = self.sample_image_depths(self.min_r, self.max_r, self.res_r, min_depth, max_depth, max_samples)
        self.range_step = self.depths[1] - self.depths[0]

    @staticmethod
    def sample_ping_times(pings, max_pings=10000, min_time = np.nan, max_time = np.nan, min_delta_t = np.nan, min_delta_t_quantile = 0.05, verbose=True):

        min_time = np.nanmax([pings[0].get_timestamp(), min_time])
        max_time = np.nanmin([pings[-1].get_timestamp(), max_time])

        if verbose:
            print(f'- Min time  : {dt.datetime.fromtimestamp(min_time,dt.UTC)}\n- Max time  : {dt.datetime.fromtimestamp(max_time,dt.UTC)}\n- Diff {max_time-min_time}')

        # filter pings by time range and find delta t per transducer
        pings_filtered = pingprocessing.filter_pings.by_time(pings, min_time, max_time)
        pings_per_channel = pingprocessing.split_pings.by_channel_id(pings_filtered)

        ping_delta_t = []
        for cid, P in pings_per_channel.items():
            for i in range(1,len(P)):
                ping_delta_t.append(P[i].get_timestamp() - P[i-1].get_timestamp())

        min_delta_t = np.nanmax([np.nanquantile(ping_delta_t,min_delta_t_quantile), min_delta_t])

        # get ping_times
        ping_times = np.array([p.get_timestamp() for p in tqdm(pings, delay=1)])
        ping_numbers = np.array(list(range(len(pings))))

        echo_times = np.linspace(min_time,max_time,max_pings)
        delta_t = (echo_times[1]-echo_times[0])
        if delta_t < min_delta_t:
            delta_t = min_delta_t
            echo_times = np.arange(min_time, max_time + delta_t, delta_t)


        # get nearest neighbor for echo_times
        interplator = Ping.tools.vectorinterpolators.NearestInterpolator(ping_times,ping_numbers)
        echopingnumbers = np.array(interplator(echo_times)).astype(int)
        echopingtimes = ping_times[echopingnumbers]

        # exclude all times where the time difference is larger than delta_t/2
        echopingdiffs = np.abs(echo_times - echopingtimes)
        excl = np.argwhere(echopingdiffs > delta_t * 0.5)

        echopingtimes[excl]   = np.nan
        echopingnumbers[excl] = -1

        # remove duplicate numbers
        u, c = np.unique(echopingnumbers, return_counts=True)
        C = int(np.quantile(c,0.75,method='linear'))

        if C > 1:
            max_pings = int(max_pings/C)

            if max_pings > 10:
                print("MAX PINGS:", max_pings, C)
                return EchoPingParameters.sample_ping_times(
                      pings = pings,
                      max_pings = max_pings,
                      min_time = min_time,
                      max_time = max_time,
                      verbose = verbose)

        return echo_times, echopingnumbers

    @staticmethod
    def sample_image_depths(min_d, max_d, res_r, min_depth, max_depth, max_samples = 5000):
        # filter
        if min_depth is None: min_depth = np.nanquantile(min_d,0.25)/1.5
        if max_depth is None: max_depth = np.nanquantile(max_d,0.75)*1.5
        min_resolution = np.nanquantile(res_r,0.25)/1.5

        res = np.nanmax([np.nanmin(res_r), min_resolution])
        mind = np.nanmax([np.nanmin(min_d), min_depth])
        maxd = np.nanmin([np.nanmax(max_d), max_depth])

        depths = np.arange(mind, maxd + res, res)

        if len(depths) > max_samples:
            depths = np.linspace(mind, maxd, max_samples)

        return depths

    def build_image(self, use_range = False, use_datetime = True):

        i_echo = []
        i_av   = []

        if use_range:
            min_d = self.min_r
            max_d = self.max_r
            depths = self.ranges
        else:
            min_d = self.min_d
            max_d = self.max_d
            depths = self.depths

        image = np.empty((len(self.echopingnumbers),len(depths)))
        image.fill(np.nan)

        for i in tqdm(range(len(self.AV)), disable=(not self.verbose), delay=1):
            interpolator = Ping.tools.vectorinterpolators.LinearInterpolator([min_d[i],max_d[i]],[0,len(self.AV[i])])
            index_av = np.round(interpolator(depths)).astype(int)
            index_echo = np.array(range(len(depths)))

            index_echo = index_echo[index_av >= 0]
            index_av = index_av[index_av >= 0]

            i_echo.append(index_echo[index_av < len(self.AV[i])])
            i_av.append(index_av[index_av < len(self.AV[i])])

        for i,nr in enumerate(tqdm(self.echopingnumbers, disable=(not self.verbose), delay=1)):
            if nr == -1:
                continue

            if nr not in self.nrs.keys():
                continue

            NR = self.nrs[nr]

            index_echo = i_echo[NR]
            index_av = i_av[NR]
            image[i][index_echo] = self.AV[NR][index_av]

        extent = [
            self.echopingtimes[0]  - self.echopingtimestep*0.5,
            self.echopingtimes[-1] + self.echopingtimestep*0.5,
            depths[-1] + self.range_step*0.5,
            depths[0]  - self.range_step*0.5
        ]

        if use_datetime:
            extent[0] = dt.datetime.fromtimestamp(extent[0], dt.UTC)
            extent[1] = dt.datetime.fromtimestamp(extent[1], dt.UTC)

        return image, extent