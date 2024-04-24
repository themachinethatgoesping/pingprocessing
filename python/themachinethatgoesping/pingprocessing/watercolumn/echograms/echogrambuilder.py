import numpy as np

from typing import Tuple
from collections import defaultdict
import datetime as dt
from tqdm.auto import tqdm

import matplotlib as mpl
import matplotlib.dates as mdates

from themachinethatgoesping import echosounders, pingprocessing

from .echoimagebuilders import TimeImageBuilder



class EchogramBuilder:
    def __init__(self, pings, pingsampleselector = echosounders.pingtools.PingSampleSelector(),
                 apply_pss_to_bottom = False):
        self.pings = np.array(pings)
        self.pss = pingsampleselector
        self.apply_pss_to_bottom = apply_pss_to_bottom

    def build_time_echogram(
        self,
        max_pings   = 10000, 
        max_samples = None,
        max_image_size = 10000 * 3000,
        min_time = np.nan,
        max_time = np.nan, 
        min_depth = np.nan,
        max_depth = np.nan,
        min_delta_t = np.nan,
        min_delta_t_quantile = 0.05,
        pss = None,
        linear_mean = True,
        use_range = False,
        verbose=True
        ):

        if pss is None:
            pss = self.pss
        
        # build echogram parameters
        imagebuilder = TimeImageBuilder( 
            pings = self.pings, 
            max_pings   = max_pings,
            max_samples = max_samples,
            max_image_size = max_image_size,
            min_time = min_time,
            max_time = max_time, 
            min_depth = min_depth,
            max_depth = max_depth,
            min_delta_t = min_delta_t,
            min_delta_t_quantile = min_delta_t_quantile,
            pss = pss,
            apply_pss_to_bottom = self.apply_pss_to_bottom,
            linear_mean = linear_mean,
            verbose=verbose)
        self.last_imagebuilder = imagebuilder
        
        image, extent = imagebuilder.build_image(use_datetime = True, use_range = use_range)
        echo = pingprocessing.watercolumn.echograms.EchogramSection(image)
        
        bottom_depth = imagebuilder.bottom_d
        if use_range:
            heave = imagebuilder.min_d - imagebuilder.min_r
            bottom_depth -= heave

        echo.set_bottom_depths(bottom_depth[np.isfinite(bottom_depth)], imagebuilder.times[np.isfinite(bottom_depth)])

        echo.set_echosounder_depths(imagebuilder.min_d[np.isfinite(imagebuilder.min_d)], imagebuilder.times[np.isfinite(imagebuilder.min_d)])
        echo.set_ping_times(imagebuilder.echopingtimes)

        if use_range:
            echo.set_sample_depths(imagebuilder.ranges)
        else:
            echo.set_sample_depths(imagebuilder.depths)
        #echo.set_ping_distances
        #echo.set_ping_numbers
        #echo.set_sample_numbers

        return echo