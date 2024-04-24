import numpy as np

from typing import Tuple
from collections import defaultdict
import datetime as dt

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

import themachinethatgoesping.tools as ptools
import themachinethatgoesping.pingprocessing.core as pcore

class EchogramSection(object):
    
    def __init__(self, data):
        self._data = data
        
        #self._pings = []
        self._ping_numbers = []
        self._ping_times_unix = []
        self._ping_distances = []
        self._sample_numbers = []
        self._sample_depths = []

        self._bottom_depths = []
        self._bottom_depths_times = []

        self._echosounder_depths = []
        self._echosounder_depths_times = []

        self._ping_parameters = defaultdict(list)
        
        #interpolators ping parameter -> ping index
        self._interpolator_pi_pnr = None
        self._interpolator_pnr_pi = None
        self._interpolator_pi_time = None
        self._interpolator_time_pi = None
        self._interpolator_pi_distance = None
        #self._interpolator_distance_pi = None

        #interpolators sample parameter -> sample index
        self._interpolator_si_snr = None
        self._interpolator_snr_si = None
        self._interpolator_si_depth = None
        self._interpolator_depth_si = None

        #interpolators index -> bottom depth
        self._interpolator_time_bottom_depth = None
        self._interpolator_time_echosounder_depth = None

    def get_downsampled(self, downsample_factor):
        echogram_section = EchogramSection(self._data[:,::downsample_factor])
        
        if len(self._ping_numbers) > 0:
            echogram_section.set_ping_numbers(self.get_ping_numbers())
        if len(self._ping_times_unix) > 0:
            echogram_section.set_ping_times(self.get_ping_times_unixtimes())
        if len(self._ping_distances) > 0:
            echogram_section.set_ping_distances(self.get_ping_distances())
        if len(self._sample_numbers) > 0:
            echogram_section.set_sample_numbers(self.get_sample_numbers()[::downsample_factor])
        if len(self._sample_depths) > 0:
            echogram_section.set_sample_depths(self.get_sample_depths()[::downsample_factor])
        if len(self._bottom_depths) > 0:
            echogram_section.set_bottom_depths(self._bottom_depths, self._bottom_depths_times)
        if len(self._echosounder_depths) > 0:
            echogram_section.set_echosounder_depths(self._echosounder_depths, self._echosounder_depths_times)
        if len(list(self._ping_parameters.keys())) > 0:
            echogram_section._ping_parameters = self._ping_parameters
        return echogram_section

    # --- plotting ---
    def get_echogram(
        self,
        ping_numbers=None,
        ping_axis='index', 
        sample_axis='index'):
        
        if ping_numbers is None:
            return self.get_data(), self.get_extent(ping_axis, sample_axis)  

        return self.get_data()[ping_numbers], self.get_extent(ping_axis, sample_axis, min_ping_index = ping_numbers[0], max_ping_index = ping_numbers[-1])

    def get_echogram_layer(
        self,
        upper_depth,
        lower_depth,
        depth_times=None,
        ping_indices=None,
        ping_axis='index', 
        sample_axis='index'):

        if ping_indices is None:
            ping_indices = self.get_ping_indices()

        if depth_times is not None:
            assert len(depth_times) == len(upper_depth), f"ERROR: len(depth_times) [{depth_times}] != len(upper_depth) [{upper_depth}]"
            i_depth_up = ptools.vectorinterpolators.LinearInterpolator(depth_times,upper_depth)
            i_depth_lo = ptools.vectorinterpolators.LinearInterpolator(depth_times,lower_depth)
            upper_depth = i_depth_up(self.get_ping_times_unixtimes()[ping_indices])
            lower_depth = i_depth_lo(self.get_ping_times_unixtimes()[ping_indices])


        assert len(upper_depth) == len(lower_depth), f"ERROR: len(upper_depth) [{upper_depth}] != len(lower_depth) [{lower_depth}]"
        assert len(ping_indices) == len(lower_depth), f"ERROR: len(ping_indices) [{ping_indices}] != len(lower_depth) [{lower_depth}]"      

        upper_sn_bound = self.sample_depth_to_sample_index(upper_depth)
        lower_sn_bound = self.sample_depth_to_sample_index(lower_depth)

        upper_sn_bound[upper_sn_bound < 0] = 0
        lower_sn_bound[lower_sn_bound < 0] = 0
        upper_sn_bound[upper_sn_bound >= self._data.shape[1]] = self._data.shape[1]-1
        lower_sn_bound[lower_sn_bound >= self._data.shape[1]] = self._data.shape[1]-1

        max_sn = np.nanmax((upper_sn_bound,lower_sn_bound))
        min_sn = np.nanmin((upper_sn_bound,lower_sn_bound))
        upper_sn_bound = upper_sn_bound - min_sn
        lower_sn_bound = lower_sn_bound - min_sn

        echo_filtered = (self.get_data()[:,min_sn:max_sn+1])[ping_indices].copy()
        extent_filtered = self.get_extent(
            ping_axis=ping_axis,
            sample_axis=sample_axis,
            min_sample_index=min_sn,
            max_sample_index=max_sn,
            min_ping_index = ping_indices[0], 
            max_ping_index = ping_indices[-1])
        
        mask = \
            np.greater.outer(upper_sn_bound, np.arange(echo_filtered.shape[1])) \
            | np.less.outer(lower_sn_bound, np.arange(echo_filtered.shape[1]))

        #print(len(ping_indices),mask.shape,echo_filtered.shape,upper_sn_bound.shape,lower_sn_bound.shape,np.min(upper_sn_bound),np.max(upper_sn_bound),np.min(lower_sn_bound),np.max(lower_sn_bound),np.max(upper_sn_bound>=lower_sn_bound))
        
        echo_filtered[mask] = np.nan

        return echo_filtered, extent_filtered


    def get_extent(self, 
        ping_axis='index', 
        sample_axis='index', 
        min_ping_index = 0, 
        max_ping_index = -1,
        min_sample_index = 0, 
        max_sample_index = -1):
        extent = []
        match ping_axis:
            case 'index':
                # if min_ping_index is negative
                if max_ping_index < 0:
                    max_ping_index += self._data.shape[0]
                    
                extent.append(min_ping_index - 0.5)
                extent.append(max_ping_index + 0.5)
            case 'number':
                ping_numbers = self.get_ping_numbers()
                extent.append(ping_numbers[min_ping_index]-0.5)
                extent.append(ping_numbers[max_ping_index]+0.5)
            case 'time':
                ping_times = self.get_ping_times_unixtimes()
                t0 = ping_times[min_ping_index]
                t1 = ping_times[max_ping_index]
                delta_t = 0.5*(t1-t0)/len(ping_times)
                x_lims = [dt.datetime.fromtimestamp(tmp_t, dt.timezone.utc) for tmp_t in [t0-delta_t,t1+delta_t]]
                x_lims = mdates.date2num(x_lims)
                extent.extend(x_lims)
            case 'distance_m':
                ping_distances = self.get_ping_distances()
                d0 = ping_distances[min_ping_index]
                d1 = ping_distances[max_ping_index]
                delta_d = 0.5*(d1-d0)/len(ping_distances)
                extent.extend([d0-delta_d,d1+delta_d])
            case 'distance_km':
                ping_distances = self.get_ping_distances()
                d0 = ping_distances[min_ping_index]/1000
                d1 = ping_distances[max_ping_index]/1000
                delta_d = 0.5*(d1-d0)/len(ping_distances)
                extent.extend([d0-delta_d,d1+delta_d])
            case _:
                raise RuntimeError(f"Unknown ping_axis: {ping_axis}\n Expected: 'index', 'number', 'time', 'distance_m' or 'distance_km'")

        match sample_axis:
            case 'index':
                # if min_ping_index is negative
                if max_ping_index < 0:
                    max_sample_index += self._data.shape[1]
                    
                extent.append(min_sample_index - 0.5)
                extent.append(max_sample_index + 0.5)
            case 'number':
                sample_numbers = self.get_sample_numbers()
                extent.append(sample_numbers[min_sample_index]+0.5)
                extent.append(sample_numbers[max_sample_index]-0.5)
            case 'depth':
                sample_depths = self.get_sample_depths()
                r0 = sample_depths[min_sample_index]
                r1 = sample_depths[max_sample_index]
                delta_r = 0.5*(r1-r0)/len(sample_depths)
                extent.extend([r1+delta_r,r0-delta_r])
            case _:
                raise RuntimeError(f"Unknown sample_axis: {sample_axis}\n Expected: ''index', number' or 'depth'")

        return extent
    @property
    def shape(self):
        return self._data.shape

    def plot(
        self, 
        ping_axis = 'time', 
        sample_axis = 'depth',
        ax = None,
        fig_size = (15,4),
        name = 'Echogram',
        colorbar = True,
        plot_bottom = True,
        **kwargs):
    
        image,extent = self.get_echogram(ping_axis=ping_axis, sample_axis=sample_axis)
        
        plot_args = {
        "vmin" : np.nanquantile(image, 0.05),
        "vmax" : np.nanquantile(image, 0.95),
        "aspect" : "auto",
        "cmap" : "YlGnBu_r"
        }
        plot_args.update(kwargs)

        if ax is None:
            fig,ax = pcore.helper.create_figure(name)
            fig.set_size_inches(15,4)
            pcore.helper.set_ax_timeformat(ax)
            
        mapable = ax.imshow(image.transpose(), extent = extent, **plot_args)

        if plot_bottom and len(self._bottom_depths) > 0:
            times = self.get_ping_times_unixtimes()
            bottom = self.bottom_depth_per_ping_time(times)
            ax.plot(self.get_ping_times_datetimes(), bottom, color='black')

        if colorbar:
            ax.get_figure().colorbar(mapable,ax=ax)

        return ax.get_figure(),ax
        
    # --- setters ---
    # def set_pings(self, pings):
    #     assert len(pings) == self._data.shape[0], f"ERROR[set_pings]: len(pings) != self._data.shape[0]! [{len(pings)} != {self._data.shape[0]}]"
    #     self._pings = pings

    def set_ping_parameters(self, parameter_key, parameters):
        assert len(parameters) == self._data.shape[0], f"ERROR[set_ping_parameter]: len(set_ping_parameter) != self._data.shape[0]! [{len(parameters)} != {self._data.shape[0]}]"
        self._ping_parameters[parameter_key] = parameters

    def set_ping_numbers(self, ping_numbers):
        assert len(ping_numbers) == self._data.shape[0], f"ERROR[set_ping_numbers]: len(ping_numbers) != self._data.shape[0]! [{len(ping_numbers)} != {self._data.shape[0]}]"
        self._ping_numbers = ping_numbers
        self._interpolator_pi_pnr = ptools.vectorinterpolators.NearestInterpolator(range(len(self.get_ping_numbers())),self.get_ping_numbers())
        self._interpolator_pnr_pi = ptools.vectorinterpolators.NearestInterpolator(self.get_ping_numbers(),range(len(self.get_ping_numbers())))

    def set_ping_times(self, ping_unixtimes):
        assert len(ping_unixtimes) == self._data.shape[0], f"ERROR[set_ping_times]: len(ping_unixtimes) != self._data.shape[0]! [{len(ping_unixtimes)} != {self._data.shape[0]}]"
        self._ping_times_unix = np.array(ping_unixtimes)
        self._interpolator_pi_time = ptools.vectorinterpolators.NearestInterpolator(range(len(self.get_ping_times_unixtimes())),self.get_ping_times_unixtimes())
        self._interpolator_time_pi = ptools.vectorinterpolators.NearestInterpolator(self.get_ping_times_unixtimes(),range(len(self.get_ping_times_unixtimes())))

    def set_ping_distances(self, ping_distances):
        assert len(ping_distances) == self._data.shape[0], f"ERROR[set_ping_distances]: len(ping_distances) != self._data.shape[0]! [{len(ping_distances)} != {self._data.shape[0]}]"
        self._ping_distances = ping_distances
        self._interpolator_pi_distance = ptools.vectorinterpolators.NearestInterpolator(range(len(self.get_ping_distances())),self.get_ping_distances())
        #self._interpolator_distance_pi = ptools.vectorinterpolators.NearestInterpolator(self.get_ping_distances(),range(len(self.get_ping_distances())))

    def set_sample_numbers(self, sample_numbers):
        assert len(sample_numbers) == self._data.shape[1], f"ERROR[set_sample_numbers]: len(sample_numbers) != self._data.shape[1]! [{len(sample_numbers)} != {self._data.shape[1]}]"
        self._sample_numbers = sample_numbers
        self._interpolator_si_snr = ptools.vectorinterpolators.NearestInterpolator(range(len(self.get_sample_numbers())),self.get_sample_numbers())
        self._interpolator_snr_si = ptools.vectorinterpolators.NearestInterpolator(self.get_sample_numbers(),range(len(self.get_sample_numbers())))

    def set_sample_depths(self, sample_depths):
        assert len(sample_depths) == self._data.shape[1], f"ERROR[set_sample_depths]: len(sample_depths) != self._data.shape[1]! [{len(sample_depths)} != {self._data.shape[1]}]"
        self._sample_depths = sample_depths
        self._interpolator_si_depth = ptools.vectorinterpolators.NearestInterpolator(range(len(self.get_sample_depths())),self.get_sample_depths())
        self._interpolator_depth_si = ptools.vectorinterpolators.NearestInterpolator(self.get_sample_depths(),range(len(self.get_sample_depths())))

    def set_bottom_depths(self, bottom_depths, bottom_depth_times):
        assert len(bottom_depths) == len(bottom_depth_times), f"ERROR[set_bottom_depths]: len(bottom_depths) != len(bottom_depth_times)! [{len(bottom_depths)} != {len(bottom_depth_times)}]"
        self._bottom_depths = bottom_depths
        self._bottom_depths_times = bottom_depth_times
        self._interpolator_time_bottom_depth = ptools.vectorinterpolators.LinearInterpolator(self._bottom_depths_times,self._bottom_depths)
    
    def set_echosounder_depths(self, echosounder_depths, echosounder_depth_times):
        assert len(echosounder_depths) == len(echosounder_depth_times), f"ERROR[set_echosounder_depths]: len(echosounder_depths) != len(echosounder_depth_times)! [{len(echosounder_depths)} != {len(echosounder_depth_times)}]"
        self._echosounder_depths = echosounder_depths
        self._echosounder_depths_times = echosounder_depth_times
        self._interpolator_time_echosounder_depth = ptools.vectorinterpolators.LinearInterpolator(self._echosounder_depths_times,self._echosounder_depths)
    
    # --- getters ---
    def get_data(self):
        return self._data

    # def get_pings(self):
    #     if len(self._pings) > 0:
    #         return self._pings
            
    #     raise RuntimeError("ERROR[get_pings]: pings have not been set!")

    def get_ping_indices(self):
        return np.arange(self._data.shape[0])

    def get_sample_indices(self):
        return np.arange(self._data.shape[1])

    def get_ping_parameters(self):
        if len(list(self._ping_parameters.keys())) > 0:
            return self._ping_parameters
            
        raise RuntimeError("ERROR[get_ping_parameters]: No ping parameters have been set!")

    def get_ping_numbers(self):
        if len(self._ping_numbers) > 0:
            return self._ping_numbers
        
        raise RuntimeError("ERROR[get_ping_numbers]: ping numbers have not been set!")

    def get_ping_times_unixtimes(self):
        if len(self._ping_times_unix) > 0:
            return self._ping_times_unix
        
        raise RuntimeError("ERROR[get_ping_times_unixtimes]: ping times have not been set!")

    def get_ping_times_datetimes(self):
        if len(self._ping_times_unix) > 0:
            return [dt.datetime.fromtimestamp(tmp_t, dt.timezone.utc) for tmp_t in self.get_ping_times_unixtimes()]
            
        raise RuntimeError("ERROR[get_ping_times_datetimes]: ping times have not been set!")

    def get_ping_times_mdates(self):
        if len(self._ping_times_unix) > 0:
            return mdates.date2num(self.get_ping_times_datetimes())
        
        raise RuntimeError("ERROR[get_ping_times_mdates]: ping times have not been set!")

    def get_ping_distances(self):
        if len(self._ping_distances) > 0:
            return self._ping_distances
            
        raise RuntimeError("ERROR[get_ping_distances]: ping distances have not been set!")

    def get_sample_numbers(self):
        if len(self._sample_numbers) > 0:
            return self._sample_numbers
            
        raise RuntimeError("ERROR[get_sample_numbers]: sample numbers have not been set!")

    def get_sample_depths(self):
        if len(self._sample_depths) > 0:
            return self._sample_depths
            
        raise RuntimeError("ERROR[get_sample_depths]: sample ranges have not been set!")

    # ----- interpolator getters -----
    
    def ping_index_to_ping_number(self, ping_index):
        self.get_ping_numbers(); # check if ping numbers have been set
        return np.array(self._interpolator_pi_pnr(ping_index)).astype(int)

    def ping_number_to_ping_index(self, ping_number):
        self.get_ping_numbers(); # check if ping numbers have been set
        return np.array(self._interpolator_pnr_pi(ping_number)).astype(int)

    def ping_index_to_ping_time(self, ping_index):
        self.get_ping_times_unixtimes(); # check if ping times have been set
        return np.array(self._interpolator_pi_time(ping_index))

    def ping_time_to_ping_index(self, ping_time):
        self.get_ping_times_unixtimes(); # check if ping times have been set
        return np.array(self._interpolator_time_pi(ping_time)).astype(int)

    def ping_time_to_ping_number(self, ping_time):
        return self.ping_index_to_ping_number(self.ping_time_to_ping_index(ping_time))

    def ping_number_to_ping_time(self, ping_number):
        return self.ping_index_to_ping_time(self.ping_number_to_ping_index(ping_number))

    def ping_distance_to_ping_index(self, ping_distance):
        get_ping_distances(); # check if ping distances have been set
        return np.array(self._interpolator_distance_pi(ping_distance)).astype(int)

    def ping_index_to_ping_distance(self, ping_index):
        get_ping_distances(); # check if ping distances have been set
        return np.array(self._interpolator_pi_distance(ping_index))

    def ping_distance_to_ping_number(self, ping_distance):
        return self.ping_index_to_ping_number(self.ping_distance_to_ping_index(ping_distance))

    def ping_number_to_ping_distance(self, ping_number):
        return self.ping_index_to_ping_distance(self.ping_number_to_ping_index(ping_number))

    def ping_distance_to_ping_time(self, ping_distance):
        return self.ping_index_to_ping_time(self.ping_distance_to_ping_index(ping_distance))

    def ping_time_to_ping_distance(self, ping_time):
        return self.ping_index_to_ping_distance(self.ping_time_to_ping_index(ping_time))

    def sample_number_sample_index(self, sample_number):
        self.get_sample_numbers(); # check if sample numbers have been set
        return np.array(self._interpolator_si_snr(sample_number)).astype(int)

    def sample_index_to_sample_number(self, sample_index):
        self.get_sample_numbers(); # check if sample numbers have been set
        return np.array(self._interpolator_snr_si(sample_index)).astype(int)

    def sample_depth_to_sample_index(self, sample_depth):
        self.get_sample_depths(); # check if sample depths have been set
        return np.array(self._interpolator_depth_si(sample_depth)).astype(int)

    def sample_index_to_sample_depth(self, sample_index):
        self.get_sample_depths(); # check if sample depths have been set
        return np.array(self._interpolator_si_depth(sample_index))

    def sample_depth_to_sample_number(self, sample_depth):
        return self.sample_index_to_sample_number(self.sample_depth_to_sample_index(sample_depth))

    def sample_number_to_sample_depth(self, sample_number):
        return self.sample_index_to_sample_depth(self.sample_number_to_sample_index(sample_number))

    def bottom_depth_per_ping_time(self, ping_time):
        if self._interpolator_time_bottom_depth:
            return np.array(self._interpolator_time_bottom_depth(ping_time))

        raise RuntimeError("ERROR[bottom_depth_per_ping_time]: bottom depths have not been set!")

    def bottom_depth_per_ping_index(self, ping_index):
        return self.bottom_depth_per_ping_time(self.ping_index_to_ping_time(ping_index))

    def bottom_depth_per_ping_number(self, ping_number):
        return self.bottom_depth_per_ping_time(self.ping_number_to_ping_time(ping_number))

    def bottom_depth_per_ping_distance(self, ping_distance):
        return self.bottom_depth_per_ping_time(self.ping_distance_to_ping_time(ping_distance))

    def echosounder_depth_per_ping_time(self, ping_time):
        if self._interpolator_time_echosounder_depth:
            return np.array(self._interpolator_time_echosounder_depth(ping_time))

        raise RuntimeError("ERROR[echosounder_depth_per_ping_time]: echosounder depths have not been set!")

    def echosounder_depth_per_ping_index(self, ping_index):
        return self.echosounder_depth_per_ping_time(self.ping_index_to_ping_time(ping_index))

    def echosounder_depth_per_ping_number(self, ping_number):
        return self.echosounder_depth_per_ping_time(self.ping_number_to_ping_time(ping_number))

    def echosounder_depth_per_ping_distance(self, ping_distance):
        return self.echosounder_depth_per_ping_time(self.ping_distance_to_ping_time(ping_distance))
