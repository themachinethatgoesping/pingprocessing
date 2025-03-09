# This is an internal class used by the echogram class to represent a layer in the echogram.

import datetime as dt
import numpy as np

# external Ping packages
from themachinethatgoesping import tools

# internal pingprocessing imports
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.pingprocessing.core.asserts import assert_length, assert_valid_argument

class EchoLayer:
    def __init__(self, echodata, vec_x_val, vec_min_y, vec_max_y):
        if vec_min_y is None:
            vec_min_y = np.zeros(len(vec_x_val))

        if vec_max_y is None:
            vec_max_y = np.empty(len(vec_x_val))
            vec_max_y.fill(echodata.x_coordinates[-1])
            
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
        #vec_min_y = tools.vectorinterpolators.AkimaInterpolator(vec_x_val, vec_min_y, extrapolation_mode = 'nearest')(echodata.vec_x_val)
        #vec_max_y = tools.vectorinterpolators.AkimaInterpolator(vec_x_val, vec_max_y, extrapolation_mode = 'nearest')(echodata.vec_x_val)       
        vec_min_y = tools.vectorinterpolators.LinearInterpolator(vec_x_val, vec_min_y, extrapolation_mode = 'nearest')(echodata.vec_x_val)
        vec_max_y = tools.vectorinterpolators.LinearInterpolator(vec_x_val, vec_max_y, extrapolation_mode = 'nearest')(echodata.vec_x_val)       

        self.echodata = echodata
        # create layer indices representing the range (i1 = last element +1_
        i0 = np.empty(len(echodata.ping_times),dtype = int)
        i1 = np.empty(len(echodata.ping_times),dtype = int)
        for nr,interpolator in enumerate(echodata.y_coordinate_indice_interpolator):
            if interpolator is not None:
                i0[nr] = interpolator(vec_min_y[nr]) + 0.5
                i1[nr] = interpolator(vec_max_y[nr]) + 1.5

        self.set_indices(i0, i1)

    def set_indices(self, i0, i1):
        bss = self.echodata.beam_sample_selections
        assert_length("set_indices", bss, [i0, i1])

        self.i0 = np.array(i0)
        self.i1 = np.array(i1)

        self.i0 = np.maximum(self.i0, 0)
        self.i1 = np.maximum(self.i1, self.i0)
        self.i1 = np.minimum(self.i1, [bss[i].get_number_of_samples_ensemble() for i in range(len(i1))])

    @classmethod
    def from_static_layer(cls, echodata, min_y, max_y):
        min_y = [min_y, min_y] if min_y is not None else None
        max_y = [max_y, max_y] if max_y is not None else None
        return cls(echodata, [echodata.vec_x_val[0], echodata.vec_x_val[-1]], min_y, max_y)

    @classmethod
    def from_ping_param_offsets_absolute(cls, echodata, ping_param_name, offset_0, offset_1):
        x,y = echodata.get_ping_param(ping_param_name)
        y0 = np.array(y) + offset_0 if offset_0 is not None else None
        y1 = np.array(y) + offset_1 if offset_1 is not None else None
        return cls(echodata,x,y0,y1)
        
    @classmethod
    def from_ping_param_offsets_relative(cls, echodata, ping_param_name, offset_0, offset_1):
        x,y = echodata.get_ping_param(ping_param_name)
        y0 = np.array(y) * offset_0 if offset_0 is not None else None
        y1 = np.array(y) * offset_1 if offset_1 is not None else None
        return cls(echodata, x, y0, y1)

    def get_y_indices(self, wci_nr):        
        n_samples = self.echodata.beam_sample_selections[wci_nr].get_number_of_samples_ensemble()
        y_indices_image = np.arange(len(self.echodata.y_coordinates))
        y_indices_wci = np.round(self.echodata.y_coordinate_indice_interpolator[wci_nr](self.echodata.y_coordinates)).astype(int)

        start_y = np.max([0, self.i0[wci_nr]])
        end_y = np.min([n_samples, self.i1[wci_nr]])

        if start_y >= end_y:
            return None, None        
        valid_coordinates = np.where(np.logical_and(y_indices_wci >= start_y, y_indices_wci < end_y))[0]

        return y_indices_image[valid_coordinates], y_indices_wci[valid_coordinates]

    def combine(self, other):
        assert_length("get_filtered_by_y_extent", self.i0, [other.i0, self.i1, other.i1])
        i0 = np.maximum(self.i0, other.i0)
        i1 = np.minimum(self.i1, other.i1)

        self.set_indices(i0, i1)
    
class PingData:
    def __init__(self, echodata, nr):
        self.echodata = echodata
        self.nr = nr

    def get_wci(self):
        return self.echodata.get_wci(self.nr)        
    
    def get_wci_layers(self):
        return self.echodata.get_wci_layers(self.nr)

    def get_extent_layers(self, axis_name=None):
        return self.echodata.get_extent_layers(self.nr, axis_name=axis_name)

    def get_limits_layers(self, axis_name=None):
        return self.echodata.get_limits_layers(self.nr, axis_name=axis_name)

    def get_ping_time(self):
        return self.echodata.ping_times[self.nr]

    def get_datetime(self):
        return dt.datetime.fromtimestamp(self.get_ping_time(), self.echodata.time_zone)
