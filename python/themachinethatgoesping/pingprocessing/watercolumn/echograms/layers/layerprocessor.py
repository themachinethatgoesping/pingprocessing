from copy import copy, deepcopy
import pytimeparse2
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from tqdm.auto import tqdm
import warnings

# external Ping packages
from themachinethatgoesping import tools

# internal pingprocessing imports
from themachinethatgoesping.pingprocessing.core.asserts import assert_length

class LayerProcessor:
    def __init__(self, 
                 echograms, 
                 names, 
                 base_name = None,
                 layers=None, 
                 deltaT="1min", 
                 step=1, 
                 min_val_qmin=0.02,
                 min_val_dmin=3,
                 only_process_visible=False,
                 show_progress=True):
        
        assert_length("LayerProcessor", echograms, [names])

        self.__base_name = names[-1] if base_name is None else base_name
        self.__deltaT = deltaT
        self.__compare_name = None
        self.__echograms = {}
        for e, n in zip(echograms, names):
            self.__echograms[n] = e
            if n != self.__base_name:
                if self.__compare_name is None:
                    self.__compare_name = n

        if self.__compare_name is None:
            raise RuntimeError("LayerProcessor: No second echogram")

        if layers is None:
            self.__layers = list(echograms[0].layers.keys())
        else:
            self.__layers = layers

        self.__data = self.__make_timeblocks_dataframe__(echograms, deltaT)

        total_pings = sum([len(e.iterate_ping_data(only_process_visible)[::step]) for e in echograms])
        if isinstance(show_progress, bool):
            if show_progress == True:
                progress = tqdm(total=total_pings, desc=f"Processing layers for {names}")
            else:
                progress = None
        elif show_progress is None:
            progress = None
        else:
            progress = show_progress
            progress.reset(total=total_pings)

        for name, echogram in self.__echograms.items():
            self.__add_layer_vals__(name, echogram, step, progress)

        if isinstance(progress, tqdm):
            progress.set_description_str("Filtering data")     
        self.reset_filters(min_val_qmin, min_val_dmin)
        if isinstance(progress, tqdm):
            progress.set_description_str("Done") 

            if not isinstance(show_progress, tqdm):
                progress.close()
            else:
                progress.refresh()

    def reset_filters(self, min_val_qmin=0.02, min_val_dmin=3):
        for layer in self.get_layers():
            for name in self.get_names():
                val_key = f"{layer}-{name}-val"
                val_key_all = f"{layer}-{name}-val-all"
                num_key = f"{layer}-{name}-num"
                num_key_all = f"{layer}-{name}-num-all"

                self.__data[val_key] = self.__data[val_key_all]
                self.__data[num_key] = self.__data[num_key_all]
                self.__data = self.__data.copy()

        self.__qmin = min_val_qmin
        self.__dmin = min_val_dmin
        self.__data = self.__filter_by_layer_size__(self.__data, self.get_layers(), self.get_names())
        self.__data = self.__filter_by_min_values__(self.__data, self.get_layers(), self.get_names(), self.__qmin, self.__dmin)
        self.__mark_outliers__()  

    def __str__(self):
        printer = tools.classhelper.ObjectPrinter("LayerProcessor", float_precission=3, superscript_exponents=False)
        printer.register_container("Echograms", list(self.__echograms.keys()))
        printer.register_string("Base name", self.__base_name)
        printer.register_string("Compare name", self.__compare_name)
        printer.register_container("Layers", list(self.__layers))
        printer.register_string("deltaT", self.__deltaT)
        printer.register_value("Min val qmin", self.__qmin)
        printer.register_value("Min val __dmin", self.__dmin)
        return printer.create_str()

    def __repr__(self):
        return f"LayerProcessor: {self.__echograms.keys()}"

        from scipy import stats

    def get_calibration_per_range(self, 
                                  data = None,
                                  layers = None,
                                  name = None,
                                  bootstrap_resamples = 100, 
                                  show_progress = True,
                                  min_n=None) :   
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')

            if layers is None:
                layers = self.get_layers()
            if name is None:
                name = self.__compare_name
            if data is None:
                data = self.get_data()


            med_mbes = []
            med_sbes = []
            med_c = []
            num_c = []
            upper = []
            lower = []
            iqr_m =[]
            iqr_s = []
            r = []
            
            if True:
                stat = np.nanmedian
                stat_name = 'median'
            elif False:
                stat = lambda x: np.nanquantile(x, 0.33) 
                stat_name = 'q33'
            else:
                stat = np.nanmean
                stat_name = 'mean'

            if show_progress:
                layers = tqdm(layers)
            else:
                layers = layers
                    
            for layer in layers:            
                r.append(float(layer.split('m')[0]))

                v_sbes = data[f'{layer}-{self.__base_name}-val'].values
                v_mbes = data[f'{layer}-{name}-val'].values
                
                iqr_m.append(np.nanquantile(v_mbes, 0.90) - np.nanquantile(v_mbes, 0.10))
                iqr_s.append(np.nanquantile(v_sbes, 0.90) - np.nanquantile(v_sbes, 0.10))

                c = np.array(v_sbes) - np.array(v_mbes)
                
                med_c.append(stat(c))
                num_c.append(len(c[np.isfinite(c)]))
                
                # Bootstrap resampling
            
                res = stats.bootstrap(
                    (c,  ),          # Tuple of data arrays
                    stat,          # Statistic function
                    n_resamples=bootstrap_resamples,  # Number of bootstrap samples
                    confidence_level=0.95,  # Confidence level for the interval
                    method='percentile',    # Method for confidence interval calculation
                    random_state=42     # For reproducibility
                )
                upper.append(res.confidence_interval.high)
                lower.append(res.confidence_interval.low)
            
            med_c = np.array(med_c)
            num_c = np.array(num_c)
            lower = np.array(lower)
            upper = np.array(upper)
            iqr_m = np.array(iqr_m)
            iqr_s = np.array(iqr_s)
            r = np.array(r)
            if min_n is not None:
                args = num_c >= min_n
                med_c = med_c[args]
                num_c = num_c[args]
                lower = lower[args]
                upper = upper[args]
                iqr_m = iqr_m[args]
                iqr_s = iqr_s[args]
                r = r[args]

            return med_c, r, (med_c - lower, upper-med_c), num_c, iqr_m, iqr_s #x,y,xerr,num


    def get_calibration_per_range2(self, 
                                  data = None,
                                  layers = None,
                                  name = None,
                                  bootstrap_resamples = 100, 
                                  show_progress = True) :   
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')

            if layers is None:
                layers = self.get_layers()
            if name is None:
                name = self.__compare_name
            if data is None:
                data = self.get_data()


            med_m = []
            med_s = []
            upper_m = []
            upper_s = []
            lower_s = []
            lower_m = []
            r = []
            
            if True:
                stat = np.nanmedian
                stat_name = 'median'
            elif False:
                stat = lambda x: np.nanquantile(x, 0.33) 
                stat_name = 'q33'
            else:
                stat = np.nanmean
                stat_name = 'mean'

            if show_progress:
                layers = tqdm(layers)
            else:
                layers = layers
                    
            for layer in layers:            
                r.append(float(layer.split('m')[0]))

                v_sbes = data[f'{layer}-{self.__base_name}-val'].values
                v_mbes = data[f'{layer}-{name}-val'].values
                                
                med_s.append(stat(v_sbes))
                med_m.append(stat(v_mbes))
                
                # Bootstrap resampling
            
                res = stats.bootstrap(
                    (v_sbes,  ),          # Tuple of data arrays
                    stat,          # Statistic function
                    n_resamples=bootstrap_resamples,  # Number of bootstrap samples
                    confidence_level=0.95,  # Confidence level for the interval
                    method='percentile',    # Method for confidence interval calculation
                    #random_state=42     # For reproducibility
                )
                upper_s.append(res.confidence_interval.high)
                lower_s.append(res.confidence_interval.low)

                res = stats.bootstrap(
                    (v_mbes,  ),          # Tuple of data arrays
                    stat,          # Statistic function
                    n_resamples=bootstrap_resamples,  # Number of bootstrap samples
                    confidence_level=0.95,  # Confidence level for the interval
                    method='percentile',    # Method for confidence interval calculation
                    #random_state=42     # For reproducibility
                )
                upper_m.append(res.confidence_interval.high)
                lower_m.append(res.confidence_interval.low)
            
            med_c = np.array(med_s) - np.array(med_m)
            lower = np.array(lower_s) - np.array(upper_m)
            upper = np.array(upper_s) - np.array(lower_m)
            r = np.array(r)
        
            return med_c, r, (med_c - lower, upper-med_c) #x,y,xerr,num

    def get_data(self):
        return self.__data.copy()

    def get_layers(self):
        return self.__layers

    def get_echograms(self):
        return self.__echograms

    def get_names(self):
        return list(self.__echograms.keys())

    def get_base_name(self):
        return self.__base_name

    def get_deltaT(self):
        return self.__deltaT

    def get_val(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{name}-val"]

    def get_base_val(self, layer):
        return self.get_val(layer, self.__base_name)

    def get_val_outliers(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{name}-val-out"]

    def get_base_val_outliers(self, layer):
        return self.get_val_outliers(layer, self.__base_name)

    def get_val_all(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{name}-val-all"]

    def get_base_val_all(self, layer):
        return self.get_val_all(layer, self.__base_name)

    def get_num(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{name}-num"]

    def get_base_num(self, layer):
        return self.get_num(layer, self.__base_name)

    def get_num_outliers(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{name}-num-out"]

    def get_base_num_outliers(self, layer):
        return self.get_num_outliers(layer, self.__base_name)

    def get_num_all(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{name}-num-all"]

    def get_base_num_all(self, layer):
        return self.get_num_all(layer, self.__base_name)

    def get_c(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{self.__base_name}-val"] - self.__data[f"{layer}-{name}-val"]

    def get_c_outliers(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{self.__base_name}-val-out"] - self.__data[f"{layer}-{name}-val-out"]

    def get_c_all(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return self.__data[f"{layer}-{self.__base_name}-val-all"] - self.__data[f"{layer}-{name}-val-all"]

    def get_valid_indices(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return np.isfinite(self.get_c(layer, name))

    def get_valid_indices_outliers(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return np.isfinite(self.get_c_outliers(layer, name))

    def get_valid_indices_all(self, layer, name = None):
        if name is None:
            name = self.__compare_name
        return np.isfinite(self.get_c_all(layer, name))

    def get_t(self):
        return self.__data["Datetime"]
    

    def add_param(self, name, times, param, step=1):
        parameter = np.empty(len(self.__data))
        parameter.fill(np.nan)
        times_ = []
        param_ = []
        ti = 0

        t_min = self.__data["Datetime_min"]
        t_max = self.__data["Datetime_max"]

        for t, p in zip(tqdm(times), param):
            while True:
                if t_min.iloc[ti] <= t < t_max.iloc[ti]:
                    times_.append(t)
                    param_.append(p)
                    break

                if len(times_) > 0:
                    parameter[ti] = np.nanmedian(param_)
                    times_ = []
                    param_ = []

                ti += 1
                if ti >= len(self.__data):
                    raise RuntimeError("add_param: Aaaaah")

        self.__data[name] = parameter
        self.__data = self.__data.copy()

    def remove_station(self, pm, station):
        processor = deepcopy(self)
        t0 = pm.get_start_time(station)
        t1 = pm.get_end_time(station)
        processor.__data = processor.__data[processor.__data.index < t0]
        pd.concat([processor.__data, self.__data[self.__data.index > t1]])

        return processor

    def split_per_station(self, pm):
        processor_per_station = {}
        
        for station in tqdm(pm.get_stations()):
            t0 = pm.get_start_time(station)
            t1 = pm.get_end_time(station)
            station_data = self.__data[self.__data.index >= t0]
            station_data = station_data[station_data.index <= t1]

            processor_per_station[station] = deepcopy(self)
            processor_per_station[station].__data = station_data.copy()
            
        return processor_per_station

    def split_per_param(self, param_name, param_ranges):
        processor_per_param = {}
        
        for name, r0, r1 in param_ranges:
            param_data = self.__data[self.__data[param_name] >= r0]
            param_data = param_data[param_data[param_name]<= r1]
            processor_per_param[name] = deepcopy(self)
            processor_per_param[name].__data = param_data.copy()
            
        return processor_per_param



    def __add_layer_vals__(self, name, echogram, step=1, progress=None):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            warnings.filterwarnings('ignore', r'Mean of empty slice')

            def make_nan():
                arr = np.empty((len(self.__data)))
                arr.fill(np.nan)
                return arr

            def make_null():
                return np.zeros(len(self.__data)).astype(int)

            vals = defaultdict(make_nan)
            nums = defaultdict(make_null)

            times_ = []
            vals_ = defaultdict(list)
            ti = 0

            t_min = self.__data["Datetime_min"]
            t_max = self.__data["Datetime_max"]

            if progress is not None:
                progress.set_description(f"Processing {name}")
                

            for pi in echogram.iterate_ping_data(False)[::step]:
                pt = pi.get_datetime()

                while True:
                    if t_min.iloc[ti] <= pt < t_max.iloc[ti]:
                        times_.append(pt)
                        for k, v in pi.get_wci_layers().items():
                            vals_[k].extend(v)
                        break

                    if len(times_) > 0:
                        for k, v in vals_.items():
                            vals[k][ti] = np.nanmedian(v)
                            nums[k][ti] = len(v)
                        times_ = []
                        vals_ = defaultdict(list)

                    ti += 1
                    if ti >= len(self.__data):
                        raise RuntimeError("get_layer_vals: Aaaaah")

                if progress is not None:
                    progress.update(1)

            for k in vals.keys():
                self.__data[f"{k}-{name}-val-all"] = vals[k]
                self.__data[f"{k}-{name}-num-all"] = nums[k]
                self.__data[f"{k}-{name}-val"] = vals[k]
                self.__data[f"{k}-{name}-num"] = nums[k]
                self.__data = self.__data.copy()

    def __mark_outliers__(self):
        for layer in self.get_layers():
            for name in self.get_names():
                val_key = f"{layer}-{name}-val"
                val_key_all = f"{layer}-{name}-val-all"
                val_key_out = f"{layer}-{name}-val-out"
                num_key = f"{layer}-{name}-num"
                num_key_all = f"{layer}-{name}-num-all"
                num_key_out = f"{layer}-{name}-num-out"

                self.__data[val_key_out] = self.__data[val_key_all]
                self.__data[num_key_out] = self.__data[num_key_all]
                self.__data = self.__data.copy()
                
                valid = np.isfinite(self.__data[val_key])
                self.__data.loc[valid, val_key_out] = np.nan
                self.__data.loc[valid, num_key_out] = 0

    @staticmethod
    def __filter_by_min_values__(data, layers, names=["mbes", "sbes"], qmin=0.02, dmin=3):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')

            new_data = data.copy()

            n = defaultdict(list)
            for name in names:
                for k in layers:
                    num_key = f"{k}-{name}-num"
                    val_key = f"{k}-{name}-val"

                    vmin = np.nanquantile(data[val_key], qmin)
                    vmin += dmin

                    invalid = data[val_key] < vmin

                    new_data.loc[invalid, val_key] = np.nan
                    #new_data.loc[invalid, num_key] = 0

            return new_data

    @staticmethod
    def __filter_by_layer_size__(data, layers, names=["mbes", "sbes"]):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')

            new_data = data.copy()

            n = defaultdict(list)
            for name in names:
                for k in layers:
                    num_key = f"{k}-{name}-num"

                    n[name].extend(data[num_key][data[num_key] > 0])

                n[name] = np.array(n[name])
                iqr = np.nanquantile(n[name], 0.75) - np.nanquantile(n[name], 0.25)
                q = np.median(n[name]) - iqr * 1.5

                new_data[f"min_num-{name}"] = int(q)

            num_keys = defaultdict(list)
            for name in names:
                min_num_key = f"min_num-{name}"

                for k in layers:
                    num_key = f"{k}-{name}-num"

                    # new_data[num_key][new_data[num_key] < new_data[min_num_key]] = 0
                    new_data.loc[new_data[num_key] < new_data[min_num_key], num_key] = 0
                    num_keys[k].append(num_key)

            for name in names:
                for k in layers:
                    val_key = f"{k}-{name}-val"

                    for num_key in num_keys[k]:
                        # new_data[val_key][new_data[num_key] <= 0] = np.nan
                        new_data.loc[new_data[num_key] <= 0, val_key] = np.nan

            return new_data

    @staticmethod
    def __make_timeblocks_dataframe__(echograms, deltaT="1min"):
        deltaT = pytimeparse2.parse(deltaT, as_timedelta=False)

        t0 = np.nanmin(
            [datetime.datetime.fromtimestamp(echogram.ping_times[0], datetime.timezone.utc) for echogram in echograms]
        )
        t1 = np.nanmax(
            [datetime.datetime.fromtimestamp(echogram.ping_times[-1], datetime.timezone.utc) for echogram in echograms]
        )

        t0 = (t0.replace(minute=0, second=0, microsecond=0)).timestamp()
        t1 = (t1.replace(minute=0, second=0, microsecond=0, hour=t1.hour + 1)).timestamp()

        T = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc) for t in np.arange(t0, t1, deltaT)]
        T_min = [
            datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
            for t in np.arange(t0 - deltaT / 2, t1 - deltaT / 2, deltaT)
        ]
        T_max = [
            datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
            for t in np.arange(t0 + deltaT / 2, t1 + deltaT / 2, deltaT)
        ]

        data = pd.DataFrame()
        data.index = T
        data["Datetime"] = T
        data["Unixtime"] = [t.timestamp() for t in T]
        data["Datetime_min"] = T_min
        data["Datetime_max"] = T_max

        return data
