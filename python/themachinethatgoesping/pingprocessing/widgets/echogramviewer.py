from time import time
import types
import numpy as np

import ipywidgets
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display
import asyncio

import themachinethatgoesping as theping
import themachinethatgoesping.pingprocessing.watercolumn.echograms as echograms


class EchogramViewer:
    def __init__(self, 
                 echogramdata, 
                 name="Echogram", 
                 names = None, 
                 figure=None, 
                 progress=None, 
                 show=True, 
                 voffsets=None,
                 cmap="YlGnBu_r", 
                 cmap_layer="jet", 
                 **kwargs):

        self.mapables = []
        if not isinstance(echogramdata, list):
            echogramdata = [echogramdata]
            
        self.echogramdata = echogramdata
        self.colorbar = [None for _ in self.echogramdata]
        self.pingline = [None for _ in self.echogramdata]
        self.fig_events = {}
        self.pingviewer = None
        self.echogram_axes = []
        
        self.voffsets = voffsets if voffsets is not None else [0 for _ in self.echogramdata]

        self.names = []
        for i in range(len(self.echogramdata)):
            if names is not None and len(names) >= i:
                self.names.append(names[i])
            else:
                self.names.append(None)
            
        self.nechograms = len(self.echogramdata)

        if isinstance(cmap, str):
            self.cmap = plt.get_cmap(cmap)
        else:
            self.cmap = cmap

        if isinstance(cmap_layer, str):
            self.cmap_layer = plt.get_cmap(cmap_layer)
        else:
            self.cmap_layer = cmap_layer
            
        # plot arguments
        self.args_plot = {
            "cmap": self.cmap,
            "aspect": "auto", 
            "vmin": -100, 
            "vmax": -25, 
            "interpolation": "nearest"
        }
        self.args_plot.update(kwargs)
        self.args_plot_layer = self.args_plot.copy()
        self.args_plot_layer["cmap"] = self.cmap_layer
        
        if figure is None:
            plt.ioff()
            self.fig = plt.figure(name, clear=True)
            self.axes = self.fig.subplots(nrows=self.nechograms, sharex=True, sharey=True)

            self.fig.set_tight_layout(True)
            self.fig.set_size_inches(10, 3 * self.nechograms)
            plt.ion()
        else:
            self.fig = figure
            if len(self.fig.axes) >= self.nechograms:
                self.axes = self.fig.axes[:self.nechograms]
            else:
                self.axes = self.fig.subplots(nrows=lenself.nechograms, sharex=True, sharey=True)

        try:
            iter(self.axes)
        except:
            self.axes = [self.axes]
        
        # initialize progressbar and buttons
        self.update_button = ipywidgets.Button(description="update")
        self.clear_button = ipywidgets.Button(description="clear output")
        self.update_button.on_click(self.show_background_zoom)
        self.clear_button.on_click(self.clear_output)

        # progressbar
        if progress is None:
            self.progress = theping.pingprocessing.widgets.TqdmWidget()
            self.display_progress = True
        else:
            self.progress = progress
            self.display_progress = False

        # sliders
        self.w_vmin = ipywidgets.FloatSlider(
            description="vmin", min=-150, max=100, step=5, value=self.args_plot["vmin"]
        )
        self.w_vmax = ipywidgets.FloatSlider(
            description="vmax", min=-150, max=100, step=5, value=self.args_plot["vmax"]
        )
        self.w_interpolation = ipywidgets.Dropdown(
            description="interpolation",
            options=[
                "antialiased",
                "none",
                "nearest",
                "bilinear",
                "bicubic",
                "spline16",
                "spline36",
                "hanning",
                "hamming",
                "hermite",
                "kaiser",
                "quadric",
                "catrom",
                "gaussian",
                "bessel",
                "mitchell",
                "sinc",
                "lanczos",
                "blackman",
            ],
            value=self.args_plot["interpolation"],
        )
        
        self.output = ipywidgets.Output()
        
        # observers for view changers
        for w in [self.w_vmin, self.w_vmax, self.w_interpolation]:
            w.observe(self.update_view, names=["value"])

        self.box_buttons = ipywidgets.HBox([
                self.update_button, 
                self.clear_button,
        ])
        self.box_sliders = ipywidgets.HBox([
                self.w_vmin, 
                self.w_vmax,
                self.w_interpolation
        ])
        

        if show:
            self.show()

        self.show_background_echogram()

    def show(self):        
        if self.display_progress:
            self.layout = ipywidgets.VBox([
                ipywidgets.HBox(children=[self.fig.canvas]),
                ipywidgets.HBox([self.progress]),
                self.box_sliders, 
                self.box_buttons, 
                self.output
            ])
        else:
            self.layout = ipywidgets.VBox([
                ipywidgets.HBox(children=[self.fig.canvas]),
                self.box_sliders, 
                self.box_buttons, 
                self.output
            ])
        display(self.layout)
    
    def init_ax(self, adapt_axis_names=True):
        with self.output:
            if adapt_axis_names:
                self.x_axis_name = self.echogramdata[-1].x_axis_name
                self.y_axis_name = self.echogramdata[-1].y_axis_name
                
            for i,ax in enumerate(self.axes):
                ax.clear()
                ax.set_title(self.names[i])
                self.mapables = []
    
    
                ax.set_xlabel(self.x_axis_name)
                ax.set_ylabel(self.y_axis_name)

            if self.x_axis_name == 'Date time':
                theping.pingprocessing.core.set_ax_timeformat(self.axes[-1])
    
    def show_background_echogram(self):
        with self.output:
            self.init_ax()
            
            self.images_background, self.extents_background = [],[]
            self.high_res_images, self.high_res_extents = [],[]
            self.layer_images, self.layer_extents = [],[]
            for i,echogram in enumerate(self.echogramdata):
            
                self.progress.set_description(f'Updating echogram [{i},{len(self.echogramdata)}]')
                
                if len(echogram.layers.keys()) == 0 and echogram.main_layer is None:
                    im,ex = echogram.build_image(progress=self.progress)   
                    self.images_background.append(im)
                    self.extents_background.append(ex)
                else:
                    im, im_layer, ex = echogram.build_image_and_layer_image(progress=self.progress)
                    self.layer_images.append(im_layer)
                    self.layer_extents.append(ex)
                    self.images_background.append(im)
                    self.extents_background.append(ex)
                
            self.update_view(reset=True)
            self.progress.set_description('Idle')

    def clear_output(self,event=0):
        with self.output:
            self.output.clear_output()
            
    def show_background_zoom(self, event = 0):
        with self.output:
            self.high_res_images, self.high_res_extents = [],[]
            self.layer_images, self.layer_extents = [],[]

            #check x/y axis
            for i,echogram in enumerate(self.echogramdata):
                self.progress.set_description('Updating echogram')
                if echogram.x_axis_name != self.x_axis_name or echogram.y_axis_name != self.y_axis_name:
                    self.show_background_echogram()
                    break
                    
            for i,echogram in enumerate(self.echogramdata):
                self.progress.set_description(f'Updating echogram [{i},{len(self.echogramdata)}]')
                
                xmin,xmax = self.axes[i].get_xlim()
                ymin,ymax = sorted(self.axes[i].get_ylim())
                x_kwargs = echogram.get_x_kwargs()
                y_kwargs = echogram.get_y_kwargs()
        
                match self.x_axis_name:
                    case 'Date time':
                        tmin,tmax = mdates.num2date([xmin, xmax])
                        x_kwargs['min_ping_time'] = tmin
                        x_kwargs['max_ping_time'] = tmax
                        echogram.set_x_axis_date_time(**x_kwargs)
                    case 'Ping number':
                        x_kwargs['min_ping_nr'] = xmin
                        x_kwargs['max_ping_nr'] = xmax
                        echogram.set_x_axis_ping_nr(**x_kwargs)
                    case 'Ping time':
                        x_kwargs['min_timestamp'] = xmin
                        x_kwargs['max_timestamp'] = xmax
                        echogram.set_x_axis_ping_time(**x_kwargs)
                    case _:
                        raise RuntimeError(f"ERROR: unknown x axis name '{self.x_axis_name}'")
                
                match self.y_axis_name:
                    case 'Depth (m)':
                        y_kwargs['min_depth'] = ymin
                        y_kwargs['max_depth'] = ymax
                        echogram.set_y_axis_depth(**y_kwargs)
                    case 'Range (m)':
                        y_kwargs['min_range'] = ymin
                        y_kwargs['max_range'] = ymax
                        echogram.set_y_axis_range(**y_kwargs)
                    case 'Sample number':
                        y_kwargs['min_sample_nr'] = ymin
                        y_kwargs['max_sample_nr'] = ymax
                        echogram.set_y_axis_sample_nr(**y_kwargs)
                    case 'Y indice':
                        y_kwargs['min_sample_nr'] = ymin
                        y_kwargs['max_sample_nr'] = ymax
                        echogram.set_y_axis_y_indice(**y_kwargs)
                    case _:
                        raise RuntimeError(f"ERROR: unknown y axis name '{self.y_axis_name}'")
                
                if len(echogram.layers.keys()) == 0 and echogram.main_layer is None:
                    im,ex = echogram.build_image(progress=self.progress)
                    self.high_res_images.append(im)
                    self.high_res_extents.append(ex)
                else:
                    im,im_layer,ex = echogram.build_image_and_layer_image(progress=self.progress)

                    self.high_res_images.append(im)
                    self.high_res_extents.append(ex)
                    self.layer_images.append(im_layer)
                    self.layer_extents.append(ex)
        self.update_view()
        
        self.progress.description = 'Idle'

    def invert_y_axis(self):
        with self.output:

            for ax in self.axes:
                ax.invert_yaxis()
            self.fig.canvas.draw_idle()

    def get_args_plot(self, axis_nr, layer=False):
        # detect changes in view settings

        args_plot = {
            "vmin": self.w_vmin.value + self.voffsets[axis_nr],
            "vmax": self.w_vmax.value + self.voffsets[axis_nr],
            "interpolation": self.w_interpolation.value,
            "cmap": self.cmap if not layer else self.cmap_layer,
            }

        if layer:
            self.args_plot_layer.update(args_plot)
            return self.args_plot_layer
        else:
            self.args_plot.update(args_plot)
            return self.args_plot


    def update_view(self, w=None, reset=False):
        with self.output:
                
            try:
                self.xlim = self.axes[-1].get_xlim()
                self.ylim = self.axes[-1].get_ylim()

                self.init_ax(reset)
                minx,maxx,miny,maxy = np.nan,np.nan,np.nan,np.nan
                
                for i,ax in enumerate(self.axes):
                    #zorder=1
                    self.mapables.append(ax.imshow(
                        self.images_background[i].transpose(), 
                        extent=self.extents_background[i], 
                        #zorder=zorder,  
                        **self.get_args_plot(i)))

                    if reset:
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        minx = np.nanmin([xlim[0],minx])
                        maxx = np.nanmax([xlim[1],maxx])
                        miny = np.nanmin([ylim[1],miny])
                        maxy = np.nanmax([ylim[0],maxy])
                    
                    if len(self.high_res_images) > i:
                        #zorder+=1
                        self.mapables.append(
                            ax.imshow(self.high_res_images[i].transpose(), 
                                        extent=self.high_res_extents[i], 
                                        #zorder=zorder, 
                                        **self.get_args_plot(i)))

                    if len(self.layer_images) > i:
                        #zorder+=1
                        self.mapables.append(
                            ax.imshow(self.layer_images[i].transpose(), 
                                        extent=self.layer_extents[i], 
                                        #zorder=zorder, 
                                        **self.get_args_plot(i,layer=True)))
                    

                    if self.colorbar[i] is None:
                        self.colorbar[i] = self.fig.colorbar(self.mapables[-1],ax=ax)
                    else:
                        self.colorbar[i].update_normal(self.mapables[-1])

                self.callback_view()

                if reset:
                    ax.set_xlim(minx,maxx)
                    ax.set_ylim(maxy,miny)
                else:
                    ax.set_xlim(self.xlim)
                    ax.set_ylim(self.ylim)
                    
                if len(self.mapables) > len(self.echogramdata)*3:
                    for m in self.mapables[len(self.echogramdata)*3-1:]:
                        m.remove()
                    self.mapables = self.mapables[:len(self.echogramdata)*3]

                self.fig.canvas.draw_idle()

            except Exception as e:
                raise (e)

    def callback_view(self):
        pass

    def click_echogram(self, event):
        with self.output:
            if self.pingviewer is None:
                return
            #global e
            #e = event
            with self.output:
                #print(event)
                if event.button == 2:
                    match self.x_axis_name:
                        case 'Date time':
                            t = mdates.num2date(event.xdata).timestamp()
                            for pn,ping in enumerate(self.pingviewer.imagebuilder.pings):
                                if isinstance(ping,dict):
                                    ping = next(iter(ping.values()))

                                if ping.get_timestamp() > t:
                                    if pn > 0:
                                        pn -= 1
                                    break
                        case 'Ping number':
                            pn = event.xdata
                        case 'Ping time':
                            t = event.xdata
                            for pn,ping in enumerate(self.pingviewer.imagebuilder.pings):
                                if isinstance(ping,dict):
                                    ping = next(iter(ping.values()))

                                if ping.get_timestamp() > t:
                                    if pn > 0:
                                        pn -= 1
                                    break
                        case _:
                            raise RuntimeError(f"ERROR: unknown x axis name '{self.x_axis_name}'")
                        
                    if pn < 0: 
                        pn = 0
                    if pn >= len(self.pingviewer.imagebuilder.pings):
                        pn = len(self.pingviewer.imagebuilder.pings)-1
                            
                    self.pingviewer.w_index.value = pn
            
            self.update_ping_line()
    
    def update_ping_line(self, event = 0):
        with self.output:
            if self.pingviewer is not None:
                with self.output:            
                    match self.x_axis_name:
                        case 'Ping number':
                            x = self.pingviewer.w_index.value
                        case 'Date time':
                            ping = self.pingviewer.imagebuilder.pings[self.pingviewer.w_index.value]
                            if isinstance(ping,dict):
                                ping = next(iter(ping.values()))                    
                            x = ping.get_datetime()
                        case 'Ping time':
                            ping = self.pingviewer.imagebuilder.pings[self.pingviewer.w_index.value]
                            if isinstance(ping,dict):
                                ping = next(iter(ping.values()))                        
                            x = ping.get_timestamp()
                        case _:
                            raise RuntimeError(f"ERROR: unknown x axis name '{self.x_axis_name}'")
                                
                    for i,ax in enumerate(self.axes):
                        try:
                            if self.pingline[i] is not None:
                                self.pingline[i].remove()
                        except:
                            pass
                        self.pingline[i] = ax.axvline(x,c='black',linestyle='dashed')
                
    def disconnect_pingviewer(self):
        with self.output:
            if 'on_click' in self.fig_events.keys():
                self.fig.canvas.mpl_disconnect(self.fig_events['on_click'])

            self.box_buttons = ipywidgets.HBox([
                    self.update_button, 
                    self.clear_button,
            ])
            children = list(self.layout.children)
            children[3] = self.box_buttons
            self.layout.children = children

            self.pingviewer = None

    def connect_pingviewer(self,pingviewer):   
        with self.output:       
            self.disconnect_pingviewer()
            
            self.pingviewer = pingviewer

            self.update_ping_line_button = ipywidgets.Button(description="update pingline")
            self.update_ping_line_button.on_click(self.update_ping_line)
            
            self.box_buttons = ipywidgets.HBox([
                    self.update_button, 
                    self.clear_button,
                    self.update_ping_line_button, 
            ])

            children = list(self.layout.children)
            children[3] = self.box_buttons
            self.layout.children = children
                
            self.fig_events['on_click'] = self.fig.canvas.mpl_connect("button_press_event", self.click_echogram)
        