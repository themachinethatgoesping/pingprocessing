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
    def __init__(self, echogramdata, name="Echogram", names = None, figure=None, progress=None, show=True, cmap="YlGnBu_r", **kwargs):
        self.mapables = []
        if not isinstance(echogramdata, list):
            echogramdata = [echogramdata]
            
        self.echogramdata = echogramdata
        self.colorbar = [None for _ in self.echogramdata]
        self.pingline = [None for _ in self.echogramdata]
        self.fig_events = {}
        self.pingviewer = None
        self.echogram_axes = []

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
            
        # plot arguments
        self.args_plot = {
            "cmap": self.cmap,
            "aspect": "auto", 
            "vmin": -100, 
            "vmax": -25, 
            "interpolation": "nearest"
        }
        self.args_plot.update(kwargs)
        
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
        self.loop = asyncio.get_event_loop()
        self.task = self.loop.create_task(self.event_loop())

        
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
            display(
                ipywidgets.HBox(children=[self.fig.canvas]),
                ipywidgets.HBox([self.progress]),
                self.box_sliders, 
                self.box_buttons, 
                self.output
            )
        else:
            display(
                ipywidgets.HBox(children=[self.fig.canvas]),
                self.box_sliders, 
                self.box_buttons, 
                self.output
            )
    
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
            for i,echogram in enumerate(self.echogramdata):
            
                self.progress.set_description(f'Updating echogram [{i},{len(self.echogramdata)}]')
                
                im,ex = echogram.build_image(progress=self.progress)   
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
                x_kwargs = echogram.x_kwargs
                y_kwargs = echogram.y_kwargs
        
                match self.x_axis_name:
                    case 'Date time':
                        tmin,tmax = mdates.num2date(xmin).timestamp(),mdates.num2date(xmax).timestamp()
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
                    case _:
                        raise RuntimeError(f"ERROR: unknown y axis name '{self.y_axis_name}'")
                
                im,ex = echogram.build_image(progress=self.progress)
                self.high_res_images.append(im)
                self.high_res_extents.append(ex)
        self.update_view()
        
        self.progress.description = 'Idle'

    def invert_y_axis(self):
        with self.output:
            for ax in self.axes:
                ax.invert_yaxis()
            self.fig.canvas.draw()

    def update_view(self, w=None, reset=False):
        with self.output:
            # detect changes in view settings
            for n, v in [
                ("vmin", self.w_vmin.value),
                ("vmax", self.w_vmax.value),
                ("interpolation", self.w_interpolation.value),
                ("cmap", self.cmap),
            ]:
                if self.args_plot[n] != v:
                    self.args_plot[n] = v

                
            try:
                self.xlim = self.axes[-1].get_xlim()
                self.ylim = self.axes[-1].get_ylim()

                self.init_ax(reset)
                minx,maxx,miny,maxy = np.nan,np.nan,np.nan,np.nan
                
                for i,ax in enumerate(self.axes):
                    zorder=1
                    self.mapables.append(ax.imshow(
                        self.images_background[i].transpose(), 
                        extent=self.extents_background[i], 
                        zorder=zorder,  
                        **self.args_plot))
    
                    if reset:
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        minx = np.nanmin([xlim[0],minx])
                        maxx = np.nanmax([xlim[1],maxx])
                        miny = np.nanmin([ylim[1],miny])
                        maxy = np.nanmax([ylim[0],maxy])
                    
                    if len(self.high_res_images) > i:
                        zorder+=1
                        self.mapables.append(
                            ax.imshow(self.high_res_images[i].transpose(), 
                                           extent=self.high_res_extents[i], 
                                           zorder=zorder, 
                                           **self.args_plot))
                    
    
                    if self.colorbar[i] is None:
                        self.colorbar[i] = self.fig.colorbar(self.mapables[0],ax=ax)
                    else:
                        self.colorbar[i].update_normal(self.mapables[0])

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
    
                self.fig.canvas.draw()

            except Exception as e:
                raise (e)

    def click_echogram(self, event):
        if self.pingviewer is None:
            return
        #global e
        #e = event
        with self.output:
            #print(event)
            if event.button == 1:
                match self.x_axis_name:
                    case 'Date time':
                        t = mdates.num2date(event.xdata).timestamp()
                        for pn,ping in enumerate(self.pingviewer.imagebuilder.pings):
                            if ping.get_timestamp() > t:
                                if pn > 0:
                                    pn -= 1
                                break
                    case 'Ping number':
                        pn = event.xdata
                    case 'Ping time':
                        t = event.xdata
                        for pn,ping in enumerate(self.pingviewer.imagebuilder.pings):
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

    async def event_loop(self):
        while True:
            self.update_ping_line()
            await asyncio.sleep(0.25)

    
    def update_ping_line(self):
        if self.pingviewer is not None:
            with self.output:
                for i,ax in enumerate(self.axes):
                    try:
                        if self.pingline[i] is not None:
                            self.pingline[i].remove()
                    except:
                        pass
    
                    match self.x_axis_name:
                        case 'Date time':
                            self.pingline[i] = ax.axvline(self.pingviewer.imagebuilder.pings[self.pingviewer.w_index.value].get_datetime(),c='black',linestyle='dashed')
                        case 'Ping number':
                            self.pingline[i] = ax.axvline(self.pingviewer.w_index.value,c='black',linestyle='dashed')
                        case 'Ping time':
                            self.pingline[i] = ax.axvline(self.pingviewer.imagebuilder.pings[self.pingviewer.w_index.value].get_timestamp(),c='black',linestyle='dashed')
                        case _:
                            raise RuntimeError(f"ERROR: unknown x axis name '{self.x_axis_name}'")
                

    def connect_pingviewer(self,pingviewer):
        self.pingviewer = pingviewer

        if 'on_click' in self.fig_events.keys():
            self.fig.canvas.mpl_disconnect(self.fig_events['on_click'])
            
        self.fig_events['on_click'] = self.fig.canvas.mpl_connect("button_press_event", self.click_echogram)
      