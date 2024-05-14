import ipywidgets
from time import time

import matplotlib.pyplot as plt
from IPython.display import display

import themachinethatgoesping as Ping
import themachinethatgoesping.pingprocessing.watercolumn.image as mi
import themachinethatgoesping.pingprocessing.watercolumn.helper.make_image_helper as mi_hlp

class WCIViewer:
    def __init__(self, 
                 pings, 
                 horizontal_pixels=1024, 
                 name = 'WCI', 
                 figure=None, 
                 progress=None, 
                 show=True, 
                 **kwargs):
        
        self.args_imagebuilder = {
            "horizontal_pixels" : horizontal_pixels,
            "stack_linear" : True,
            "hmin" : None,
            "hmax" : None,
            "vmin" : None,
            "vmax" : None,
            "from_bottom_xyz" : False,
            "wci_value" : 'sv/av',
            "ping_sample_selector" : Ping.echosounders.pingtools.PingSampleSelector(),
            "mp_cores" : 1
            
        }
        self.args_imagebuilder.update((k, kwargs[k]) for k in self.args_imagebuilder.keys() & kwargs.keys())
        for k in self.args_imagebuilder.keys():
            if k in kwargs.keys():
                kwargs.pop(k)
            
        self.args_plot = {            
                "cmap" : 'YlGnBu_r',
                "aspect" : 'equal', 
                "vmin" : -90, 
                "vmax" : -50,
                "interpolation" : "nearest"
        }
        self.args_plot.update(kwargs)

        self.output = ipywidgets.Output()

        #setup figure
        if figure is None:
            plt.ioff()
            self.fig = plt.figure(name, clear=True)
            self.ax = self.fig.subplots()
                        
            self.fig.set_tight_layout(True)
            self.fig.set_size_inches(10,4)
            plt.ion()
        else:
            self.fig = figure
            if len(self.fig.axes) > 0:
                self.ax = self.fig.axes[0]
            else:
                self.ax = self.fig.subplots()

        #setup progressbar
        if progress is None:
            self.progress = Ping.pingprocessing.widgets.TqdmWidget()
            self.display_progress = True
        else:
            self.progress = progress
            self.display_progress = False

        #setup image builder
        self.imagebuilder = Ping.pingprocessing.watercolumn.image.ImageBuilder(
            pings, 
            horizontal_pixels=horizontal_pixels,
            progress=self.progress)

        #setup widgets
        # basic display control
        self.w_index = ipywidgets.IntSlider(
            layout=ipywidgets.Layout(width='50%'),
            description = 'ping nr',
            min=0, 
            max=len(pings)-1, 
            step=1, 
            value = 0)
        
        self.w_stack = ipywidgets.IntText(value=1, description='stack:',layout=ipywidgets.Layout(width='15%'))
        self.w_stack_step = ipywidgets.IntText(value=1, description='stack step:',layout=ipywidgets.Layout(width='15%'))
        self.w_mp_cores = ipywidgets.IntText(value=1, description='mp_cores:',layout=ipywidgets.Layout(width='15%'))
        
        box_index = ipywidgets.HBox([self.w_index, self.w_stack, self.w_stack_step, self.w_mp_cores])

        # basic plotting setup
        self.w_vmin = ipywidgets.FloatSlider(description='vmin', min=-150, max=100, step=5, value = self.args_plot['vmin'])
        self.w_vmax = ipywidgets.FloatSlider(description='vmax', min=-150, max=100, step=5, value = self.args_plot['vmax'])
        self.w_aspect = ipywidgets.Dropdown(description="stack_linear", 
                                            options=['auto', 'equal'], 
                                            value=self.args_plot["aspect"])
        self.w_interpolation = ipywidgets.Dropdown(description="interpolation", 
                                               options=['antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'], 
                                               value=self.args_plot["interpolation"])
        
        box_plot = ipywidgets.HBox([self.w_vmin, self.w_vmax, self.w_aspect, self.w_interpolation])

        #self.w_from_bottom = ipywidgets.Checkbox(description="from bottom", value=False)
        self.w_horizontal_pixels = ipywidgets.IntSlider(description="horizontal pixels", min=2, max=2048, step=1, value=self.args_imagebuilder["horizontal_pixels"])
        self.w_stack_linear = ipywidgets.Checkbox(description="stack_linear", 
                                                  value=self.args_imagebuilder["stack_linear"])
        self.w_wci_value = ipywidgets.Dropdown(description="wci value", 
                                               options=['sv/av', 'av', 'sv', 'amp'], 
                                               value=self.args_imagebuilder["wci_value"])
        
        box_process = ipywidgets.HBox([self.w_stack_linear, self.w_wci_value, self.w_horizontal_pixels])


        layout = [self.fig.canvas]
        if self.display_progress: layout.append(self.progress)
        layout.append(box_process)
        layout.append(box_plot)
        layout.append(box_index)
            
        layout.append(self.output)
        self.layout = ipywidgets.VBox(layout)

        # observers for data changers
        for w in [self.w_index, self.w_stack, self.w_stack_step, self.w_mp_cores, self.w_stack_linear, self.w_wci_value, self.w_horizontal_pixels]:
            w.observe(self.update_data, names=['value'])
            
        # observers for view changers
        for w in [self.w_vmin,self.w_vmax,self.w_aspect,self.w_interpolation]:
            w.observe(self.update_view, names=['value'])

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.colorbar = None

        self.update_data(0)
        
        if show:
            display(self.layout)
        

    #@self.output.capture()
    def update_data(self,w):
        self.output.clear_output()
        t = time()

        self.args_imagebuilder['wci_value'] = self.w_wci_value.value
        self.args_imagebuilder['linear_mean'] = self.w_stack_linear.value
        self.args_imagebuilder['horizontal_pixels'] = self.w_horizontal_pixels.value
        self.args_imagebuilder['mp_cores'] = self.w_mp_cores.value
        self.imagebuilder.update_args(**self.args_imagebuilder)
        
        try:
            self.wci, self.extent = self.imagebuilder.build(
                index = self.w_index.value, 
                stack = self.w_stack.value, 
                stack_step = self.w_stack_step.value)        
                                   
            #w_text_execution_time.value = str(round(time()-t,3))
    
        except Exception as e:
            with self.output:
                raise(e)

        self.update_view(w)

            
    #@self.output.capture()
    def update_view(self,w):      
        self.args_plot['vmin'] = self.w_vmin.value
        self.args_plot['vmax'] = self.w_vmax.value
        self.args_plot['interpolation'] = self.w_interpolation.value
        self.args_plot['aspect'] = self.w_aspect.value
        
        try:               
            self.ax.clear()
                
            self.mapable = self.ax.imshow(
                self.wci.transpose(),
                extent = self.extent, 
                **self.args_plot
            )

            self.ax.set_xlim(self.xmin, self.xmax)
            self.ax.set_ylim(self.ymax, self.ymin)
                                   
            #w_text_num_active.value = str(int(w_text_num_active.value) -1)
            #w_text_execution_time.value = str(round(time()-t,3))
            if self.colorbar is None:
                self.colorbar = self.fig.colorbar(self.mapable)
            else:
                self.colorbar.update_normal(self.mapable)
            self.fig.canvas.draw()
            
        except Exception as e:
            with self.output:
                raise (e)
