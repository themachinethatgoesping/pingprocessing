from time import time
import types
import numpy as np

import ipywidgets
import matplotlib.pyplot as plt
from IPython.display import display

from themachinethatgoesping import echosounders

import themachinethatgoesping.pingprocessing.watercolumn.image as mi
import themachinethatgoesping.pingprocessing.watercolumn.helper.make_image_helper as mi_hlp

from themachinethatgoesping.pingprocessing.widgets import TqdmWidget

class WCIViewer:
    def __init__(self, pings, horizontal_pixels=1024, name="WCI", figure=None, progress=None, show=True, cmap="YlGnBu_r", **kwargs):

        self.args_imagebuilder = {
            "horizontal_pixels": horizontal_pixels,
            "stack_linear": True,
            "hmin": None,
            "hmax": None,
            "vmin": None,
            "vmax": None,
            "from_bottom_xyz": False,
            "wci_value": "sv/av/pv/rv",
            "wci_render": "linear",
            "ping_sample_selector": echosounders.pingtools.PingSampleSelector(),
            "apply_pss_to_bottom": False,
            "mp_cores": 1,
        }

        if len(pings) < 1:
            raise ValueError("No pings provided")

        self.mapable = None
        self.wci = None
        self.extent = None
        self.wci_value = None # if set, will replace value in self.w_wci_value

        if isinstance(cmap, str):
            self.cmap = plt.get_cmap(cmap)
        else:
            self.cmap = cmap

        self.args_imagebuilder.update((k, kwargs[k]) for k in self.args_imagebuilder.keys() & kwargs.keys())
        for k in self.args_imagebuilder.keys():
            if k in kwargs.keys():
                kwargs.pop(k)

        self.args_plot = {"cmap": self.cmap, "aspect": "equal", "vmin": -90, "vmax": -25, "interpolation": "nearest"}
        self.args_plot.update(kwargs)

        self.output = ipywidgets.Output()

        # setup figure
        if figure is None:
            plt.ioff()
            self.fig = plt.figure(name, clear=True)
            self.ax = self.fig.subplots()

            self.fig.set_tight_layout(True)
            self.fig.set_size_inches(10, 4)
            plt.ion()
        else:
            self.fig = figure
            if len(self.fig.axes) > 0:
                self.ax = self.fig.axes[0]
            else:
                self.ax = self.fig.subplots()

        # setup progressbar and buttons
        if progress is None:
            self.progress = TqdmWidget()
            self.display_progress = True
        else:
            self.progress = progress
            self.display_progress = False

        self.w_fix_xy = ipywidgets.Button(description="fix x/y")
        self.w_unfix_xy = ipywidgets.Button(description="unfix x/y")
        self.w_proctime = ipywidgets.Text(description="proc time")
        self.w_procrate = ipywidgets.Text(description="proc rate")

        self.w_fix_xy.on_click(self.fix_xy)
        self.w_unfix_xy.on_click(self.unfix_xy)

        if self.display_progress:
            box_progress = ipywidgets.VBox([
                ipywidgets.HBox([self.progress]), 
                ipywidgets.HBox([self.w_fix_xy, self.w_unfix_xy, self.w_proctime, self.w_procrate])
            ])
        else:
            box_progress = ipywidgets.HBox([self.w_fix_xy, self.w_unfix_xy, self.w_time])

        # setup image builder
        self.imagebuilder = mi.ImageBuilder(
            pings, horizontal_pixels=horizontal_pixels, progress=self.progress
        )

        # setup widgets
        # basic display control
        self.w_index = ipywidgets.IntSlider(
            layout=ipywidgets.Layout(width="50%"), description="ping nr", min=0, max=len(pings) - 1, step=1, value=0
        )

        self.w_date = ipywidgets.Text(layout=ipywidgets.Layout(width="10%"))
        self.w_time = ipywidgets.Text(layout=ipywidgets.Layout(width="10%"))

        self.w_stack = ipywidgets.IntText(value=1, description="stack:", layout=ipywidgets.Layout(width="15%"))
        self.w_stack_step = ipywidgets.IntText(
            value=1, description="stack step:", layout=ipywidgets.Layout(width="15%")
        )
        self.w_mp_cores = ipywidgets.IntText(value=1, description="mp_cores:", layout=ipywidgets.Layout(width="15%"))

        box_index = ipywidgets.HBox(
            [self.w_index, self.w_date, self.w_time, self.w_stack, self.w_stack_step, self.w_mp_cores]
        )

        # basic plotting setup
        self.w_vmin = ipywidgets.FloatSlider(
            description="vmin", min=-150, max=100, step=5, value=self.args_plot["vmin"]
        )
        self.w_vmax = ipywidgets.FloatSlider(
            description="vmax", min=-150, max=100, step=5, value=self.args_plot["vmax"]
        )
        self.w_aspect = ipywidgets.Dropdown(
            description="aspect", options=["auto", "equal"], value=self.args_plot["aspect"]
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

        box_plot = ipywidgets.HBox([self.w_vmin, self.w_vmax, self.w_aspect, self.w_interpolation])

        # self.w_from_bottom = ipywidgets.Checkbox(description="from bottom", value=False)
        self.w_horizontal_pixels = ipywidgets.IntSlider(
            description="horizontal pixels", min=2, max=2048, step=1, value=self.args_imagebuilder["horizontal_pixels"]
        )
        self.w_stack_linear = ipywidgets.Checkbox(
            description="stack_linear", value=self.args_imagebuilder["stack_linear"]
        )
        self.w_wci_value = ipywidgets.Dropdown(
            description="wci value",
            options=[
                "sv/av/pv/rv",
                "sv/av/pv",
                "sv/av",
                "sp/ap/pp/rp",
                "sp/ap/pp",
                "sp/ap",
                "power/amp",
                "av",
                "ap",
                "amp",
                "sv",
                "sp",
                "pv",
                "pp",
                "rv",
                "rp",
                "power",
            ],
            value=self.args_imagebuilder["wci_value"],
        )
        self.w_wci_render = ipywidgets.Dropdown(
            description="wci render", options=["linear", "beamsample"], value=self.args_imagebuilder["wci_render"]
        )

        box_process = ipywidgets.HBox(
            [self.w_stack_linear, self.w_wci_value, self.w_wci_render, self.w_horizontal_pixels]
        )

        layout = [self.fig.canvas]
        layout.append(box_progress)
        layout.append(box_process)
        layout.append(box_plot)
        layout.append(box_index)

        layout.append(self.output)
        self.layout = ipywidgets.VBox(layout)

        # observers for data changers
        for w in [
            self.w_index,
            self.w_stack,
            self.w_stack_step,
            self.w_mp_cores,
            self.w_stack_linear,
            self.w_wci_value,
            self.w_wci_render,
            self.w_horizontal_pixels,
        ]:
            w.observe(self.update_data, names=["value"])

        # observers for view changers
        for w in [self.w_vmin, self.w_vmax, self.w_aspect, self.w_interpolation]:
            w.observe(self.update_view, names=["value"])

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.colorbar = None

        self.update_data(0)

        if show:
            display(self.layout)

    def set_ping_sample_selector(self, ping_sample_selector, apply_pss_to_bottom=False):
        with self.output:
            self.args_imagebuilder["ping_sample_selector"] = ping_sample_selector
            self.args_imagebuilder["apply_pss_to_bottom"] = apply_pss_to_bottom
            self.update_data(0)

    def fix_xy(self, w):
        with self.output:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.args_imagebuilder["hmin"] = xlim[0]
            self.args_imagebuilder["hmax"] = xlim[1]
            self.args_imagebuilder["vmin"] = ylim[1]
            self.args_imagebuilder["vmax"] = ylim[0]

            self.update_data(0)

    def unfix_xy(self, w):
        with self.output:
            self.args_imagebuilder["hmin"] = None
            self.args_imagebuilder["hmax"] = None
            self.args_imagebuilder["vmin"] = None
            self.args_imagebuilder["vmax"] = None

            self.update_data(0)

    # @self.output.capture()
    def update_data(self, w=None):
        with self.output:
            self.output.clear_output()
            t0 = time()

            if self.wci_value is None:
                self.args_imagebuilder["wci_value"] = self.w_wci_value.value
            else:
                self.args_imagebuilder["wci_value"] = self.wci_value
            self.args_imagebuilder["wci_render"] = self.w_wci_render.value
            self.args_imagebuilder["linear_mean"] = self.w_stack_linear.value
            self.args_imagebuilder["horizontal_pixels"] = self.w_horizontal_pixels.value
            self.args_imagebuilder["mp_cores"] = self.w_mp_cores.value
            self.imagebuilder.update_args(**self.args_imagebuilder)

            try:
                self.wci, self.extent = self.imagebuilder.build(
                    index=self.w_index.value, stack=self.w_stack.value, stack_step=self.w_stack_step.value
                )
                self.callback_data()

                # w_text_execution_time.value = str(round(time()-t,3))

            except Exception as e:
                with self.output:
                    raise (e)

            t1 = time()
            self.update_view(w)
            t2 = time()
            ping = self.imagebuilder.pings[self.w_index.value]
            if not isinstance(ping, echosounders.filetemplates.I_Ping):
                ping = next(iter(ping.values()))

            self.w_date.value = ping.get_datetime().strftime("%Y-%m-%d")
            self.w_time.value = ping.get_datetime().strftime("%H:%M:%S")

            self.w_proctime.value = f"{round(t1-t0,3)} / {round(t2-t1,3)} / [{round(t2-t0,3)}] s"
            r1 = 1/(t1-t0) if t1-t0 > 0 else 0
            r2 = 1/(t2-t1) if t2-t1 > 0 else 0
            r3 = 1/(t2-t0) if t2-t0 > 0 else 0

            self.w_procrate.value = f"r1: {round(r1,1)} / r2: {round(r2,1)} / r3: [{round(r3,1)}] Hz"

    def save_background(self):
        with self.output:
            empty = np.empty(self.wci.transpose().shape)
            empty.fill(np.nan)
            self.mapable.set_data(empty)
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            # self.mapable.set_data(self.wci.transpose())

    def update_view(self, w=None):
        with self.output:
            # detect changes in view settings
            for n, v in [
                ("vmin", self.w_vmin.value),
                ("vmax", self.w_vmax.value),
                ("interpolation", self.w_interpolation.value),
                ("aspect", self.w_aspect.value),
                ("cmap", self.cmap),
            ]:
                if self.args_plot[n] != v:
                    self.args_plot[n] = v
                    self.mapable = None

            try:
                self.w_fix_xy.button_style = "warning"
                if self.mapable is not None:
                    if self.mapable.get_array().shape == self.wci.transpose().shape:
                        if self.mapable.get_extent() == list(self.extent):
                            if self.first_blit:
                                self.save_background()
                                self.first_blit = False

                            self.fig.canvas.restore_region(self.background)
                            self.w_fix_xy.button_style = "success"
                            self.mapable.set_data(self.wci.transpose())
                            # self.fig.canvas.draw()
                            self.ax.draw_artist(self.mapable)
                            self.fig.canvas.blit(self.fig.bbox)
                            self.fig.canvas.flush_events()
                            self.callback_view()
                            return

                self.ax.clear()
                self.first_blit = True

                # carch empty wci
                if len(self.wci) > 0:
                    self.mapable = self.ax.imshow(self.wci.transpose(), extent=self.extent, **self.args_plot, animated=True)

                    self.ax.set_xlim(self.xmin, self.xmax)
                    self.ax.set_ylim(self.ymax, self.ymin)

                    if self.colorbar is None:
                        self.colorbar = self.fig.colorbar(self.mapable)
                    else:
                        self.colorbar.update_normal(self.mapable)

                self.fig.canvas.draw()

                self.callback_view()

            except Exception as e:
                raise (e)

    def callback_view(self):
        pass

    def callback_data(self):
        pass
