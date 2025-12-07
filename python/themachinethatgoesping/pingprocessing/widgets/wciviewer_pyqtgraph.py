"""PyQtGraph-based Water Column Image (WCI) viewer."""
from __future__ import annotations

from time import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ipywidgets
import numpy as np
import pyqtgraph as pg
from IPython.display import display
from pyqtgraph.jupyter import GraphicsLayoutWidget
from pyqtgraph.Qt import QtCore, QtWidgets

from themachinethatgoesping import echosounders
import themachinethatgoesping.pingprocessing.watercolumn.image as mi
from themachinethatgoesping.pingprocessing.widgets import TqdmWidget

from . import pyqtgraph_helpers as pgh

WCI_VALUE_CHOICES = [
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
    "sv_vs_av",
    "sp_vs_ap"
]

class WCIViewerPyQtGraph:
    """Replacement for the Matplotlib-based WCI viewer using PyQtGraph."""

    def __init__(
        self,
        pings: Sequence[Any],
        horizontal_pixels: int = 1024,
        name: str = "WCI",
        figure: Optional[Any] = None,  # retained for API compatibility, unused
        progress: Optional[Any] = None,
        show: bool = True,
        cmap: str = "YlGnBu_r",
        widget_height_px: Optional[int] = None,
        widget_width_px: int = 900,
        **kwargs: Any,
    ) -> None:
        if len(pings) < 1:
            raise ValueError("No pings provided")

        pgh.ensure_qapp()
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.args_imagebuilder: Dict[str, Any] = {
            "horizontal_pixels": horizontal_pixels,
            "linear_mean": True,
            "hmin": None,
            "hmax": None,
            "vmin": None,
            "vmax": None,
            "wci_value": "sv/av/pv/rv",
            "wci_render": "linear",
            "ping_sample_selector": echosounders.pingtools.PingSampleSelector(),
            "apply_pss_to_bottom": False,
            "mp_cores": 1,
        }
        self.args_plot: Dict[str, Any] = {
            "vmin": kwargs.pop("vmin", -90),
            "vmax": kwargs.pop("vmax", -25),
        }

        # allow callers to override builder args via kwargs
        for key in list(kwargs.keys()):
            if key in self.args_imagebuilder:
                self.args_imagebuilder[key] = kwargs.pop(key)
        self.args_imagebuilder.update(kwargs)

        self.pings = pings
        self.name = name
        self.name = name
        self.progress = progress or TqdmWidget()
        self.display_progress = progress is None
        self.wci: Optional[np.ndarray] = None
        self.extent: Optional[Tuple[float, float, float, float]] = None
        self.hover_label = ipywidgets.HTML(value="&nbsp;")
        self.output = ipywidgets.Output()

        self.cmap = pgh.resolve_colormap(cmap)
        self.widget_height_px = widget_height_px or 400
        self.widget_width_px = widget_width_px

        self.graphics = GraphicsLayoutWidget(
            css_width=f"{widget_width_px}px",
            css_height=f"{self.widget_height_px}px",
        )
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, self.widget_height_px)
        if hasattr(self.graphics, "gfxView"):
            self.graphics.gfxView.setBackground("w")
        self.plot: pg.PlotItem = self.graphics.addPlot(row=0, col=0)
        self.plot.setLabels(left="Depth (m)", bottom="Horizontal distance")
        self.plot.getViewBox().invertY(True)
        self.plot.getViewBox().setBackgroundColor("w")
        self.plot.getViewBox().setAspectLocked(True, ratio=1)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        self.colorbar: Optional[pg.ColorBarItem] = None
        try:
            self.colorbar = pg.ColorBarItem(label="(dB)", values=(self.args_plot["vmin"], self.args_plot["vmax"]))
            self.colorbar.setImageItem(self.image_item, insert_in=self.plot)
            if hasattr(self.colorbar, "setColorMap"):
                self.colorbar.setColorMap(self.cmap)
        except AttributeError:
            self.colorbar = None

        self._connect_scene_events()

        self._init_controls()
        self.imagebuilder = mi.ImageBuilder(
            pings,
            horizontal_pixels=self.args_imagebuilder["horizontal_pixels"],
            progress=self.progress,
        )
        self.wci_value_override: Optional[str] = None
        self.first_draw = True

        if show:
            display(self.layout)
        self.update_data()

    def _init_controls(self) -> None:
        self.w_fix_xy = ipywidgets.Button(description="fix x/y")
        self.w_unfix_xy = ipywidgets.Button(description="unfix x/y")
        self.w_proctime = ipywidgets.Text(description="proc time")
        self.w_procrate = ipywidgets.Text(description="proc rate")
        self.w_fix_xy.on_click(self.fix_xy)
        self.w_unfix_xy.on_click(self.unfix_xy)

        self.w_index = ipywidgets.IntSlider(
            layout=ipywidgets.Layout(width="50%"),
            description="ping nr",
            min=0,
            max=len(self.pings) - 1,
            step=1,
            value=0,
        )
        self.w_date = ipywidgets.Text(layout=ipywidgets.Layout(width="10%"))
        self.w_time = ipywidgets.Text(layout=ipywidgets.Layout(width="10%"))
        self.w_stack = ipywidgets.IntText(value=1, description="stack", layout=ipywidgets.Layout(width="15%"))
        self.w_stack_step = ipywidgets.IntText(value=1, description="stack step", layout=ipywidgets.Layout(width="15%"))
        self.w_mp_cores = ipywidgets.IntText(value=self.args_imagebuilder["mp_cores"], description="mp_cores", layout=ipywidgets.Layout(width="15%"))

        self.w_vmin = ipywidgets.FloatSlider(description="vmin", min=-150, max=100, step=0.5, value=self.args_plot["vmin"])
        self.w_vmax = ipywidgets.FloatSlider(description="vmax", min=-150, max=100, step=0.5, value=self.args_plot["vmax"])

        self.w_stack_linear = ipywidgets.Checkbox(
            description="stack_linear",
            value=self.args_imagebuilder["linear_mean"],
        )
        self.w_wci_value = ipywidgets.Dropdown(
            description="wci value",
            options=WCI_VALUE_CHOICES,
            value=self.args_imagebuilder["wci_value"],
        )
        self.w_wci_render = ipywidgets.Dropdown(
            description="wci render",
            options=["linear", "beamsample"],
            value=self.args_imagebuilder["wci_render"],
        )
        self.w_horizontal_pixels = ipywidgets.IntSlider(
            description="horizontal px",
            min=2,
            max=2048,
            step=1,
            value=self.args_imagebuilder["horizontal_pixels"],
        )

        box_progress = ipywidgets.HBox([self.w_fix_xy, self.w_unfix_xy, self.w_proctime, self.w_procrate])
        if self.display_progress:
            box_progress = ipywidgets.VBox([ipywidgets.HBox([self.progress]), box_progress])

        box_plot = ipywidgets.HBox([self.w_vmin, self.w_vmax])
        box_process = ipywidgets.HBox([self.w_stack_linear, self.w_wci_value, self.w_wci_render, self.w_horizontal_pixels])
        box_index = ipywidgets.HBox([
            self.w_index,
            self.w_date,
            self.w_time,
            self.w_stack,
            self.w_stack_step,
            self.w_mp_cores,
        ])

        controls: List[ipywidgets.Widget] = [
            ipywidgets.HBox([self.graphics]),
            box_progress,
            box_process,
            box_plot,
            box_index,
            self.hover_label,
            self.output,
        ]
        self.layout = ipywidgets.VBox(controls)

        for widget in [
            self.w_index,
            self.w_stack,
            self.w_stack_step,
            self.w_mp_cores,
            self.w_stack_linear,
            self.w_wci_value,
            self.w_wci_render,
            self.w_horizontal_pixels,
        ]:
            widget.observe(self.update_data, names="value")
        for widget in [self.w_vmin, self.w_vmax]:
            widget.observe(self.update_view, names="value")

    def _connect_scene_events(self) -> None:
        gfx_view = getattr(self.graphics, "gfxView", None)
        scene = gfx_view.scene() if gfx_view is not None else None
        if scene is None:
            return
        scene.sigMouseMoved.connect(self._handle_scene_move)

    def fix_xy(self, _event: Any = None) -> None:
        view = self.plot.getViewBox()
        (xmin, xmax), (ymin, ymax) = view.viewRange()
        xmin, xmax = sorted((float(xmin), float(xmax)))
        ymin, ymax = sorted((float(ymin), float(ymax)))
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            # Nothing to lock; keep auto-scaling to avoid zero-size extents.
            return
        self.args_imagebuilder["hmin"] = xmin
        self.args_imagebuilder["hmax"] = xmax
        self.args_imagebuilder["vmin"] = ymin
        self.args_imagebuilder["vmax"] = ymax
        self.update_data()

    def unfix_xy(self, _event: Any = None) -> None:
        for key in ("hmin", "hmax", "vmin", "vmax"):
            self.args_imagebuilder[key] = None
        self.update_data()

    def update_data(self, _change: Any = None) -> None:
        with self.output:
            self.output.clear_output()
            t0 = time()
            self._sync_builder_args()
            try:
                self.wci, self.extent = self.imagebuilder.build(
                    index=self.w_index.value,
                    stack=self.w_stack.value,
                    stack_step=self.w_stack_step.value,
                )
            except Exception as error:  # pragma: no cover - bubble up to notebook
                raise error
            self._update_metadata()
            t1 = time()
            self.update_view()
            t2 = time()
            self._update_timing_fields(t0, t1, t2)

    def _sync_builder_args(self) -> None:
        self.args_imagebuilder["linear_mean"] = self.w_stack_linear.value
        self.args_imagebuilder["wci_value"] = self.wci_value_override or self.w_wci_value.value
        self.args_imagebuilder["wci_render"] = self.w_wci_render.value
        self.args_imagebuilder["horizontal_pixels"] = self.w_horizontal_pixels.value
        self.args_imagebuilder["mp_cores"] = self.w_mp_cores.value
        self.imagebuilder.update_args(**self.args_imagebuilder)

    def _update_metadata(self) -> None:
        ping = self.imagebuilder.pings[self.w_index.value]
        if isinstance(ping, dict):
            ping = next(iter(ping.values()))
        dt = ping.get_datetime()
        self.w_date.value = dt.strftime("%Y-%m-%d")
        self.w_time.value = dt.strftime("%H:%M:%S")

    def _update_timing_fields(self, t0: float, t1: float, t2: float) -> None:
        build_time = t1 - t0
        draw_time = t2 - t1
        total_time = t2 - t0
        self.w_proctime.value = f"{build_time:0.3f} / {draw_time:0.3f} / [{total_time:0.3f}] s"
        r1 = 1 / build_time if build_time > 0 else 0
        r2 = 1 / draw_time if draw_time > 0 else 0
        r3 = 1 / total_time if total_time > 0 else 0
        self.w_procrate.value = f"r1: {r1:0.1f} / r2: {r2:0.1f} / r3: [{r3:0.1f}] Hz"

    def update_view(self, _change: Any = None) -> None:
        with self.output:
            if self.wci is None or self.extent is None:
                self.image_item.hide()
                return
            array = self.wci.transpose()
            if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
                self.image_item.hide()
                self.hover_label.value = "&nbsp;"
                return
            self.image_item.setImage(array, autoLevels=False)
            img_width = self.image_item.width() or 0
            img_height = self.image_item.height() or 0
            if img_width == 0 or img_height == 0:
                self.image_item.hide()
                self.hover_label.value = "&nbsp;"
                return
            x0, x1, y0, y1 = self.extent
            vb = self.plot.getViewBox()
            if vb.yInverted():
                y0, y1 = y1, y0
            width = x1 - x0
            height = y1 - y0
            if width == 0 or height == 0:
                # Degenerate extent (typically no valid data). Skip drawing to avoid divide-by-zero.
                self.image_item.hide()
                self.hover_label.value = "&nbsp;"
                return
            rect = QtCore.QRectF(x0, y0, width, height)
            self.image_item.setRect(rect)
            if hasattr(self.image_item, "setColorMap"):
                self.image_item.setColorMap(self.cmap)
            else:  # pragma: no cover - compatibility
                lut = self.cmap.getLookupTable(256)
                self.image_item.setLookupTable(lut)
            vmin = float(self.w_vmin.value)
            vmax = float(self.w_vmax.value)
            self.image_item.setLevels((vmin, vmax))
            if self.colorbar is not None:
                self.colorbar.setLevels((vmin, vmax))
            if self.first_draw or self.args_imagebuilder["hmin"] is None:
                vb.setXRange(x0, x1, padding=0)
            if self.first_draw or self.args_imagebuilder["vmin"] is None:
                vb.setYRange(min(y0, y1), max(y0, y1), padding=0)
            self.first_draw = False
            self.image_item.show()
            self._process_qt_events()
            self._request_remote_draw()

    def _handle_scene_move(self, pos: QtCore.QPointF) -> None:
        vb = self.plot.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            point = vb.mapSceneToView(pos)
            value = self._sample_value(point.x(), point.y())
            label = (
                f"<b>x</b>: {point.x():0.2f} | <b>y</b>: {point.y():0.2f} | "
                f"<b>value</b>: {value:0.2f}" if value is not None else "--"
            )
            self.hover_label.value = label
        else:
            self.hover_label.value = "&nbsp;"

    def _sample_value(self, x_coord: float, y_coord: float) -> Optional[float]:
        if self.wci is None or self.extent is None:
            return None
        x0, x1, y0, y1 = self.extent
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0 or dy == 0:
            return None
        col = (x_coord - x0) / dx * (self.wci.shape[0] - 1)
        row = (y_coord - y0) / dy * (self.wci.shape[1] - 1)
        if 0 <= col < self.wci.shape[0] and 0 <= row < self.wci.shape[1]:
            return float(self.wci[int(col), int(row)])
        return None

    @staticmethod
    def _process_qt_events() -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def _request_remote_draw(self) -> None:
        request_draw = getattr(self.graphics, "request_draw", None)
        if callable(request_draw):
            request_draw()

    def process_events(self) -> None:
        """Process pending Qt events.
        
        Call this method when updating the viewer from a loop to ensure
        the UI remains responsive and updates are displayed.
        
        Example:
            for i in range(len(viewer.pings)):
                viewer.w_index.value = i
                viewer.process_events()
        """
        self._process_qt_events()

    def redraw(self, force: bool = True) -> None:
        """Force an immediate redraw of the widget.
        
        Processes pending Qt events and sends frame to browser.
        Use this when updating the viewer from a loop to ensure changes
        are immediately visible.
        
        Parameters
        ----------
        force : bool, default True
            If True, bypasses the normal frame scheduling and sends the
            frame directly (required for updates in tight loops).
            If False, uses the standard request_draw mechanism.
        
        Example:
            for i in range(len(viewer.pings)):
                viewer.w_index.value = i
                viewer.redraw()
        """
        self._process_qt_events()
        if force:
            self._force_send_frame()
        else:
            self._request_remote_draw()

    def _force_send_frame(self) -> None:
        """Bypass RFB scheduling and send frame directly to browser."""
        graphics = self.graphics
        # Get the frame from the widget
        get_frame = getattr(graphics, "get_frame", None)
        if not callable(get_frame):
            return
        try:
            array = get_frame()
            if array is None:
                return
            # Access the internal _rfb_send_frame method
            send_frame = getattr(graphics, "_rfb_send_frame", None)
            if callable(send_frame):
                send_frame(array)
        except Exception:
            # Fall back to request_draw if direct send fails
            self._request_remote_draw()

    def set_widget_height(self, height_px: int) -> None:
        self.widget_height_px = max(1, int(height_px))
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, self.widget_height_px)
        self._request_remote_draw()

    # -------------------------------------------------------------------------
    # Export functionality
    # -------------------------------------------------------------------------

    def export_image(self, filename: str, width: Optional[int] = None, height: Optional[int] = None) -> None:
        """Export the current view as an image file (PNG, JPG, TIFF, etc.).
        
        Parameters
        ----------
        filename : str
            Output filename. Format is determined by extension.
        width : int, optional
            Image width in pixels. If None, uses current widget size.
        height : int, optional  
            Image height in pixels. If None, uses current widget size.
        
        Example:
            viewer.export_image('watercolumn.png')
            viewer.export_image('watercolumn.png', width=1920, height=1080)
        """
        from pyqtgraph.exporters import ImageExporter
        exporter = ImageExporter(self.plot)
        if width is not None:
            exporter.parameters()['width'] = width
        if height is not None:
            exporter.parameters()['height'] = height
        exporter.export(filename)

    def export_svg(self, filename: str) -> None:
        """Export the current view as an SVG file.
        
        Parameters
        ----------
        filename : str
            Output filename (should end with .svg).
        
        Example:
            viewer.export_svg('watercolumn.svg')
        """
        from pyqtgraph.exporters import SVGExporter
        exporter = SVGExporter(self.plot)
        exporter.export(filename)

    def export_matplotlib(self) -> "matplotlib.figure.Figure":
        """Export the current view to a matplotlib figure.
        
        Note: MatplotlibExporter works best with line plots. For image data
        like water column images, consider using `get_figure_data()` instead
        to recreate the plot in matplotlib directly.
        
        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure containing the exported plot.
        
        Example:
            fig = viewer.export_matplotlib()
            fig.savefig('watercolumn_mpl.png', dpi=300)
        """
        from pyqtgraph.exporters import MatplotlibExporter
        exporter = MatplotlibExporter(self.plot)
        exporter.export()
        # The exporter opens a window; get the figure from it
        if exporter.windows:
            return exporter.windows[-1].getFigure()
        return None

    def get_figure_data(self) -> Dict[str, Any]:
        """Get the current image data and metadata for external plotting.
        
        This is useful for recreating the plot in matplotlib or other
        plotting libraries with full control over the output.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'image': The 2D numpy array of the water column image
            - 'extent': Tuple (x0, x1, y0, y1) for image positioning
            - 'vmin': Colorbar minimum value
            - 'vmax': Colorbar maximum value
            - 'cmap': Colormap name (if available)
            - 'xlabel': X-axis label
            - 'ylabel': Y-axis label
            - 'title': Plot title (ping info)
        
        Example:
            import matplotlib.pyplot as plt
            data = viewer.get_figure_data()
            fig, ax = plt.subplots()
            im = ax.imshow(data['image'].T, extent=data['extent'],
                          aspect='auto', origin='upper',
                          vmin=data['vmin'], vmax=data['vmax'],
                          cmap='YlGnBu_r')
            ax.set_xlabel(data['xlabel'])
            ax.set_ylabel(data['ylabel'])
            plt.colorbar(im, label='dB')
            plt.savefig('watercolumn.png', dpi=300)
        """
        return {
            'image': self.wci.copy() if self.wci is not None else None,
            'extent': self.extent,
            'vmin': float(self.w_vmin.value),
            'vmax': float(self.w_vmax.value),
            'xlabel': 'Horizontal distance (m)',
            'ylabel': 'Depth (m)',
            'title': f'{self.name} - Ping {self.w_index.value}',
            'date': self.w_date.value,
            'time': self.w_time.value,
        }

    def to_matplotlib(self, ax: Optional["matplotlib.axes.Axes"] = None, 
                      cmap: str = 'YlGnBu_r',
                      colorbar: bool = True,
                      **kwargs: Any) -> "matplotlib.axes.Axes":
        """Render the current water column image using matplotlib.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        cmap : str, default 'YlGnBu_r'
            Matplotlib colormap name.
        colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs
            Additional arguments passed to ax.imshow().
        
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        
        Example:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            viewer.to_matplotlib(ax=ax)
            plt.savefig('watercolumn.png', dpi=300)
        """
        import matplotlib.pyplot as plt
        
        if self.wci is None or self.extent is None:
            raise ValueError("No image data available. Call update_data() first.")
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        
        data = self.get_figure_data()
        x0, x1, y0, y1 = data['extent']
        
        # imshow expects extent as [left, right, bottom, top]
        im_kwargs = {
            'extent': [x0, x1, y1, y0],  # y1, y0 for origin='upper'
            'aspect': 'auto',
            'origin': 'upper',
            'vmin': data['vmin'],
            'vmax': data['vmax'],
            'cmap': cmap,
        }
        im_kwargs.update(kwargs)
        
        im = ax.imshow(data['image'].T, **im_kwargs)
        ax.set_xlabel(data['xlabel'])
        ax.set_ylabel(data['ylabel'])
        ax.set_title(f"{data['title']} ({data['date']} {data['time']})")
        
        if colorbar:
            plt.colorbar(im, ax=ax, label='dB')
        
        return ax

    def copy_to_clipboard(self) -> None:
        """Copy the current view as an image to the clipboard.
        
        Example:
            viewer.copy_to_clipboard()
            # Now paste into any application
        """
        from pyqtgraph.exporters import ImageExporter
        exporter = ImageExporter(self.plot)
        exporter.export(copy=True)

    def get_image_bytes(self, format: str = 'png') -> bytes:
        """Get the current view as image bytes.
        
        Parameters
        ----------
        format : str, default 'png'
            Image format ('png', 'jpg', etc.)
        
        Returns
        -------
        bytes
            The image data as bytes.
        
        Example:
            img_bytes = viewer.get_image_bytes()
            with open('watercolumn.png', 'wb') as f:
                f.write(img_bytes)
        """
        from pyqtgraph.exporters import ImageExporter
        exporter = ImageExporter(self.plot)
        return exporter.export(toBytes=True)