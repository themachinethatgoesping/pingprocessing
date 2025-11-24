"""PyQtGraph-based echogram viewer that streams via pyqtgraph's Jupyter widgets."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import ipywidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.jupyter import GraphicsLayoutWidget
from pyqtgraph.Qt import QtCore, QtWidgets

import themachinethatgoesping as theping
from . import pyqtgraph_helpers as pgh


class EchogramViewerPyQtGraph:
    """Replacement for Matplotlib-based EchogramViewer using PyQtGraph."""

    def __init__(
        self,
        echogramdata: Sequence[Any],
        name: str = "Echogram",
        names: Optional[Sequence[Optional[str]]] = None,
        progress: Optional[Any] = None,
        show: bool = True,
        voffsets: Optional[Sequence[float]] = None,
        cmap: str = "Greys_r",
        cmap_layer: str = "YlGnBu_r",
        fps: int = 25,
        widget_height_px: Optional[int] = None,
        widget_width_px: int = 900,
        **kwargs: Any,
    ) -> None:
        pg.setConfigOptions(imageAxisOrder="row-major")
        pgh.ensure_qapp()
        self.args_plot: Dict[str, Any] = {
            "vmin": kwargs.pop("vmin", -100),
            "vmax": kwargs.pop("vmax", -25),
        }
        self.args_plot.update(kwargs)
        self.args_plot_layer = dict(self.args_plot)
        self.cmap_name = cmap
        self.cmap_layer_name = cmap_layer
        self._colormap = pgh.resolve_colormap(cmap)
        self._colormap_layer = pgh.resolve_colormap(cmap_layer)

        if isinstance(echogramdata, dict):
            names = list(echogramdata.keys()) if names is None else names
            echogramdata = list(echogramdata.values())
        elif not isinstance(echogramdata, list):
            echogramdata = [echogramdata]

        self.echogramdata = list(echogramdata)
        if voffsets is not None:
            self.voffsets = list(voffsets)
        else:
            self.voffsets = [0.0] * len(self.echogramdata)
        if len(self.voffsets) < len(self.echogramdata):
            self.voffsets.extend([0.0] * (len(self.echogramdata) - len(self.voffsets)))
        self.names: List[Optional[str]] = []
        for idx in range(len(self.echogramdata)):
            if names is not None and idx < len(names):
                self.names.append(names[idx])
            else:
                self.names.append(None)
        self.nechograms = len(self.echogramdata)
        if self.echogramdata:
            self.x_axis_name = self.echogramdata[-1].x_axis_name
            self.y_axis_name = self.echogramdata[-1].y_axis_name
        else:
            self.x_axis_name = "Ping number"
            self.y_axis_name = "Depth (m)"
        self._x_axis_is_datetime = self.x_axis_name == "Date time"

        self.progress = progress or theping.pingprocessing.widgets.TqdmWidget()
        self.display_progress = progress is None

        self.output = ipywidgets.Output()
        self.hover_label = ipywidgets.HTML(value="&nbsp;")
        self.update_button = ipywidgets.Button(description="update")
        self.clear_button = ipywidgets.Button(description="clear output")
        self.update_button.on_click(self.show_background_zoom)
        self.clear_button.on_click(self.clear_output)

        self.w_vmin = ipywidgets.FloatSlider(description="vmin", min=-150, max=100, step=5, value=self.args_plot["vmin"])
        self.w_vmax = ipywidgets.FloatSlider(description="vmax", min=-150, max=100, step=5, value=self.args_plot["vmax"])
        self.w_interpolation = ipywidgets.Dropdown(
            description="interpolation",
            options=["nearest", "linear"],
            value="nearest",
        )
        for widget in (self.w_vmin, self.w_vmax, self.w_interpolation):
            widget.observe(self.update_view, names="value")

        height_px = widget_height_px or (300 * max(1, self.nechograms))
        self.widget_height_px = height_px
        self.widget_width_px = widget_width_px
        self.graphics = GraphicsLayoutWidget(css_width=f"{widget_width_px}px", css_height=f"{height_px}px")
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, self.widget_height_px)
        if hasattr(self.graphics, "gfxView"):
            self.graphics.gfxView.setBackground("w")
        self.box_sliders = ipywidgets.HBox([self.w_vmin, self.w_vmax, self.w_interpolation])
        self.update_ping_line_button = ipywidgets.Button(description="update pingline")
        self.update_ping_line_button.on_click(self.update_ping_line)
        self.box_buttons = ipywidgets.HBox([self.update_button, self.clear_button])

        self.plot_items: List[pg.PlotItem] = []
        self.image_layers: List[Dict[str, pg.ImageItem]] = []
        self.colorbars: List[Optional[pg.ColorBarItem]] = []
        self.layer_colorbars: List[Optional[pg.ColorBarItem]] = []
        self.pingline: List[Optional[pg.InfiniteLine]] = [None] * self.nechograms
        self.pingviewer = None
        self.fig_events: Dict[str, Any] = {}

        self.images_background: List[Optional[np.ndarray]] = []
        self.extents_background: List[Optional[Tuple[float, float, float, float]]] = []
        self.high_res_images: List[Optional[np.ndarray]] = []
        self.high_res_extents: List[Optional[Tuple[float, float, float, float]]] = []
        self.layer_images: List[Optional[np.ndarray]] = []
        self.layer_extents: List[Optional[Tuple[float, float, float, float]]] = []

        self._make_plot_items()
        self._connect_scene_events()
        if show:
            self.show()
        self.show_background_echogram()

    def _make_plot_items(self) -> None:
        self.plot_items.clear()
        self.image_layers.clear()
        self.colorbars.clear()
        layout = self.graphics
        for idx in range(self.nechograms):
            axis_items = None
            if self._x_axis_is_datetime:
                axis_items = {"bottom": pgh.MatplotlibDateAxis(self._mpl_num_to_datetime, orientation="bottom")}
            plot: pg.PlotItem = layout.addPlot(row=idx, col=0, axisItems=axis_items)
            plot.setLabels(bottom="", left="")
            plot.getViewBox().invertY(True)
            plot.getViewBox().setBackgroundColor("w")
            background = pg.ImageItem(axisOrder="row-major")
            plot.addItem(background)
            high_res = pg.ImageItem(axisOrder="row-major")
            high_res.setOpacity(0.85)
            high_res.hide()
            plot.addItem(high_res)
            layer = pg.ImageItem(axisOrder="row-major")
            layer.setOpacity(0.65)
            layer.hide()
            plot.addItem(layer)
            self.plot_items.append(plot)
            self.image_layers.append({"background": background, "high": high_res, "layer": layer})
            try:
                colorbar = pg.ColorBarItem(label="(dB)", values=(self.args_plot["vmin"], self.args_plot["vmax"]))
                colorbar.setImageItem(background, insert_in=plot)
                if hasattr(colorbar, "setColorMap"):
                    colorbar.setColorMap(self._colormap)
            except AttributeError:
                colorbar = None
            self.colorbars.append(colorbar)
            try:
                layer_colorbar = pg.ColorBarItem(
                    label="layer (dB)",
                    values=(self.args_plot_layer["vmin"], self.args_plot_layer["vmax"]),
                )
                layer_colorbar.setImageItem(layer)
                if hasattr(layer_colorbar, "setColorMap"):
                    layer_colorbar.setColorMap(self._colormap_layer)
                layer_colorbar.hide()
                layout.addItem(layer_colorbar, row=idx, col=1)
            except AttributeError:
                layer_colorbar = None
            self.layer_colorbars.append(layer_colorbar)
            if idx == self.nechograms - 1:
                layout.nextRow()
        self._link_axes()
        self._refresh_axis_items()

    def _connect_scene_events(self) -> None:
        gfx_view = getattr(self.graphics, "gfxView", None)
        scene = gfx_view.scene() if gfx_view is not None else None
        if scene is None:
            return
        scene.sigMouseClicked.connect(self._handle_scene_click)
        scene.sigMouseMoved.connect(self._handle_scene_move)

    def _link_axes(self) -> None:
        if not self.plot_items:
            return
        master = self.plot_items[0]
        for plot in self.plot_items[1:]:
            plot.setXLink(master)
            plot.setYLink(master)

    def _refresh_axis_items(self) -> None:
        if not self.plot_items:
            return
        for plot in self.plot_items:
            axis_items: Dict[str, pg.AxisItem] = {}
            if self._x_axis_is_datetime:
                axis_items["bottom"] = pgh.MatplotlibDateAxis(self._mpl_num_to_datetime, orientation="bottom")
                axis_items["top"] = pgh.MatplotlibDateAxis(self._mpl_num_to_datetime, orientation="top")
            else:
                axis_items["bottom"] = pg.AxisItem(orientation="bottom")
                axis_items["top"] = pg.AxisItem(orientation="top")
            plot.setAxisItems(axis_items)
        self._configure_x_axis_visibility()

    def _configure_x_axis_visibility(self) -> None:
        if not self.plot_items:
            return
        for idx, plot in enumerate(self.plot_items):
            show_top = idx == 0
            show_bottom = idx == self.nechograms - 1
            plot.showAxis("top", show_top)
            plot.showAxis("bottom", show_bottom)
            top_axis = plot.getAxis("top")
            bottom_axis = plot.getAxis("bottom")
            if top_axis is not None:
                top_axis.setStyle(showValues=show_top)
            if bottom_axis is not None:
                bottom_axis.setStyle(showValues=show_bottom)

    def show(self) -> None:
        widgets = [ipywidgets.HBox([self.graphics])]
        if self.display_progress:
            widgets.append(ipywidgets.HBox([self.progress]))
        widgets.extend([self.box_sliders, self.box_buttons, self.hover_label, self.output])
        self.layout = ipywidgets.VBox(widgets)
        display(self.layout)

    def init_ax(self, adapt_axis_names: bool = True) -> None:
        with self.output:
            if adapt_axis_names and self.echogramdata:
                new_x = self.echogramdata[-1].x_axis_name
                new_y = self.echogramdata[-1].y_axis_name
                self.x_axis_name = new_x
                self.y_axis_name = new_y
                new_flag = self.x_axis_name == "Date time"
                if new_flag != self._x_axis_is_datetime:
                    self._x_axis_is_datetime = new_flag
                    self._refresh_axis_items()
            for idx, plot in enumerate(self.plot_items):
                plot.setTitle(self.names[idx] or "")
                plot.setLabel("left", self.y_axis_name)
                if idx == self.nechograms - 1:
                    plot.setLabel("bottom", self.x_axis_name)
                else:
                    plot.setLabel("bottom", "")
                if idx == 0:
                    plot.setLabel("top", self.x_axis_name)
                else:
                    plot.setLabel("top", "")
            self._configure_x_axis_visibility()

    def show_background_echogram(self) -> None:
        with self.output:
            self.init_ax()
            self.images_background, self.extents_background = [], []
            self.high_res_images, self.high_res_extents = [], []
            self.layer_images, self.layer_extents = [], []
            for idx, echogram in enumerate(self.echogramdata):
                self.progress.set_description(f"Updating echogram [{idx},{self.nechograms}]")
                if not echogram.layers and echogram.main_layer is None:
                    image, extent = echogram.build_image(progress=self.progress)
                    self.images_background.append(image)
                    self.extents_background.append(extent)
                    self.high_res_images.append(None)
                    self.high_res_extents.append(None)
                    self.layer_images.append(None)
                    self.layer_extents.append(None)
                else:
                    image, layer_img, extent = echogram.build_image_and_layer_image(progress=self.progress)
                    self.images_background.append(image)
                    self.extents_background.append(extent)
                    self.layer_images.append(layer_img)
                    self.layer_extents.append(extent)
                    self.high_res_images.append(None)
                    self.high_res_extents.append(None)
            self.update_view(reset=True)
            self._process_qt_events()
            self.progress.set_description("Idle")

    def clear_output(self, _event: Any = None) -> None:
        with self.output:
            self.output.clear_output()

    def show_background_zoom(self, _event: Any = None) -> None:
        with self.output:
            self.high_res_images = [None] * self.nechograms
            self.high_res_extents = [None] * self.nechograms
            self.layer_images = [None] * self.nechograms
            self.layer_extents = [None] * self.nechograms
            for idx, echogram in enumerate(self.echogramdata):
                self.progress.set_description(f"Updating echogram [{idx},{self.nechograms}]")
                view = self.plot_items[idx].getViewBox()
                xmin, xmax = view.viewRange()[0]
                ymin, ymax = view.viewRange()[1]
                self._apply_axis_limits(echogram, xmin, xmax, ymin, ymax)
                if not echogram.layers and echogram.main_layer is None:
                    image, extent = echogram.build_image(progress=self.progress)
                    self.high_res_images[idx] = image
                    self.high_res_extents[idx] = extent
                else:
                    image, layer_img, extent = echogram.build_image_and_layer_image(progress=self.progress)
                    self.high_res_images[idx] = image
                    self.high_res_extents[idx] = extent
                    self.layer_images[idx] = layer_img
                    self.layer_extents[idx] = extent
            self.update_view()
            self._process_qt_events()
            self.progress.set_description("Idle")

    def _apply_axis_limits(self, echogram: Any, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        x_kwargs = echogram.get_x_kwargs()
        y_kwargs = echogram.get_y_kwargs()
        match self.x_axis_name:
            case "Date time":
                tmin, tmax = self._mpl_num_to_datetime([xmin, xmax])
                x_kwargs["min_ping_time"] = tmin
                x_kwargs["max_ping_time"] = tmax
                echogram.set_x_axis_date_time(**x_kwargs)
            case "Ping number":
                x_kwargs["min_ping_nr"] = xmin
                x_kwargs["max_ping_nr"] = xmax
                echogram.set_x_axis_ping_nr(**x_kwargs)
            case "Ping index":
                x_kwargs["min_ping_index"] = xmin
                x_kwargs["max_ping_index"] = xmax
                echogram.set_x_axis_ping_index(**x_kwargs)
            case "Ping time":
                x_kwargs["min_timestamp"] = xmin
                x_kwargs["max_timestamp"] = xmax
                echogram.set_x_axis_ping_time(**x_kwargs)
            case _:
                raise RuntimeError(f"unknown x axis name '{self.x_axis_name}'")
        match self.y_axis_name:
            case "Depth (m)":
                y_kwargs["min_depth"] = ymin
                y_kwargs["max_depth"] = ymax
                echogram.set_y_axis_depth(**y_kwargs)
            case "Range (m)":
                y_kwargs["min_range"] = ymin
                y_kwargs["max_range"] = ymax
                echogram.set_y_axis_range(**y_kwargs)
            case "Sample number":
                y_kwargs["min_sample_nr"] = ymin
                y_kwargs["max_sample_nr"] = ymax
                echogram.set_y_axis_sample_nr(**y_kwargs)
            case "Y indice":
                y_kwargs["min_sample_nr"] = ymin
                y_kwargs["max_sample_nr"] = ymax
                echogram.set_y_axis_y_indice(**y_kwargs)
            case _:
                raise RuntimeError(f"unknown y axis name '{self.y_axis_name}'")

    def _current_levels(self, idx: int) -> Tuple[float, float]:
        vmin = self.w_vmin.value + self.voffsets[idx]
        vmax = self.w_vmax.value + self.voffsets[idx]
        return float(vmin), float(vmax)

    def update_view(self, _widget: Any = None, reset: bool = False) -> None:
        with self.output:
            try:
                minx = np.inf
                maxx = -np.inf
                miny = np.inf
                maxy = -np.inf
                for idx in range(self.nechograms):
                    background = self.images_background[idx]
                    extent = self.extents_background[idx]
                    if background is None or extent is None:
                        continue
                    self._update_plot_image(idx, "background", background, extent)
                    if reset:
                        x0, x1, y0, y1 = self._numeric_extent(extent)
                        minx = min(minx, x0)
                        maxx = max(maxx, x1)
                        miny = min(miny, y0)
                        maxy = max(maxy, y1)
                    if self.high_res_images[idx] is not None and self.high_res_extents[idx] is not None:
                        self._update_plot_image(idx, "high", self.high_res_images[idx], self.high_res_extents[idx])
                    else:
                        self.image_layers[idx]["high"].hide()
                    if self.layer_images[idx] is not None and self.layer_extents[idx] is not None:
                        self._update_plot_image(idx, "layer", self.layer_images[idx], self.layer_extents[idx])
                        layer_cbar = self.layer_colorbars[idx]
                        if layer_cbar is not None:
                            layer_cbar.show()
                    else:
                        self.image_layers[idx]["layer"].hide()
                        layer_cbar = self.layer_colorbars[idx]
                        if layer_cbar is not None:
                            layer_cbar.hide()
                if reset and np.all(np.isfinite([minx, maxx, miny, maxy])) and self.plot_items:
                    master_plot = self.plot_items[0]
                    master_plot.setXRange(minx, maxx, padding=0)
                    master_plot.setYRange(miny, maxy, padding=0)
                self.callback_view()
                self._process_qt_events()
                self._request_remote_draw()
            except Exception as error:  # pragma: no cover
                raise error

    def _update_plot_image(self, idx: int, key: str, data: np.ndarray, extent: Tuple[float, float, float, float]) -> None:
        image_item = self.image_layers[idx][key]
        array = data.transpose()
        image_item.setImage(array, autoLevels=False)
        x0, x1, y0, y1 = self._numeric_extent(extent)
        plot = self.plot_items[idx]
        vb = plot.getViewBox()
        if vb.yInverted():
            y0, y1 = y1, y0
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        image_item.setRect(rect)
        colormap = self._colormap_layer if key != "background" else self._colormap
        if hasattr(image_item, "setColorMap"):
            image_item.setColorMap(colormap)
        else:  # pragma: no cover - compatibility with older pyqtgraph
            lut = colormap.getLookupTable(256)
            image_item.setLookupTable(lut)
        image_item.setLevels(self._current_levels(idx))
        image_item.show()
        colorbar = self.colorbars[idx]
        if colorbar is not None and key == "background":
            if hasattr(colorbar, "setColorMap"):
                colorbar.setColorMap(colormap)
            colorbar.setLevels(self._current_levels(idx))
        layer_colorbar = self.layer_colorbars[idx]
        if layer_colorbar is not None and key == "layer":
            if hasattr(layer_colorbar, "setColorMap"):
                layer_colorbar.setColorMap(colormap)
            layer_colorbar.setLevels(self._current_levels(idx))

    def invert_y_axis(self) -> None:
        with self.output:
            for plot in self.plot_items:
                vb = plot.getViewBox()
                vb.invertY(not vb.yInverted())
            self._request_remote_draw()

    def callback_view(self) -> None:
        pass

    def _handle_scene_click(self, event: Any) -> None:
        if self.pingviewer is None:
            return
        pos = event.scenePos()
        for idx, plot in enumerate(self.plot_items):
            vb = plot.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                point = vb.mapSceneToView(pos)
                self._update_pingviewer_from_coordinate(point.x())
                self.update_ping_line()
                break

    def _handle_scene_move(self, pos: QtCore.QPointF) -> None:
        for idx, plot in enumerate(self.plot_items):
            vb = plot.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                point = vb.mapSceneToView(pos)
                value = self._sample_value(idx, point.x(), point.y())
                self._update_hover_label(point.x(), point.y(), value)
                return
        self.hover_label.value = "&nbsp;"

    def _update_pingviewer_from_coordinate(self, coord: float) -> None:
        pingviewer = self.pingviewer
        if pingviewer is None:
            return
        match self.x_axis_name:
            case "Ping number":
                pingviewer.w_index.value = int(max(0, coord))
            case "Ping index":
                pingviewer.w_index.value = int(max(0, coord))
            case "Date time":
                target = self._mpl_num_to_datetime(coord).timestamp()
                for idx, ping in enumerate(pingviewer.imagebuilder.pings):
                    ping_obj = ping if not isinstance(ping, dict) else next(iter(ping.values()))
                    if ping_obj.get_datetime().timestamp() > target:
                        pingviewer.w_index.value = max(0, idx - 1)
                        return
                pingviewer.w_index.value = len(pingviewer.imagebuilder.pings) - 1
            case "Ping time":
                target = coord
                for idx, ping in enumerate(pingviewer.imagebuilder.pings):
                    ping_obj = ping if not isinstance(ping, dict) else next(iter(ping.values()))
                    if ping_obj.get_timestamp() > target:
                        pingviewer.w_index.value = max(0, idx - 1)
                        return
                pingviewer.w_index.value = len(pingviewer.imagebuilder.pings) - 1
            case _:
                raise RuntimeError(f"unknown x axis name '{self.x_axis_name}'")

    def update_ping_line(self, _event: Any = None) -> None:
        if self.pingviewer is None:
            return
        match self.x_axis_name:
            case "Ping number":
                value = float(self.pingviewer.w_index.value)
            case "Ping index":
                value = float(self.pingviewer.w_index.value)
            case "Date time":
                ping = self._get_current_ping()
                value = self._datetime_to_mpl_num(ping.get_datetime())
            case "Ping time":
                ping = self._get_current_ping()
                value = ping.get_timestamp()
            case _:
                raise RuntimeError(f"unknown x axis name '{self.x_axis_name}'")
        for idx, plot in enumerate(self.plot_items):
            if self.pingline[idx] is None:
                line = pg.InfiniteLine(angle=90, pen=pg.mkPen(color="k", style=QtCore.Qt.PenStyle.DashLine))
                plot.addItem(line)
                self.pingline[idx] = line
            self.pingline[idx].setValue(value)
        self._request_remote_draw()

    def _get_current_ping(self) -> Any:
        ping = self.pingviewer.imagebuilder.pings[self.pingviewer.w_index.value]
        if isinstance(ping, dict):
            return next(iter(ping.values()))
        return ping

    def disconnect_pingviewer(self) -> None:
        self.pingviewer = None
        for line in self.pingline:
            if line is not None:
                line.hide()
        self.box_buttons.children = (self.update_button, self.clear_button)

    def connect_pingviewer(self, pingviewer: Any) -> None:
        self.pingviewer = pingviewer
        self.box_buttons.children = (self.update_button, self.clear_button, self.update_ping_line_button)
        self.update_ping_line()
        self._process_qt_events()
        self._process_qt_events()

    def set_widget_height(self, height_px: int) -> None:
        """Adjust the CSS height of the embedded GraphicsLayoutWidget."""
        self.widget_height_px = max(1, int(height_px))
        pgh.apply_widget_layout(self.graphics, self.widget_width_px, self.widget_height_px)
        self._request_remote_draw()

    @staticmethod
    def _mpl_num_to_datetime(value: Union[float, Sequence[float]]) -> Union[datetime, List[datetime]]:
        base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return [base + timedelta(days=float(v)) for v in value]
        return base + timedelta(days=float(value))

    @staticmethod
    def _datetime_to_mpl_num(value: datetime) -> float:
        base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        dt_value = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        delta = dt_value - base
        return delta.total_seconds() / 86400.0

    @staticmethod
    def _process_qt_events() -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def _request_remote_draw(self) -> None:
        request_draw = getattr(self.graphics, "request_draw", None)
        if callable(request_draw):
            request_draw()

    def _format_x_value(self, coord: float) -> str:
        match self.x_axis_name:
            case "Date time":
                dt_value = self._mpl_num_to_datetime(coord)
                return dt_value.isoformat(sep=" ")
            case "Ping time":
                return f"{coord:0.2f} s"
            case "Ping index":
                return f"{coord:0.0f}"
            case _:
                return f"{coord:0.2f}"

    def _sample_value(self, idx: int, x_coord: float, y_coord: float) -> Optional[float]:
        sources = [
            (self.high_res_images[idx], self.high_res_extents[idx]),
            (self.images_background[idx], self.extents_background[idx]),
        ]
        for image, extent in sources:
            if image is None or extent is None:
                continue
            x0, x1, y0, y1 = self._numeric_extent(extent)
            dx = x1 - x0
            dy = y1 - y0
            if dx == 0 or dy == 0:
                continue
            col = (x_coord - x0) / dx * (image.shape[1] - 1)
            row = (y_coord - y0) / dy * (image.shape[0] - 1)
            if 0 <= col < image.shape[1] and 0 <= row < image.shape[0]:
                return float(image[int(row), int(col)])
        return None

    def _update_hover_label(self, x_coord: float, y_coord: float, value: Optional[float]) -> None:
        x_text = self._format_x_value(x_coord)
        y_text = f"{y_coord:0.2f}"
        if value is None:
            value_text = "--"
        else:
            value_text = f"{value:0.2f}"
        self.hover_label.value = (
            f"<b>x</b>: {x_text} | <b>y</b>: {y_text} | <b>value</b>: {value_text}"
        )

    def _numeric_extent(self, extent: Tuple[Any, Any, Any, Any]) -> Tuple[float, float, float, float]:
        x0 = self._extent_value_to_float(extent[0])
        x1 = self._extent_value_to_float(extent[1])
        y0 = self._extent_value_to_float(extent[2])
        y1 = self._extent_value_to_float(extent[3])
        return x0, x1, y0, y1

    def _extent_value_to_float(self, value: Any) -> float:
        if isinstance(value, datetime):
            return self._datetime_to_mpl_num(value)
        if isinstance(value, timedelta):
            return value.total_seconds() / 86400.0
        if isinstance(value, np.datetime64):
            delta = value - np.datetime64("1970-01-01T00:00:00Z")
            seconds = delta / np.timedelta64(1, "s")
            return float(seconds) / 86400.0
        if isinstance(value, np.timedelta64):
            seconds = value / np.timedelta64(1, "s")
            return float(seconds) / 86400.0
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)

