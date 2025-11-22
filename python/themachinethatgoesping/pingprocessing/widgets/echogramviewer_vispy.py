"""VisPy-based echogram viewer with ipywidgets integration.

This module mirrors the public API of :mod:`echogramviewer` but swaps out the
Matplotlib backend for VisPy to improve interactivity and rendering
performance inside Jupyter notebooks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ipywidgets as widgets
import numpy as np
from IPython.display import display

import themachinethatgoesping as theping
import themachinethatgoesping.pingprocessing.watercolumn.echograms as echograms

try:  # pragma: no cover - import guard required at runtime
    from vispy import app, scene
    from vispy.color import Colormap, get_colormap
    from vispy.scene import AxisWidget
    from vispy.scene.widgets import ColorBarWidget
    from vispy.visuals.transforms import STTransform
except ImportError as exc:  # pragma: no cover - executed when vispy is missing
    raise ImportError(
        "EchogramViewerVispy requires the 'vispy' extra. Install via 'pip install vispy' "
        "or include the '[vispy]' extra when installing themachinethatgoesping."
    ) from exc


def _as_float(value: Any) -> float:
    """Best-effort conversion of extent coordinates to plain floats."""

    if isinstance(value, (int, float, np.floating)):
        return float(value)

    try:
        import datetime as _dt

        if isinstance(value, (_dt.datetime, _dt.date)):
            try:
                import matplotlib.dates as _mdates  # type: ignore

                return float(_mdates.date2num(value))
            except Exception:
                if isinstance(value, _dt.datetime):
                    return value.timestamp()
                return _dt.datetime.combine(value, _dt.time()).timestamp()
        if isinstance(value, _dt.timedelta):
            return value.total_seconds()
    except Exception:  # pragma: no cover - datetime always available
        pass

    try:
        import matplotlib.dates as _mdates  # type: ignore

        return float(_mdates.date2num(value))
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass

    return float(value)  # may still raise, but ensures consistent error path


def _normalize_extent(extent: Tuple[Any, Any, Any, Any]) -> Tuple[float, float, float, float]:
    return tuple(_as_float(coord) for coord in extent)  # type: ignore[return-value]


def _compute_image_transform(
    extent: Tuple[float, float, float, float],
    shape: Tuple[int, int],
) -> STTransform:
    xmin, xmax, ymin, ymax = extent
    height = max(shape[0], 1)
    width = max(shape[1], 1)

    delta_x = xmax - xmin
    delta_y = ymax - ymin

    scale_x = delta_x / (width - 1) if width > 1 else delta_x or 1.0
    scale_y = delta_y / (height - 1) if height > 1 else delta_y or 1.0

    if scale_x == 0:
        scale_x = 1e-9
    if scale_y == 0:
        scale_y = 1e-9

    return STTransform(scale=(scale_x, scale_y), translate=(xmin, ymin))


def _resolve_colormap(value: Any, fallback: str = "viridis") -> Colormap:

    if isinstance(value, Colormap):
        return value

    if isinstance(value, str):
        try:
            return get_colormap(value)
        except KeyError:
            pass

    try:
        import matplotlib.cm as cm  # type: ignore

        mpl_cmap = cm.get_cmap(value if isinstance(value, str) else fallback)
        colors = mpl_cmap(np.linspace(0.0, 1.0, mpl_cmap.N))[:, :4]
        return Colormap(colors)
    except Exception:  # pragma: no cover - matplotlib optional
        return get_colormap(fallback)


@dataclass
class _ImagePayload:
    data: np.ndarray
    extent: Tuple[float, float, float, float]


class EchogramViewerVispy:
    """Interactive echogram viewer rendered with VisPy."""

    def __init__(
        self,
        echogramdata: Sequence[echograms.EchogramData] | Dict[str, Any],
        name: str = "Echogram",
        names: Optional[Sequence[Optional[str]]] = None,
        canvas: Optional[scene.SceneCanvas] = None,
        progress: Optional[widgets.Widget] = None,
        show: bool = True,
        voffsets: Optional[Sequence[float]] = None,
        cmap: Any = "YlGnBu_r",
        cmap_layer: Any = "jet",
        **kwargs: Any,
    ) -> None:
        try:
            app.use_app("jupyter_rfb")
        except Exception:  # pragma: no cover - backend fallback
            pass

        if isinstance(echogramdata, dict):
            keys = list(echogramdata.keys())
            names = keys if names is None else names
            echogramdata = list(echogramdata.values())
        elif not isinstance(echogramdata, Sequence):
            echogramdata = [echogramdata]

        self.echogramdata = list(echogramdata)
        self.nechograms = len(self.echogramdata)
        self.voffsets = list(voffsets or [0.0] * self.nechograms)
        self.names = self._build_name_list(names)
        self._title_widgets = [widgets.Label(value=name or "") for name in self.names]
        self._has_titles = any(name for name in self.names)
        self.args_plot: Dict[str, Any] = {
            "vmin": kwargs.pop("vmin", -100.0),
            "vmax": kwargs.pop("vmax", -25.0),
        }
        self.args_plot_layer = dict(self.args_plot)

        self.cmap = _resolve_colormap(cmap)
        self.cmap_layer = _resolve_colormap(cmap_layer)

        self.progress = (
            progress
            if progress is not None
            else theping.pingprocessing.widgets.TqdmWidget()
        )
        self.display_progress = progress is None

        self.canvas = canvas or scene.SceneCanvas(
            keys="interactive",
            title=name,
            size=(1200, int(320 * max(1, self.nechograms))),
            bgcolor="#ffffff",
            show=False,
            resizable=True,
        )
        self.grid = self.canvas.central_widget.add_grid(margin=0)
        self._axis_y_widgets: List[AxisWidget] = []
        self._x_axis_widget: Optional[AxisWidget] = None
        self._master_camera = scene.cameras.PanZoomCamera(aspect=None)

        self.views: List[scene.widgets.ViewBox] = []
        self.colorbars: List[ColorBarWidget] = []
        for row in range(self.nechograms):
            if row == 0:
                camera = self._master_camera
            else:
                camera = scene.cameras.PanZoomCamera(aspect=None)
                camera.link = self._master_camera
            view = self.grid.add_view(row=row, col=1, camera=camera)
            view.bgcolor = "#ffffff"
            self.views.append(view)

            y_axis = AxisWidget(
                orientation="left",
                axis_color="#000000",
                text_color="#000000",
            )
            self.grid.add_widget(y_axis, row=row, col=0)
            y_axis.link_view(view)
            self._axis_y_widgets.append(y_axis)

            cbar = ColorBarWidget(
                label="(dB)",
                cmap=self.cmap,
                clim=(self.args_plot["vmin"], self.args_plot["vmax"]),
                orientation="right",
                border_color="#444444",
            )
            self.grid.add_widget(cbar, row=row, col=2)
            self.colorbars.append(cbar)

        if self.views:
            self._x_axis_widget = AxisWidget(
                orientation="bottom",
                axis_color="#000000",
                text_color="#000000",
            )
            self.grid.add_widget(self._x_axis_widget, row=self.nechograms, col=1)
            self._x_axis_widget.link_view(self.views[-1])

        self._visuals = [
            {
                "background": scene.visuals.Image(parent=view.scene, cmap=self.cmap, method="auto"),
                "hires": scene.visuals.Image(parent=view.scene, cmap=self.cmap, method="auto"),
                "layer": scene.visuals.Image(parent=view.scene, cmap=self.cmap_layer, method="auto"),
            }
            for view in self.views
        ]
        for vis in self._visuals:
            for key in vis:
                vis[key].visible = False

        self.canvas.events.key_press.connect(self.on_key_press)

        self.pingviewer = None
        self._ping_lines = [scene.visuals.Line(parent=view.scene, color="white") for view in self.views]
        for line in self._ping_lines:
            line.visible = False

        self.update_button = widgets.Button(description="update")
        self.clear_button = widgets.Button(description="clear output")
        self.update_button.on_click(self.show_background_zoom)
        self.clear_button.on_click(self.clear_output)

        self.w_vmin = widgets.FloatSlider(
            description="vmin", min=-150, max=100, step=5, value=self.args_plot["vmin"]
        )
        self.w_vmax = widgets.FloatSlider(
            description="vmax", min=-150, max=100, step=5, value=self.args_plot["vmax"]
        )
        self.w_interpolation = widgets.Dropdown(
            description="interpolation",
            options=["nearest", "linear", "cubic"],
            value="nearest",
        )
        for widget_control in (self.w_vmin, self.w_vmax, self.w_interpolation):
            widget_control.observe(self.update_view, names="value")

        self.output = widgets.Output()
        self.box_buttons = widgets.HBox([self.update_button, self.clear_button])
        self.box_sliders = widgets.HBox([self.w_vmin, self.w_vmax, self.w_interpolation])

        self._canvas_widget = self._wrap_canvas()
        self.layout: Optional[widgets.Widget] = None

        self.images_background: List[Optional[_ImagePayload]] = [None] * self.nechograms
        self.high_res_images: List[Optional[_ImagePayload]] = [None] * self.nechograms
        self.layer_images: List[Optional[_ImagePayload]] = [None] * self.nechograms

        if show:
            self.show()
        self.show_background_echogram()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _wrap_canvas(self) -> widgets.Widget:
        native = getattr(self.canvas, "native", None)
        if isinstance(native, widgets.Widget):
            layout = widgets.Layout(width="100%", height="100%", flex="1 1 auto")
            if hasattr(native, "layout"):
                native.layout = widgets.Layout(width="100%", height="100%")
            return widgets.Box([native], layout=layout)

        out = widgets.Output()
        with out:
            display(self.canvas)
        return out

    def show(self) -> None:
        children: List[widgets.Widget] = []
        if self._has_titles:
            children.append(widgets.HBox(self._title_widgets))
        children.append(
            widgets.HBox(
                [self._canvas_widget],
                layout=widgets.Layout(width="100%", height="100%", flex="1 1 auto"),
            )
        )
        if self.display_progress:
            children.append(widgets.HBox([self.progress]))
        children.extend([self.box_sliders, self.box_buttons, self.output])
        self.layout = widgets.VBox(children)
        display(self.layout)

    def clear_output(self, _event: object | None = None) -> None:
        with self.output:
            self.output.clear_output()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def init_ax(self) -> None:
        with self.output:
            self.x_axis_name = self.echogramdata[-1].x_axis_name
            self.y_axis_name = self.echogramdata[-1].y_axis_name
            if self._has_titles:
                for widget, name in zip(self._title_widgets, self.names, strict=False):
                    widget.value = name or ""
            for axis_widget in self._axis_y_widgets:
                if axis_widget.axis.axis_label is not None:
                    axis_widget.axis.axis_label.text = self.y_axis_name
            if self._x_axis_widget is not None and self._x_axis_widget.axis.axis_label is not None:
                self._x_axis_widget.axis.axis_label.text = self.x_axis_name

    def show_background_echogram(self, _event: object | None = None) -> None:
        with self.output:
            self.init_ax()
            self.images_background = [None] * self.nechograms
            self.high_res_images = [None] * self.nechograms
            self.layer_images = [None] * self.nechograms

            for idx, echogram in enumerate(self.echogramdata):
                self.progress.set_description(f"Loading echogram {idx+1}/{self.nechograms}")
                if not echogram.layers and echogram.main_layer is None:
                    img, extent = echogram.build_image(progress=self.progress)
                    self.images_background[idx] = _ImagePayload(img, _normalize_extent(extent))
                else:
                    img, img_layer, extent = echogram.build_image_and_layer_image(
                        progress=self.progress
                    )
                    norm_extent = _normalize_extent(extent)
                    self.images_background[idx] = _ImagePayload(img, norm_extent)
                    self.layer_images[idx] = _ImagePayload(img_layer, norm_extent)
            self.progress.set_description("Idle")
            self.update_view(reset=True)

    def show_background_zoom(self, _event: object | None = None) -> None:
        with self.output:
            self.high_res_images = [None] * self.nechograms
            self.layer_images = [None] * self.nechograms
            for idx, (echogram, view) in enumerate(zip(self.echogramdata, self.views, strict=False)):
                rect = getattr(view.camera, "rect", None)
                xmin = ymin = xmax = ymax = None
                if rect is not None:
                    try:
                        pos = getattr(rect, "pos", rect[:2])
                        size = getattr(rect, "size", rect[2:])
                        xmin = float(pos[0])
                        ymin = float(pos[1])
                        xmax = xmin + float(size[0])
                        ymax = ymin + float(size[1])
                    except Exception:
                        xmin = ymin = xmax = ymax = None

                if (rect is None) or (not all(np.isfinite(v) for v in (xmin, xmax, ymin, ymax))):
                    payload = self.high_res_images[idx] or self.images_background[idx]
                    if payload is None:
                        continue
                    xmin, xmax, ymin, ymax = payload.extent

                x_kwargs = echogram.get_x_kwargs()
                y_kwargs = echogram.get_y_kwargs()
                self._update_axis_selection(echogram, xmin, xmax, ymin, ymax, x_kwargs, y_kwargs)
                if not echogram.layers and echogram.main_layer is None:
                    img, extent = echogram.build_image(progress=self.progress)
                    self.high_res_images[idx] = _ImagePayload(img, _normalize_extent(extent))
                else:
                    img, img_layer, extent = echogram.build_image_and_layer_image(
                        progress=self.progress
                    )
                    norm_extent = _normalize_extent(extent)
                    self.high_res_images[idx] = _ImagePayload(img, norm_extent)
                    self.layer_images[idx] = _ImagePayload(img_layer, norm_extent)
            self.update_view()

    def _update_axis_selection(
        self,
        echogram: echograms.EchogramData,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        x_kwargs: Dict[str, Any],
        y_kwargs: Dict[str, Any],
    ) -> None:
        if self.x_axis_name == "Date time":
            import matplotlib.dates as mdates

            tmin, tmax = mdates.num2date([xmin, xmax])
            x_kwargs["min_ping_time"] = tmin
            x_kwargs["max_ping_time"] = tmax
            echogram.set_x_axis_date_time(**x_kwargs)
        elif self.x_axis_name == "Ping number":
            x_kwargs["min_ping_nr"] = xmin
            x_kwargs["max_ping_nr"] = xmax
            echogram.set_x_axis_ping_nr(**x_kwargs)
        elif self.x_axis_name == "Ping time":
            x_kwargs["min_timestamp"] = xmin
            x_kwargs["max_timestamp"] = xmax
            echogram.set_x_axis_ping_time(**x_kwargs)
        else:
            raise RuntimeError(f"Unknown x axis '{self.x_axis_name}'")

        if self.y_axis_name == "Depth (m)":
            y_kwargs["min_depth"] = ymin
            y_kwargs["max_depth"] = ymax
            echogram.set_y_axis_depth(**y_kwargs)
        elif self.y_axis_name == "Range (m)":
            y_kwargs["min_range"] = ymin
            y_kwargs["max_range"] = ymax
            echogram.set_y_axis_range(**y_kwargs)
        elif self.y_axis_name in {"Sample number", "Y indice"}:
            y_kwargs["min_sample_nr"] = ymin
            y_kwargs["max_sample_nr"] = ymax
            if self.y_axis_name == "Sample number":
                echogram.set_y_axis_sample_nr(**y_kwargs)
            else:
                echogram.set_y_axis_y_indice(**y_kwargs)
        else:
            raise RuntimeError(f"Unknown y axis '{self.y_axis_name}'")

    def update_view(self, change: Optional[dict] = None, reset: bool = False) -> None:
        del change  # not used; signature retained for ipywidgets callbacks
        with self.output:
            use_high_res = any(payload is not None for payload in self.high_res_images)
            payloads = self.high_res_images if use_high_res else self.images_background
            for idx, payload in enumerate(payloads):
                if payload is None:
                    continue
                self._apply_payload(idx, payload, layer=False, use_high_res=use_high_res)

            for idx, payload in enumerate(self.layer_images):
                if payload is None:
                    continue
                self._apply_payload(idx, payload, layer=True, use_high_res=use_high_res)

            for cbar, offset in zip(self.colorbars, self.voffsets, strict=False):
                cbar.clim = (self.w_vmin.value + offset, self.w_vmax.value + offset)

            if reset and self._master_camera is not None:
                first_payload = next((payload for payload in payloads if payload is not None), None)
                if first_payload is not None:
                    xmin, xmax, ymin, ymax = first_payload.extent
                    self._master_camera.rect = (xmin, ymin, xmax - xmin, ymax - ymin)

            self.canvas.update()

    def _apply_payload(
        self,
        idx: int,
        payload: _ImagePayload,
        layer: bool,
        use_high_res: bool,
    ) -> None:
        if idx >= len(self._visuals):
            return
        vis_key = "layer" if layer else ("hires" if use_high_res else "background")
        visual = self._visuals[idx][vis_key]
        image = np.asarray(payload.data)
        if image.ndim >= 2:
            perm = (1, 0) + tuple(range(2, image.ndim))
            image = np.transpose(image, axes=perm)
        if image.dtype != np.float32:
            image = image.astype(np.float32, copy=False)
        visual.set_data(image)
        transform = _compute_image_transform(payload.extent, image.shape[:2])
        visual.transform = transform
        offset = self.voffsets[idx]
        clim = (self.w_vmin.value + offset, self.w_vmax.value + offset)
        visual.clim = clim
        visual.interpolation = self.w_interpolation.value
        visual.visible = True

    def invert_y_axis(self) -> None:
        with self.output:
            for view in self.views:
                rect = view.camera.rect
                rect.pos = (rect.pos[0], rect.pos[1] + rect.size[1])
                rect.size = (rect.size[0], -rect.size[1])
                view.camera.rect = rect

    # ------------------------------------------------------------------
    # Ping viewer integration
    # ------------------------------------------------------------------
    def on_key_press(self, event: Any) -> None:
        if self.pingviewer is None or event.text != "p":
            return
        self._jump_to_ping_at_x(event.pos[0])
        self.update_ping_line()

    def _jump_to_ping_at_x(self, x_value: float) -> None:
        match self.x_axis_name:
            case "Date time":
                timestamps = [self._ping_timestamp(ping) for ping in self.pingviewer.imagebuilder.pings]
                differences = np.array(timestamps) - x_value
                candidates = np.where(differences > 0)[0]
                pn = candidates[0] - 1 if len(candidates) else len(timestamps) - 1
            case "Ping number":
                pn = int(np.clip(round(x_value), 0, len(self.pingviewer.imagebuilder.pings) - 1))
            case "Ping time":
                timestamps = [self._ping_timestamp(ping) for ping in self.pingviewer.imagebuilder.pings]
                pn = int(np.searchsorted(timestamps, x_value))
            case _:
                raise RuntimeError(f"Unknown x axis '{self.x_axis_name}'")
        pn = max(0, min(pn, len(self.pingviewer.imagebuilder.pings) - 1))
        self.pingviewer.w_index.value = pn

    @staticmethod
    def _ping_timestamp(ping: Any) -> float:
        if isinstance(ping, dict):
            ping = next(iter(ping.values()))
        return ping.get_timestamp()

    def update_ping_line(self, _event: object | None = None) -> None:
        if self.pingviewer is None:
            return
        x_value = self._current_ping_position()
        for view, line in zip(self.views, self._ping_lines, strict=False):
            rect = view.camera.rect
            y0 = rect.pos[1]
            y1 = y0 + rect.size[1]
            line.set_data(np.array([[x_value, y0], [x_value, y1]]))
            line.visible = True
        self.canvas.update()

    def _current_ping_position(self) -> float:
        idx = self.pingviewer.w_index.value
        ping = self.pingviewer.imagebuilder.pings[idx]
        if isinstance(ping, dict):
            ping = next(iter(ping.values()))
        if self.x_axis_name == "Ping number":
            return float(idx)
        if self.x_axis_name == "Date time":
            return ping.get_datetime().timestamp()
        return ping.get_timestamp()

    def connect_pingviewer(self, pingviewer: Any) -> None:
        self.pingviewer = pingviewer
        self.update_ping_line_button = widgets.Button(description="update pingline")
        self.update_ping_line_button.on_click(self.update_ping_line)
        self.box_buttons.children = (
            self.update_button,
            self.clear_button,
            self.update_ping_line_button,
        )

    def disconnect_pingviewer(self) -> None:
        self.pingviewer = None
        for line in self._ping_lines:
            line.visible = False
        self.box_buttons.children = (self.update_button, self.clear_button)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_name_list(self, names: Optional[Sequence[Optional[str]]]) -> List[Optional[str]]:
        resolved: List[Optional[str]] = []
        for idx in range(self.nechograms):
            if names and idx < len(names):
                resolved.append(names[idx])
            else:
                resolved.append(None)
        return resolved
