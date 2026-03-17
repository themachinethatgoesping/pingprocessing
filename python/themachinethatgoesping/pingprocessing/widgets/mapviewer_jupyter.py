"""Jupyter notebook map viewer using the extracted core + ipywidgets controls.

Drop-in replacement for ``MapViewerPyQtGraph`` — same constructor signature,
same public API — but built on top of :class:`mapviewer_core.MapCore` and
:class:`control_jupyter.JupyterControlPanel`.
"""
from __future__ import annotations

import asyncio
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import ipywidgets
import pyqtgraph as pg
from IPython.display import display
from pyqtgraph.jupyter import GraphicsLayoutWidget

from . import pyqtgraph_helpers as pgh
from .control_spec import (
    MAP_NAV_SPECS,
    MAP_MISC_SPECS,
    MAP_COLORBAR_SPECS,
    MAP_COLORMAPS,
    DropdownSpec,
    CheckboxSpec,
    FloatSliderSpec,
)
from .control_jupyter import JupyterControlPanel, JupyterControlHandle, create_jupyter_control
from .mapviewer_core import MapCore, LayerRenderSettings


class MapViewerJupyter:
    """PyQtGraph-based map viewer for Jupyter notebooks.

    Features:
    - Interactive pan/zoom with mouse
    - Layer management with visibility/opacity/colorscale controls
    - Auto-update with debouncing on pan/zoom
    - Track overlays showing navigation paths from echograms
    - Current ping position marker
    - Integration with EchogramViewer and WCIViewer
    - Background tiles from web sources

    The constructor signature is compatible with ``MapViewerPyQtGraph``.
    """

    def __init__(
        self,
        builder: Any = None,
        tile_builder: Any = None,
        width: int = 800,
        height: int = 600,
        show_controls: bool = True,
        max_render_size: Tuple[int, int] = (2000, 2000),
        auto_update: bool = True,
        auto_update_delay_ms: int = 300,
        show: bool = True,
        embedded: bool = False,
    ) -> None:
        pgh.ensure_qapp()

        self._width = width
        self._height = height
        self._show_controls = show_controls

        # Auto-update state
        self._auto_update_enabled = auto_update
        self._auto_update_delay_ms = auto_update_delay_ms
        self._debounce_task: Optional[asyncio.Task] = None
        self._last_view_range = None
        self._last_range_change_time: float = 0.0
        self._startup_complete = False

        # Loading state
        self._is_loading = False
        self._view_changed_during_load = False
        self._cancel_flag = threading.Event()
        self._loading_future: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Tile loading state
        self._is_loading_tiles = False
        self._tile_cancel_flag = threading.Event()
        self._tile_loading_future: Optional[asyncio.Task] = None
        self._tile_executor = ThreadPoolExecutor(max_workers=2)
        self._tile_view_changed_during_load = False
        self._pending_tile_bounds = None

        # Output for errors
        self.output = ipywidgets.Output()

        # Build control panel
        self.panel = JupyterControlPanel.from_specs(
            MAP_NAV_SPECS,
            MAP_MISC_SPECS,
            MAP_COLORBAR_SPECS,
        )

        # Build graphics widget
        self.graphics = GraphicsLayoutWidget(
            css_width=f"{width}px",
            css_height=f"{height}px",
        )
        pgh.apply_widget_layout(self.graphics, width, height)
        if hasattr(self.graphics, "gfxView"):
            self.graphics.gfxView.setBackground("w")

        # Create core
        self.core = MapCore(
            builder=builder,
            tile_builder=tile_builder,
            panel=self.panel,
            graphics=self.graphics,
            max_render_size=max_render_size,
        )

        # Wire adapter callbacks
        self.core._schedule_update = self._schedule_debounced_update
        self.core._request_draw = self._request_draw
        self.core._trigger_tile_load = self._trigger_tile_load
        self.core._report_error = self._report_error

        # Wire observers
        self.core.wire_observers()
        self.panel["auto_update"].on_change(self._on_auto_update_toggle)

        # Build layer controls (per-layer checkboxes, sliders, colormaps)
        self._layer_checkboxes: Dict[str, ipywidgets.Checkbox] = {}
        self._layer_sliders: Dict[str, ipywidgets.FloatSlider] = {}
        self._layer_colormap_dropdowns: Dict[str, ipywidgets.Dropdown] = {}
        self._build_layer_controls()

        # Build tile controls
        self._tile_source_dropdown = None
        self._tile_visibility_checkbox = None
        self._build_tile_controls()

        # Track legend
        self._track_legend_label = ipywidgets.HTML("<b>Tracks:</b>")
        self._track_checkboxes: Dict[str, ipywidgets.Checkbox] = {}
        self._track_legend = ipywidgets.VBox([])
        self.core._update_track_legend = self._update_track_legend

        self._embedded = embedded

        # Auto-update hook
        self._setup_auto_update_hook()

        # Assemble layout (skip in embedded mode)
        if not embedded:
            self._assemble_layout()

        # Initial render
        self.core.update_view()
        self._startup_complete = True

        if show and not embedded:
            self.show()

    # =====================================================================
    # Layer controls
    # =====================================================================

    def _build_layer_controls(self) -> None:
        """Build per-layer ipywidgets controls."""
        if self.core._builder is None:
            return

        layer_names = []
        for layer in self.core._builder.layers:
            settings = self.core._layer_render_settings.get(layer.name, LayerRenderSettings())
            layer_names.append(layer.name)

            cb = ipywidgets.Checkbox(
                value=layer.visible,
                description=layer.name,
                indent=False,
                layout=ipywidgets.Layout(width='auto'),
            )
            cb.observe(
                lambda change, name=layer.name: self.core.set_layer_visibility(name, change['new']),
                names='value',
            )
            self._layer_checkboxes[layer.name] = cb

            slider = ipywidgets.FloatSlider(
                value=settings.opacity, min=0.0, max=1.0, step=0.1,
                description='', continuous_update=True, readout=False,
                layout=ipywidgets.Layout(width='100px'),
            )
            slider.observe(
                lambda change, name=layer.name: self.core.set_layer_opacity_image(name, change['new']),
                names='value',
            )
            self._layer_sliders[layer.name] = slider

            cmap_dropdown = ipywidgets.Dropdown(
                options=MAP_COLORMAPS, value=settings.colormap,
                layout=ipywidgets.Layout(width='100px'),
            )
            cmap_dropdown.observe(
                lambda change, name=layer.name: self.core.set_layer_colormap_image(name, change['new']),
                names='value',
            )
            self._layer_colormap_dropdowns[layer.name] = cmap_dropdown

        # Update colorbar layer dropdown
        colorbar_options = [("None", None)] + [(n, n) for n in layer_names]
        self.panel["colorbar_layer"].widget.options = colorbar_options
        if layer_names:
            self.panel["colorbar_layer"].value = layer_names[0]
            self.core._active_colorbar_layer = layer_names[0]

    def _build_tile_controls(self) -> None:
        """Build tile source controls if TileBuilder is available."""
        if self.core._tile_builder is None:
            return

        from ..overview.map_builder.tile_builder import TILE_SOURCES

        self._tile_visibility_checkbox = ipywidgets.Checkbox(
            value=self.core.tile_visible,
            description="Show background tiles",
            indent=False,
        )
        self._tile_visibility_checkbox.observe(
            lambda change: self._on_tile_visibility_change(change['new']),
            names='value',
        )

        tile_source_options = ['None'] + list(TILE_SOURCES.keys())
        current_source = getattr(self.core._tile_builder, '_current_source_name', None)
        default_source = (
            current_source if current_source in tile_source_options
            else tile_source_options[1] if len(tile_source_options) > 1
            else 'None'
        )
        self._tile_source_dropdown = ipywidgets.Dropdown(
            options=tile_source_options,
            value=default_source,
            description='Tiles:',
            layout=ipywidgets.Layout(width='200px'),
        )
        self._tile_source_dropdown.observe(
            lambda change: self._on_tile_source_change(change['new']),
            names='value',
        )

    def _rebuild_layer_controls(self) -> None:
        """Rebuild layer controls after adding layers."""
        self._layer_checkboxes.clear()
        self._layer_sliders.clear()
        self._layer_colormap_dropdowns.clear()
        self._build_layer_controls()
        self._assemble_layout()

    # =====================================================================
    # Tile callbacks
    # =====================================================================

    def _on_tile_visibility_change(self, visible: bool) -> None:
        self.core.tile_visible = visible
        if self._tile_visibility_checkbox is not None:
            self._tile_visibility_checkbox.value = visible

    def _on_tile_source_change(self, source_name: str) -> None:
        self.core.change_tile_source(source_name)
        if self._tile_visibility_checkbox is not None and source_name != 'None':
            self._tile_visibility_checkbox.value = True

    # =====================================================================
    # Auto-update
    # =====================================================================

    def _on_auto_update_toggle(self, value) -> None:
        self._auto_update_enabled = value
        if not self._auto_update_enabled and self._debounce_task is not None:
            self._debounce_task.cancel()

    def _setup_auto_update_hook(self) -> None:
        original_request_draw = self.graphics.request_draw
        viewer = self

        def patched_request_draw():
            original_request_draw()
            if not viewer._startup_complete or not viewer._auto_update_enabled:
                return
            if viewer.core._ignore_range_changes or viewer._is_loading:
                return
            vb = viewer.core._plot.getViewBox()
            current_range = vb.viewRange()
            if viewer._last_view_range is not None:
                old_x, old_y = viewer._last_view_range
                new_x, new_y = current_range
                if not (np.allclose(old_x, new_x, rtol=1e-6) and
                        np.allclose(old_y, new_y, rtol=1e-6)):
                    viewer._last_view_range = current_range
                    viewer._last_range_change_time = time.time()
                    viewer._schedule_debounced_update()
            else:
                viewer._last_view_range = current_range

        self.graphics.request_draw = patched_request_draw

    def _schedule_debounced_update(self) -> None:
        if self._is_loading:
            self._view_changed_during_load = True
            return
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()

        delay_s = self._auto_update_delay_ms / 1000.0

        async def debounced():
            try:
                await asyncio.sleep(delay_s)
                elapsed = time.time() - self._last_range_change_time
                if elapsed >= delay_s - 0.01:
                    if not self._is_loading:
                        self._trigger_high_res_update()
            except asyncio.CancelledError:
                pass

        try:
            loop = asyncio.get_running_loop()
            self._debounce_task = loop.create_task(debounced())
        except RuntimeError:
            self._trigger_high_res_update()

    def _cancel_pending_load(self) -> None:
        self._cancel_flag.set()
        if self._loading_future is not None:
            try:
                self._loading_future.cancel()
            except Exception:
                pass
            self._loading_future = None
        self._is_loading = False

    def _trigger_high_res_update(self) -> None:
        self._cancel_pending_load()

        # Trigger tile loading
        if self.core._tile_builder is not None and self.core.tile_visible:
            self.core.render_tiles()

        if self.core._builder is None:
            # Just update tracks
            if self.core._tracks or self.core._overview_tracks:
                self.core._update_tracks(force=True)
                self.core._update_ping_marker()
                self._request_draw()
            return

        if self.core._current_bounds is None:
            self.core._current_bounds = self.core._builder.combined_bounds
        if self.core._current_bounds is None:
            return

        self._is_loading = True
        self._view_changed_during_load = False
        self._cancel_flag.clear()

        core = self.core
        cancel_flag = self._cancel_flag
        viewer = self

        def load_data():
            return core.build_high_res_sync(cancel_flag)

        def apply_results(results):
            viewer._is_loading = False
            if results is None:
                if viewer._view_changed_during_load:
                    viewer._view_changed_during_load = False
                    viewer._schedule_debounced_update()
                return
            core.apply_high_res_results(results)
            if viewer._view_changed_during_load:
                viewer._view_changed_during_load = False
                viewer._schedule_debounced_update()

        async def run_async():
            try:
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(viewer._executor, load_data)
                apply_results(results)
            except Exception as e:
                viewer._is_loading = False
                viewer._report_error(f"Map update error: {e}")

        try:
            loop = asyncio.get_running_loop()
            self._loading_future = loop.create_task(run_async())
        except RuntimeError:
            results = load_data()
            apply_results(results)

    # =====================================================================
    # Tile loading (async)
    # =====================================================================

    def _trigger_tile_load(self, bbox, target_size, cache_key) -> None:
        """Load tiles asynchronously."""
        self._cancel_tile_load()

        self._is_loading_tiles = True
        self._tile_view_changed_during_load = False
        self._tile_cancel_flag.clear()

        viewer = self
        tile_builder = self.core._tile_builder

        def load_tiles():
            if viewer._tile_cancel_flag.is_set():
                return None
            try:
                tile_image, actual_bounds = tile_builder.get_image_with_bounds(
                    bounds=bbox, target_size=target_size,
                )
                if viewer._tile_cancel_flag.is_set():
                    return None
                return {
                    'image': tile_image, 'bounds': actual_bounds,
                    'requested_bounds': bbox, 'cache_key': cache_key,
                }
            except AttributeError:
                tile_image, _ = tile_builder.get_image(
                    bounds=bbox, target_size=target_size,
                )
                if viewer._tile_cancel_flag.is_set():
                    return None
                return {
                    'image': tile_image, 'bounds': bbox,
                    'requested_bounds': bbox, 'cache_key': cache_key,
                }
            except Exception as e:
                warnings.warn(f"Tile loading error: {e}")
                return None

        def apply_tiles(result):
            viewer._is_loading_tiles = False
            if result is None:
                if viewer._tile_view_changed_during_load:
                    viewer._tile_view_changed_during_load = False
                    if viewer._pending_tile_bounds is not None:
                        viewer.core.render_tiles()
                return
            if result['image'] is not None:
                viewer.core.apply_tile_result(result)
            if viewer._tile_view_changed_during_load:
                viewer._tile_view_changed_during_load = False
                if viewer._pending_tile_bounds is not None:
                    viewer.core.render_tiles()

        async def run_tile_async():
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(viewer._tile_executor, load_tiles)
                apply_tiles(result)
            except Exception as e:
                viewer._is_loading_tiles = False
                warnings.warn(f"Async tile load error: {e}")

        try:
            loop = asyncio.get_running_loop()
            self._tile_loading_future = loop.create_task(run_tile_async())
        except RuntimeError:
            result = load_tiles()
            apply_tiles(result)

    def _cancel_tile_load(self) -> None:
        self._tile_cancel_flag.set()
        if self._tile_loading_future is not None:
            try:
                self._tile_loading_future.cancel()
            except Exception:
                pass
            self._tile_loading_future = None
        self._is_loading_tiles = False

    # =====================================================================
    # Track legend
    # =====================================================================

    def _update_track_legend(self) -> None:
        all_tracks = {**self.core._tracks, **self.core._overview_tracks}
        if not all_tracks:
            self._track_legend.children = []
            return

        checkbox_widgets = [self._track_legend_label]
        self._track_checkboxes.clear()

        for name, track in all_tracks.items():
            checkbox = ipywidgets.Checkbox(
                value=track.visible, description='', indent=False,
                layout=ipywidgets.Layout(width='20px'),
            )

            def make_vis_handler(track_name):
                def handler(change):
                    self.core.set_track_visibility(track_name, change['new'])
                return handler
            checkbox.observe(make_vis_handler(name), names='value')
            self._track_checkboxes[name] = checkbox

            color_picker = ipywidgets.ColorPicker(
                value=track.color, description='', concise=True,
                layout=ipywidgets.Layout(width='28px', height='24px'),
            )

            def make_color_handler(track_name):
                def handler(change):
                    self.core.set_track_color(track_name, change['new'])
                return handler
            color_picker.observe(make_color_handler(name), names='value')

            name_label = ipywidgets.HTML(f'{name}')
            row = ipywidgets.HBox(
                [checkbox, color_picker, name_label],
                layout=ipywidgets.Layout(align_items='center'),
            )
            checkbox_widgets.append(row)

        self._track_legend.children = checkbox_widgets

    # =====================================================================
    # Embeddable control widget
    # =====================================================================

    def build_control_widget(self) -> ipywidgets.Widget:
        """Return all controls as a single embeddable ipywidget."""
        p = self.panel
        parts: list = []

        # Tile controls
        if self._tile_source_dropdown is not None:
            parts.append(ipywidgets.HTML("<b>Background Tiles</b>"))
            parts.append(ipywidgets.HBox([
                self._tile_visibility_checkbox, self._tile_source_dropdown
            ]))

        # Layer controls
        if self.core._builder is not None:
            layer_widgets = []
            for layer in self.core._builder.layers:
                if layer.name in self._layer_checkboxes:
                    layer_widgets.append(ipywidgets.HBox([
                        self._layer_checkboxes[layer.name],
                        self._layer_sliders[layer.name],
                        self._layer_colormap_dropdowns[layer.name],
                    ]))
            if layer_widgets:
                parts.append(ipywidgets.HTML("<b>Data Layers</b>"))
                parts.append(ipywidgets.VBox(layer_widgets))
                parts.append(p.widget("colorbar_layer"))

        # Navigation
        parts.append(ipywidgets.HBox([
            p.widget("btn_zoom_fit"), p.widget("btn_zoom_track"),
            p.widget("btn_zoom_wci"), p.widget("btn_refresh_tracks"),
        ]))
        parts.append(ipywidgets.HBox([
            p.widget("auto_update"), p.widget("auto_center_wci"),
        ]))
        parts.append(p.widget("lbl_coords"))
        parts.append(self._track_legend)

        return ipywidgets.VBox(parts)

    # =====================================================================
    # Layout
    # =====================================================================

    def _assemble_layout(self) -> None:
        p = self.panel

        # Layer widgets
        layer_widgets = []
        if self.core._builder is not None:
            for layer in self.core._builder.layers:
                if layer.name in self._layer_checkboxes:
                    layer_widgets.append(ipywidgets.HBox([
                        self._layer_checkboxes[layer.name],
                        self._layer_sliders[layer.name],
                        self._layer_colormap_dropdowns[layer.name],
                    ]))

        controls_list = []

        # Tile controls
        if self._tile_source_dropdown is not None:
            controls_list.append(ipywidgets.HTML("<b>Background Tiles</b>"))
            controls_list.append(ipywidgets.HBox([
                self._tile_visibility_checkbox, self._tile_source_dropdown
            ]))

        # Layer controls
        if layer_widgets:
            controls_list.append(ipywidgets.HTML("<b>Data Layers</b>"))
            controls_list.append(ipywidgets.VBox(layer_widgets))
            controls_list.append(p.widget("colorbar_layer"))

        # Navigation controls
        controls_list.append(ipywidgets.HTML("<b>Navigation</b>"))
        controls_list.append(ipywidgets.HBox([
            p.widget("btn_zoom_fit"), p.widget("btn_zoom_track"),
            p.widget("btn_zoom_wci"), p.widget("btn_refresh_tracks"),
        ]))
        controls_list.append(ipywidgets.HBox([
            p.widget("auto_update"), p.widget("auto_center_wci"),
        ]))
        controls_list.append(p.widget("lbl_coords"))

        self._controls_box = ipywidgets.VBox(controls_list)

        self.layout = ipywidgets.VBox([
            ipywidgets.HBox([self.graphics]),
            self._controls_box,
            self._track_legend,
            self.output,
        ])

    # =====================================================================
    # Display
    # =====================================================================

    def show(self) -> None:
        """Display the viewer widget."""
        self.core.create_colorbar()
        display(self.layout)
        self.core.zoom_to_fit()

    def _request_draw(self) -> None:
        if hasattr(self.graphics, 'request_draw'):
            self.graphics.request_draw()

    def _report_error(self, msg: str) -> None:
        with self.output:
            print(msg)

    # =====================================================================
    # Public API (forwarded to core)
    # =====================================================================

    def set_layer_colormap(self, layer_name: str, colormap: str) -> "MapViewerJupyter":
        self.core.set_layer_colormap(layer_name, colormap)
        if layer_name in self._layer_colormap_dropdowns:
            self._layer_colormap_dropdowns[layer_name].value = colormap
        return self

    def set_layer_opacity(self, layer_name: str, opacity: float) -> "MapViewerJupyter":
        self.core.set_layer_opacity(layer_name, opacity)
        if layer_name in self._layer_sliders:
            self._layer_sliders[layer_name].value = opacity
        return self

    def set_layer_range(self, layer_name: str, vmin: float, vmax: float) -> "MapViewerJupyter":
        self.core.set_layer_range(layer_name, vmin, vmax)
        return self

    def set_layer_blend_mode(self, layer_name: str, blend_mode: str) -> "MapViewerJupyter":
        self.core.set_layer_blend_mode(layer_name, blend_mode)
        return self

    def set_tile_source(self, source_name: str) -> "MapViewerJupyter":
        self.core.change_tile_source(source_name)
        if self._tile_source_dropdown is not None:
            self._tile_source_dropdown.value = source_name
        return self

    def set_tile_visible(self, visible: bool) -> "MapViewerJupyter":
        self.core.tile_visible = visible
        if self._tile_visibility_checkbox is not None:
            self._tile_visibility_checkbox.value = visible
        return self

    @property
    def tile_builder(self):
        return self.core.tile_builder

    @tile_builder.setter
    def tile_builder(self, builder):
        self.core.tile_builder = builder

    def list_tile_sources(self) -> List[str]:
        return self.core.list_tile_sources()

    def add_geotiff(self, path: str, name: Optional[str] = None, band: int = 1, **kwargs) -> "MapViewerJupyter":
        self.core.add_geotiff(path, name=name, band=band, **kwargs)
        self._rebuild_layer_controls()
        return self

    def add_layer(self, backend: Any, name: Optional[str] = None,
                  visible: bool = True, z_order: Optional[int] = None) -> "MapViewerJupyter":
        self.core.add_layer(backend, name=name, visible=visible, z_order=z_order)
        self._rebuild_layer_controls()
        return self

    def add_track(self, latitudes, longitudes, name: str = "Track",
                  color: Optional[str] = None, line_width: float = 2.0,
                  is_active: bool = False, slot_idx: Optional[int] = None) -> None:
        self.core.add_track(latitudes, longitudes, name, color, line_width, is_active, slot_idx)

    def add_overview_track(self, overview, name: str = "Overview",
                           color: Optional[str] = None, **kwargs) -> None:
        self.core.add_overview_track(overview, name, color=color, **kwargs)

    def set_active_track(self, name: str) -> None:
        self.core.set_active_track(name)

    def clear_tracks(self) -> None:
        self.core.clear_tracks()

    def set_track_color(self, name: str, color: str) -> None:
        self.core.set_track_color(name, color)

    def add_markers(self, latitudes, longitudes, name: str = "markers", **kwargs):
        return self.core.add_markers(latitudes, longitudes, name, **kwargs)

    def add_markers_tuples(self, positions, name: str = "markers", **kwargs):
        return self.core.add_markers_tuples(positions, name, **kwargs)

    def remove_markers(self, name: str) -> None:
        self.core.remove_markers(name)

    def clear_markers(self) -> None:
        self.core.clear_markers()

    def update_ping_position(self, lat: float, lon: float) -> None:
        self.core.update_ping_position(lat, lon)

    def zoom_to_fit(self) -> None:
        self.core.zoom_to_fit()

    def zoom_to_track(self) -> None:
        self.core.zoom_to_track()

    def zoom_to_position(self, lat: float, lon: float, radius_deg: float = 0.01) -> None:
        self.core.zoom_to_position(lat, lon, radius_deg)

    def pan_to_wci_position(self) -> None:
        self.core.pan_to_wci_position()

    def pan_to_position(self, lat: float, lon: float) -> None:
        self.core.pan_to_position(lat, lon)

    def is_position_near_edge(self, lat: float, lon: float, edge_fraction: float = 0.2) -> bool:
        return self.core.is_position_near_edge(lat, lon, edge_fraction)

    def pan_to_position_if_near_edge(self, lat: float, lon: float, edge_fraction: float = 0.2) -> None:
        self.core.pan_to_position_if_near_edge(lat, lon, edge_fraction)

    def connect_echogram_viewer(self, echogram_viewer) -> None:
        self.core.connect_echogram_viewer(echogram_viewer)

    def connect_wci_viewer(self, wci_viewer) -> None:
        self.core.connect_wci_viewer(wci_viewer)

    def refresh_tracks(self) -> None:
        self.core.refresh_tracks()

    def register_click_callback(self, callback: Callable[[float, float], None]) -> None:
        self.core.register_click_callback(callback)

    def register_view_change_callback(self, callback: Callable) -> None:
        self.core.register_view_change_callback(callback)

    # =====================================================================
    # Cleanup
    # =====================================================================

    def cleanup(self) -> None:
        self._cancel_pending_load()
        self._cancel_tile_load()
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)
        try:
            self._tile_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._tile_executor.shutdown(wait=False)

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
