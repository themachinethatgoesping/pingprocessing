from collections import defaultdict
from typing import Tuple

import numpy as np

from matplotlib import pyplot as plt

import rasterio.plot as rioplt
import rasterio.warp as riowarp
import rasterio as rio
from rasterio.io import MemoryFile
from contextlib import contextmanager  

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

#Adapted from: https://gis.stackexchange.com/questions/443822/how-do-you-reproject-a-raster-using-rasterio-in-memory
@contextmanager  
def reproject_raster(src, dst_crs):
    
    src_crs = src.crs
    transform, width, height = riowarp.calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()

    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height})

    with MemoryFile() as memfile:
        with memfile.open(**kwargs) as dst:
            for i in range(1, src.count + 1):
                riowarp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=riowarp.Resampling.nearest)
        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return as we're a contextmanager


def create_figure(
    name: str, 
    aspect: str = "equal", 
    close_plots: bool = True, 
    background_image_path: str = None, 
    background_colorbar_label = None,
    colorbar_orientation = 'vertical',
    add_grid = True,
    dst_crs = 'EPSG:4326', 
    return_crs = False,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with a given name and aspect ratio.

    Parameters:
        name (str): The name of the figure.
        aspect (str, optional): The aspect ratio of the figure. Defaults to "equal".
        close_plots (bool, optional): Whether to close existing plots. Defaults to True.
        background_image_path (str, optional): Path to the background image that can be opened
                                               with rastio and contains navigation reference(e.g. geotif).
                                               Defaults to None.
        **kwargs: Additional keyword arguments for the background_image (e.g. cmap).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The created figure and axes.
    """
    global ax,fig,mapable
    if close_plots:
        plt.close(name)
    fig = plt.figure(name)
    fig.suptitle = name

    ax = fig.subplots()

    # initialize axis
    if add_grid:
        ax.grid(True, linestyle="--", color="gray", alpha=0.5)
    ax.set_title(name)
    ax.set_aspect(aspect)

    if background_image_path:
        background_map = rio.open(background_image_path)
        _kwargs = {"cmap": "Greys_r", "adjust": False}
        _kwargs.update(kwargs)


        if dst_crs is None or dst_crs == background_map.crs:
            rioplt.show(background_map, ax=ax, **_kwargs)
            dst_crs = background_map.crs
        else:
            with reproject_raster(background_map, dst_crs) as reprojected_map:            
                rioplt.show(reprojected_map, ax=ax, **_kwargs)
                dst_crs = reprojected_map.crs

        if background_colorbar_label:
            fig.colorbar(ax.get_images()[-1], ax=ax, label=background_colorbar_label, shrink=0.7, orientation=colorbar_orientation)

        if dst_crs.is_geographic:
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        elif dst_crs.is_projected:
            ax.set_xlabel("easting")
            ax.set_ylabel("northing")

    if return_crs:
        return fig, ax, dst_crs
    return fig, ax


def _format_dd(degrees, decimal_places=4):
    """Format decimal degrees (DD)."""
    if not np.isfinite(degrees):
        return ""
    return f"{degrees:.{decimal_places}f}°"


def _format_ddm(degrees, decimal_places=2):
    """Format degrees decimal minutes (DDM)."""
    if not np.isfinite(degrees):
        return ""
    sign = "-" if degrees < 0 else ""
    degrees = abs(degrees)
    d = int(degrees)
    m = (degrees - d) * 60
    return f"{sign}{d}°{m:0{decimal_places + 3}.{decimal_places}f}'"


def _format_dms(degrees, decimal_places=1):
    """Format degrees minutes decimal seconds (DMS)."""
    if not np.isfinite(degrees):
        return ""
    sign = "-" if degrees < 0 else ""
    degrees = abs(degrees)
    d = int(degrees)
    m_full = (degrees - d) * 60
    m = int(m_full)
    s = (m_full - m) * 60
    return f"{sign}{d}°{m:02d}'{s:0{decimal_places + 3}.{decimal_places}f}\""


# Nice tick intervals in degrees for each coordinate format
_NICE_INTERVALS_DD = [
    0.0001, 0.0002, 0.0005,
    0.001, 0.002, 0.005,
    0.01, 0.02, 0.05,
    0.1, 0.2, 0.5,
    1, 2, 5, 10, 20, 45, 90,
]

_NICE_INTERVALS_DDM = [
    1/60, 2/60, 5/60, 10/60, 15/60, 30/60,   # 1', 2', 5', 10', 15', 30'
    1, 2, 5, 10, 20, 45, 90,
]

_NICE_INTERVALS_DMS = [
    1/3600, 2/3600, 5/3600, 10/3600, 15/3600, 30/3600,  # 1", 2", 5", 10", 15", 30"
    1/60, 2/60, 5/60, 10/60, 15/60, 30/60,               # 1', 2', 5', 10', 15', 30'
    1, 2, 5, 10, 20, 45, 90,
]


def _get_nice_intervals(coord_format):
    """Return the list of nice tick intervals for a coordinate format."""
    coord_format = coord_format.upper()
    if coord_format == "DD":
        return _NICE_INTERVALS_DD
    elif coord_format == "DDM":
        return _NICE_INTERVALS_DDM
    elif coord_format == "DMS":
        return _NICE_INTERVALS_DMS
    else:
        return _NICE_INTERVALS_DD


def _pick_nice_interval(data_range, target_ticks=6, coord_format="DD"):
    """Choose a nice tick interval for the given data range and format.

    Parameters:
        data_range (float): The span of the axis in degrees.
        target_ticks (int): Desired approximate number of ticks.
        coord_format (str): ``'DD'``, ``'DDM'``, or ``'DMS'``.

    Returns:
        float: The chosen interval in degrees.
    """
    if data_range <= 0:
        return 1.0
    ideal = data_range / target_ticks
    intervals = _get_nice_intervals(coord_format)
    # Pick the interval closest to the ideal
    best = min(intervals, key=lambda iv: abs(iv - ideal))
    return best


from matplotlib.ticker import Locator


class _CoordLocator(Locator):
    """Matplotlib tick locator that places ticks at nice
    coordinate boundaries (full degrees, minutes, or seconds).

    Parameters:
        coord_format (str): ``'DD'``, ``'DDM'``, or ``'DMS'``.
        target_ticks (int): Approximate number of ticks desired.
    """

    def __init__(self, coord_format="DD", target_ticks=6):
        super().__init__()
        self.coord_format = coord_format
        self.target_ticks = target_ticks

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        import math
        data_range = vmax - vmin
        if not np.isfinite(data_range) or data_range <= 0:
            return np.array([])
        interval = _pick_nice_interval(data_range, self.target_ticks, self.coord_format)
        start = math.ceil(vmin / interval) * interval
        ticks = []
        v = start
        while v <= vmax + interval * 1e-9:
            ticks.append(round(v, 12))
            v += interval
        return np.array(ticks)


def _apply_coord_format(ax, coord_format, decimal_places=None):
    """Apply coordinate format and nice tick locator to both axes.

    Parameters:
        ax: matplotlib axes.
        coord_format (str): ``'DD'``, ``'DDM'``, or ``'DMS'``.
        decimal_places (int, optional): Override default decimal places.
    """
    from matplotlib.ticker import FuncFormatter

    fmt = _get_coord_formatter(coord_format, decimal_places)
    formatter = FuncFormatter(lambda v, _pos: fmt(v))

    for axis in (ax.xaxis, ax.yaxis):
        locator = _CoordLocator(coord_format)
        axis.set_major_locator(locator)
        axis.set_major_formatter(formatter)


def _get_coord_formatter(coord_format, decimal_places=None):
    """Return a formatting function for the given coordinate format.

    Parameters:
        coord_format (str): One of ``'DD'``, ``'DDM'``, ``'DMS'``.
        decimal_places (int, optional): Override the default decimal places
            for the chosen format.

    Returns:
        callable: A function ``(degrees) -> str``.
    """
    coord_format = coord_format.upper()
    if coord_format == "DD":
        dp = decimal_places if decimal_places is not None else 4
        return lambda v: _format_dd(v, dp)
    elif coord_format == "DDM":
        dp = decimal_places if decimal_places is not None else 2
        return lambda v: _format_ddm(v, dp)
    elif coord_format == "DMS":
        dp = decimal_places if decimal_places is not None else 1
        return lambda v: _format_dms(v, dp)
    else:
        raise ValueError(f"Unknown coord_format '{coord_format}'. Use 'DD', 'DDM', or 'DMS'.")


def plot_latlon(lat, lon, ax, label="survey", annotate=True, max_points=100000,
                coord_format=None, decimal_places=None, **kwargs):
    """
    Plot latitude and longitude coordinates on a given axis.

    Parameters:
        lat (list): List of latitude coordinates.
        lon (list): List of longitude coordinates.
        ax (matplotlib.axes.Axes): The axis on which to plot the coordinates.
        label (str, optional): Name of the survey. Defaults to 'survey'.
        annotate (bool, optional): Whether to annotate the plot with the survey name. Defaults to True.
        max_points (int, optional): Maximum number of points to plot. Defaults to 100000.
        coord_format (str, optional): Coordinate display format for axis ticks
            and annotations. One of ``'DD'`` (decimal degrees),
            ``'DDM'`` (degrees decimal minutes), or ``'DMS'`` (degrees
            minutes decimal seconds). ``None`` leaves the default numeric
            tick labels unchanged. Tick positions are automatically
            adjusted to fall on nice boundaries (full minutes, seconds,
            etc.).
        decimal_places (int, optional): Number of decimal places for the
            coordinate format. Defaults depend on *coord_format*:
            4 for DD, 2 for DDM, 1 for DMS.
        **kwargs: Additional keyword arguments to be passed to the plot function.

    Returns:
        None
    """

    # Reduce the number of points if necessary
    if len(lat) > max_points:
        plot_lat = np.array(lat)[np.round(np.linspace(0, len(lat) - 1, max_points)).astype(int)]
        plot_lon = np.array(lon)[np.round(np.linspace(0, len(lon) - 1, max_points)).astype(int)]
    else:
        plot_lat = lat
        plot_lon = lon

    _kwargs = {
        "linewidth": 0.5,
        "marker": "o",
        "markersize": 2,
        "markevery": 1
        }
    _kwargs.update(kwargs)

    # Plot the coordinates
    ax.plot(plot_lon, plot_lat, label=label, **_kwargs)

    # Add label at the first point
    if annotate:
        if coord_format is not None:
            fmt = _get_coord_formatter(coord_format, decimal_places)
            annotation = f"Start {label}\n{fmt(plot_lat[0])}, {fmt(plot_lon[0])}"
        else:
            annotation = f"Start {label}"
        ax.annotate(annotation, xy=(plot_lon[0], plot_lat[0]), xytext=(plot_lon[0], plot_lat[0]))

    # Apply coordinate formatters and locators to axis ticks
    if coord_format is not None:
        _apply_coord_format(ax, coord_format, decimal_places)


def set_latlon_axes_labels(ax, src_crs, coord_format="DD", decimal_places=None):
    """Replace projected axis tick labels with lat/lon values.

    After plotting in a projected CRS (e.g. UTM), call this function to
    convert the axis tick labels back to latitude/longitude while keeping
    the spatial scaling of the projection. Tick positions are
    automatically adjusted to fall on nice coordinate boundaries.

    Parameters:
        ax (matplotlib.axes.Axes): The axes whose tick labels should be
            converted.
        src_crs: The CRS of the data currently shown on the axes.
            Anything accepted by ``pyproj.CRS`` (e.g. ``'EPSG:32631'``).
        coord_format (str, optional): Coordinate display format. One of
            ``'DD'`` (decimal degrees), ``'DDM'`` (degrees decimal
            minutes), or ``'DMS'`` (degrees minutes decimal seconds).
            Defaults to ``'DD'``.
        decimal_places (int, optional): Number of decimal places. Defaults
            depend on *coord_format*: 4 for DD, 2 for DDM, 1 for DMS.
    """
    from pyproj import Transformer, CRS
    from matplotlib.ticker import FuncFormatter
    import math

    crs = CRS(src_crs)
    transformer_to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    transformer_from_latlon = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    fmt = _get_coord_formatter(coord_format, decimal_places)

    class _ProjectedCoordLocator(Locator):
        """Locator that places ticks at nice lat/lon boundaries in projected space."""

        def __init__(self, is_x):
            super().__init__()
            self.is_x = is_x

        def __call__(self):
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)

        def tick_values(self, vmin, vmax):
            # Sample multiple points along the axis to get a robust lat/lon range
            n_samples = 20
            samples = np.linspace(vmin, vmax, n_samples)

            if self.is_x:
                ymid_proj = np.mean(ax.get_ylim())
                xmid_proj = np.mean([vmin, vmax])
                lons, lats = transformer_to_latlon.transform(
                    samples, np.full(n_samples, ymid_proj)
                )
                valid = np.isfinite(lons)
                if not np.any(valid):
                    return np.array([])
                deg_min, deg_max = np.min(lons[valid]), np.max(lons[valid])
                # Get the lat/lon midpoint for the perpendicular axis
                _, lat_mid = transformer_to_latlon.transform(xmid_proj, ymid_proj)
            else:
                xmid_proj = np.mean(ax.get_xlim())
                ymid_proj = np.mean([vmin, vmax])
                lons, lats = transformer_to_latlon.transform(
                    np.full(n_samples, xmid_proj), samples
                )
                valid = np.isfinite(lats)
                if not np.any(valid):
                    return np.array([])
                deg_min, deg_max = np.min(lats[valid]), np.max(lats[valid])
                # Get the lat/lon midpoint for the perpendicular axis
                lon_mid, _ = transformer_to_latlon.transform(xmid_proj, ymid_proj)

            if deg_min > deg_max:
                deg_min, deg_max = deg_max, deg_min

            data_range = deg_max - deg_min
            if not np.isfinite(data_range) or data_range <= 0:
                return np.array([])

            interval = _pick_nice_interval(data_range, 6, coord_format)
            start = math.ceil(deg_min / interval) * interval
            ticks = []
            v = start
            while v <= deg_max + interval * 1e-9:
                # Convert back to projected coordinates using lat/lon midpoints
                if self.is_x:
                    proj_x, _ = transformer_from_latlon.transform(v, lat_mid)
                    if np.isfinite(proj_x):
                        ticks.append(proj_x)
                else:
                    _, proj_y = transformer_from_latlon.transform(lon_mid, v)
                    if np.isfinite(proj_y):
                        ticks.append(proj_y)
                v += interval
            return np.array(ticks)

    def _lon_formatter(x, _pos):
        lon, _ = transformer_to_latlon.transform(x, sum(ax.get_ylim()) / 2)
        return fmt(lon)

    def _lat_formatter(y, _pos):
        _, lat = transformer_to_latlon.transform(sum(ax.get_xlim()) / 2, y)
        return fmt(lat)

    ax.xaxis.set_major_locator(_ProjectedCoordLocator(is_x=True))
    ax.yaxis.set_major_locator(_ProjectedCoordLocator(is_x=False))
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_formatter))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
