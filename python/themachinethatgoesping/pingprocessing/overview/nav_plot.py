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
    if close_plots:
        plt.close(name)
    fig = plt.figure(name)
    fig.suptitle = name

    ax = fig.subplots()

    # initialize axis
    ax.grid(True, linestyle="--", color="gray", alpha=0.5)
    ax.set_title(name)
    ax.set_aspect(aspect)

    if background_image_path:
        background_map = rio.open(background_image_path)
        _kwargs = {"cmap": "Greys_r"}
        _kwargs.update(kwargs)


        if dst_crs is None or dst_crs == background_map.crs:
            rioplt.show(background_map, ax=ax, **_kwargs)
            dst_crs = background_map.crs
        else:
            with reproject_raster(background_map, dst_crs) as reprojected_map:            
                rioplt.show(reprojected_map, ax=ax, **_kwargs)
                dst_crs = reprojected_map.crs

        if dst_crs.is_geographic:
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        elif dst_crs.is_projected:
            ax.set_xlabel("easting")
            ax.set_ylabel("northing")

    if return_crs:
        return fig, ax, dst_crs
    return fig, ax


def plot_latlon(lat, lon, ax, label="survey", annotate=True, max_points=100000, **kwargs):
    """
    Plot latitude and longitude coordinates on a given axis.

    Parameters:
        lat (list): List of latitude coordinates.
        lon (list): List of longitude coordinates.
        ax (matplotlib.axes.Axes): The axis on which to plot the coordinates.
        label (str, optional): Name of the survey. Defaults to 'survey'.
        annotate (bool, optional): Whether to annotate the plot with the survey name. Defaults to True.
        max_points (int, optional): Maximum number of points to plot. Defaults to 100000.
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

    # Plot the coordinates
    ax.plot(plot_lon, plot_lat, label=label, linewidth=0.5, marker="o", markersize=2, markevery=1, **kwargs)

    # Add label at the first point
    if annotate:
        ax.annotate(f"Start {label}", xy=(plot_lon[0], plot_lat[0]), xytext=(plot_lon[0], plot_lat[0]))
