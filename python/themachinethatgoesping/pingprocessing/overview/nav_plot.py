from collections import defaultdict

import numpy as np

from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from themachinethatgoesping.pingprocessing.overview import get_ping_overview

from matplotlib import pyplot as plt



def create_figure(    
    name,     
    aspect = 'equal', 
    close_plots = True):

    if close_plots:
        plt.close(name)
    fig = plt.figure(name)
    fig.suptitle = name

    ax = fig.subplots()


    # initialze axis
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(name)
    ax.set_aspect(aspect)

    return fig, ax


def plot_latlon(
    lat,
    lon,
    ax,
    survey_name='survey',
    annotate=True,
    max_points=100000,
    **kwargs):
    """
    Plot latitude and longitude coordinates on a given axis.

    Parameters:
        lat (list): List of latitude coordinates.
        lon (list): List of longitude coordinates.
        ax (matplotlib.axes.Axes): The axis on which to plot the coordinates.
        survey_name (str, optional): Name of the survey. Defaults to 'survey'.
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
    ax.plot(plot_lon, plot_lat, label=survey_name, linewidth=0.5, marker='o', markersize=2, markevery=1, **kwargs)

    # Add label at the first point
    if annotate:
        ax.annotate(f'Start {survey_name}', xy=(plot_lon[0], plot_lat[0]), xytext=(plot_lon[0], plot_lat[0]))