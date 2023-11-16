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

# def plot_navigation_pings(
#     pings, 
#     progress = False):

#     overview = get_ping_overview(pings, progress)

#     if isinstance(overview, dict):

#     it = get_progress_iterator(pings, progress, desc = "Plot navigation")

#     plot_data = defaultdict(list)

#     for ping in it:
#         g = ping.get_geolocation()
#         plot_data['latitude'].append(g.latitude)
#         plot_data['longitude'].append(g.longitude)

#     return plot_data


def plot_latlon(
    lat,
    lon,
    ax,
    survey_name = 'survey',
    annotate = True,
    max_points = 100000, 
    **kwargs):

    if len(lat) > max_points:
        plot_lat = lat[::int(len(lat)//max_points)]
        plot_lon = lat[::int(len(lat)//max_points)]
    else:
        plot_lat = lat
        plot_lon = lon

    ax.plot(plot_lat, plot_lon, label=survey_name, linewidth=0.5, marker='o', markersize=2, markevery=1, **kwargs)

    # Add arrows with labels that indicate the beginning and the end of the survey
    ax.annotate(f'Start {survey_name}', xy=(lon[0], lat[0]), xytext=(lon[0]+0.01, lat[0]+0.01),
                arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10)
    ax.annotate(f'Start {survey_name}', xy=(lon[-1], lat[-1]), xytext=(lon[-1]-0.01, lat[-1]-0.01),
                arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10)
