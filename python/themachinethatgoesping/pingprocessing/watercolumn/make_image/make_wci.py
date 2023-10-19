import numpy as np

from typing import Tuple

import themachinethatgoesping.echosounders as es
import themachinethatgoesping.algorithms.geoprocessing as gp

import themachinethatgoesping.pingprocessing.watercolumn.helper.make_image_helper as mi_hlp

def make_wci(
    ping: es.filetemplates.I_Ping,
    horizontal_res: int, 
    from_bottom_xyz: bool = True,
    mp_cores: int = True) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    
    """Create a water column image from a ping.

    This function creates a water column image from a ping object using the
    get_bottom_directions_bottom or get_bottom_directions_wci function from the
    mi_hlp module, depending on the value of the from_bottom_xyz parameter.

    Note: this function is an approximation (for performance reasons). As such it:
    - uses nearest neighbor instead of real interpolation
    - uses backtracing instead of raytracing and assumes a constant sound velocity profile

    Parameters
    ----------
    ping : es.filetemplates.I_Ping
        The ping object to create the image from.
    horizontal_res : int
        The number of horizontal pixels in the resulting image.
    from_bottom_xyz : bool, optional
        If True, attempt to correct the beam launch angles using the ping's bottom detection (if existent).
        If False, do not attempt to correct the beam launch angles.
        Defaults to True.
    mp_cores : int, optional
        The number of cores to use for parallel processing.
        Defaults to 1.

    Returns
    -------
    Tuple[np.ndarray, Tuple[float, float, float, float]]
        A tuple containing the resulting water column image and the extent of the image (in meters, for plotting).
    """

    # get sensor configuration
    sc = ping.get_sensor_configuration()
    pingoff = sc.get_target("Transducer")
    posoff = sc.get_position_source()
    geolocation = ping.get_geolocation()

    # get bottom positions/directions/sample numbers
    if from_bottom_xyz:
        xyz, bottom_directions, bottom_direction_sample_numbers = mi_hlp.get_bottom_directions_bottom(ping)
    else:
        xyz, bottom_directions, bottom_direction_sample_numbers = mi_hlp.get_bottom_directions_wci(ping)

    # compute limits of the created image
    hmin = 1.1 * np.nanmin(xyz.y)
    hmax = 1.1 * np.nanmax(xyz.y)
    vmin = geolocation.z
    vmax = np.nanmax(xyz.z)
    vmax += (vmax-vmin)*0.1

    res = (hmax-hmin)/horizontal_res   

    # build array with backtraced positions (beam angle, range from transducer)
    y = np.arange(hmin,hmax,res)
    z = np.arange(vmin,vmax,res)

    bt = gp.backtracers.BTConstantSVP(geolocation, pingoff.x, pingoff.y)

    sd_grid = bt.backtrace_image(y,z, mp_cores = mp_cores)

    # lookup beam/sample numbers for each pixel
    maxsn = ping.watercolumn.get_number_of_samples_per_beam()-1
    bisi = bt.lookup_indices(bottom_directions,bottom_direction_sample_numbers, maxsn,sd_grid)
    bi = np.array(bisi.beam_numbers)
    si = np.array(bisi.sample_numbers )

    # get amplitudes for each pixel
    wci = ping.watercolumn.get_amplitudes()[bi,si]
    wci [bi==np.nanmin(bi)] = np.nan
    wci [si==0] = np.nan
    wci [bi==np.nanmax(bi)] = np.nan

    # compute the extent
    extent = [
        np.min(y)-res*0.5,np.max(y)+res*0.5,
        np.max(z)+res*0.5,np.min(z)-res*0.5
    ]

    # return the resulting water column image and the extent of the image
    return wci, tuple(extent)

def make_wci_dual_head(
    ping1: es.filetemplates.I_Ping, 
    ping2: es.filetemplates.I_Ping, 
    horizontal_res: int, 
    from_bottom_xyz: bool = False,
    mp_cores: int = 1) -> Tuple[np.ndarray, Tuple[float, float, float, float]]: 
    """
    Create a water column image from two pings.

    Note: this function is an approximation (for performance reasons). As such it:
    - uses nearest neighbor instead of real interpolation
    - uses backtracing instead of raytracing and assumes a constant sound velocity profile
    - does not plot samples that overlap

    Parameters
    ----------
    ping1 : es.filetemplates.I_Ping
        Left ping to create image from.
    ping2 : es.filetemplates.I_Ping
        Right ping to create image from.
    horizontal_res : int
        Number of horizontal pixels in the image.
    from_bottom_xyz : bool, optional
        Attempt to correct the beam launch angles using the pings' bottom detection (if available).
        Default is False.
    mp_cores : int, optional
        Number of cores to use for parallel processing. Default is 1.

    Returns
    -------
    Tuple[np.ndarray, Tuple[float, float, float, float]]
        A tuple containing the water column image as a numpy array and the extent of the image for plotting.
        The extent is a tuple of four floats: (xmin, xmax, ymax, ymin).
    """

    # Get sensor configurations
    sc1 = ping1.get_sensor_configuration()
    sc2 = ping2.get_sensor_configuration()
    pingoff1 = sc1.get_target("Transducer")
    pingoff2 = sc2.get_target("Transducer")
    geolocation1 = ping1.get_geolocation()
    geolocation2 = ping2.get_geolocation()

    # Get bottom positions/directions/sample numbers
    if from_bottom_xyz:
        xyz1, bottom_directions1, bottom_direction_sample_numbers1 = mi_hlp.get_bottom_directions_bottom(ping1)
        xyz2, bottom_directions2, bottom_direction_sample_numbers2 = mi_hlp.get_bottom_directions_bottom(ping2)
    else:
        xyz1, bottom_directions1, bottom_direction_sample_numbers1 = mi_hlp.get_bottom_directions_wci(ping1)
        xyz2, bottom_directions2, bottom_direction_sample_numbers2 = mi_hlp.get_bottom_directions_wci(ping2)

    # Compute limits of the created image
    hmin = 1.1 * np.nanmin(xyz1.y)
    hmax = 1.1 * np.nanmax(xyz2.y)
    vmin = np.nanmin([geolocation1.z,geolocation2.z])
    vmax = np.nanmax([np.nanmax(xyz1.z),np.nanmax(xyz2.z)])
    vmax += (vmax-vmin)*0.1

    res = (hmax-hmin)/horizontal_res   

    # Build array with backtraced positions (beam angle, range from transducer)
    y = np.arange(hmin,hmax,res)
    z = np.arange(vmin,vmax,res)
        
    bt1 = gp.backtracers.BTConstantSVP(geolocation1, pingoff1.x, pingoff1.y)
    bt2 = gp.backtracers.BTConstantSVP(geolocation2, pingoff2.x, pingoff2.y)

    sd_grid1 = bt1.backtrace_image(y[y<=0],z, mp_cores = mp_cores)
    sd_grid2 = bt2.backtrace_image(y[y>0],z, mp_cores = mp_cores)
        
    # Lookup beam/sample numbers for each pixel
    maxsn1 = ping1.watercolumn.get_number_of_samples_per_beam()-1
    maxsn2 = ping2.watercolumn.get_number_of_samples_per_beam()-1
    bisi1 = bt1.lookup_indices(bottom_directions1,bottom_direction_sample_numbers1, maxsn1,sd_grid1)
    bisi2 = bt2.lookup_indices(bottom_directions2,bottom_direction_sample_numbers2, maxsn2,sd_grid2)
    bi1 = np.array(bisi1.beam_numbers)
    bi2 = np.array(bisi2.beam_numbers)
    si1 = np.array(bisi1.sample_numbers )
    si2 = np.array(bisi2.sample_numbers )

    # Get amplitudes for each pixel
    # TODO: speed up by only reading the beams that are necessary
    wci1 = ping1.watercolumn.get_amplitudes()[bi1,si1]
    wci1 [bi1==np.nanmin(bi1)] = np.nan
    wci1 [si1==0] = np.nan

    wci2 = ping2.watercolumn.get_amplitudes()[bi2,si2]
    wci2 [bi2==np.nanmax(bi2)] = np.nan
    wci2 [si2==0] = np.nan

    # Compute the extent
    extent = [
        np.min(y)-res*0.5,np.max(y)+res*0.5,
        np.max(z)+res*0.5,np.min(z)-res*0.5
    ]

    # Return
    return np.append(wci1,wci2, axis=0), tuple(extent)

def make_wci_stack(
    pings: list,
    horizontal_res: int, 
    linear_mean: bool = True,
    from_bottom_xyz: bool = False,
    progress_bar = None,
    mp_cores: int = 1):
    
    """
    Create a water column image from a list of pings.

    Note: this function is an approximation (for performance reasons). As such it:
    - uses nearest neighbor instead of real interpolation
    - uses backtracing instead of raytracing and assumes a constant sound velocity profile
    - does not plot samples that overlap

    Parameters
    ----------
    pings : list(es.filetemplates.I_Ping)
        List of pings to stack.
    horizontal_res : int
        Number of horizontal pixels.
    linear_mean : bool, optional
        Use linear mean instead of geometric mean (mean of dB values), by default True.
    progress_bar : progress_bar, optional
        tqdm style progress bar to use, by default None.
    from_bottom_xyz : bool, optional
        Attempt to correct the beam launch angles using the ping's bottom detection (if existent), by default True.
    mp_cores : int, optional
        Number of cores to use for parallel processing, by default 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        water column image, extent of the image (in m, for plotting)
    """
    
    # get sensor configurations, geolocations and ping offsets
    scs = [ping.get_sensor_configuration() for ping in pings]
    pingoffs = [sc.get_target("Transducer") for sc in scs]
    geolocations = [ping.get_geolocation() for ping in pings]
    
    # get bottom positions/directions/sample numbers
    xyzs = []
    bottom_directions = []
    bottom_direction_sample_numbers = []
    hmin = np.nan
    hmax = np.nan
    vmin = np.nan
    vmax = np.nan
    
    for ping in pings:
        if from_bottom_xyz:            
            xyz, bd, bdsn = mi_hlp.get_bottom_directions_bottom(ping)
        else:
            xyz, bd, bdsn = mi_hlp.get_bottom_directions_wci(ping)
            
        xyzs.append(xyz)
        bottom_directions.append(bd)
        bottom_direction_sample_numbers.append(bdsn)

        # compute limits of the create image
        hmin = np.nanmin([hmin,np.nanmin(xyz.y)])
        hmax = np.nanmax([hmax,np.nanmax(xyz.y)])
        vmax = np.nanmax([vmax,np.nanmax(xyz.z)])
      
    hmin *= 1.1
    hmax *= 1.1
    vmin = np.nanmin([g.z for g in geolocations])  
    vmax += (vmax-vmin)*0.1

    res = (hmax-hmin)/horizontal_res   
    
    # build array with backtraced positions (beam angle, range from transducer)
    y = np.arange(hmin,hmax,res)
    z = np.arange(vmin,vmax,res)
    
    WCI=None
    NUM=None
    
    # initialize progress bar
    if progress_bar is None:
        progress_bar = pings
    else:
        progress_bar = progress_bar(pings)
        
    # loop through each ping
    for pn,ping in enumerate(progress_bar):        
        # create backtracer object
        bt = gp.backtracers.BTConstantSVP(geolocations[pn], pingoffs[pn].x, pingoffs[pn].y)
    
        # backtrace image
        sd_grid = bt.backtrace_image(y,z)
        
        # lookup beam/sample numbers for each pixel
        maxsn = ping.watercolumn.get_number_of_samples_per_beam()-1
        bisi = bt.lookup_indices(bottom_directions[pn],bottom_direction_sample_numbers[pn], maxsn, sd_grid)
        bi = np.array(bisi.beam_numbers)
        si = np.array(bisi.sample_numbers )
        
        # TODO: this should be done in the backtracer
        use = np.ones_like(bi)
        use[bi==np.nanmin(bi)] = 0
        use[si==0] = 0
        use[bi==np.nanmax(bi)] = 0

        # get amplitudes for each pixel
        # TODO: speed up by only reading the beams that are necessary
        wci = np.empty_like(bi,dtype=np.float32)
        wci.fill(np.nan)
        wci[use==1] = ping.watercolumn.get_amplitudes()[bi[use==1],si[use==1]]
                
        # apply linear mean if specified
        if linear_mean:
            wci[use==1] = np.power(10,wci[use==1]*0.1)
        
        # initialize WCI and NUM arrays
        if WCI is None:
            WCI = np.empty_like(wci)
            WCI.fill(np.nan)
            NUM = np.zeros_like(wci)
            
        # accumulate WCI and NUM arrays
        WCI[use==1] = np.nansum([WCI[use==1],wci[use==1]],axis=0)
        NUM[use==1] += 1
    
    # compute the extent
    extent = [
        hmin-res*0.5,hmax+res*0.5,
        vmax+res*0.5,vmin-res*0.5
    ]
    
    # compute the final WCI array
    WCI = WCI/NUM
    
    # apply logarithmic scaling if specified
    if linear_mean:
        WCI = 10*np.log10(WCI)
    
    # return the WCI array and extent
    return WCI, extent