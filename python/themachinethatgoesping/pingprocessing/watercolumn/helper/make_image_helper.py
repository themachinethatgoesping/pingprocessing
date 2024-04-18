import numpy as np
import themachinethatgoesping.echosounders as es
import themachinethatgoesping.algorithms.geoprocessing as gp

def get_bottom_directions_wci(
    ping: es.filetemplates.I_Ping, 
    selection: es.pingtools.BeamSelection = None) -> (gp.datastructures.XYZ_1, gp.datastructures.SampleDirectionsRange_1, np.ndarray):
    """
    Retrieve bottom positions/directions/sample numbers from a water column ping.
    Note: this function is an approximation (for performance reasons). As such, it 
    assumes a constant sound velocity profile.

    Parameters
    ----------
    ping : es.filetemplates.I_Ping
        Ping to retrieve bottom positions/directions/sample numbers from.
    selection : es.pingtools.BeamSelection, optional
        A beam selection to retrieve the directions from, by default None.

    Returns
    -------
    Tuple[gp.datastructures.XYZ_1, gp.datastructures.SampleDirectionsRange_1, np.ndarray]
        A tuple containing the bottom positions, directions, and sample numbers.
    """
    
    # Create select all selection if necessary.
    if selection is None:
        selection = ping.watercolumn.get_beam_sample_selection_all()
        
    # Get sensor configuration.
    sc = ping.get_sensor_configuration()
    pingoff = sc.get_target("Transducer")
    posoff = sc.get_position_source()
    geolocation = ping.get_geolocation()

    if ping.watercolumn.has_bottom_range_samples():
        bottom_direction_sample_numbers = ping.watercolumn.get_bottom_range_samples(selection)

        # If the bottom direction sample numbers are 0, set them to the number of samples per beam.
        bottom_direction_sample_numbers_is_0 = np.nonzero(bottom_direction_sample_numbers == 0)
        if len(bottom_direction_sample_numbers_is_0) > 0:
            bottom_direction_sample_numbers[bottom_direction_sample_numbers_is_0] = ping.watercolumn.get_number_of_samples_per_beam(selection)[bottom_direction_sample_numbers_is_0]
    else:
        bottom_direction_sample_numbers = ping.watercolumn.get_number_of_samples_per_beam(selection)
    
    bottomdirections = gp.datastructures.SampleDirectionsTime_1([selection.get_number_of_beams()])
    bottomdirections.crosstrack_angle = ping.watercolumn.get_beam_crosstrack_angles(selection) - geolocation.roll
    bottomdirections.alongtrack_angle = ping.watercolumn.get_beam_alongtrack_angles(selection)
    bottomdirections.two_way_travel_time = bottom_direction_sample_numbers * ping.watercolumn.get_sample_interval()

    # TODO: Get sound velocity from SVP.
    c = ping.watercolumn.get_sound_speed_at_transducer()

    # Raytrace to bottom assuming constant sound velocity profile.
    rt = gp.raytracers.RTConstantSVP(geolocation, c)
    xyz = rt.trace_points(bottomdirections)
    bottom_directions = gp.datastructures.SampleDirectionsRange_1([ping.watercolumn.get_number_of_beams()])
    bottom_directions.crosstrack_angle = bottomdirections.crosstrack_angle
    bottom_directions.alongtrack_angle = bottomdirections.alongtrack_angle
    bottom_directions.range = xyz.true_range    
    
    return xyz, bottom_directions, bottom_direction_sample_numbers

def get_bottom_directions_bottom(ping: es.filetemplates.I_Ping) -> (gp.datastructures.XYZ_1, gp.datastructures.SampleDirectionsRange_1, np.ndarray):
    """
    Retrieve bottom positions/directions/sample numbers from a bottom ping.

    Parameters
    ----------
    ping : es.filetemplates.I_Ping
        Ping to retrieve bottom positions/directions/sample numbers from.

    Returns
    -------
    Tuple[gp.datastructures.XYZ_1, gp.datastructures.SampleDirectionsRange_1, np.ndarray]
        A tuple containing the bottom positions, directions, and sample numbers.
    """
    sc = ping.get_sensor_configuration()
    pingoff = sc.get_target("Transducer")
    posoff = sc.get_position_source()
    geolocation = ping.get_geolocation()
    
    xyz = ping.bottom.get_xyz()
    xyz.x = xyz.x + posoff.x
    xyz.y = xyz.y + posoff.y
    xyz.z = xyz.z + geolocation.z
    
    bt = gp.backtracers.BTConstantSVP(geolocation, pingoff.x, pingoff.y)
    bottom_directions = bt.backtrace_points(xyz)
    bottom_direction_sample_numbers = ping.watercolumn.get_bottom_range_samples()
    
    return xyz, bottom_directions, bottom_direction_sample_numbers


