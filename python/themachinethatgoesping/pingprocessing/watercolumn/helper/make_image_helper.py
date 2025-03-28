import numpy as np
import themachinethatgoesping.navigation as navigation
import themachinethatgoesping.echosounders as echosounders
import themachinethatgoesping.algorithms.geoprocessing as geoprocessing

def get_bottom_directions_wci(
    ping: echosounders.filetemplates.I_Ping, 
    selection: echosounders.pingtools.BeamSelection = None) -> (geoprocessing.datastructures.XYZ_1, geoprocessing.datastructures.SampleDirectionsRange_1, np.ndarray):
    """
    Retrieve bottom positions/directions/sample numbers from a water column ping.
    Note: this function is an approximation (for performance reasons). As such, it 
    assumes a constant sound velocity profile.

    Parameters
    ----------
    ping : echosounders.filetemplates.I_Ping
        Ping to retrieve bottom positions/directions/sample numbers from.
    selection : echosounders.pingtools.BeamSelection, optional
        A beam selection to retrieve the directions from, by default None.

    Returns
    -------
    Tuple[geoprocessing.datastructures.XYZ_1, geoprocessing.datastructures.SampleDirectionsRange_1, np.ndarray]
        A tuple containing the bottom positions, directions, and sample numbers.
    """
    
    # Create select all selection if necessary.
    if selection is None:
        selection = ping.watercolumn.get_beam_sample_selection_all()
        
    # Get sensor configuration.
    sc = ping.get_sensor_configuration()
    try:
        pingoff = sc.get_target("Transducer")
    except:
        print("Warning: No transducer target found in sensor configuration. Using default values.")
        pingoff = navigation.datastructures.PositionalOffsets()
    posoff = sc.get_position_source()
    geolocation = ping.get_geolocation()

    bottom_direction_sample_numbers = np.array(selection.get_last_sample_number_per_beam())
    
    bottomdirections = geoprocessing.datastructures.SampleDirectionsTime_1([selection.get_number_of_beams()])
    bottomdirections.crosstrack_angle = ping.watercolumn.get_beam_crosstrack_angles(selection) - geolocation.roll
    bottomdirections.alongtrack_angle = ping.watercolumn.get_beam_alongtrack_angles(selection)
    bottomdirections.two_way_travel_time = bottom_direction_sample_numbers * ping.watercolumn.get_sample_interval()

    # TODO: Get sound velocity from SVP.
    c = ping.watercolumn.get_sound_speed_at_transducer()

    # Raytrace to bottom assuming constant sound velocity profile.
    rt = geoprocessing.raytracers.RTConstantSVP(geolocation, c)
    xyz = rt.trace_points(bottomdirections)
    bottom_directions = geoprocessing.datastructures.SampleDirectionsRange_1([ping.watercolumn.get_number_of_beams()])
    bottom_directions.crosstrack_angle = bottomdirections.crosstrack_angle
    bottom_directions.alongtrack_angle = bottomdirections.alongtrack_angle
    bottom_directions.range = xyz.true_range    
    
    return xyz, bottom_directions, bottom_direction_sample_numbers

def get_bottom_directions_bottom(ping: echosounders.filetemplates.I_Ping) -> (geoprocessing.datastructures.XYZ_1, geoprocessing.datastructures.SampleDirectionsRange_1, np.ndarray):
    """
    Retrieve bottom positions/directions/sample numbers from a bottom ping.

    Parameters
    ----------
    ping : echosounders.filetemplates.I_Ping
        Ping to retrieve bottom positions/directions/sample numbers from.

    Returns
    -------
    Tuple[geoprocessing.datastructures.XYZ_1, geoprocessing.datastructures.SampleDirectionsRange_1, np.ndarray]
        A tuple containing the bottom positions, directions, and sample numbers.
    """
    sc = ping.get_sensor_configuration()
    try:
        pingoff = sc.get_target("Transducer")
    except:
        print("Warning: No transducer target found in sensor configuration. Using default values.")
        pingoff = navigation.datastructures.PositionalOffsets()

    posoff = sc.get_position_source()
    geolocation = ping.get_geolocation()
    
    xyz = ping.bottom.get_xyz()
    xyz.x = xyz.x + posoff.x
    xyz.y = xyz.y + posoff.y
    xyz.z = xyz.z + geolocation.z
    
    bt = geoprocessing.backtracers.BTConstantSVP(geolocation, pingoff.x, pingoff.y)
    bottom_directions = bt.backtrace_points(xyz)
    bottom_direction_sample_numbers = ping.watercolumn.get_bottom_range_samples()
    
    return xyz, bottom_directions, bottom_direction_sample_numbers


