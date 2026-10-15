import numpy as np
import themachinethatgoesping.navigation as navigation
from themachinethatgoesping.pingprocessing_nanopy import datatypes
import themachinethatgoesping.algorithms as algorithms
import themachinethatgoesping.algorithms.geoprocessing as geoprocessing

def get_bottom_directions_wci(
    ping: datatypes.I_Ping, 
    selection: algorithms.pingtools.BeamSelection = None) -> (geoprocessing.datastructures.XYZ_1, geoprocessing.datastructures.SampleDirectionsRange_1, np.ndarray):
    """
    Retrieve bottom positions/directions/sample numbers from a water column ping.
    Note: this function is an approximation (for performance reasons). As such, it 
    assumes a constant sound velocity profile.

    Parameters
    ----------
    ping : datatypes.I_Ping
        Ping to retrieve bottom positions/directions/sample numbers from.
    selection : algorithms.pingtools.BeamSelection, optional
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
        pingoff = sc.get_target(*sc.get_transducer_transmit_receive_id(ping.get_channel_id()))
    except Exception as e:
        print("Warning: No transducer target found in sensor configuration. Using default values.")
        print(f"Exception occurred: {e}")
        pingoff = navigation.datastructures.SensorPose()
    posoff = sc.get_position_source()

    # deal with unavailable geolocation by using default values.
    # TODO: this could issue a warning in a log file
    if ping.has_geolocation():
        geolocation = ping.get_geolocation()
    else:
        geolocation = navigation.datastructures.Geolocation()

    bottom_direction_sample_numbers = np.array(selection.get_last_sample_number_per_beam())
    
    bottomdirections = geoprocessing.datastructures.SampleDirectionsTime_1([selection.get_number_of_beams()])
    bottomdirections.crosstrack_angle = ping.watercolumn.get_beam_crosstrack_angles_legacy(selection)
    if ping.watercolumn.has_beam_crosstrack_angles_in_world_frame():
        # World-frame angles (e.g. kmall/kongsbergall re-vertical): subtract roll so the
        # raytracer's sensor-orientation rotation restores the original gravity-frame direction.
        bottomdirections.crosstrack_angle = bottomdirections.crosstrack_angle - geolocation.roll
    bottomdirections.alongtrack_angle = ping.watercolumn.get_beam_alongtrack_angles(selection)
    bottomdirections.two_way_travel_time = bottom_direction_sample_numbers * ping.watercolumn.get_sample_interval()

    # TODO: Get sound velocity from SVP.
    c = ping.watercolumn.get_sound_speed_at_transducer_legacy()

    # Raytrace to bottom assuming constant sound velocity profile.
    rt = geoprocessing.raytracers.RTConstantSVP(geolocation, c)
    xyz = rt.trace_points(bottomdirections)
    bottom_directions = geoprocessing.datastructures.SampleDirectionsRange_1([selection.get_number_of_beams()])
    bottom_directions.crosstrack_angle = bottomdirections.crosstrack_angle
    bottom_directions.alongtrack_angle = bottomdirections.alongtrack_angle

    # A beam without a valid bottom detection yields a range of 0. The water-column
    # backtracer (BacktracedWCI) builds a per-beam range->sample interpolator from
    # {0, range}; a zero range makes that {0, 0} (duplicate x), which raises
    # ERROR[Interpolation::_check_XY] and aborts the ENTIRE image build. Such beams
    # have no in-range samples anyway, so replace non-positive/invalid ranges with a
    # tiny positive value: the affected beams then simply contribute no samples
    # instead of crashing the build (important when a PingSampleSelector such as
    # select_max_bottom_range_percent includes outer beams that lack a bottom).
    bottom_range = np.asarray(xyz.true_range).copy()
    invalid_range = ~np.isfinite(bottom_range) | (bottom_range <= 0.0)
    if invalid_range.any():
        bottom_range[invalid_range] = 1e-3
    bottom_directions.range = bottom_range

    return xyz, bottom_directions, bottom_direction_sample_numbers

def get_bottom_directions_bottom(
    ping: datatypes.I_Ping,
    selection: algorithms.pingtools.BeamSelection = None) -> (geoprocessing.datastructures.XYZ_1, geoprocessing.datastructures.SampleDirectionsRange_1, np.ndarray):
    """
    Retrieve bottom positions/directions/sample numbers from a bottom ping.

    Parameters
    ----------
    ping : datatypes.I_Ping
        Ping to retrieve bottom positions/directions/sample numbers from.
    selection : algorithms.pingtools.BeamSelection, optional
        A beam selection to filter beams, by default None (all beams).

    Returns
    -------
    Tuple[geoprocessing.datastructures.XYZ_1, geoprocessing.datastructures.SampleDirectionsRange_1, np.ndarray]
        A tuple containing the bottom positions, directions, and sample numbers.
    """
    sc = ping.get_sensor_configuration()
    try:
        pingoff = sc.get_target(*sc.get_transducer_transmit_receive_id(ping.get_channel_id()))
    except Exception as e:
        print("Warning: No transducer target found in sensor configuration. Using default values.")
        print(f"Exception occurred: {e}")
        pingoff = navigation.datastructures.SensorPose()

    posoff = sc.get_position_source()
    # deal with unavailable geolocation by using default values.
    # TODO: this could issue a warning in a log file
    if ping.has_geolocation():
        geolocation = ping.get_geolocation()
    else:
        geolocation = navigation.datastructures.Geolocation()
    
    if selection is not None:
        xyz = ping.bottom.get_xyz(selection)
    else:
        xyz = ping.bottom.get_xyz()
    xyz.x = xyz.x + posoff.x
    xyz.y = xyz.y + posoff.y
    xyz.z = xyz.z + geolocation.z
    
    bt = geoprocessing.backtracers.BTConstantSVP(geolocation, pingoff.x, pingoff.y)
    bottom_directions = bt.backtrace_points(xyz)
    # Guard against degenerate (zero / non-finite) ranges that would make the
    # backtracer's per-beam {0, range} interpolator collapse to {0, 0} and abort
    # the whole image build (see get_bottom_directions_wci for details).
    bottom_range = np.asarray(bottom_directions.range).copy()
    invalid_range = ~np.isfinite(bottom_range) | (bottom_range <= 0.0)
    if invalid_range.any():
        bottom_range[invalid_range] = 1e-3
        bottom_directions.range = bottom_range
    if selection is not None:
        bottom_direction_sample_numbers = ping.watercolumn.get_bottom_range_samples(selection)
    else:
        bottom_direction_sample_numbers = ping.watercolumn.get_bottom_range_samples()
    
    return xyz, bottom_directions, bottom_direction_sample_numbers


