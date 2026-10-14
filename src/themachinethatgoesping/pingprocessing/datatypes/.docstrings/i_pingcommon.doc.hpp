//sourcehash: c32a5f357e073e028e36a7e632c11ff31258ded29cbfb6ca15ab43a1d01966ee

/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.

  This is a modified version which allows for more than 8 arguments and includes def-guard
 */

#pragma once

#ifndef __DOCSTRINGS_HPP__
#define __DOCSTRINGS_HPP__

#define MKD_EXPAND(x)                                      x
#define MKD_COUNT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, COUNT, ...)  COUNT
#define MKD_VA_SIZE(...)                                   MKD_EXPAND(MKD_COUNT(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define MKD_CAT1(a, b)                                     a ## b
#define MKD_CAT2(a, b)                                     MKD_CAT1(a, b)
#define MKD_DOC1(n1)                                       mkd_doc_##n1
#define MKD_DOC2(n1, n2)                                   mkd_doc_##n1##_##n2
#define MKD_DOC3(n1, n2, n3)                               mkd_doc_##n1##_##n2##_##n3
#define MKD_DOC4(n1, n2, n3, n4)                           mkd_doc_##n1##_##n2##_##n3##_##n4
#define MKD_DOC5(n1, n2, n3, n4, n5)                       mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5
#define MKD_DOC6(n1, n2, n3, n4, n5, n6)                   mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define MKD_DOC7(n1, n2, n3, n4, n5, n6, n7)               mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define MKD_DOC8(n1, n2, n3, n4, n5, n6, n7, n8)           mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7##_##n8
#define MKD_DOC9(n1, n2, n3, n4, n5, n6, n7, n8, n9)       mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7##_##n8##_##n9
#define MKD_DOC10(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10) mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7##_##n8##_##n9##_##n10
#define DOC(...)                                           MKD_EXPAND(MKD_EXPAND(MKD_CAT2(MKD_DOC, MKD_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#endif // __DOCSTRINGS_HPP__
#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *mkd_doc_Option = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_I_PingCommon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_available_feature_groups = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_available_features = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_class_name = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_feature_group_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_feature_groups =
R"doc(Get a string of all registered feature groups for this ping class

Returns:
    std::string

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_feature_groups_string =
R"doc(Get a string of all registered feature groups that are available or
not available

Args:
    available: if True (default) return available features, else
               return not available features

Returns:
    std::string

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_feature_string =
R"doc(Get a string of all registered features that are available or not
available

Args:
    available: if True (default) return available features, else
               return not available features

Returns:
    std::string

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_get_ping_info =
R"doc(Get the shared PingInfo object (channel id, timestamp, sensor
configuration, navigation).

Returns:
    const std::shared_ptr<PingInfo>&

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_has_all_of_features =
R"doc(Check if all of the specified features are available

Returns:
    true false

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_has_any_of_features =
R"doc(Check if any of the specified features is available

Returns:
    true false

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_has_feature =
R"doc(Check if any of the registered features is available

Returns:
    true false

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_has_feature_groups = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_has_features =
R"doc(Check if any of the registered features is available

Returns:
    true false

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_has_ping_info =
R"doc(Check if a PingInfo object is attached to this ping object.

Returns:
    true if a PingInfo is attached, false otherwise

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_has_primary_features =
R"doc(Check if any of the registered main features is available

Returns:
    true false

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_has_secondary_features = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_info_string =
R"doc(                                                                                           \
return an info string using the class __printer__ object
\

Args:
    float_precision: number of digits for floating point values
                     \
    superscript_exponents: print exponents in superscript
                           \

Returns:
    std::string
    \

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_load = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_loaded = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_not_implemented = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_not_implemented_not_implemented = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_ping_info =
R"doc(Per-ping meta information (channel id, timestamp, sensor
configuration, navigation).

This is a shared pointer so that an I_Ping and its I_PingBottom /
I_PingWatercolumn sub-objects can share the exact same PingInfo
instance (the echosounder ping implementations hand the same pointer
to the bottom and watercolumn during creation).)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_possible_feature_groups = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_possible_features = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_primary_feature_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_primary_features =
R"doc(Get a string of all registered primary features for this ping class

Returns:
    std::string

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_print =
R"doc(                                                                                           \
print the object information to the given outpustream
\ \

Args:
    os: output stream, e.g. file stream or std::out or std::cerr
        \
    float_precision: number of digits for floating point values
                     \
    superscript_exponents: print exponents in superscript
                           \

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_print_feature_groups = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_print_features = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_printer = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_registered_features =
R"doc(Get a string of all registered features for this ping class

Returns:
    std::string

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_release = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_secondary_feature_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_secondary_features =
R"doc(Get a string of all registered secondary features for this ping class

Returns:
    std::string

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingCommon_set_ping_info =
R"doc(Set (share) the PingInfo object.

Args:
    ping_info: shared pointer to the PingInfo object

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_amplitudes = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_ap = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_av = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_beam_crosstrack_angles_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_beam_numbers_per_tx_sector_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_beam_selection_all = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_bottom = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_bottom_range_samples = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_channel_id = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_datetime = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_file_data = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_geolocation = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_multisectorwatercolumn_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_number_of_beams_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_number_of_soundings = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_number_of_tx_sectors_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_ping_info = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_power = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_pp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_pv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_ref_pingtime = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_ref_pose = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_rp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_rv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_rx_offsets = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_sector_beam_numbers = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_sector_tx_offsets = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_sector_tx_steering_angles = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_sector_tx_time_delays = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_sensor_data_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_soundings_rx_steering_angles = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_soundings_sector_numbers = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_soundings_two_way_travel_times = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_sp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_surface_sound_speed = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_sv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_timestamp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_two_way_travel_times_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_tx_sector_per_beam_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_tx_signal_parameters_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_watercolumn = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_watercolumn_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_t_pingfeature_xyz = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


