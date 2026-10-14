//sourcehash: 19d9665e1b48f1d767b90a6fbb65da187d1cbf91d53e0e2be1db23668b8c7859

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn =
R"doc(Interface for all ping watercolumn functions



Interface for watercolumn ping data.

This class represents an interface for accessing watercolumn ping
data. It inherits from the I_PingCommon class and provides additional
functions and variables specific to watercolumn pings.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_I_PingWatercolumn = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_beam_sample_selection_all = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_beam_sample_selection_all_is_initialized = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_class_name = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_feature_group_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_amplitudes = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_amplitudes_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_ap = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_ap_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_approximate_ranges = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_approximate_ranges_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_av = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_av_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_beam_alongtrack_angles = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_beam_alongtrack_angles_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_beam_crosstrack_angles_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_beam_crosstrack_angles_legacy_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_beam_numbers_per_tx_sector_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_beam_numbers_per_tx_sector_legacy_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_beam_sample_selection_all = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_beam_selection_all = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_bottom_range_samples = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_bottom_range_samples_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_calibrated_wci = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_first_sample_offset_per_beam = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_minslant_sample_nr = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_minslant_sample_nr_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_multisectorwatercolumn_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_number_of_beams_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_number_of_samples_per_beam = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_number_of_samples_per_beam_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_number_of_tx_sectors_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_power = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_power_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_pp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_pp_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_pv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_pv_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_rp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_rp_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_rv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_rv_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_sample_interval = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_sound_speed_at_transducer_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_sp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_sp_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_sv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_sv_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_test_mode = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_tx_sector_per_beam_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_tx_sector_per_beam_legacy_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_tx_signal_parameters_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_watercolumn_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_watercolumn_calibration_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_wci = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_wci_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_wci_correction = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_wci_correction_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_get_wci_correction_3 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_amplitudes = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_ap = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_av = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_beam_crosstrack_angles_in_world_frame =
R"doc(Returns true if get_beam_crosstrack_angles_legacy() produces gravity-
referenced (world-frame) angles. When false (default), angles are in
the vessel/transducer frame and the raytracer must apply the sensor
orientation (roll/pitch) to convert to world-frame.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_beam_crosstrack_angles_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_beam_numbers_per_tx_sector_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_beam_selection_all = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_bottom_range_samples = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_multisectorwatercolumn_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_number_of_beams_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_number_of_tx_sectors_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_power = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_pp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_pv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_rp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_rv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_sp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_sv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_tx_sector_per_beam_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_tx_signal_parameters_legacy = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_has_watercolumn_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_info_string =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_primary_feature_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_print =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_printer = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_secondary_feature_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_set_test_mode = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumn_test_mode = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


