//sourcehash: 310b8a8eaaba368509c63705bea72528496df6d1af5d59ae4a5c36db7033040e

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_I_MultiSectorCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_apply_beam_sample_correction = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_apply_beam_sample_correction_2 =
R"doc(Apply beam/sample correction with per-beam absorption coefficients.

This overload supports per-beam absorption for multi-sector sonars
(e.g., Kongsberg) where each transmit sector may have a different
absorption value.

Args:
    wci: Water column image to correct.
    beam_angles: Beam angles for each beam.
    ranges: Sample ranges in meters.
    absorption_db_m_per_beam: Per-beam absorption coefficients in
                              dB/m.
    beam_numbers_per_tx_sector: Beam indices for each transmit sector.
    mp_cores: Number of parallel cores.

Returns:
    Corrected water column image.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_calibration_for_sector = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_calibration_for_sector_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_get_number_of_sectors = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_has_ap_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_has_av_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_has_power_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_has_sp_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_has_sv_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_has_valid_absorption_db_m = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_initialized = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_inplace_beam_sample_correction = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_inplace_beam_sample_correction_2 =
R"doc(Inplace apply beam/sample correction with per-beam absorption
coefficients.

This overload supports per-beam absorption for multi-sector sonars
(e.g., Kongsberg) where each transmit sector may have a different
absorption value.

Args:
    wci: Water column image to correct in-place.
    beam_angles: Beam angles for each beam.
    ranges: Sample ranges in meters.
    absorption_db_m_per_beam: Per-beam absorption coefficients in
                              dB/m.
    beam_numbers_per_tx_sector: Beam indices for each transmit sector.
    mp_cores: Number of parallel cores.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_I_MultiSectorCalibration_operator_eq = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_T_MultiSectorCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_T_MultiSectorCalibration_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_T_MultiSectorCalibration_3 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_binary_hash =
R"doc(compute a 64 bit hash of the object using xxhash and the       \
to_binary function. This  function is called binary because the
\ to_binary  function of the object is used

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_calibration_for_sector = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_calibration_for_sector_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_calibration_per_sector = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_from_binary =
R"doc(convert object to vector of bytes
\ \

Args:
    check_buffer_is_read_completely: variable for interface
                                     compatibility, does not do    \
                                     anything
                                     \ \

Returns:
    vector of bytes
    \

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_from_stream = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_get_calibrations = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_get_number_of_sectors = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_has_ap_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_has_av_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_has_power_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_has_sp_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_has_sv_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_has_valid_absorption_db_m = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_info_string =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_initialized = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_operator_assign = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_operator_eq = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_print =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_printer = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_to_binary =
R"doc(convert object to vector of bytes
\ \

Args:
    resize_buffer: variable for interface compatibility, does not do
                   anything             \ \

Returns:
    vector of bytes
    \

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_T_MultiSectorCalibration_to_stream = R"doc()doc";


#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


