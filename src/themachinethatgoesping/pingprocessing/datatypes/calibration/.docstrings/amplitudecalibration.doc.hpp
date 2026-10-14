//sourcehash: 831faa42e88ca1ca6b9b5bce33c3be34e4b965550f2c72ddaef37b65ba3968db

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_AmplitudeCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_AmplitudeCalibration_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_add_hash = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_apply_beam_sample_correction = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_apply_beam_sample_correction_2 =
R"doc(Apply beam and sample corrections with per-beam absorption
coefficients.

This overload supports per-beam absorption coefficients for multi-
sector sonars (e.g., Kongsberg) where each transmit sector may have a
different absorption value.

Args:
    wci: The input 2D tensor (beams x samples).
    beam_angles: A 1D tensor of beam angles (size = number of beams).
    ranges: A 1D tensor of ranges in meters (size = number of
            samples).
    absorption_db_m_per_beam: A 1D tensor of absorption coefficients
                              in dB/m per beam.
    tvg_factor: Optional time-varying gain factor.
    mp_cores: Number of parallel cores (default = 1).

Template Args:
    t_xtensor_2d: Type of the 2D water column image tensor.
    t_xtensor_1d: Type of the 1D tensor for beam angles, ranges, and
                  absorption.

Returns:
    Corrected 2D tensor.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_binary_hash = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_from_binary =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_from_stream = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_interpolator_offset_per_beamangle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_interpolator_offset_per_beamangle_and_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_interpolator_offset_per_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_offset_per_beamangle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_offset_per_beamangle_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_offset_per_beamangle_and_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_offset_per_beamangle_and_range_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_offset_per_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_offset_per_range_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_per_beam_offsets = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_per_sample_offsets = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_get_system_offset = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_has_offset_per_beamangle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_has_offset_per_beamangle_and_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_has_offset_per_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_has_system_offset = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_info_string =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_inplace_beam_sample_correction = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_inplace_beam_sample_correction_2 =
R"doc(Inplace apply beam and sample corrections with per-beam absorption
coefficients.

This overload supports per-beam absorption coefficients for multi-
sector sonars (e.g., Kongsberg) where each transmit sector may have a
different absorption value.

Args:
    wci: The input 2D tensor to be modified in-place (beams x
         samples).
    beam_angles: A 1D tensor of beam angles (size = number of beams).
    ranges: A 1D tensor of ranges in meters (size = number of
            samples).
    absorption_db_m_per_beam: A 1D tensor of absorption coefficients
                              in dB/m per beam.
    tvg_factor: Optional time-varying gain factor.
    min_beam_index: Optional minimum beam index.
    max_beam_index: Optional maximum beam index.
    mp_cores: Number of parallel cores (default = 1).

Template Args:
    t_xtensor_2d: Type of the 2D water column image tensor.
    t_xtensor_1d: Type of the 1D tensor for beam angles, ranges, and
                  absorption.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_offset_per_beamangle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_offset_per_beamangle_and_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_offset_per_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_operator_eq = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_print =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_printer = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_set_offset_per_beamangle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_set_offset_per_beamangle_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_set_offset_per_beamangle_and_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_set_offset_per_beamangle_and_range_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_set_offset_per_beamangle_and_range_3 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_set_offset_per_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_set_offset_per_range_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_set_system_offset = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_system_offset = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_to_binary =
R"doc(convert object to vector of bytes
\ \

Args:
    resize_buffer: variable for interface compatibility, does not do
                   anything             \ \

Returns:
    vector of bytes
    \

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_AmplitudeCalibration_to_stream = R"doc()doc";


#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


