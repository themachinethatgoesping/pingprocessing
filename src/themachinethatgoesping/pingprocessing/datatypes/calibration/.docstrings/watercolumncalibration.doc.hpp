//sourcehash: 328859b389afe9daf2fb302b5fe53088114aef844c06e56767997cb9c0f644bb

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


static const char *mkd_doc_OptionFrozen = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_WaterColumnCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_WaterColumnCalibration_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_WaterColumnCalibration_3 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_absorption_db_m = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_ap_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_apply_beam_sample_correction = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_apply_beam_sample_correction_2 =
R"doc(Apply beam and sample correction with per-beam absorption
coefficients.

This overload supports per-beam absorption for multi-sector sonars
(e.g., Kongsberg) where each transmit sector may have a different
absorption value.

Note:
    For calibration types that don't apply absorption (power, rp, rv),
    this function ignores the per-beam absorption parameter.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_av_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_binary_hash =
R"doc(compute a 64 bit hash of the object using xxhash and the       \
to_binary function. This  function is called binary because the
\ to_binary  function of the object is used

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_check_calibration_initialized = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_check_initialized = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_check_modifying_base_calibration_allowed = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_from_binary =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_from_stream = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_absorption_db_m = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_absorption_to_apply = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_ap_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_av_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_calibration_ptr =
R"doc(Get the calibration pointer for a given calibration kind.

Template Args:
    kind: The calibration kind (constexpr).

Returns:
    Reference to the unique_ptr for that calibration.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_power_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_sp_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_sv_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_tvg_absorption_db_m = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_tvg_factor = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_get_tvg_factor_to_apply = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_has_ap_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_has_av_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_has_power_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_has_sp_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_has_sv_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_has_valid_absorption_db_m = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_info_string =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_inplace_beam_sample_correction = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_inplace_beam_sample_correction_2 =
R"doc(Inplace apply beam and sample correction with per-beam absorption
coefficients.

This overload supports per-beam absorption for multi-sector sonars
(e.g., Kongsberg) where each transmit sector may have a different
absorption value.

Note:
    For calibration types that don't apply absorption (power, rp, rv),
    this function ignores the per-beam absorption parameter.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_operator_assign = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_operator_eq = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_power_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_print =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_printer = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_set_absorption_db_m = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_set_ap_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_set_av_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_set_power_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_set_sp_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_set_sv_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_sp_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_sv_calibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_to_binary =
R"doc(convert object to vector of bytes
\ \

Args:
    resize_buffer: variable for interface compatibility, does not do
                   anything             \ \

Returns:
    vector of bytes
    \

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_to_stream = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_tvg_absorption_db_m = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_WaterColumnCalibration_tvg_factor = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_applies_absorption =
R"doc(Check if absorption should be applied for a given calibration type.

For power, rp, rv: absorption is explicitly NOT applied (use 0). For
all others: absorption is applied.

Args:
    ct: The calibration type.

Returns:
    true if absorption should be applied, false otherwise.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_get_calibration_kind =
R"doc(Get the calibration kind (which AmplitudeCalibration pointer) for a
given calibration type.

Args:
    ct: The calibration type.

Returns:
    The calibration kind.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_get_calibration_kind_name =
R"doc(Get the name of a calibration kind for error messages.

Args:
    ck: The calibration kind.

Returns:
    Human-readable name.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_get_target_tvg_factor =
R"doc(Get the target TVG factor for a given calibration type.

Args:
    ct: The calibration type.

Returns:
    The TVG factor (0, 20, or 40).

)doc";


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_kind = R"doc(Calibration kind enum for selecting which AmplitudeCalibration to use.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_kind_ap = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_kind_av = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_kind_power = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_kind_sp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_kind_sv = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type = R"doc(Calibration type for water column data processing)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_ap = R"doc(amplitude derived point scattering (uncalibrated TS))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_av = R"doc(amplitude derived volume scattering (uncalibrated SV))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_power =
R"doc(power calibration (Received amplitudes computed to power, no
absorption, no spreading loss))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_pp =
R"doc(power derived point scattering (uncalibrated TS without parameter
compensation))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_pv =
R"doc(power derived volume scattering (uncalibrated SV without parameter
compensation))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_rp =
R"doc(power derived point scattering (uncalibrated TS without parameter and
absorption compensation))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_rv =
R"doc(power derived volume scattering (uncalibrated SV without parameter and
absorption compensation))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_sp = R"doc(point scattering (uncompensated TS))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_t_calibration_type_sv = R"doc(volume scattering SV)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


