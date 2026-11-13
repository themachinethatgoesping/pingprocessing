//sourcehash: 1ecb54ac94011a54e31fad5fa1e6beae843a01fea6923f803b950a0e52e5904c

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumnCalibration =
R"doc(CRTP base adding a covariant single-sector water column calibration
accessor.

Template Args:
    t_derived: the concrete ping water column class (must expose a
               `file_data()` accessor)
    t_watercolumncalibration: the format's water column calibration
                              type (derived from
                              WaterColumnCalibration))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumnCalibration_I_PingWatercolumnCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumnCalibration_calibration_file_data =
R"doc(Lazily initialize the calibration (a cache held by the shared file
data) and return the (non-const) file data.

The ping water column handle is never truly const - the calibration is
a lazily built cache on the shared file-data object - so the
const_cast here only re-selects the non-const file_data() accessor,
mirroring the pre-CRTP behavior where the lazy build was triggered
through the shared_ptr (which propagates const only shallowly).

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumnCalibration_get_watercolumn_calibration =
R"doc(Water column calibration of this ping.

Covariant override: returns the format-specific calibration type
through the concrete class and the generic WaterColumnCalibration
through the I_PingWatercolumn interface.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumnMultiSectorCalibration =
R"doc(CRTP base adding covariant single- and multi-sector water column
calibration accessors.

Template Args:
    t_derived: the concrete ping water column class (must expose a
               `file_data()` accessor)
    t_watercolumncalibration: the format's water column calibration
                              type
    t_multisectorcalibration: the format's multi-sector water column
                              calibration type (derived from
                              I_MultiSectorCalibration))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumnMultiSectorCalibration_I_PingWatercolumnMultiSectorCalibration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumnMultiSectorCalibration_get_multisectorwatercolumn_calibration =
R"doc(Multi-sector water column calibration of this ping.

Covariant override: returns the format-specific multi-sector
calibration type through the concrete class and the generic
I_MultiSectorCalibration through the I_PingWatercolumn interface.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingWatercolumnMultiSectorCalibration_update_calibration =
R"doc(Replace the ping's calibration with the given (deduplicated) multi-
sector calibration.

)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


