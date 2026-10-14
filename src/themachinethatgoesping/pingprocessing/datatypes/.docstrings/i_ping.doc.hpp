//sourcehash: 868d02518ea4c63fec9516ef5219aa045ac39bff2a3d76db00244fc886706d52

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_I_Ping = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_bottom = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_bottom_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_channel_id = R"doc(channel id of the transducer)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_class_name = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_feature_group_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_get_channel_id = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_get_geolocation = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_get_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_get_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_get_sensor_configuration_base_hash =
R"doc(Returns the hash of the base sensor configuraiton.

This hash can be used to get the correct navigation interpolator from
the navigation_data_interface Note: This function is for testing and
finding errors. It is rather slow.

Returns:
    uint64_t

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_get_sensor_data_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_get_timestamp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_has_bottom = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_has_channel_id = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_has_geolocation = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_has_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_has_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_has_sensor_data_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_has_timestamp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_has_watercolumn = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_info_string =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_load = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_loaded = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_navigation_interpolator_set = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_not_implemented = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_not_implemented_not_implemented = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_primary_feature_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_print =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_printer = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_release = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_secondary_feature_functions = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_sensor_configuration_set = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_set_channel_id = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_set_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_set_navigation_interpolator_latlon_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_set_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_set_sensor_configuration_flyweight = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_set_timestamp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_timestamp = R"doc(Unix timestamp in seconds (saved in UTC0))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_watercolumn = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_Ping_watercolumn_2 = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


