//sourcehash: f68129b4b78effed3d282444f927e89d64b90854924a1576016071f1f69c4b19

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo =
R"doc(Per-ping meta information shared between a ping and its
bottom/watercolumn sub-objects.

PingInfo stores the information that is identical for a ping and its
I_PingBottom / I_PingWatercolumn (channel id, timestamp, sensor
configuration and navigation interpolator). An I_Ping creates one
PingInfo and hands the same shared pointer to its bottom and
watercolumn (done by the echosounder ping implementations) so that the
sub-objects can access the sensor configuration / navigation /
geolocation without going back through the owning ping.

Memory layout: a single flyweight for the (deduplicated) string
metadata plus the (already deduplicated) sensor configuration and
navigation interpolator flyweights and the per-ping timestamp.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfoData =
R"doc(Small, deduplicated bundle of the rarely-changing string metadata of a
ping.

This structure is stored inside PingInfo as a single boost::flyweight
so that all pings that share the same (channel_id, ...) combination
only store one copy (one pointer) of it. It is intentionally kept
lightweight and holds only string metadata - more string fields can be
added here later without changing the number of pointers held by
PingInfo. The sensor configuration and the navigation interpolator are
deliberately NOT part of this structure; they are stored as their own
flyweights directly on PingInfo.

The structure is an implementation detail of PingInfo and is not
exposed to Python.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfoData_channel_id = R"doc(channel id of the transducer)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfoData_operator_eq = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_PingInfo = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_data = R"doc(Deduplicated string metadata (channel id, ...). Not exposed directly.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_get_channel_id = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_get_geolocation = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_get_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_get_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_get_sensor_configuration_base_hash =
R"doc(Returns the hash of the base sensor configuration.

This hash can be used to get the correct navigation interpolator from
the navigation_data_interface. Note: This function is for testing and
finding errors. It is rather slow.

Returns:
    uint64_t

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_get_sensor_data_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_get_timestamp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_has_channel_id = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_has_geolocation = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_has_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_has_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_has_sensor_data_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_has_timestamp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_info_string =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_operator_eq = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_operator_ne = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_print =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_printer = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_set_channel_id = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_set_data =
R"doc(Set channel id, sensor configuration and navigation interpolator at
once.

This is the setter that should be used during initialization. The
sensor configuration and the navigation interpolator are passed as
flyweights so that no hashes have to be recomputed.

Args:
    channel_id: channel id of the transducer
    sensor_configuration: sensor configuration flyweight
    navigation_interpolator_latlon: navigation interpolator flyweight

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_set_navigation_interpolator_latlon = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_set_navigation_interpolator_latlon_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_set_sensor_configuration = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_set_sensor_configuration_flyweight = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_set_timestamp = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_PingInfo_timestamp = R"doc(Unix timestamp in seconds (saved in UTC0))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_hash_value =
R"doc(Boost hash function (used by boost::flyweight to deduplicate
PingInfoData).

Args:
    data: object to hash

Returns:
    std::size_t

)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


