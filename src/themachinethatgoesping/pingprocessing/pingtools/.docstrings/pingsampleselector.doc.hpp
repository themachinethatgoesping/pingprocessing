//sourcehash: 027eeba5d102757a5255f9e941e85a2778289a8363afec68448817ca0a612aaa

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_PingSampleSelector = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_apply_selection = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_apply_selection_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_beam_step = R"doc(step size for beam numbers)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_binary_hash =
R"doc(compute a 64 bit hash of the object using xxhash and the       \
to_binary function. This  function is called binary because the
\ to_binary  function of the object is used

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_beam_angle_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_beam_number_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_beam_step = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_bottom_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_minslant_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_sample_number_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_sample_range_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_sample_step = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_transmit_sector_beam_angle_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_clear_transmit_sectors = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_from_binary =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_from_stream =
R"doc(Return a PingSampleSelector read from a binary stream

Args:
    is: input stream

Returns:
    PingSampleSelector

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_beam_numbers_selected_by_transmit_sector = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_beam_step = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_max_beam_angle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_max_beam_number = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_max_bottom_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_max_minslant_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_max_sample_number = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_max_sample_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_min_beam_angle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_min_beam_number = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_min_bottom_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_min_minslant_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_min_sample_number = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_min_sample_range = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_sample_step = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_transmit_sector_max_beam_angle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_transmit_sector_min_beam_angle = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_get_transmit_sectors = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_info_string =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_max_beam_angle = R"doc(max beam angle to select (°))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_max_beam_number = R"doc(max beam number to select)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_max_bottom_range_percent = R"doc(max samples before bottom detection to consider (%))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_max_minslant_range_percent = R"doc(max samples before minslant to consider (%))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_max_sample_number = R"doc(max sample number to select)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_max_sample_range = R"doc(max sample range to select (m))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_min_beam_angle = R"doc(min beam angle to select (°))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_min_beam_number = R"doc(min beam number to select)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_min_bottom_range_percent = R"doc(max samples before bottom detection to consider (%))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_min_minslant_range_percent = R"doc(min samples after minslant to consider (%))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_min_sample_number = R"doc(min sample number to select)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_min_sample_range = R"doc(min sample range to select (m))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_operator_eq = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_print =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_printer =
R"doc(Print function, needs __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__ macro
See also: classhelper/objectprinter.hpp

Args:
    float_precision: Precision of floating point numbers

Returns:
    classhelper::ObjectPrinter

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_register_filter = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_register_filter_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_sample_step = R"doc(step size for sample numbers)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_beam_range_by_angles = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_beam_range_by_numbers = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_bottom_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_max_bottom_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_max_minslant_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_min_bottom_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_min_minslant_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_minslant_range_percent = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_sample_range_by_numbers = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_sample_range_by_ranges = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_transmit_sectors = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_select_transmit_sectors_by_beam_angles = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_set_beam_step = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_set_sample_step = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_to_binary =
R"doc(convert object to vector of bytes
\ \

Args:
    resize_buffer: variable for interface compatibility, does not do
                   anything             \ \

Returns:
    vector of bytes
    \

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_to_stream =
R"doc(Write a PingSampleSelector to a binary stream

Args:
    os: output stream

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_transmit_sector_max_beam_angle = R"doc(select transmit sectors with angles <= _transmit_sector_max_beam_angle)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_transmit_sector_min_beam_angle = R"doc(select transmit sectors with angles >= _transmit_sector_min_beam_angle)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_pingtools_PingSampleSelector_transmit_sectors = R"doc(transmit_sectors to select)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


