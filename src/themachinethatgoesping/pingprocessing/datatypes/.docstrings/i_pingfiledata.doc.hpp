//sourcehash: e14257fd77653cd2e1626825324e46402e8f291065812a361d77885d75b49994

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData =
R"doc(Interface for raw ping data.

This class defines an interface for raw ping data. It provides methods
to access and manipulate the properties of the ping data, such as the
name, file ping counter, primary file number, file numbers, primary
file path, and file paths.

The class also includes a nested exception class, `not_implemented`,
which is thrown when a method is not implemented.

The class provides a `__printer__` function for object printing, which
can be used to print the properties of the ping data.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_I_PingFileData =
R"doc(Constructor for I_PingFileData.

Args:
    name: The name of the ping data.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_class_name = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_file_ping_counter = R"doc(Counter of the ping in the file (starting at 0))doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_get_file_numbers =
R"doc(Get the file numbers of the contained datagrams.

Returns:
    std::vector<size_t> The file numbers.

Raises:
    not_implemented: Exception if not implemented.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_get_file_paths =
R"doc(Get the file paths.

Returns:
    std::vector<std::string> The file paths associated with the
    contained datagrams.

Raises:
    not_implemented: Exception if not implemented.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_get_file_ping_counter =
R"doc(Get the file ping counter.

Returns:
    size_t The file ping counter.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_get_primary_file_nr =
R"doc(Get the primary file number of this ping.

Returns:
    size_t The primary file number.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_get_primary_file_path =
R"doc(Get the primary file path of this ping.

Returns:
    std::string The primary file path of this ping.

Raises:
    not_implemented: Exception if not implemented.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_info_string =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_not_implemented = R"doc(Exception thrown when a method is not implemented.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_not_implemented_not_implemented =
R"doc(Constructor for not_implemented.

Args:
    method_name: The name of the method.
    name: The name of the ping data.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_primary_file_nr = R"doc(File number of the primary file)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_print =
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

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_printer =
R"doc(ObjectPrinter for I_PingFileData.

Args:
    float_precision: The float precision for printing.

Returns:
    tools::classhelper::ObjectPrinter The ObjectPrinter instance.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_set_file_ping_counter =
R"doc(Set the file ping counter.

Args:
    file_ping_counter: The file ping counter to set.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_I_PingFileData_set_primary_file_nr =
R"doc(Set the primary file number for this ping.

Args:
    primary_file_nr: The primary file number to set.

)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


