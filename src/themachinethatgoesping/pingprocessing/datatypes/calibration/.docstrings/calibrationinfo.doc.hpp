//sourcehash: 144a81d9502259f2c84a6630a5ca55bf8c6ea69c2640b60ec0eb0a343c2227a2

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


static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo =
R"doc(Base class for storing calibration documentation / provenance
information.

Records two kinds of information:
 - **setup_steps**: Python script snippets that document how each
   calibration parameter was set (e.g. from a datagram object or
   manually).
 - **formulas**: Python script snippets that document the exact math
   applied to the raw amplitude data (e.g. power, av, sv calibration
   formulas).

A suppress-guard mechanism ensures that when an outer method (e.g.
`set_environment_parameters(XML_Environment)`) registers a step and
then calls the value-based overload internally, the inner call does
not overwrite the outer registration.

Steps are stored in insertion order.  Registering a step with an
existing name **replaces** the previous entry (unless suppressed).)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_CalibrationInfo = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_SuppressGuard =
R"doc(RAII guard that temporarily suppresses step registration.

Create via `CalibrationInfo::suppress()`.  While the guard is alive,
all calls to `register_step()` are silently ignored.  Nesting is
supported (counted).)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_SuppressGuard_SuppressGuard = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_SuppressGuard_SuppressGuard_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_SuppressGuard_SuppressGuard_3 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_SuppressGuard_info = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_SuppressGuard_operator_assign = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_SuppressGuard_operator_assign_2 = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_clear = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_clear_formulas = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_clear_steps = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_formulas = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_get_formula_script = R"doc(Get script for a single formula by name (empty if not found).)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_get_formulas = R"doc(Get the formulas list (name→script pairs, in insertion order).)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_get_formulas_script = R"doc(Concatenate all formulas into one script.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_get_full_script = R"doc(Concatenate everything (setup steps + formulas).)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_get_setup_script = R"doc(Concatenate all setup steps into one script.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_get_setup_steps = R"doc(Get the steps list (name→script pairs, in insertion order).)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_get_step_script = R"doc(Get script for a single setup step by name (empty if not found).)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_has_formula = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_has_formulas = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_has_step = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_has_steps = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_is_suppressed = R"doc(Returns true when step registration is currently suppressed.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_register_formula = R"doc(Register (or replace) a formula description.)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_register_step =
R"doc(Register (or replace) a setup step.

If a step with the same ``name`` already exists it is overwritten in-
place (preserving insertion order).  No-op when suppressed.

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_setup_steps = R"doc()doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_suppress =
R"doc(Create a guard that suppresses step registration until destroyed.

Typical usage:
```
_cal_info.register_step("foo", script);   // registers
auto guard = _cal_info.suppress();
inner_method();                             // inner register_step calls are suppressed
```

)doc";

static const char *mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_CalibrationInfo_suppress_depth = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


