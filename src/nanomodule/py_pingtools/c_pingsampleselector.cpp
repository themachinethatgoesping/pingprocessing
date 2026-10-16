// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "../../themachinethatgoesping/pingprocessing/pingtools/pingsampleselector.hpp"
#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_pingtools {

namespace nb = nanobind;
using namespace themachinethatgoesping::pingprocessing::pingtools;

#define DOC_PingSampleSelector(arg)                                                                \
    DOC(themachinethatgoesping, pingprocessing, pingtools, PingSampleSelector, arg)

void init_c_pingsampleselector(nanobind::module_& m)
{
          nb::class_<PingSampleSelector>(
                    m,
                    "PingSampleSelector",
                    DOC(themachinethatgoesping, pingprocessing, pingtools, PingSampleSelector))
                    .def(nb::init<>(), DOC_PingSampleSelector(PingSampleSelector))

        // --- main interface ---
          .def("apply_selection",
                nb::overload_cast<pingprocessing::datatypes::I_PingWatercolumn&>(
                     &PingSampleSelector::apply_selection),
                DOC_PingSampleSelector(apply_selection),
                nb::arg("ping_watercolumn"))
          .def("apply_selection",
                nb::overload_cast<pingprocessing::datatypes::I_PingBottom&>(
                     &PingSampleSelector::apply_selection),
                DOC_PingSampleSelector(apply_selection),
                nb::arg("ping_bottom"))

        // --- convenient data access ---
        .def("get_min_beam_number",
             &PingSampleSelector::get_min_beam_number,
             DOC_PingSampleSelector(min_beam_number))
        .def("get_max_beam_number",
             &PingSampleSelector::get_max_beam_number,
             DOC_PingSampleSelector(max_beam_number))
        .def("get_min_sample_number",
             &PingSampleSelector::get_min_sample_number,
             DOC_PingSampleSelector(min_sample_number))
        .def("get_max_sample_number",
             &PingSampleSelector::get_max_sample_number,
             DOC_PingSampleSelector(max_sample_number))
        .def("get_min_beam_angle",
             &PingSampleSelector::get_min_beam_angle,
             DOC_PingSampleSelector(min_beam_angle))
        .def("get_max_beam_angle",
             &PingSampleSelector::get_max_beam_angle,
             DOC_PingSampleSelector(max_beam_angle))
        .def("get_min_sample_range",
             &PingSampleSelector::get_min_sample_range,
             DOC_PingSampleSelector(min_sample_range))
        .def("get_max_sample_range",
             &PingSampleSelector::get_max_sample_range,
             DOC_PingSampleSelector(max_sample_range))
        .def("get_transmit_sectors",
             &PingSampleSelector::get_transmit_sectors,
             DOC_PingSampleSelector(transmit_sectors))
        .def("get_transmit_sector_min_beam_angle",
             &PingSampleSelector::get_transmit_sector_min_beam_angle,
             DOC_PingSampleSelector(transmit_sector_min_beam_angle))
        .def("get_transmit_sector_max_beam_angle",
             &PingSampleSelector::get_transmit_sector_max_beam_angle,
             DOC_PingSampleSelector(transmit_sector_max_beam_angle))
        .def("get_beam_step", &PingSampleSelector::get_beam_step, DOC_PingSampleSelector(beam_step))
        .def("get_sample_step",
             &PingSampleSelector::get_sample_step,
             DOC_PingSampleSelector(sample_step))
        .def("get_min_minslant_range_percent",
             &PingSampleSelector::get_min_minslant_range_percent,
             DOC_PingSampleSelector(min_minslant_range_percent))
        .def("get_max_minslant_range_percent",
             &PingSampleSelector::get_max_minslant_range_percent,
             DOC_PingSampleSelector(max_minslant_range_percent))
        .def("get_min_bottom_range_percent",
             &PingSampleSelector::get_min_bottom_range_percent,
             DOC_PingSampleSelector(min_bottom_range_percent))
        .def("get_max_bottom_range_percent",
             &PingSampleSelector::get_max_bottom_range_percent,
             DOC_PingSampleSelector(max_bottom_range_percent))

        // ----- clear range filters -----
        .def("clear_beam_number_range",
             &PingSampleSelector::clear_beam_number_range,
             DOC_PingSampleSelector(clear_beam_number_range))
        .def("clear_sample_number_range",
             &PingSampleSelector::clear_sample_number_range,
             DOC_PingSampleSelector(clear_sample_number_range))
        .def("clear_beam_angle_range",
             &PingSampleSelector::clear_beam_angle_range,
             DOC_PingSampleSelector(clear_beam_angle_range))
        .def("clear_sample_range_range",
             &PingSampleSelector::clear_sample_range_range,
             DOC_PingSampleSelector(clear_sample_range_range))
        .def("clear_transmit_sectors",
             &PingSampleSelector::clear_transmit_sectors,
             DOC_PingSampleSelector(clear_transmit_sectors))
        .def("clear_transmit_sector_beam_angle_range",
             &PingSampleSelector::clear_transmit_sector_beam_angle_range,
             DOC_PingSampleSelector(clear_transmit_sector_beam_angle_range))
        .def("clear_beam_step",
             &PingSampleSelector::clear_beam_step,
             DOC_PingSampleSelector(clear_beam_step))
        .def("clear_sample_step",
             &PingSampleSelector::clear_sample_step,
             DOC_PingSampleSelector(clear_sample_step))
        .def("clear_minslant_range",
             &PingSampleSelector::clear_minslant_range,
             DOC_PingSampleSelector(clear_minslant_range))
        .def("clear_bottom_range",
             &PingSampleSelector::clear_bottom_range,
             DOC_PingSampleSelector(clear_bottom_range))
        .def("clear", &PingSampleSelector::clear, DOC_PingSampleSelector(clear))

        // ----- selectors -----
        .def("select_beam_range_by_numbers",
             &PingSampleSelector::select_beam_range_by_numbers,
             DOC_PingSampleSelector(select_beam_range_by_numbers),
             nb::arg("min_beam_number") = std::nullopt,
             nb::arg("max_beam_number") = std::nullopt,
             nb::arg("beam_step")       = std::nullopt)
        .def("select_sample_range_by_numbers",
             &PingSampleSelector::select_sample_range_by_numbers,
             DOC_PingSampleSelector(select_sample_range_by_numbers),
             nb::arg("min_sample_number") = std::nullopt,
             nb::arg("max_sample_number") = std::nullopt,
             nb::arg("sample_step")       = std::nullopt)
        .def("select_beam_range_by_angles",
             &PingSampleSelector::select_beam_range_by_angles,
             DOC_PingSampleSelector(select_beam_range_by_angles),
             nb::arg("min_beam_angle") = std::nullopt,
             nb::arg("max_beam_angle") = std::nullopt,
             nb::arg("beam_step")      = std::nullopt)
        .def("select_sample_range_by_ranges",
             &PingSampleSelector::select_sample_range_by_ranges,
             DOC_PingSampleSelector(select_sample_range_by_ranges),
             nb::arg("min_sample_range") = std::nullopt,
             nb::arg("max_sample_range") = std::nullopt,
             nb::arg("sample_step")      = std::nullopt)
        .def("select_transmit_sectors",
             &PingSampleSelector::select_transmit_sectors,
             DOC_PingSampleSelector(select_transmit_sectors),
             nb::arg("transmit_sectors") = std::nullopt)
        .def("select_transmit_sectors_by_beam_angles",
             &PingSampleSelector::select_transmit_sectors_by_beam_angles,
             DOC_PingSampleSelector(select_transmit_sectors_by_beam_angles),
             nb::arg("transmit_sector_min_beam_angle") = std::nullopt,
             nb::arg("transmit_sector_max_beam_angle") = std::nullopt)
        .def("select_min_minslant_range_percent",
             &PingSampleSelector::select_min_minslant_range_percent,
             DOC_PingSampleSelector(select_min_minslant_range_percent),
             nb::arg("min_minslant_range_percent") = std::nullopt)
        .def("select_max_minslant_range_percent",
             &PingSampleSelector::select_max_minslant_range_percent,
             DOC_PingSampleSelector(select_max_minslant_range_percent),
             nb::arg("max_minslant_range_percent") = std::nullopt)
        .def("select_minslant_range_percent",
             &PingSampleSelector::select_minslant_range_percent,
             DOC_PingSampleSelector(select_minslant_range_percent),
             nb::arg("min_minslant_range_percent") = std::nullopt,
             nb::arg("max_minslant_range_percent") = std::nullopt)
        .def("select_min_bottom_range_percent",
             &PingSampleSelector::select_min_bottom_range_percent,
             DOC_PingSampleSelector(select_min_bottom_range_percent),
             nb::arg("min_bottom_range_percent") = std::nullopt)
        .def("select_max_bottom_range_percent",
             &PingSampleSelector::select_max_bottom_range_percent,
             DOC_PingSampleSelector(select_max_bottom_range_percent),
             nb::arg("max_bottom_range_percent") = std::nullopt)
        .def("select_bottom_range_percent",
             &PingSampleSelector::select_bottom_range_percent,
             DOC_PingSampleSelector(select_bottom_range_percent),
             nb::arg("min_bottom_range_percent") = std::nullopt,
             nb::arg("max_bottom_range_percent") = std::nullopt)

        .def("set_sample_step",
             &PingSampleSelector::set_sample_step,
             DOC_PingSampleSelector(sample_step),
             nb::arg("sample_step"))
        .def("set_beam_step",
             &PingSampleSelector::set_beam_step,
             DOC_PingSampleSelector(beam_step),
             nb::arg("beam_step"))

        // ----- operators -----
        .def(nanobind::self == nanobind::self, DOC_PingSampleSelector(operator_eq))

        // ----- nanobind macros -----
        // default copy functions
        __PYCLASS_DEFAULT_COPY__(PingSampleSelector)
        // default binary functions
        __PYCLASS_DEFAULT_BINARY__(PingSampleSelector)
        // default printing functions
        __PYCLASS_DEFAULT_PRINTING__(PingSampleSelector);
}

}
}
}
}