// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>
#include <xtensor-python/nanobind/pytensor.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/i_pingbottom.hpp>
#include "i_mbesrawdetectioninfos.hpp"
#include "i_pingcommon.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {
namespace py_i_pingbottom {

#define DOC_I_PingBottom(ARG)                                                                     \
    DOC(themachinethatgoesping, pingprocessing, datatypes, I_PingBottom, ARG)

namespace pingtools = themachinethatgoesping::algorithms::pingtools;

template<typename T_BaseClass, typename T_PyClass>
void I_PingBottom_add_interface(T_PyClass& cls)
{
    namespace nb = nanobind;

    py_i_pingcommon::I_PingCommon_add_interface<T_BaseClass>(cls);

    // I_MBESRawDetectionInfos interface (nanobind has no multiple inheritance)
    py_i_mbesrawdetectioninfos::add_interface<T_BaseClass>(cls);

    cls.def("has_tx_signal_parameters_legacy",
            &T_BaseClass::has_tx_signal_parameters_legacy,
            DOC_I_PingBottom(has_tx_signal_parameters_legacy));
    cls.def("has_number_of_tx_sectors_legacy",
            &T_BaseClass::has_number_of_tx_sectors_legacy,
            DOC_I_PingBottom(has_number_of_tx_sectors_legacy));
    cls.def("has_beam_numbers_per_tx_sector_legacy",
            &T_BaseClass::has_beam_numbers_per_tx_sector_legacy,
            DOC_I_PingBottom(has_beam_numbers_per_tx_sector_legacy));
    cls.def("has_beam_selection_all",
            &T_BaseClass::has_beam_selection_all,
            DOC_I_PingBottom(has_beam_selection_all));
    cls.def("has_number_of_beams_legacy",
            &T_BaseClass::has_number_of_beams_legacy,
            DOC_I_PingBottom(has_number_of_beams_legacy));
    cls.def("has_tx_sector_per_beam_legacy",
            &T_BaseClass::has_tx_sector_per_beam_legacy,
            DOC_I_PingBottom(has_tx_sector_per_beam_legacy));

    cls.def("get_tx_signal_parameters_legacy",
            &T_BaseClass::get_tx_signal_parameters_legacy,
            DOC_I_PingBottom(get_tx_signal_parameters_legacy));
    cls.def("get_number_of_tx_sectors_legacy",
            &T_BaseClass::get_number_of_tx_sectors_legacy,
            DOC_I_PingBottom(get_number_of_tx_sectors_legacy));

    cls.def("get_tx_sector_per_beam_legacy",
            nb::overload_cast<>(&T_BaseClass::get_tx_sector_per_beam_legacy),
            DOC_I_PingBottom(get_tx_sector_per_beam_legacy));
    cls.def("get_tx_sector_per_beam_legacy",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_tx_sector_per_beam_legacy),
            DOC_I_PingBottom(get_tx_sector_per_beam_legacy),
            nb::arg("beam_selection"));

    cls.def("get_beam_numbers_per_tx_sector_legacy",
            nb::overload_cast<>(&T_BaseClass::get_beam_numbers_per_tx_sector_legacy),
            DOC_I_PingBottom(get_beam_numbers_per_tx_sector_legacy));
    cls.def("get_beam_numbers_per_tx_sector_legacy",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_beam_numbers_per_tx_sector_legacy),
            DOC_I_PingBottom(get_beam_numbers_per_tx_sector_legacy),
            nb::arg("beam_selection"));

    cls.def("get_beam_selection_all",
            nb::overload_cast<>(&T_BaseClass::get_beam_selection_all),
            DOC_I_PingBottom(get_beam_selection_all));
    cls.def("get_number_of_beams_legacy",
            &T_BaseClass::get_number_of_beams_legacy,
            DOC_I_PingBottom(get_number_of_beams_legacy));

    cls.def("get_beam_crosstrack_angles_legacy",
            nb::overload_cast<>(&T_BaseClass::get_beam_crosstrack_angles_legacy),
            DOC_I_PingBottom(get_beam_crosstrack_angles_legacy));
    cls.def("get_beam_crosstrack_angles_legacy",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_beam_crosstrack_angles_legacy),
            DOC_I_PingBottom(get_beam_crosstrack_angles_legacy),
            nb::arg("beam_selection"));

    cls.def("has_beam_crosstrack_angles_legacy",
            nb::overload_cast<>(&T_BaseClass::has_beam_crosstrack_angles_legacy, nb::const_),
            DOC_I_PingBottom(has_beam_crosstrack_angles_legacy));
    cls.def("has_xyz",
            nb::overload_cast<>(&T_BaseClass::has_xyz, nb::const_),
            DOC_I_PingBottom(has_xyz));
    cls.def("has_two_way_travel_times_legacy",
            nb::overload_cast<>(&T_BaseClass::has_two_way_travel_times_legacy, nb::const_),
            DOC_I_PingBottom(has_two_way_travel_times_legacy));

    cls.def("get_xyz",
            nb::overload_cast<>(&T_BaseClass::get_xyz),
            DOC_I_PingBottom(get_xyz));
    cls.def("get_xyz",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_xyz),
            DOC_I_PingBottom(get_xyz_2),
            nb::arg("beam_selection"));

    cls.def("get_bottom_z",
            nb::overload_cast<>(&T_BaseClass::get_bottom_z),
            DOC_I_PingBottom(get_bottom_z));
    cls.def("get_bottom_z",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_bottom_z),
            DOC_I_PingBottom(get_bottom_z_2),
            nb::arg("beam_selection"));

    cls.def("get_two_way_travel_times_legacy",
            nb::overload_cast<>(&T_BaseClass::get_two_way_travel_times_legacy),
            DOC_I_PingBottom(get_two_way_travel_times_legacy));
    cls.def("get_two_way_travel_times_legacy",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_two_way_travel_times_legacy),
            DOC_I_PingBottom(get_two_way_travel_times_legacy),
            nb::arg("beam_selection"));
}

}
}
}
}
}
