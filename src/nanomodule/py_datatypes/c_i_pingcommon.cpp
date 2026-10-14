// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>
#include <themachinethatgoesping/tools_nanobind/enumhelper.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/i_pingcommon.hpp>
#include "c_i_pingcommon.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {

#define DOC_I_PingCommon(ARG)                                                                      \
    DOC(themachinethatgoesping, pingprocessing, datatypes, I_PingCommon, ARG)

void init_c_i_pingcommon(nanobind::module_& m)
{
    namespace nb = nanobind;
    using namespace themachinethatgoesping::pingprocessing::datatypes;

    nb::enum_<t_pingfeature>(
        m,
        "t_pingfeature",
        DOC(themachinethatgoesping, pingprocessing, datatypes, t_pingfeature))
        .value("timestamp", t_pingfeature::timestamp)
        .value("datetime", t_pingfeature::datetime)
        .value("channel_id", t_pingfeature::channel_id)
        .value("sensor_configuration", t_pingfeature::sensor_configuration)
        .value("navigation_interpolator_latlon", t_pingfeature::navigation_interpolator_latlon)
        .value("sensor_data_latlon", t_pingfeature::sensor_data_latlon)
        .value("geolocation", t_pingfeature::geolocation)
        .value("bottom", t_pingfeature::bottom)
        .value("watercolumn", t_pingfeature::watercolumn)
        .value("tx_signal_parameters_legacy", t_pingfeature::tx_signal_parameters_legacy)
        .value("beam_numbers_per_tx_sector_legacy", t_pingfeature::beam_numbers_per_tx_sector_legacy)
        .value("beam_selection_all", t_pingfeature::beam_selection_all)
        .value("number_of_beams_legacy", t_pingfeature::number_of_beams_legacy)
        .value("tx_sector_per_beam_legacy", t_pingfeature::tx_sector_per_beam_legacy)
        .value("number_of_tx_sectors_legacy", t_pingfeature::number_of_tx_sectors_legacy)
        .value("beam_crosstrack_angles_legacy", t_pingfeature::beam_crosstrack_angles_legacy)
        .value("two_way_travel_times_legacy", t_pingfeature::two_way_travel_times_legacy)
        .value("xyz", t_pingfeature::xyz)
        .value("ref_pingtime", t_pingfeature::ref_pingtime)
        .value("ref_pose", t_pingfeature::ref_pose)
        .value("surface_sound_speed", t_pingfeature::surface_sound_speed)
        .value("sector_beam_numbers", t_pingfeature::sector_beam_numbers)
        .value("sector_tx_time_delays", t_pingfeature::sector_tx_time_delays)
        .value("sector_tx_steering_angles", t_pingfeature::sector_tx_steering_angles)
        .value("sector_tx_offsets", t_pingfeature::sector_tx_offsets)
        .value("rx_offsets", t_pingfeature::rx_offsets)
        .value("number_of_soundings", t_pingfeature::number_of_soundings)
        .value("soundings_two_way_travel_times", t_pingfeature::soundings_two_way_travel_times)
        .value("soundings_rx_steering_angles", t_pingfeature::soundings_rx_steering_angles)
        .value("soundings_sector_numbers", t_pingfeature::soundings_sector_numbers)
        .value("bottom_range_samples", t_pingfeature::bottom_range_samples)
        .value("amplitudes", t_pingfeature::amplitudes)
        .value("ap", t_pingfeature::ap)
        .value("av", t_pingfeature::av)
        .value("power", t_pingfeature::power)
        .value("rp", t_pingfeature::rp)
        .value("rv", t_pingfeature::rv)
        .value("pp", t_pingfeature::pp)
        .value("pv", t_pingfeature::pv)
        .value("sp", t_pingfeature::sp)
        .value("sv", t_pingfeature::sv)
        .value("watercolumn_calibration", t_pingfeature::watercolumn_calibration)
        .value("multisectorwatercolumn_calibration", t_pingfeature::multisectorwatercolumn_calibration);

    tools::nanobind_helper::make_option_class<o_pingfeature>(m, "o_pingfeature");

    auto cls =
        nb::class_<I_PingCommon>(
            m,
            "I_PingCommon",
            DOC(themachinethatgoesping, pingprocessing, datatypes, I_PingCommon));

    py_i_pingcommon::I_PingCommon_add_interface<I_PingCommon>(cls);

    cls __PYCLASS_DEFAULT_COPY__(I_PingCommon)
        __PYCLASS_DEFAULT_PRINTING__(I_PingCommon);
}

}
}
}
}
