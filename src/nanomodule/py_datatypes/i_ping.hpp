// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>
#include <themachinethatgoesping/tools_nanobind/datetime.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/i_ping.hpp>
#include "i_pingcommon.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {
namespace py_i_ping {

#define DOC_I_Ping(ARG)                                                                            \
    DOC(themachinethatgoesping, pingprocessing, datatypes, I_Ping, ARG)

namespace navigation = themachinethatgoesping::navigation;

template<typename T_BaseClass, typename T_PyClass>
void I_Ping_add_interface(T_PyClass& cls)
{
    namespace nb = nanobind;

    py_i_pingcommon::I_PingCommon_add_interface<T_BaseClass>(cls);

    cls.def("has_timestamp", &T_BaseClass::has_timestamp, DOC_I_Ping(has_timestamp));
    cls.def("get_timestamp", &T_BaseClass::get_timestamp, DOC_I_Ping(timestamp));
    cls.def(
        "set_timestamp", &T_BaseClass::set_timestamp, DOC_I_Ping(timestamp), nb::arg("timestamp"));
    cls.def("has_datetime",
            &T_BaseClass::has_timestamp,
            "Return true if the timestamp is available that can be converted to a datetime");
    cls.def(
        "get_datetime",
        [](const T_BaseClass& self, double timezone_offset_hours) {
            return themachinethatgoesping::tools::nanobind_helper::unixtime_to_datetime(
                self.get_timestamp(), timezone_offset_hours);
        },
        nb::arg("timezone_offset_hours") = 0.,
        "Return the timestamp as datetime object");
    cls.def(
        "set_datetime",
        [](T_BaseClass& self, const nb::handle& datetime) {
            self.set_timestamp(
                themachinethatgoesping::tools::nanobind_helper::datetime_to_unixtime(datetime));
        },
        nb::arg("datetime"),
        "Set the timestamp using a datetime object");

    cls.def("has_channel_id", &T_BaseClass::has_channel_id, DOC_I_Ping(has_channel_id));
    cls.def("get_channel_id", &T_BaseClass::get_channel_id, DOC_I_Ping(channel_id));
    cls.def("set_channel_id",
            &T_BaseClass::set_channel_id,
            DOC_I_Ping(channel_id),
            nb::arg("channel_id"));

    cls.def("has_sensor_configuration",
            &T_BaseClass::has_sensor_configuration,
            DOC_I_Ping(has_sensor_configuration));
    cls.def("has_navigation_interpolator_latlon",
            &T_BaseClass::has_navigation_interpolator_latlon,
            DOC_I_Ping(has_navigation_interpolator_latlon));
    cls.def("has_sensor_data_latlon",
            &T_BaseClass::has_sensor_data_latlon,
            DOC_I_Ping(has_sensor_data_latlon));

    cls.def("get_sensor_configuration",
            &T_BaseClass::get_sensor_configuration,
            DOC_I_Ping(sensor_configuration),
            nb::rv_policy::reference_internal);
    cls.def("get_sensor_configuration_base_hash",
            &T_BaseClass::get_sensor_configuration_base_hash,
            DOC_I_Ping(get_sensor_configuration_base_hash));
    cls.def("set_sensor_configuration",
            &T_BaseClass::set_sensor_configuration,
            DOC_I_Ping(sensor_configuration),
            nb::arg("sensor_configuration"));

    cls.def("get_navigation_interpolator_latlon",
            &T_BaseClass::get_navigation_interpolator_latlon,
            DOC_I_Ping(navigation_interpolator_latlon),
            nb::rv_policy::reference_internal);
    cls.def("set_navigation_interpolator_latlon",
            nb::overload_cast<const navigation::NavigationInterpolatorLatLon&>(
                &T_BaseClass::set_navigation_interpolator_latlon),
            DOC_I_Ping(set_navigation_interpolator_latlon),
            nb::arg("nav_interpolator"));

    cls.def("get_sensor_data_latlon",
            &T_BaseClass::get_sensor_data_latlon,
            DOC_I_Ping(get_sensor_data_latlon));
    cls.def("has_geolocation", &T_BaseClass::has_geolocation, DOC_I_Ping(has_geolocation));
    cls.def(
        "get_geolocation",
        nb::overload_cast<std::optional<std::string>>(&T_BaseClass::get_geolocation, nb::const_),
        DOC_I_Ping(get_geolocation),
        nb::arg("target_id") = std::nullopt);

    cls.def_prop_ro("bottom",
                    nb::overload_cast<>(&T_BaseClass::bottom),
                    DOC_I_Ping(bottom),
                    nb::rv_policy::reference_internal);
    cls.def_prop_ro("watercolumn",
                    nb::overload_cast<>(&T_BaseClass::watercolumn),
                    DOC_I_Ping(watercolumn),
                    nb::rv_policy::reference_internal);

    cls.def("has_bottom", &T_BaseClass::has_bottom, DOC_I_Ping(has_bottom));
    cls.def("has_watercolumn", &T_BaseClass::has_watercolumn, DOC_I_Ping(has_watercolumn));
}

}
}
}
}
}
