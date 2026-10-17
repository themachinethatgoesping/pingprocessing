// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <boost/flyweight.hpp>

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>
#include <themachinethatgoesping/tools_nanobind/datetime.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/pinginfo.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {

#define DOC_PingInfo(ARG) DOC(themachinethatgoesping, pingprocessing, datatypes, PingInfo, ARG)

void init_c_pinginfo(nanobind::module_& m)
{
    namespace nb         = nanobind;
    namespace navigation = themachinethatgoesping::navigation;
    using namespace themachinethatgoesping::pingprocessing::datatypes;

    auto cls = nb::class_<PingInfo>(
        m, "PingInfo", DOC(themachinethatgoesping, pingprocessing, datatypes, PingInfo));

    cls.def(nb::init<>(), DOC_PingInfo(PingInfo));

    // ----- timestamp / datetime -----
    cls.def("has_timestamp", &PingInfo::has_timestamp, DOC_PingInfo(has_timestamp));
    cls.def("get_timestamp", &PingInfo::get_timestamp, DOC_PingInfo(get_timestamp));
    cls.def("set_timestamp", &PingInfo::set_timestamp, DOC_PingInfo(set_timestamp), nb::arg("timestamp"));
    cls.def("has_datetime",
            &PingInfo::has_timestamp,
            "Return true if the timestamp is available that can be converted to a datetime");
    cls.def(
        "get_datetime",
        [](const PingInfo& self, double timezone_offset_hours) {
            return themachinethatgoesping::tools::nanobind_helper::unixtime_to_datetime(
                self.get_timestamp(), timezone_offset_hours);
        },
        nb::arg("timezone_offset_hours") = 0.,
        "Return the timestamp as datetime object");
    cls.def(
        "set_datetime",
        [](PingInfo& self, const nb::handle& datetime) {
            self.set_timestamp(
                themachinethatgoesping::tools::nanobind_helper::datetime_to_unixtime(datetime));
        },
        nb::arg("datetime"),
        "Set the timestamp using a datetime object");

    // ----- channel id -----
    cls.def("has_channel_id", &PingInfo::has_channel_id, DOC_PingInfo(has_channel_id));
    cls.def("get_channel_id", &PingInfo::get_channel_id, DOC_PingInfo(get_channel_id));
    cls.def("set_channel_id", &PingInfo::set_channel_id, DOC_PingInfo(set_channel_id), nb::arg("channel_id"));

    // ----- combined setter -----
    cls.def(
        "set_data",
        [](PingInfo&                                       self,
           std::string_view                                channel_id,
           const navigation::SensorConfiguration&          sensor_configuration,
           const navigation::NavigationInterpolatorLatLon& navigation_interpolator_latlon) {
            self.set_data(
                channel_id,
                SensorConfigurationFlyweight(sensor_configuration),
                NavigationInterpolatorLatLonFlyweight(navigation_interpolator_latlon));
        },
        DOC_PingInfo(set_data),
        nb::arg("channel_id"),
        nb::arg("sensor_configuration"),
        nb::arg("navigation_interpolator_latlon"));

    // ----- sensor configuration -----
    cls.def("has_sensor_configuration",
            &PingInfo::has_sensor_configuration,
            DOC_PingInfo(has_sensor_configuration));
    cls.def("get_sensor_configuration",
            &PingInfo::get_sensor_configuration,
            DOC_PingInfo(get_sensor_configuration),
            nb::rv_policy::reference_internal);
    cls.def("get_sensor_configuration_base_hash",
            &PingInfo::get_sensor_configuration_base_hash,
            DOC_PingInfo(get_sensor_configuration_base_hash));
    cls.def("set_sensor_configuration",
            &PingInfo::set_sensor_configuration,
            DOC_PingInfo(set_sensor_configuration),
            nb::arg("sensor_configuration"));

    // ----- navigation interpolator / sensor data / geolocation -----
    cls.def("has_navigation_interpolator_latlon",
            &PingInfo::has_navigation_interpolator_latlon,
            DOC_PingInfo(has_navigation_interpolator_latlon));
    cls.def("get_navigation_interpolator_latlon",
            &PingInfo::get_navigation_interpolator_latlon,
            DOC_PingInfo(get_navigation_interpolator_latlon),
            nb::rv_policy::reference_internal);
    cls.def("set_navigation_interpolator_latlon",
            nb::overload_cast<const navigation::NavigationInterpolatorLatLon&>(
                &PingInfo::set_navigation_interpolator_latlon),
            DOC_PingInfo(set_navigation_interpolator_latlon),
            nb::arg("navigation_interpolator_latlon"));

    cls.def("has_sensor_data_latlon",
            &PingInfo::has_sensor_data_latlon,
            DOC_PingInfo(has_sensor_data_latlon));
    cls.def("get_sensor_data_latlon",
            &PingInfo::get_sensor_data_latlon,
            DOC_PingInfo(get_sensor_data_latlon));
    cls.def("has_geolocation", &PingInfo::has_geolocation, DOC_PingInfo(has_geolocation));
    cls.def(
        "get_geolocation",
        nb::overload_cast<std::optional<std::string>>(&PingInfo::get_geolocation, nb::const_),
        DOC_PingInfo(get_geolocation),
        nb::arg("target_id") = std::nullopt);

    cls __PYCLASS_DEFAULT_COPY__(PingInfo)
        __PYCLASS_DEFAULT_PRINTING__(PingInfo);
}

}
}
}
}
