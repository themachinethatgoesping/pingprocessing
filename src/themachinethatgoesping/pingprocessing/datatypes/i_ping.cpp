// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "i_ping.hpp"

#include <memory>
#include <utility>

#include <themachinethatgoesping/tools/timeconv.hpp>

#include "i_pingbottom.hpp"
#include "i_pingwatercolumn.hpp"
#include "pinginfo.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

// ----- constructors -----
I_Ping::I_Ping()
{
    _ping_info = std::make_shared<PingInfo>();
}

I_Ping::I_Ping(const I_Ping& other)
    : I_PingCommon(other)
{
    // Give the copy its own, independent PingInfo (deep copy of the data). This is required by the
    // echosounder deep_copy() pattern (each ping gets its own channel id / sensor configuration) and
    // keeps the previous "copy == independent ping info" semantics.
    _ping_info = std::make_shared<PingInfo>(other._ping_info ? *other._ping_info : PingInfo());
}

I_Ping& I_Ping::operator=(const I_Ping& other)
{
    if (this != &other)
    {
        I_PingCommon::operator=(other);
        _ping_info = std::make_shared<PingInfo>(other._ping_info ? *other._ping_info : PingInfo());
    }
    return *this;
}

// ----- feature functions -----
std::map<t_pingfeature, std::function<bool()>> I_Ping::primary_feature_functions() const
{
    auto features = t_base ::primary_feature_functions();

    features[t_pingfeature::timestamp] = std::bind(&I_Ping::has_timestamp, this);
    // datetime is a timestamp alias that only exists in the python module (implemented as lambda)
    features[t_pingfeature::datetime]   = std::bind(&I_Ping::has_timestamp, this);
    features[t_pingfeature::channel_id] = std::bind(&I_Ping::has_channel_id, this);
    features[t_pingfeature::sensor_configuration] =
        std::bind(&I_Ping::has_sensor_configuration, this);
    features[t_pingfeature::sensor_data_latlon] = std::bind(&I_Ping::has_sensor_data_latlon, this);
    features[t_pingfeature::navigation_interpolator_latlon] =
        std::bind(&I_Ping::has_navigation_interpolator_latlon, this);
    features[t_pingfeature::geolocation] = std::bind(&I_Ping::has_geolocation, this);

    return features;
}

std::map<t_pingfeature, std::function<bool()>> I_Ping::secondary_feature_functions() const
{
    return t_base ::secondary_feature_functions();
}

std::map<t_pingfeature, std::function<bool()>> I_Ping::feature_group_functions() const
{
    auto features                        = t_base ::feature_group_functions();
    features[t_pingfeature::bottom]      = std::bind(&I_Ping::has_bottom, this);
    features[t_pingfeature::watercolumn] = std::bind(&I_Ping::has_watercolumn, this);
    return features;
}

std::string I_Ping::class_name() const
{
    return "I_Ping";
}

// ----- features (delegated to the shared PingInfo) -----
bool I_Ping::has_timestamp() const
{
    return _ping_info->has_timestamp();
}

void I_Ping::set_timestamp(double timestamp)
{
    _ping_info->set_timestamp(timestamp);
}

double I_Ping::get_timestamp() const
{
    return _ping_info->get_timestamp();
}

bool I_Ping::has_channel_id() const
{
    return _ping_info->has_channel_id();
}
const std::string& I_Ping::get_channel_id() const
{
    return _ping_info->get_channel_id();
}
void I_Ping::set_channel_id(const std::string& channel_id)
{
    _ping_info->set_channel_id(channel_id);
}
bool I_Ping::has_geolocation() const
{
    return _ping_info->has_geolocation();
}
navigation::datastructures::GeolocationLatLon I_Ping::get_geolocation(
    std::optional<std::string> target_id) const
{
    return _ping_info->get_geolocation(std::move(target_id));
}
const navigation::SensorConfiguration& I_Ping::get_sensor_configuration() const
{
    return _ping_info->get_sensor_configuration();
}
uint64_t I_Ping::get_sensor_configuration_base_hash() const
{
    return _ping_info->get_sensor_configuration_base_hash();
}
bool I_Ping::has_sensor_configuration() const
{
    return _ping_info->has_sensor_configuration();
}
void I_Ping::set_sensor_configuration(const navigation::SensorConfiguration& sensor_configuration)
{
    _ping_info->set_sensor_configuration(sensor_configuration);
}
void I_Ping::set_sensor_configuration_flyweight(
    SensorConfigurationFlyweight sensor_configuration)
{
    _ping_info->set_sensor_configuration_flyweight(std::move(sensor_configuration));
}
bool I_Ping::has_navigation_interpolator_latlon() const
{
    return _ping_info->has_navigation_interpolator_latlon();
}
bool I_Ping::has_sensor_data_latlon() const
{
    return _ping_info->has_sensor_data_latlon();
}
navigation::datastructures::SensordataLatLon I_Ping::get_sensor_data_latlon() const
{
    return _ping_info->get_sensor_data_latlon();
}
const navigation::NavigationInterpolatorLatLon& I_Ping::get_navigation_interpolator_latlon() const
{
    return _ping_info->get_navigation_interpolator_latlon();
}
void I_Ping::set_navigation_interpolator_latlon(
    const navigation::NavigationInterpolatorLatLon& nav_interpolator)
{
    _ping_info->set_navigation_interpolator_latlon(nav_interpolator);
}
void I_Ping::set_navigation_interpolator_latlon(
    NavigationInterpolatorLatLonFlyweight nav_interpolator)
{
    _ping_info->set_navigation_interpolator_latlon(std::move(nav_interpolator));
}
void I_Ping::load(bool force)
{
    if (has_bottom())
        bottom().load(force);
    if (has_watercolumn())
        watercolumn().load(force);
}
void I_Ping::release()
{
    if (has_bottom())
        bottom().release();
    if (has_watercolumn())
        watercolumn().release();
}
bool I_Ping::loaded()
{
    if (has_bottom() && !bottom().loaded())
        return false;
    if (has_watercolumn() && !watercolumn().loaded())
        return false;
    return true;
}
I_PingBottom& I_Ping::bottom()
{
    throw not_implemented("bottom", this->class_name());
}
const I_PingBottom& I_Ping::bottom() const
{
    return const_cast<I_Ping*>(this)->bottom();
}
I_PingWatercolumn& I_Ping::watercolumn()
{
    throw not_implemented("watercolumn", this->class_name());
}
const I_PingWatercolumn& I_Ping::watercolumn() const
{
    return const_cast<I_Ping*>(this)->watercolumn();
}

bool I_Ping::has_bottom() const
{
    try
    {
        return bottom().has_primary_features();
    }
    catch (const not_implemented&)
    {
        return false;
    }
}

bool I_Ping::has_watercolumn() const
{
    try
    {
        return watercolumn().has_primary_features();
    }
    catch (const not_implemented&)
    {
        return false;
    }
}
I_Ping::not_implemented::not_implemented(std::string_view method_name, std::string_view name)
    : std::runtime_error(
          fmt::format("method {} not implemented for ping type '{}'", method_name, name))
{
}
tools::classhelper::ObjectPrinter I_Ping::__printer__(unsigned int float_precision,
                                                      bool         superscript_exponents) const
{
    tools::classhelper::ObjectPrinter printer(
        this->class_name(), float_precision, superscript_exponents);

    printer.register_section("Ping infos");

    std::string time_str =
        tools::timeconv::unixtime_to_datestring(this->get_timestamp(), 2, "%d/%m/%Y %H:%M:%S");

    printer.register_string("Channel id", this->get_channel_id());
    printer.register_value("Time info", time_str, std::to_string(this->get_timestamp()));

    // print features
    printer.append(t_base::__printer__(float_precision, superscript_exponents));
    if (has_bottom())
        bottom().print_features(printer, "bottom");
    if (has_watercolumn())
        watercolumn().print_features(printer, "watercolumn");

    if (has_geolocation())
    {
        printer.register_section("Geolocation");
        printer.append(get_geolocation().__printer__(float_precision, superscript_exponents));
    }
    else
    {
        printer.register_string(
            "Geolocation",
            "not available",
            fmt::format("Sensor configuration: {}, Sensor data: {}",
                        has_sensor_configuration() ? "available" : "not available",
                        has_navigation_interpolator_latlon() ? "available" : "not available"));
    }

    return printer;
}

} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping
