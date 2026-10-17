// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "pinginfo.hpp"

#include <string>
#include <utility>

#include <themachinethatgoesping/tools/timeconv.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

// ----- timestamp -----
bool PingInfo::has_timestamp() const
{
    return _timestamp > 0;
}
void PingInfo::set_timestamp(double timestamp)
{
    _timestamp = timestamp;
}
double PingInfo::get_timestamp() const
{
    return _timestamp;
}

// ----- channel id -----
bool PingInfo::has_channel_id() const
{
    return !_data.get().channel_id.empty();
}
const std::string& PingInfo::get_channel_id() const
{
    return _data.get().channel_id;
}
void PingInfo::set_channel_id(std::string_view channel_id)
{
    // copy the current (deduplicated) string metadata, update the channel id and re-intern the
    // struct as a whole (this keeps any future string fields of PingInfoData intact)
    PingInfoData data = _data.get();
    data.channel_id.assign(channel_id);
    _data = std::move(data);
}

// ----- combined setter -----
void PingInfo::set_data(std::string_view             channel_id,
                        SensorConfigurationFlyweight sensor_configuration,
                        NavigationInterpolatorLatLonFlyweight navigation_interpolator_latlon)
{
    _data                           = PingInfoData{ std::string(channel_id) };
    _sensor_configuration           = std::move(sensor_configuration); // avoid reference counting
    _navigation_interpolator_latlon = std::move(navigation_interpolator_latlon);
}

// ----- sensor configuration -----
bool PingInfo::has_sensor_configuration() const
{
    // "empty" == a default constructed SensorConfiguration. The cached binary hash makes this a
    // cheap (value based, module independent) comparison that replaces the previous _..._set bool.
    static const auto empty_hash = navigation::SensorConfiguration().binary_hash();
    return _sensor_configuration.get().binary_hash() != empty_hash;
}
const navigation::SensorConfiguration& PingInfo::get_sensor_configuration() const
{
    return _sensor_configuration.get();
}
void PingInfo::set_sensor_configuration(
    const navigation::SensorConfiguration& sensor_configuration)
{
    _sensor_configuration = sensor_configuration;
}
void PingInfo::set_sensor_configuration_flyweight(
    SensorConfigurationFlyweight sensor_configuration)
{
    _sensor_configuration = std::move(sensor_configuration); // avoid reference counting
}
uint64_t PingInfo::get_sensor_configuration_base_hash() const
{
    return get_sensor_configuration().binary_hash();
}

// ----- navigation interpolator / sensor data / geolocation -----
bool PingInfo::has_navigation_interpolator_latlon() const
{
    static const auto empty_hash = navigation::NavigationInterpolatorLatLon().binary_hash();
    return _navigation_interpolator_latlon.get().binary_hash() != empty_hash;
}
const navigation::NavigationInterpolatorLatLon& PingInfo::get_navigation_interpolator_latlon() const
{
    return _navigation_interpolator_latlon.get();
}
void PingInfo::set_navigation_interpolator_latlon(
    const navigation::NavigationInterpolatorLatLon& navigation_interpolator_latlon)
{
    _navigation_interpolator_latlon = navigation_interpolator_latlon;
}
void PingInfo::set_navigation_interpolator_latlon(
    NavigationInterpolatorLatLonFlyweight navigation_interpolator_latlon)
{
    _navigation_interpolator_latlon = std::move(navigation_interpolator_latlon);
}

bool PingInfo::has_sensor_data_latlon() const
{
    return has_navigation_interpolator_latlon();
}
navigation::datastructures::SensordataLatLon PingInfo::get_sensor_data_latlon() const
{
    return get_navigation_interpolator_latlon().get_sensor_data(get_timestamp());
}

bool PingInfo::has_geolocation() const
{
    return has_sensor_configuration() && has_navigation_interpolator_latlon();
}
navigation::datastructures::GeolocationLatLon PingInfo::get_geolocation(
    std::optional<std::string> target_id) const
{
    const auto& sc = get_sensor_configuration();

    return sc.compute_target_position(
        target_id.value_or(sc.get_transducer_transmit_receive_id(get_channel_id()).first),
        get_sensor_data_latlon());
}

// ----- operators -----
bool PingInfo::operator==(const PingInfo& other) const
{
    return _timestamp == other._timestamp && _data == other._data &&
           _sensor_configuration == other._sensor_configuration &&
           _navigation_interpolator_latlon == other._navigation_interpolator_latlon;
}

// ----- objectprinter -----
tools::classhelper::ObjectPrinter PingInfo::__printer__(unsigned int float_precision,
                                                        bool         superscript_exponents) const
{
    tools::classhelper::ObjectPrinter printer("PingInfo", float_precision, superscript_exponents);

    printer.register_section("Ping info");

    std::string time_str =
        tools::timeconv::unixtime_to_datestring(_timestamp, 2, "%d/%m/%Y %H:%M:%S");

    printer.register_string("Channel id", this->get_channel_id());
    printer.register_value("Time info", time_str, std::to_string(_timestamp));
    printer.register_string("Sensor configuration",
                            has_sensor_configuration() ? "available" : "not available");
    printer.register_string("Navigation",
                            has_navigation_interpolator_latlon() ? "available" : "not available");

    if (has_geolocation())
    {
        printer.register_section("Geolocation");
        printer.append(get_geolocation().__printer__(float_precision, superscript_exponents));
    }

    return printer;
}

} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping
