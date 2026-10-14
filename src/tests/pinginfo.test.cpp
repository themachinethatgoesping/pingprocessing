// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_test_macros.hpp>

#include <boost/flyweight.hpp>

#include <themachinethatgoesping/navigation/navigationinterpolatorlatlon.hpp>
#include <themachinethatgoesping/navigation/sensorconfiguration.hpp>
#include <themachinethatgoesping/pingprocessing/datatypes/pinginfo.hpp>

using namespace themachinethatgoesping::pingprocessing::datatypes;
namespace navigation = themachinethatgoesping::navigation;

namespace {

/// A sensor configuration that differs from a default-constructed one.
navigation::SensorConfiguration make_sensor_configuration(float waterline_offset = 2.5f)
{
    navigation::SensorConfiguration sc;
    sc.set_waterline_offset(waterline_offset);
    return sc;
}

/// A navigation interpolator that differs from a default-constructed one.
navigation::NavigationInterpolatorLatLon make_navigation_interpolator(
    const navigation::SensorConfiguration& sc)
{
    navigation::NavigationInterpolatorLatLon nav(sc);
    nav.set_data_position({ 0., 10. }, { 50.0, 50.1 }, { 8.0, 8.1 });
    return nav;
}

} // namespace

TEST_CASE("PingInfo: a default PingInfo has no features", "[pinginfo]")
{
    PingInfo pi;

    REQUIRE_FALSE(pi.has_timestamp());
    REQUIRE_FALSE(pi.has_channel_id());
    REQUIRE_FALSE(pi.has_sensor_configuration());
    REQUIRE_FALSE(pi.has_navigation_interpolator_latlon());
    REQUIRE_FALSE(pi.has_sensor_data_latlon());
    REQUIRE_FALSE(pi.has_geolocation());

    REQUIRE(pi.get_channel_id().empty());
    REQUIRE(pi.get_timestamp() == 0.0);
}

TEST_CASE("PingInfo: timestamp set/get/has", "[pinginfo]")
{
    PingInfo pi;
    pi.set_timestamp(1234.5);

    REQUIRE(pi.has_timestamp());
    REQUIRE(pi.get_timestamp() == 1234.5);
}

TEST_CASE("PingInfo: channel id set/get/has", "[pinginfo]")
{
    PingInfo pi;
    pi.set_channel_id("TRX-1");

    REQUIRE(pi.has_channel_id());
    REQUIRE(pi.get_channel_id() == "TRX-1");
}

TEST_CASE("PingInfo: has_ flags are derived from the (flyweight) content (no _set bools)",
          "[pinginfo]")
{
    auto sc  = make_sensor_configuration();
    auto nav = make_navigation_interpolator(sc);

    PingInfo pi;

    REQUIRE_FALSE(pi.has_sensor_configuration());
    pi.set_sensor_configuration(sc);
    REQUIRE(pi.has_sensor_configuration());
    REQUIRE(pi.get_sensor_configuration().binary_hash() == sc.binary_hash());
    REQUIRE(pi.get_sensor_configuration_base_hash() == sc.binary_hash());

    REQUIRE_FALSE(pi.has_navigation_interpolator_latlon());
    pi.set_navigation_interpolator_latlon(nav);
    REQUIRE(pi.has_navigation_interpolator_latlon());
    REQUIRE(pi.has_sensor_data_latlon());

    // geolocation requires both a sensor configuration and a navigation interpolator
    REQUIRE(pi.has_geolocation());
}

TEST_CASE("PingInfo: combined set_data sets channel id, sensor configuration and navigation",
          "[pinginfo]")
{
    auto sc  = make_sensor_configuration(1.0f);
    auto nav = make_navigation_interpolator(sc);

    PingInfo pi;
    pi.set_data("TRX-2",
                boost::flyweights::flyweight<navigation::SensorConfiguration>(sc),
                boost::flyweights::flyweight<navigation::NavigationInterpolatorLatLon>(nav));

    REQUIRE(pi.get_channel_id() == "TRX-2");
    REQUIRE(pi.has_sensor_configuration());
    REQUIRE(pi.has_navigation_interpolator_latlon());
    REQUIRE(pi.has_geolocation());
}

TEST_CASE("PingInfo: a copy is independent (deep copy of the data)", "[pinginfo]")
{
    PingInfo pi;
    pi.set_channel_id("A");
    pi.set_timestamp(100.0);

    PingInfo copy = pi;
    copy.set_channel_id("B");
    copy.set_timestamp(200.0);

    REQUIRE(pi.get_channel_id() == "A");
    REQUIRE(pi.get_timestamp() == 100.0);
    REQUIRE(copy.get_channel_id() == "B");
    REQUIRE(copy.get_timestamp() == 200.0);
}

TEST_CASE("PingInfo: equality operator", "[pinginfo]")
{
    PingInfo a;
    PingInfo b;

    a.set_channel_id("X");
    a.set_timestamp(5.0);
    b.set_channel_id("X");
    b.set_timestamp(5.0);
    REQUIRE(a == b);

    b.set_channel_id("Y");
    REQUIRE(a != b);
}
