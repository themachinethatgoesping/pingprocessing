// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 Sven Schorge, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
//

#pragma once

/* generated doc strings */
#include ".docstrings/pinginfo.doc.hpp"

/* std includes */
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <boost/container_hash/hash.hpp>
#include <boost/flyweight.hpp>
#include <boost/flyweight/intermodule_holder.hpp>

/* themachinethatgoesping includes */
#include <themachinethatgoesping/navigation/navigationinterpolatorlatlon.hpp>
#include <themachinethatgoesping/navigation/sensorconfiguration.hpp>
#include <themachinethatgoesping/tools/classhelper/objectprinter.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

/**
 * @brief Small, deduplicated bundle of the rarely-changing string metadata of a ping.
 *
 * This structure is stored inside PingInfo as a single boost::flyweight so that all pings that
 * share the same (channel_id, ...) combination only store one copy (one pointer) of it. It is
 * intentionally kept lightweight and holds only string metadata - more string fields can be added
 * here later without changing the number of pointers held by PingInfo. The sensor configuration and
 * the navigation interpolator are deliberately NOT part of this structure; they are stored as their
 * own flyweights directly on PingInfo.
 *
 * The structure is an implementation detail of PingInfo and is not exposed to Python.
 */
struct PingInfoData
{
    std::string channel_id; ///< channel id of the transducer

    bool operator==(const PingInfoData& other) const = default;
};

/**
 * @brief Boost hash function (used by boost::flyweight to deduplicate PingInfoData).
 *
 * @param data object to hash
 * @return std::size_t
 */
inline std::size_t hash_value(const PingInfoData& data)
{
    std::size_t seed = 0;
    boost::hash_combine(seed, data.channel_id);
    return seed;
}

/**
 * @brief Flyweight types for the (deduplicated) per-ping payloads.
 *
 * These flyweights are created in the echosounders module and handed to PingInfo, which lives in
 * the pingprocessing static library that is linked into BOTH pingprocessing_nanopy and
 * echosounders_nanopy. A default (static_holder) flyweight factory is hidden-visibility => one copy
 * per .so, so interning in one module and destroying across the mismatched factory of the other
 * corrupts the heap (std::bad_array_new_length on macOS/Windows; latent on Linux via
 * STB_GNU_UNIQUE). intermodule_holder shares a single process-wide factory across modules, keeping
 * the flyweight dedup while fixing this (same reasoning as I_PingWatercolumn).
 *
 * Using the SAME intermodule type end-to-end (interface -> setter -> member) is also what keeps the
 * hand-off a cheap handle copy: a flyweight must be passed AS a flyweight, never converted
 * value<->flyweight, so the (potentially expensive) hash is computed only once.
 */
using PingInfoDataFlyweight =
    boost::flyweights::flyweight<PingInfoData, boost::flyweights::intermodule_holder>;
using SensorConfigurationFlyweight =
    boost::flyweights::flyweight<navigation::SensorConfiguration,
                                 boost::flyweights::intermodule_holder>;
using NavigationInterpolatorLatLonFlyweight =
    boost::flyweights::flyweight<navigation::NavigationInterpolatorLatLon,
                                 boost::flyweights::intermodule_holder>;

/**
 * @brief Per-ping meta information shared between a ping and its bottom/watercolumn sub-objects.
 *
 * PingInfo stores the information that is identical for a ping and its I_PingBottom /
 * I_PingWatercolumn (channel id, timestamp, sensor configuration and navigation interpolator). An
 * I_Ping creates one PingInfo and hands the same shared pointer to its bottom and watercolumn
 * (done by the echosounder ping implementations) so that the sub-objects can access the sensor
 * configuration / navigation / geolocation without going back through the owning ping.
 *
 * Memory layout: a single flyweight for the (deduplicated) string metadata plus the (already
 * deduplicated) sensor configuration and navigation interpolator flyweights and the per-ping
 * timestamp.
 */
class PingInfo
{
    double _timestamp = 0; ///< Unix timestamp in seconds (saved in UTC0)

    /// Deduplicated string metadata (channel id, ...). Not exposed directly.
    PingInfoDataFlyweight _data;

    SensorConfigurationFlyweight          _sensor_configuration;
    NavigationInterpolatorLatLonFlyweight _navigation_interpolator_latlon;

  public:
    PingInfo()  = default;
    ~PingInfo() = default;

    //------ timestamp -----
    bool   has_timestamp() const;
    void   set_timestamp(double timestamp);
    double get_timestamp() const;

    //------ channel id -----
    bool               has_channel_id() const;
    const std::string& get_channel_id() const;
    void               set_channel_id(std::string_view channel_id);

    //------ combined setter (efficient initialization) -----

    /**
     * @brief Set channel id, sensor configuration and navigation interpolator at once.
     *
     * This is the setter that should be used during initialization. The sensor configuration and
     * the navigation interpolator are passed as flyweights so that no hashes have to be recomputed.
     *
     * @param channel_id channel id of the transducer
     * @param sensor_configuration sensor configuration flyweight
     * @param navigation_interpolator_latlon navigation interpolator flyweight
     */
    void set_data(std::string_view             channel_id,
                  SensorConfigurationFlyweight sensor_configuration,
                  NavigationInterpolatorLatLonFlyweight navigation_interpolator_latlon);

    //------ sensor configuration -----
    bool                                   has_sensor_configuration() const;
    const navigation::SensorConfiguration& get_sensor_configuration() const;
    void set_sensor_configuration(const navigation::SensorConfiguration& sensor_configuration);
    void set_sensor_configuration_flyweight(SensorConfigurationFlyweight sensor_configuration);

    /**
     * @brief Returns the hash of the base sensor configuration.
     * This hash can be used to get the correct navigation interpolator from the
     * navigation_data_interface. Note: This function is for testing and finding errors. It is
     * rather slow.
     *
     * @return uint64_t
     */
    uint64_t get_sensor_configuration_base_hash() const;

    //------ navigation interpolator / sensor data / geolocation -----
    bool                                            has_navigation_interpolator_latlon() const;
    const navigation::NavigationInterpolatorLatLon& get_navigation_interpolator_latlon() const;
    void                                            set_navigation_interpolator_latlon(
                                                   const navigation::NavigationInterpolatorLatLon& navigation_interpolator_latlon);
    void set_navigation_interpolator_latlon(
        NavigationInterpolatorLatLonFlyweight navigation_interpolator_latlon);

    bool                                         has_sensor_data_latlon() const;
    navigation::datastructures::SensordataLatLon get_sensor_data_latlon() const;

    bool                                          has_geolocation() const;
    navigation::datastructures::GeolocationLatLon get_geolocation(
        std::optional<std::string> target_id = std::nullopt) const;

    // ----- operators -----
    bool operator==(const PingInfo& other) const;
    bool operator!=(const PingInfo& other) const { return !(*this == other); }

    // ----- objectprinter -----
    tools::classhelper::ObjectPrinter __printer__(unsigned int float_precision,
                                                  bool         superscript_exponents) const;

    // -- class helper function macros --
    // define info_string and print functions (needs the __printer__ function)
    __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__
};

} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping
