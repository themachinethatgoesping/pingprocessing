// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 Sven Schorge, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
//

#pragma once

/* generated doc strings */
#include ".docstrings/i_ping.doc.hpp"

/* std includes */
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <boost/flyweight.hpp>
#include <fmt/format.h>

/* themachinethatgoesping includes */
#include "i_pingbottom.hpp"
#include "i_pingcommon.hpp"
#include "i_pingwatercolumn.hpp"
#include "pinginfo.hpp"
#include <themachinethatgoesping/navigation/navigationinterpolatorlatlon.hpp>
#include <themachinethatgoesping/tools/classhelper/objectprinter.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

/**
 * @brief Interface for all ping objects.
 *
 * The channel id, timestamp, sensor configuration and navigation interpolator are stored in a
 * shared PingInfo object (I_PingCommon::_ping_info). The accessors below are thin native wrappers
 * that delegate to that PingInfo. The same PingInfo pointer is shared with the bottom() and
 * watercolumn() sub-objects (done by the echosounder ping implementations), so they can access the
 * same sensor configuration / navigation / geolocation.
 */
class I_Ping : public I_PingCommon
{
  public:
    using t_base = I_PingCommon;

  protected:
    std::string class_name() const override;

    std::map<t_pingfeature, std::function<bool()>> primary_feature_functions() const override;
    std::map<t_pingfeature, std::function<bool()>> secondary_feature_functions() const override;
    std::map<t_pingfeature, std::function<bool()>> feature_group_functions() const override;

  public:
    I_Ping();
    I_Ping(const I_Ping& other);
    I_Ping(I_Ping&& other)            = default;
    I_Ping& operator=(const I_Ping& other);
    I_Ping& operator=(I_Ping&& other) = default;
    ~I_Ping() override                = default;

    //------ features (delegated to the shared PingInfo) -----
    bool   has_timestamp() const;
    void   set_timestamp(double timestamp);
    double get_timestamp() const;

    bool               has_channel_id() const;
    const std::string& get_channel_id() const;
    void               set_channel_id(const std::string& channel_id);

    bool                                          has_geolocation() const;
    navigation::datastructures::GeolocationLatLon get_geolocation(
        std::optional<std::string> target_id = std::nullopt) const;

    const navigation::SensorConfiguration& get_sensor_configuration() const;

    /**
     * @brief Returns the hash of the base sensor configuraiton.
     * This hash can be used to get the correct navigation interpolator from the
     * navigation_data_interface Note: This function is for testing and finding errors. It is rather
     * slow.
     *
     * @return uint64_t
     */
    uint64_t get_sensor_configuration_base_hash() const;

    bool has_sensor_configuration() const;
    void set_sensor_configuration(const navigation::SensorConfiguration& sensor_configuration);
    void set_sensor_configuration_flyweight(
        boost::flyweights::flyweight<navigation::SensorConfiguration> sensor_configuration);

    bool                                            has_navigation_interpolator_latlon() const;
    bool                                            has_sensor_data_latlon() const;
    navigation::datastructures::SensordataLatLon    get_sensor_data_latlon() const;
    const navigation::NavigationInterpolatorLatLon& get_navigation_interpolator_latlon() const;

    void set_navigation_interpolator_latlon(
        const navigation::NavigationInterpolatorLatLon& nav_interpolator);

    void set_navigation_interpolator_latlon(
        boost::flyweights::flyweight<navigation::NavigationInterpolatorLatLon> nav_interpolator);

    void load(bool force = false) override;
    void release() override;
    bool loaded() override;

    // ----- ping interface -----
    virtual I_PingBottom& bottom();
    const I_PingBottom&   bottom() const;

    virtual I_PingWatercolumn& watercolumn();
    const I_PingWatercolumn&   watercolumn() const;

    virtual bool has_bottom() const;
    virtual bool has_watercolumn() const;

  protected:
    struct not_implemented : public std::runtime_error
    {
        not_implemented(std::string_view method_name, std::string_view name);
    };

  public:
    // ----- objectprinter -----
    tools::classhelper::ObjectPrinter __printer__(unsigned int float_precision,
                                                  bool         superscript_exponents) const;

    // -- class helper function macros --
    // define info_string and print functions (needs the __printer__ function)
    __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__
};
}
} // namespace pingprocessing
} // namespace themachinethatgoesping
