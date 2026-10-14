// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "i_pingcommon.hpp"

#include <algorithm>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <magic_enum/magic_enum.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

I_PingCommon::I_PingCommon() = default;

bool I_PingCommon::has_ping_info() const
{
    return bool(_ping_info);
}

const std::shared_ptr<PingInfo>& I_PingCommon::get_ping_info() const
{
    return _ping_info;
}

void I_PingCommon::set_ping_info(std::shared_ptr<PingInfo> ping_info)
{
    _ping_info = std::move(ping_info);
}

std::string I_PingCommon::class_name() const
{
    return "I_PingCommon";
}

std::map<t_pingfeature, std::function<bool()>> I_PingCommon::primary_feature_functions() const
{
    return {};
}

std::map<t_pingfeature, std::function<bool()>> I_PingCommon::secondary_feature_functions() const
{
    std::map<t_pingfeature, std::function<bool()>> features;
    // the shared ping info (channel id, timestamp, sensor configuration, navigation) is a feature
    // common to all ping objects (I_Ping, I_PingBottom, I_PingWatercolumn)
    __PING_REGISTER_FEATURE__(features, ping_info);
    return features;
}

std::map<t_pingfeature, std::function<bool()>> I_PingCommon::feature_group_functions() const
{
    return {};
}

bool I_PingCommon::has_features() const
{
    for (const auto& [feature_name, has_feature] : this->primary_feature_functions())
        if (has_feature())
            return true;

    for (const auto& [feature_name, has_feature] : this->secondary_feature_functions())
        if (has_feature())
            return true;

    for (const auto& [feature_name, has_feature] : this->feature_group_functions())
        if (has_feature())
            return true;

    return false;
}

bool I_PingCommon::has_any_of_features(const std::vector<o_pingfeature>& features) const
{
    for (const auto& feature : features)
    {
        if (has_feature(feature))
            return true;
    }

    return false;
}

bool I_PingCommon::has_all_of_features(const std::vector<o_pingfeature>& features) const
{
    for (const auto& feature : features)
    {
        if (!has_feature(feature))
            return false;
    }

    return true;
}

bool I_PingCommon::has_primary_features() const
{
    for (const auto& [feature_name, has_feature] : this->primary_feature_functions())
        if (has_feature())
            return true;

    return false;
}

bool I_PingCommon::has_secondary_features() const
{
    for (const auto& [feature_name, has_feature] : this->secondary_feature_functions())
        if (has_feature())
            return true;

    return false;
}

bool I_PingCommon::has_feature_groups() const
{
    for (const auto& [feature_name, has_feature] : this->feature_group_functions())
        if (has_feature())
            return true;

    return false;
}

bool I_PingCommon::has_feature(o_pingfeature feature) const
{
    auto primary    = this->primary_feature_functions();
    auto it_primary = primary.find(feature.value);
    if (it_primary != primary.end())
        return it_primary->second();

    auto secondary    = this->secondary_feature_functions();
    auto it_secondary = secondary.find(feature.value);
    if (it_secondary != secondary.end())
        return it_secondary->second();

    auto groups   = this->feature_group_functions();
    auto it_group = groups.find(feature.value);
    if (it_group != groups.end())
        return it_group->second();

    // The requested value is a valid t_pingfeature, but this particular ping class does not register
    // it (e.g. a watercolumn feature queried on a bottom ping, or vice versa). This is a usage error,
    // not an internal bug, so report which features this class actually provides.
    std::vector<t_pingfeature> registered;
    for (const auto& feature_map : { primary, secondary, groups })
        for (const auto& [registered_feature, has_registered_feature] : feature_map)
            registered.push_back(registered_feature);
    std::sort(registered.begin(), registered.end());

    std::vector<std::string_view> registered_names;
    registered_names.reserve(registered.size());
    for (auto f : registered)
        registered_names.push_back(magic_enum::enum_name(f));

    throw std::runtime_error(fmt::format(
        "Feature '{}' is not available on '{}': it is a valid ping feature but is not registered for "
        "this class (it may belong to a different ping component, e.g. a watercolumn feature queried "
        "on a bottom ping). Features registered for '{}': [{}]",
        magic_enum::enum_name(feature.value),
        class_name(),
        class_name(),
        fmt::join(registered_names, ", ")));
}

std::string I_PingCommon::feature_string(bool available, const std::string& prefix) const
{
    std::string features;

    for (const auto& features_map :
         { this->primary_feature_functions(), this->secondary_feature_functions() })
        for (const auto& [feature, has_feature] : features_map)
        {
            if (has_feature() == available)
            {
                if (!features.empty())
                    features += ", ";
                features += prefix;
                features += magic_enum::enum_name(feature);
            }
        }

    return features;
}

std::vector<t_pingfeature> I_PingCommon::possible_features()
{
    std::vector<t_pingfeature> features;

    for (const auto& features_map :
         { this->primary_feature_functions(), this->secondary_feature_functions() })
        for (const auto& [feature, has_feature] : features_map)
            features.push_back(feature);

    return features;
}

std::vector<t_pingfeature> I_PingCommon::possible_feature_groups()
{
    std::vector<t_pingfeature> features;

    for (const auto& [feature, has_feature] : this->feature_group_functions())
        features.push_back(feature);

    return features;
}

std::vector<t_pingfeature> I_PingCommon::available_features(bool available)
{
    std::vector<t_pingfeature> features;

    for (const auto& features_map :
         { this->primary_feature_functions(), this->secondary_feature_functions() })
        for (const auto& [feature, has_feature] : features_map)
            if (has_feature() == available)
                features.push_back(feature);

    return features;
}

std::vector<t_pingfeature> I_PingCommon::available_feature_groups(bool available)
{
    std::vector<t_pingfeature> features;

    for (const auto& [feature, has_feature] : this->feature_group_functions())
        if (has_feature() == available)
            features.push_back(feature);

    return features;
}

std::string I_PingCommon::feature_groups_string(bool available, const std::string& prefix) const
{
    std::string features;

    for (const auto& [feature, has_feature] : this->feature_group_functions())
    {
        if (has_feature() == available)
        {
            if (!features.empty())
                features += ", ";
            features += prefix;
            features += magic_enum::enum_name(feature);
        }
    }

    return features;
}

std::string I_PingCommon::registered_features() const
{
    std::string features = primary_features();

    auto secondary_features_map = this->secondary_feature_functions();
    if (!secondary_features_map.empty())
    {
        if (!features.empty())
            features += ", ";
        features += this->secondary_features();
    }

    return features;
}

std::string I_PingCommon::primary_features() const
{
    std::string features;

    for (const auto& [feature, has_feature] : this->primary_feature_functions())
    {
        if (!features.empty())
            features += ", ";
        features += magic_enum::enum_name(feature);
    }

    return features;
}

std::string I_PingCommon::secondary_features() const
{
    std::string features;

    for (const auto& [feature, has_feature] : this->secondary_feature_functions())
    {
        if (!features.empty())
            features += ", ";
        features += magic_enum::enum_name(feature);
    }

    return features;
}

std::string I_PingCommon::feature_groups() const
{
    std::string features;

    for (const auto& [feature, has_feature] : this->feature_group_functions())
    {
        if (!features.empty())
            features += ", ";
        features += magic_enum::enum_name(feature);
    }

    return features;
}

void I_PingCommon::load([[maybe_unused]] bool force)
{
    throw not_implemented("load", this->class_name());
}

void I_PingCommon::release()
{
    throw not_implemented("release", this->class_name());
}

bool I_PingCommon::loaded()
{
    throw not_implemented("load", this->class_name());
}

I_PingCommon::not_implemented::not_implemented(std::string_view method_name, std::string_view name)
    : std::runtime_error(
          fmt::format("method {} not implemented for ping type '{}'", method_name, name))
{
}

void I_PingCommon::print_features(tools::classhelper::ObjectPrinter& printer,
                                  const std::string&                 prefix) const
{
    std::string prefix_with_points = ".get_";
    std::string feature_header =
        prefix.empty() ? "Features" : fmt::format("Features(.{})", prefix);

    auto features     = feature_string(true, prefix_with_points);
    auto not_features = feature_string(false, prefix_with_points);

    if (!prefix.empty())
        features = "." + prefix + " : " + features;

    if (!not_features.empty())
    {
        printer.register_string(feature_header, features, std::string("Not:") + not_features);
    }
    else
    {
        printer.register_string(feature_header, features);
    }
}

void I_PingCommon::print_feature_groups(tools::classhelper::ObjectPrinter& printer,
                                        const std::string&                 prefix) const
{
    std::string prefix_with_points = prefix.empty() ? "." + prefix : "." + prefix + ".";
    std::string feature_header =
        prefix.empty() ? "Feature groups" : fmt::format("Feature groups(.{})", prefix);

    auto features     = feature_groups_string(true, prefix_with_points);
    auto not_features = feature_groups_string(false, prefix_with_points);
    if (!not_features.empty())
        printer.register_string(feature_header, features, std::string("Not:") + not_features);
    else
        printer.register_string(feature_header, features);
}

tools::classhelper::ObjectPrinter
I_PingCommon::__printer__(unsigned int float_precision, bool superscript_exponents) const
{
    tools::classhelper::ObjectPrinter printer(
        this->class_name(), float_precision, superscript_exponents);

    print_features(printer);
    if (!this->feature_group_functions().empty())
        print_feature_groups(printer);

    return printer;
}

} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping

template class themachinethatgoesping::tools::classhelper::Option<
    themachinethatgoesping::pingprocessing::datatypes::t_pingfeature>;