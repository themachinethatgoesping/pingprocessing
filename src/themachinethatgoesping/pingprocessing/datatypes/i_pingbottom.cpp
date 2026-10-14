// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "i_pingbottom.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

std::string I_PingBottom::class_name() const
{
	return "I_PingBottom";
}

std::map<t_pingfeature, std::function<bool()>> I_PingBottom::primary_feature_functions() const
{
	auto features = t_base::primary_feature_functions();
	__PING_REGISTER_FEATURE__(features, two_way_travel_times_legacy);
	__PING_REGISTER_FEATURE__(features, xyz);

	return features;
}

std::map<t_pingfeature, std::function<bool()>> I_PingBottom::secondary_feature_functions() const
{
	auto features = t_base::secondary_feature_functions();
	__PING_REGISTER_FEATURE__(features, tx_signal_parameters_legacy);
	__PING_REGISTER_FEATURE__(features, number_of_tx_sectors_legacy);
	__PING_REGISTER_FEATURE__(features, beam_crosstrack_angles_legacy);

	// I_MBESRawDetectionInfos features
	__PING_REGISTER_FEATURE__(features, ref_pingtime);
	__PING_REGISTER_FEATURE__(features, ref_pose);
	__PING_REGISTER_FEATURE__(features, surface_sound_speed);
	__PING_REGISTER_FEATURE__(features, sector_beam_numbers);
	__PING_REGISTER_FEATURE__(features, sector_tx_time_delays);
	__PING_REGISTER_FEATURE__(features, sector_tx_steering_angles);
	__PING_REGISTER_FEATURE__(features, sector_tx_offsets);
	__PING_REGISTER_FEATURE__(features, rx_offsets);
	__PING_REGISTER_FEATURE__(features, number_of_soundings);
	__PING_REGISTER_FEATURE__(features, soundings_two_way_travel_times);
	__PING_REGISTER_FEATURE__(features, soundings_rx_steering_angles);
	__PING_REGISTER_FEATURE__(features, soundings_sector_numbers);

	return features;
}

std::map<t_pingfeature, std::function<bool()>> I_PingBottom::feature_group_functions() const
{
	return t_base::feature_group_functions();
}

I_PingBottom::I_PingBottom(std::shared_ptr<PingInfo> ping_info)
{
	_ping_info = std::move(ping_info);
}

I_PingBottom::~I_PingBottom() = default;

bool I_PingBottom::has_tx_signal_parameters_legacy() const
{
	return false;
}

bool I_PingBottom::has_number_of_tx_sectors_legacy() const
{
	return false;
}

bool I_PingBottom::has_beam_numbers_per_tx_sector_legacy() const
{
	return false;
}

bool I_PingBottom::has_tx_sector_per_beam_legacy() const
{
	return false;
}

bool I_PingBottom::has_beam_selection_all() const
{
	return has_number_of_beams_legacy();
}

bool I_PingBottom::has_number_of_beams_legacy() const
{
	return false;
}

bool I_PingBottom::has_beam_crosstrack_angles_legacy() const
{
	return false;
}

bool I_PingBottom::has_two_way_travel_times_legacy() const
{
	return false;
}

bool I_PingBottom::has_xyz() const
{
	return false;
}

std::vector<algorithms::signalprocessing::datastructures::TxSignalParameters>
	I_PingBottom::get_tx_signal_parameters_legacy()
{
	throw not_implemented(__func__, this->class_name());
}

size_t I_PingBottom::get_number_of_tx_sectors_legacy()
{
	throw not_implemented(__func__, this->class_name());
}

xt::xtensor<size_t, 1> I_PingBottom::get_tx_sector_per_beam_legacy()
{
	return get_tx_sector_per_beam_legacy(get_beam_selection_all());
}

xt::xtensor<size_t, 1>
	I_PingBottom::get_tx_sector_per_beam_legacy([[maybe_unused]] const algorithms::pingtools::BeamSelection& bs)
{
	throw not_implemented(__func__, class_name());
}

std::vector<std::vector<size_t>> I_PingBottom::get_beam_numbers_per_tx_sector_legacy()
{
	return get_beam_numbers_per_tx_sector_legacy(get_beam_selection_all());
}

std::vector<std::vector<size_t>> I_PingBottom::get_beam_numbers_per_tx_sector_legacy(
	[[maybe_unused]] const algorithms::pingtools::BeamSelection& bs)
{
	throw not_implemented(__func__, class_name());
}

algorithms::pingtools::BeamSelection I_PingBottom::get_beam_selection_all()
{
	return algorithms::pingtools::BeamSelection(get_number_of_beams_legacy());
}

uint32_t I_PingBottom::get_number_of_beams_legacy()
{
	throw not_implemented(__func__, class_name());
}

xt::xtensor<float, 1> I_PingBottom::get_beam_crosstrack_angles_legacy()
{
	return get_beam_crosstrack_angles_legacy(get_beam_selection_all());
}

xt::xtensor<float, 1> I_PingBottom::get_beam_crosstrack_angles_legacy(
	[[maybe_unused]] const algorithms::pingtools::BeamSelection& bs)
{
	throw not_implemented(__func__, class_name());
}

algorithms::geoprocessing::datastructures::XYZ<1> I_PingBottom::get_xyz()
{
	return get_xyz(get_beam_selection_all());
}

algorithms::geoprocessing::datastructures::XYZ<1> I_PingBottom::get_xyz(
	[[maybe_unused]] const algorithms::pingtools::BeamSelection& selection)
{
	throw not_implemented(__func__, this->class_name());
}

float I_PingBottom::get_bottom_z()
{
	return get_bottom_z(get_beam_selection_all());
}

float I_PingBottom::get_bottom_z(const algorithms::pingtools::BeamSelection& selection)
{
	auto xyz = get_xyz(selection);

	if (xyz.size() == 0)
		throw std::runtime_error(fmt::format("Error[{}]: No valid bottom z found", __func__));
	if (xyz.size() == 1)
		return xyz.z[0];

	std::vector<float> bottom_distances(xyz.z.begin(), xyz.z.end());

	auto median_it = bottom_distances.begin() + bottom_distances.size() / 2;
	std::nth_element(bottom_distances.begin(), median_it, bottom_distances.end());
	auto median = *median_it;

	auto q3_it = bottom_distances.begin() + 3 * bottom_distances.size() / 4;
	std::nth_element(median_it, q3_it, bottom_distances.end());
	auto q3 = *q3_it;

	auto q1_it = bottom_distances.begin() + bottom_distances.size() / 4;
	std::nth_element(bottom_distances.begin(), q1_it, median_it);
	auto q1      = *q1_it;
	auto min_iqr = median - (q3 - q1) * 1.5F;

	std::sort(bottom_distances.begin(), q1_it);
	float bottom_z = std::numeric_limits<float>::max();
	for (auto it = bottom_distances.begin(); it != bottom_distances.end(); ++it)
		if (*it > min_iqr && *it < bottom_z)
			bottom_z = *it;

	if (bottom_z == std::numeric_limits<float>::max())
		throw std::runtime_error(fmt::format("Error[{}]: No valid bottom z found", __func__));

	return bottom_z;
}

xt::xtensor<float, 1> I_PingBottom::get_two_way_travel_times_legacy()
{
	return get_two_way_travel_times_legacy(get_beam_selection_all());
}

xt::xtensor<float, 1> I_PingBottom::get_two_way_travel_times_legacy(
	[[maybe_unused]] const algorithms::pingtools::BeamSelection& selection)
{
	throw not_implemented(__func__, this->class_name());
}

tools::classhelper::ObjectPrinter I_PingBottom::__printer__(unsigned int float_precision,
															bool         superscript_exponents) const
{
	tools::classhelper::ObjectPrinter printer(
		this->class_name(), float_precision, superscript_exponents);

	printer.append(I_PingCommon::__printer__(float_precision, superscript_exponents));

	printer.register_section("Bottom detection infos");
	auto features     = this->feature_string();
	auto not_features = this->feature_string(false);
	if (!not_features.empty())
		printer.register_string("Features", features, std::string("Not:") + not_features);
	else
		printer.register_string("Features", features);

	return printer;
}

} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping
