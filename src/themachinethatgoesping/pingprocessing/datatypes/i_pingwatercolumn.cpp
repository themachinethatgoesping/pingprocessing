// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "i_pingwatercolumn.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

std::string I_PingWatercolumn::class_name() const
{
    return "I_PingWatercolumn";
}

I_PingWatercolumn::I_PingWatercolumn() = default;

I_PingWatercolumn::~I_PingWatercolumn() = default;

std::map<t_pingfeature, std::function<bool()>> I_PingWatercolumn::primary_feature_functions() const
{
    auto features                       = t_base::primary_feature_functions();
    features[t_pingfeature::amplitudes] = std::bind(&I_PingWatercolumn::has_amplitudes, this);
    return features;
}

std::map<t_pingfeature, std::function<bool()>> I_PingWatercolumn::secondary_feature_functions()
    const
{
    auto features = t_base::secondary_feature_functions();
    features[t_pingfeature::tx_signal_parameters_legacy] =
        std::bind(&I_PingWatercolumn::has_tx_signal_parameters_legacy, this);
    features[t_pingfeature::number_of_tx_sectors_legacy] =
        std::bind(&I_PingWatercolumn::has_number_of_tx_sectors_legacy, this);
    features[t_pingfeature::beam_numbers_per_tx_sector_legacy] =
        std::bind(&I_PingWatercolumn::has_beam_numbers_per_tx_sector_legacy, this);
    features[t_pingfeature::beam_selection_all] =
        std::bind(&I_PingWatercolumn::has_beam_selection_all, this);
    features[t_pingfeature::number_of_beams_legacy] =
        std::bind(&I_PingWatercolumn::has_number_of_beams_legacy, this);
    features[t_pingfeature::tx_sector_per_beam_legacy] =
        std::bind(&I_PingWatercolumn::has_tx_sector_per_beam_legacy, this);
    features[t_pingfeature::beam_crosstrack_angles_legacy] =
        std::bind(&I_PingWatercolumn::has_beam_crosstrack_angles_legacy, this);

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

    features[t_pingfeature::bottom_range_samples] =
        std::bind(&I_PingWatercolumn::has_bottom_range_samples, this);
    features[t_pingfeature::amplitudes] = std::bind(&I_PingWatercolumn::has_amplitudes, this);
    features[t_pingfeature::ap]         = std::bind(&I_PingWatercolumn::has_ap, this);
    features[t_pingfeature::av]         = std::bind(&I_PingWatercolumn::has_av, this);
    features[t_pingfeature::power]      = std::bind(&I_PingWatercolumn::has_power, this);
    features[t_pingfeature::rp]         = std::bind(&I_PingWatercolumn::has_rp, this);
    features[t_pingfeature::rv]         = std::bind(&I_PingWatercolumn::has_rv, this);
    features[t_pingfeature::pp]         = std::bind(&I_PingWatercolumn::has_pp, this);
    features[t_pingfeature::pv]         = std::bind(&I_PingWatercolumn::has_pv, this);
    features[t_pingfeature::sp]         = std::bind(&I_PingWatercolumn::has_sp, this);
    features[t_pingfeature::sv]         = std::bind(&I_PingWatercolumn::has_sv, this);
    features[t_pingfeature::watercolumn_calibration] =
        std::bind(&I_PingWatercolumn::has_watercolumn_calibration, this);
    features[t_pingfeature::multisectorwatercolumn_calibration] =
        std::bind(&I_PingWatercolumn::has_multisectorwatercolumn_calibration, this);
    return features;
}

std::map<t_pingfeature, std::function<bool()>> I_PingWatercolumn::feature_group_functions() const
{
    return t_base::feature_group_functions();
}

bool I_PingWatercolumn::has_tx_signal_parameters_legacy() const
{
    return false;
}

bool I_PingWatercolumn::has_number_of_tx_sectors_legacy() const
{
    return false;
}

bool I_PingWatercolumn::has_beam_numbers_per_tx_sector_legacy() const
{
    return false;
}

bool I_PingWatercolumn::has_tx_sector_per_beam_legacy() const
{
    return false;
}

bool I_PingWatercolumn::has_beam_selection_all() const
{
    return has_number_of_beams_legacy();
}

bool I_PingWatercolumn::has_number_of_beams_legacy() const
{
    return false;
}

bool I_PingWatercolumn::has_beam_crosstrack_angles_legacy() const
{
    return false;
}

bool I_PingWatercolumn::has_watercolumn_calibration() const
{
    return false;
}

bool I_PingWatercolumn::has_multisectorwatercolumn_calibration() const
{
    return false;
}

bool I_PingWatercolumn::has_bottom_range_samples() const
{
    return false;
}

bool I_PingWatercolumn::has_amplitudes() const
{
    return false;
}

bool I_PingWatercolumn::has_ap() const
{
    if (get_number_of_tx_sectors_legacy() == 1)
        return has_amplitudes() && get_watercolumn_calibration().has_ap_calibration() &&
               get_watercolumn_calibration().has_valid_absorption_db_m();

    return has_amplitudes() && get_multisectorwatercolumn_calibration().has_ap_calibration() &&
           get_multisectorwatercolumn_calibration().has_valid_absorption_db_m();
}

bool I_PingWatercolumn::has_av() const
{
    if (get_number_of_tx_sectors_legacy() == 1)
        return has_amplitudes() && get_watercolumn_calibration().has_av_calibration() &&
               get_watercolumn_calibration().has_valid_absorption_db_m();

    return has_amplitudes() && get_multisectorwatercolumn_calibration().has_av_calibration() &&
           get_multisectorwatercolumn_calibration().has_valid_absorption_db_m();
}

bool I_PingWatercolumn::has_power() const
{
    if (get_number_of_tx_sectors_legacy() == 1)
        return has_amplitudes() && get_watercolumn_calibration().has_power_calibration();

    return has_amplitudes() && get_multisectorwatercolumn_calibration().has_power_calibration();
}

bool I_PingWatercolumn::has_rp() const
{
    return has_power();
}

bool I_PingWatercolumn::has_rv() const
{
    return has_power();
}

bool I_PingWatercolumn::has_pp() const
{
    if (get_number_of_tx_sectors_legacy() == 1)
        return has_amplitudes() && get_watercolumn_calibration().has_power_calibration() &&
               get_watercolumn_calibration().has_valid_absorption_db_m();

    return has_amplitudes() && get_multisectorwatercolumn_calibration().has_power_calibration() &&
           get_multisectorwatercolumn_calibration().has_valid_absorption_db_m();
}

bool I_PingWatercolumn::has_pv() const
{
    return has_pp();
}

bool I_PingWatercolumn::has_sp() const
{
    if (get_number_of_tx_sectors_legacy() == 1)
        return has_amplitudes() && get_watercolumn_calibration().has_sp_calibration() &&
               get_watercolumn_calibration().has_valid_absorption_db_m();

    return has_amplitudes() && get_multisectorwatercolumn_calibration().has_sp_calibration() &&
           get_multisectorwatercolumn_calibration().has_valid_absorption_db_m();
}

bool I_PingWatercolumn::has_sv() const
{
    if (get_number_of_tx_sectors_legacy() == 1)
        return has_amplitudes() && get_watercolumn_calibration().has_sv_calibration() &&
               get_watercolumn_calibration().has_valid_absorption_db_m();

    return has_amplitudes() && get_multisectorwatercolumn_calibration().has_sv_calibration() &&
           get_multisectorwatercolumn_calibration().has_valid_absorption_db_m();
}

std::vector<algorithms::signalprocessing::datastructures::TxSignalParameters>
I_PingWatercolumn::get_tx_signal_parameters_legacy() const
{
    throw not_implemented(__func__, this->class_name());
}

size_t I_PingWatercolumn::get_number_of_tx_sectors_legacy() const
{
    throw not_implemented(__func__, this->class_name());
}

const calibration::WaterColumnCalibration& I_PingWatercolumn::get_watercolumn_calibration() const
{
    throw not_implemented(__func__, this->class_name());
}

const calibration::WaterColumnCalibration& I_PingWatercolumn::get_watercolumn_calibration(
    size_t sector_nr) const
{
    if (get_number_of_tx_sectors_legacy() == 1)
    {
        if (sector_nr == 0)
            return get_watercolumn_calibration();
    }

    const auto& multisector = get_multisectorwatercolumn_calibration();
    if (multisector.get_number_of_sectors() <= sector_nr)
        throw std::runtime_error(
            fmt::format("Error[{}]: Sector {} out of range", __func__, sector_nr));

    return multisector.calibration_for_sector(sector_nr);
}

const calibration::I_MultiSectorCalibration&
I_PingWatercolumn::get_multisectorwatercolumn_calibration() const
{
    throw not_implemented(__func__, this->class_name());
}

float I_PingWatercolumn::get_sample_interval()
{
    throw not_implemented(__func__, class_name());
}

xt::xtensor<size_t, 1> I_PingWatercolumn::get_tx_sector_per_beam_legacy()
{
    return get_tx_sector_per_beam_legacy(get_beam_selection_all());
}

xt::xtensor<size_t, 1> I_PingWatercolumn::get_tx_sector_per_beam_legacy(const algorithms::pingtools::BeamSelection&)
{
    throw not_implemented(__func__, class_name());
}

std::vector<std::vector<size_t>> I_PingWatercolumn::get_beam_numbers_per_tx_sector_legacy()
{
    return get_beam_numbers_per_tx_sector_legacy(get_beam_selection_all());
}

std::vector<std::vector<size_t>> I_PingWatercolumn::get_beam_numbers_per_tx_sector_legacy(
    const algorithms::pingtools::BeamSelection&)
{
    throw not_implemented(__func__, class_name());
}

algorithms::pingtools::BeamSelection I_PingWatercolumn::get_beam_selection_all()
{
    return algorithms::pingtools::BeamSelection(get_number_of_beams_legacy());
}

uint32_t I_PingWatercolumn::get_number_of_beams_legacy() const
{
    throw not_implemented(__func__, class_name());
}

xt::xtensor<float, 1> I_PingWatercolumn::get_beam_crosstrack_angles_legacy()
{
    return get_beam_crosstrack_angles_legacy(get_beam_selection_all());
}

xt::xtensor<float, 1> I_PingWatercolumn::get_beam_crosstrack_angles_legacy(const algorithms::pingtools::BeamSelection&)
{
    throw not_implemented(__func__, class_name());
}

xt::xtensor<float, 1> I_PingWatercolumn::get_beam_alongtrack_angles()
{
    return get_beam_alongtrack_angles(get_beam_selection_all());
}

xt::xtensor<float, 1> I_PingWatercolumn::get_beam_alongtrack_angles(const algorithms::pingtools::BeamSelection&)
{
    throw not_implemented(__func__, class_name());
}

xt::xtensor<uint32_t, 1> I_PingWatercolumn::get_first_sample_offset_per_beam()
{
    throw not_implemented(__func__, class_name());
}

xt::xtensor<uint32_t, 1> I_PingWatercolumn::get_number_of_samples_per_beam()
{
    return get_number_of_samples_per_beam(get_beam_selection_all());
}

xt::xtensor<uint32_t, 1> I_PingWatercolumn::get_number_of_samples_per_beam(
    const algorithms::pingtools::BeamSelection&)
{
    throw not_implemented(__func__, class_name());
}

float I_PingWatercolumn::get_sound_speed_at_transducer_legacy()
{
    throw not_implemented(__func__, class_name());
}

xt::xtensor<float, 1> I_PingWatercolumn::get_approximate_ranges(
    const algorithms::pingtools::BeamSampleSelection& selection)
{
    return algorithms::amplitudecorrection::functions::approximate_ranges<xt::xtensor<float, 1>>(
        get_sample_interval(),
        get_sound_speed_at_transducer_legacy(),
        selection.get_sample_numbers_ensemble_1d());
}

xt::xtensor<float, 1> I_PingWatercolumn::get_approximate_ranges()
{
    return get_approximate_ranges(get_beam_sample_selection_all());
}

const algorithms::pingtools::BeamSampleSelection& I_PingWatercolumn::get_beam_sample_selection_all()
{
    if (!_beam_sample_selection_all_is_initialized)
    {
        auto first_sample_offsets        = get_first_sample_offset_per_beam();
        auto number_of_samples           = get_number_of_samples_per_beam();
        auto last_sample_number_per_beam = first_sample_offsets + number_of_samples - 1;

        std::vector<uint32_t> first_snpb(first_sample_offsets.begin(), first_sample_offsets.end());
        std::vector<uint32_t> last_snpb(last_sample_number_per_beam.begin(),
                                        last_sample_number_per_beam.end());

        _beam_sample_selection_all =
            algorithms::pingtools::BeamSampleSelection(std::move(first_snpb), std::move(last_snpb));
        _beam_sample_selection_all_is_initialized = true;
    }

    return _beam_sample_selection_all;
}

xt::xtensor<float, 2> I_PingWatercolumn::get_amplitudes(int mp_cores)
{
    return get_amplitudes(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_ap(int mp_cores)
{
    return get_ap(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_av(int mp_cores)
{
    return get_av(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_rp(int mp_cores)
{
    return get_rp(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_rv(int mp_cores)
{
    return get_rv(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_pp(int mp_cores)
{
    return get_pp(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_pv(int mp_cores)
{
    return get_pv(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_power(int mp_cores)
{
    return get_power(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_sp(int mp_cores)
{
    return get_sp(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_sv(int mp_cores)
{
    return get_sv(get_beam_sample_selection_all(), mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_amplitudes(const algorithms::pingtools::BeamSampleSelection&, int)
{
    throw not_implemented(__func__, this->class_name());
}

xt::xtensor<float, 2> I_PingWatercolumn::get_rp(const algorithms::pingtools::BeamSampleSelection& selection,
                                                int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::rp>(selection, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_rv(const algorithms::pingtools::BeamSampleSelection& selection,
                                                int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::rv>(selection, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_pp(const algorithms::pingtools::BeamSampleSelection& selection,
                                                int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::pp>(selection, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_pv(const algorithms::pingtools::BeamSampleSelection& selection,
                                                int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::pv>(selection, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_ap(const algorithms::pingtools::BeamSampleSelection& selection,
                                                int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::ap>(selection, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_av(const algorithms::pingtools::BeamSampleSelection& selection,
                                                int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::av>(selection, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_power(const algorithms::pingtools::BeamSampleSelection& selection,
                                                   int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::power>(selection, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_sp(const algorithms::pingtools::BeamSampleSelection& selection,
                                                int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::sp>(selection, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_sv(const algorithms::pingtools::BeamSampleSelection& selection,
                                                int                                   mp_cores)
{
    using t_calibration_type = typename calibration::WaterColumnCalibration::t_calibration_type;
    return this->template get_calibrated_wci<t_calibration_type::sv>(selection, mp_cores);
}

void I_PingWatercolumn::set_test_mode(int test_test_mode)
{
    _test_mode = test_test_mode;
}

int I_PingWatercolumn::get_test_mode()
{
    return _test_mode;
}

xt::xtensor<float, 2> I_PingWatercolumn::get_wci_correction(
    const calibration::o_calibration_type calibration1,
    const calibration::o_calibration_type calibration_base,
    int                                   mp_cores)
{
    return get_wci_correction(
        get_beam_sample_selection_all(), calibration1, calibration_base, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_wci_correction(
    const algorithms::pingtools::BeamSampleSelection& selection,
    const calibration::o_calibration_type calibration1,
    const calibration::o_calibration_type calibration_base,
    int                                   mp_cores)
{
    xt::xtensor<float, 2> amplitudes;
    switch (calibration1.value)
    {
        case calibration::t_calibration_type::power:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::power>(selection, mp_cores);
            break;
        case calibration::t_calibration_type::rp:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::rp>(selection, mp_cores);
            break;
        case calibration::t_calibration_type::rv:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::rv>(selection, mp_cores);
            break;
        case calibration::t_calibration_type::pp:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::pp>(selection, mp_cores);
            break;
        case calibration::t_calibration_type::pv:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::pv>(selection, mp_cores);
            break;
        case calibration::t_calibration_type::ap:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::ap>(selection, mp_cores);
            break;
        case calibration::t_calibration_type::av:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::av>(selection, mp_cores);
            break;
        case calibration::t_calibration_type::sp:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::sp>(selection, mp_cores);
            break;
        case calibration::t_calibration_type::sv:
            amplitudes =
                get_wci_correction<calibration::t_calibration_type::sv>(selection, mp_cores);
            break;
        default:
            throw std::invalid_argument("Invalid calibration type");
    }

    if (calibration_base == calibration::t_calibration_type::power)
        return amplitudes;

    return amplitudes -
           get_wci_correction(
               selection, calibration_base, calibration::t_calibration_type::power, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_wci(
    const calibration::o_calibration_type calibration_type,
    int                                   mp_cores)
{
    return get_wci(get_beam_sample_selection_all(), calibration_type, mp_cores);
}

xt::xtensor<float, 2> I_PingWatercolumn::get_wci(
    const algorithms::pingtools::BeamSampleSelection& selection,
    const calibration::o_calibration_type calibration_type,
    int                                   mp_cores)
{
    switch (calibration_type.value)
    {
        case calibration::t_calibration_type::power:
            return get_power(selection, mp_cores);
        case calibration::t_calibration_type::rp:
            return get_rp(selection, mp_cores);
        case calibration::t_calibration_type::rv:
            return get_rv(selection, mp_cores);
        case calibration::t_calibration_type::pp:
            return get_pp(selection, mp_cores);
        case calibration::t_calibration_type::pv:
            return get_pv(selection, mp_cores);
        case calibration::t_calibration_type::ap:
            return get_ap(selection, mp_cores);
        case calibration::t_calibration_type::av:
            return get_av(selection, mp_cores);
        case calibration::t_calibration_type::sp:
            return get_sp(selection, mp_cores);
        case calibration::t_calibration_type::sv:
            return get_sv(selection, mp_cores);
        default:
            throw std::invalid_argument("Invalid calibration type");
    }
}

xt::xtensor<uint32_t, 1> I_PingWatercolumn::get_bottom_range_samples()
{
    return get_bottom_range_samples(get_beam_selection_all());
}

xt::xtensor<uint32_t, 1> I_PingWatercolumn::get_bottom_range_samples(
    const algorithms::pingtools::BeamSelection&)
{
    throw not_implemented(__func__, this->class_name());
}

uint32_t I_PingWatercolumn::get_minslant_sample_nr()
{
    return get_minslant_sample_nr(get_beam_selection_all());
}

uint32_t I_PingWatercolumn::get_minslant_sample_nr(const algorithms::pingtools::BeamSelection& selection)
{
    auto bottom_range_samples = get_bottom_range_samples(selection);

    if (bottom_range_samples.size() == 0)
        throw std::runtime_error(
            fmt::format("Error[{}]: No valid bottom range sample found", __func__));

    if (bottom_range_samples.size() == 1)
        return bottom_range_samples[0];

    std::vector<uint32_t> bottom_range_samples_vec(bottom_range_samples.begin(),
                                                   bottom_range_samples.end());

    auto median_it = bottom_range_samples_vec.begin() + bottom_range_samples_vec.size() / 2;
    std::nth_element(bottom_range_samples_vec.begin(), median_it, bottom_range_samples_vec.end());
    auto median = *median_it;

    auto q3_it = bottom_range_samples_vec.begin() + 3 * bottom_range_samples_vec.size() / 4;
    std::nth_element(median_it, q3_it, bottom_range_samples_vec.end());
    auto q3 = *q3_it;

    auto q1_it = bottom_range_samples_vec.begin() + bottom_range_samples_vec.size() / 4;
    std::nth_element(bottom_range_samples_vec.begin(), q1_it, median_it);
    auto q1      = *q1_it;
    auto min_iqr = median - (q3 - q1) * 1.5;

    std::sort(bottom_range_samples_vec.begin(), q1_it);
    uint32_t minslant_sample_nr = std::numeric_limits<uint32_t>::max();
    for (auto it = bottom_range_samples_vec.begin(); it != bottom_range_samples_vec.end(); ++it)
        if (*it > min_iqr && *it < minslant_sample_nr)
            minslant_sample_nr = *it;

    if (minslant_sample_nr == std::numeric_limits<uint32_t>::max())
        throw std::runtime_error(
            fmt::format("Error[{}]: No valid bottom range sample found", __func__));

    return minslant_sample_nr;
}

tools::classhelper::ObjectPrinter I_PingWatercolumn::__printer__(unsigned int float_precision,
                                                                 bool superscript_exponents) const
{
    tools::classhelper::ObjectPrinter printer(
        this->class_name(), float_precision, superscript_exponents);

    printer.append(I_PingCommon::__printer__(float_precision, superscript_exponents));

    printer.register_section("Watercolumn detection infos");
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
