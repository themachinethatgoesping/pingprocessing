// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 Sven Schorge, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
//

#pragma once

/* generated doc strings */
#include ".docstrings/i_pingbottom.doc.hpp"

/* std includes */
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

/* themachinethatgoesping includes */
#include <themachinethatgoesping/algorithms/geoprocessing/datastructures/xyz.hpp>
#include <themachinethatgoesping/algorithms/geoprocessing/raytracers2/i_mbesrawdetectioninfos.hpp>
#include <themachinethatgoesping/algorithms/signalprocessing/datastructures.hpp>
#include <themachinethatgoesping/tools/classhelper/objectprinter.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <themachinethatgoesping/algorithms/pingtools/beamselection.hpp>
#include "i_pingcommon.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

/**
 * @brief Interface for all ping bottom detection functions
 *
 */
class I_PingBottom
    : public I_PingCommon
    , public algorithms::geoprocessing::raytracers2::I_MBESRawDetectionInfos
{
  protected:
    std::string class_name() const override;

  public:
    using t_base = I_PingCommon;

    /**
     * @brief Construct a new I_PingBottom.
     *
     * @param ping_info shared PingInfo object (channel id, timestamp, sensor configuration,
     * navigation). The echosounder ping implementations hand the owning ping's PingInfo pointer
     * here so that the bottom can access the same sensor configuration / navigation / geolocation.
     */
    explicit I_PingBottom(std::shared_ptr<PingInfo> ping_info = nullptr);
    ~I_PingBottom() override;

    std::map<t_pingfeature, std::function<bool()>> primary_feature_functions() const override;
    std::map<t_pingfeature, std::function<bool()>> secondary_feature_functions() const override;
    std::map<t_pingfeature, std::function<bool()>> feature_group_functions() const override;

    virtual bool has_tx_signal_parameters_legacy() const;
    virtual bool has_number_of_tx_sectors_legacy() const;
    virtual bool has_beam_numbers_per_tx_sector_legacy() const;
    virtual bool has_tx_sector_per_beam_legacy() const;

    bool         has_beam_selection_all() const;
    virtual bool has_number_of_beams_legacy() const;
    virtual bool has_beam_crosstrack_angles_legacy() const;

    virtual bool has_two_way_travel_times_legacy() const;
    virtual bool has_xyz() const;

    virtual std::vector<algorithms::signalprocessing::datastructures::TxSignalParameters>
        get_tx_signal_parameters_legacy();
    virtual size_t get_number_of_tx_sectors_legacy();

    xt::xtensor<size_t, 1> get_tx_sector_per_beam_legacy();
    virtual xt::xtensor<size_t, 1>
        get_tx_sector_per_beam_legacy(const algorithms::pingtools::BeamSelection& bs);

    std::vector<std::vector<size_t>> get_beam_numbers_per_tx_sector_legacy();
    virtual std::vector<std::vector<size_t>>
        get_beam_numbers_per_tx_sector_legacy(const algorithms::pingtools::BeamSelection& bs);

    algorithms::pingtools::BeamSelection get_beam_selection_all();
    virtual uint32_t get_number_of_beams_legacy();

    xt::xtensor<float, 1> get_beam_crosstrack_angles_legacy();
    virtual xt::xtensor<float, 1>
        get_beam_crosstrack_angles_legacy(const algorithms::pingtools::BeamSelection& bs);

    algorithms::geoprocessing::datastructures::XYZ<1> get_xyz();
    virtual algorithms::geoprocessing::datastructures::XYZ<1>
        get_xyz(const algorithms::pingtools::BeamSelection& selection);

    float get_bottom_z();
    float get_bottom_z(const algorithms::pingtools::BeamSelection& selection);

    xt::xtensor<float, 1> get_two_way_travel_times_legacy();
    virtual xt::xtensor<float, 1>
        get_two_way_travel_times_legacy(const algorithms::pingtools::BeamSelection& selection);

    tools::classhelper::ObjectPrinter __printer__(unsigned int float_precision,
                                                  bool         superscript_exponents) const;

    __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__
};

}
} // namespace pingprocessing
} // namespace themachinethatgoesping