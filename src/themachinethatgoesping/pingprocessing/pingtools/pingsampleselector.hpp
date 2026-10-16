// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
//

#pragma once

/* generated doc strings */
#include ".docstrings/pingsampleselector.doc.hpp"

/* std includes */
#include <string>
#include <vector>

/* ping includes */
#include <themachinethatgoesping/algorithms/pingtools/beamsampleselection.hpp>
#include <themachinethatgoesping/tools/classhelper/objectprinter.hpp>

#include <themachinethatgoesping/tools/pyhelper/pyindexer.hpp>

#include "../datatypes/i_ping.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pingtools {

class PingSampleSelector
{
    std::optional<size_t> _min_beam_number; ///< min beam number to select
    std::optional<size_t> _max_beam_number; ///< max beam number to select

    std::optional<size_t> _min_sample_number; ///< min sample number to select
    std::optional<size_t> _max_sample_number; ///< max sample number to select

    std::optional<float> _min_beam_angle; ///< min beam angle to select (°)
    std::optional<float> _max_beam_angle; ///< max beam angle to select (°)

    std::optional<float> _min_sample_range; ///< min sample range to select (m)
    std::optional<float> _max_sample_range; ///< max sample range to select (m)

    std::optional<float>
        _min_minslant_range_percent; ///< min samples after minslant to consider (%)
    std::optional<float>
        _max_minslant_range_percent; ///< max samples before minslant to consider (%)
    std::optional<float>
        _max_bottom_range_percent; ///< max samples before bottom detection to consider (%)
    std::optional<float>
        _min_bottom_range_percent; ///< max samples before bottom detection to consider (%)

    // transmit sectors
    std::optional<std::vector<size_t>> _transmit_sectors; ///< transmit_sectors to select
    std::optional<float> _transmit_sector_min_beam_angle; ///< select transmit sectors with angles
                                                          ///< >= _transmit_sector_min_beam_angle
    std::optional<float> _transmit_sector_max_beam_angle; ///< select transmit sectors with angles
                                                          ///< <= _transmit_sector_max_beam_angle

    size_t _beam_step   = 1; ///< step size for beam numbers
    size_t _sample_step = 1; ///< step size for sample numbers

  public:
    PingSampleSelector() = default;

    // operators
    bool operator==(const PingSampleSelector& other) const = default;

    algorithms::pingtools::BeamSampleSelection apply_selection(
        pingprocessing::datatypes::I_PingWatercolumn& ping_watercolumn);

    // get selection
    algorithms::pingtools::BeamSelection apply_selection(pingprocessing::datatypes::I_PingBottom& ping_bottom);

    // getters
    auto        get_min_beam_number() const { return _min_beam_number; }
    auto        get_max_beam_number() const { return _max_beam_number; }
    auto        get_min_sample_number() const { return _min_sample_number; }
    auto        get_max_sample_number() const { return _max_sample_number; }
    auto        get_min_beam_angle() const { return _min_beam_angle; }
    auto        get_max_beam_angle() const { return _max_beam_angle; }
    auto        get_min_sample_range() const { return _min_sample_range; }
    auto        get_max_sample_range() const { return _max_sample_range; }
    auto        get_min_minslant_range_percent() const { return _min_minslant_range_percent; }
    auto        get_max_minslant_range_percent() const { return _max_minslant_range_percent; }
    auto        get_min_bottom_range_percent() const { return _min_bottom_range_percent; }
    auto        get_max_bottom_range_percent() const { return _max_bottom_range_percent; }
    auto        get_beam_step() const { return _beam_step; }
    auto        get_sample_step() const { return _sample_step; }
    const auto& get_transmit_sectors() const { return _transmit_sectors; }
    auto get_transmit_sector_min_beam_angle() const { return _transmit_sector_min_beam_angle; }
    auto get_transmit_sector_max_beam_angle() const { return _transmit_sector_max_beam_angle; }

    // resetters
    void clear_beam_number_range()
    {
        _min_beam_number.reset();
        _max_beam_number.reset();
    }

    void clear_sample_number_range()
    {
        _min_sample_number.reset();
        _max_sample_number.reset();
    }

    void clear_beam_angle_range()
    {
        _min_beam_angle.reset();
        _max_beam_angle.reset();
    }

    void clear_sample_range_range()
    {
        _min_sample_range.reset();
        _max_sample_range.reset();
    }

    void clear_minslant_range()
    {
        _min_minslant_range_percent.reset();
        _max_minslant_range_percent.reset();
    }

    void clear_bottom_range()
    {
        _min_bottom_range_percent.reset();
        _max_bottom_range_percent.reset();
    }

    void clear_transmit_sectors() { _transmit_sectors.reset(); }

    void clear_transmit_sector_beam_angle_range()
    {
        _transmit_sector_min_beam_angle.reset();
        _transmit_sector_max_beam_angle.reset();
    }

    void clear_beam_step() { _beam_step = 1; }
    void clear_sample_step() { _sample_step = 1; }

    void clear()
    {
        clear_beam_number_range();
        clear_sample_number_range();
        clear_beam_angle_range();
        clear_sample_range_range();
        clear_beam_step();
        clear_sample_step();
        clear_transmit_sectors();
        clear_transmit_sector_beam_angle_range();
        clear_minslant_range();
        clear_bottom_range();
    }

    // selectors
    void select_beam_range_by_numbers(std::optional<size_t> min_beam_number,
                                      std::optional<size_t> max_beam_number,
                                      std::optional<size_t> beam_step = std::nullopt)
    {
        _min_beam_number = min_beam_number;
        _max_beam_number = max_beam_number;
        if (beam_step)
            _beam_step = *beam_step;
    }

    void select_sample_range_by_numbers(std::optional<size_t> min_sample_number,
                                        std::optional<size_t> max_sample_number,
                                        std::optional<size_t> sample_step = std::nullopt)
    {
        _min_sample_number = min_sample_number;
        _max_sample_number = max_sample_number;
        if (sample_step)
            _sample_step = *sample_step;
    }

    void select_beam_range_by_angles(std::optional<float>  min_beam_angle,
                                     std::optional<float>  max_beam_angle,
                                     std::optional<size_t> beam_step = std::nullopt)
    {
        _min_beam_angle = min_beam_angle;
        _max_beam_angle = max_beam_angle;
        if (beam_step)
            _beam_step = *beam_step;
    }

    void select_sample_range_by_ranges(std::optional<float>  min_sample_range,
                                       std::optional<float>  max_sample_range,
                                       std::optional<size_t> sample_step = std::nullopt)
    {
        _min_sample_range = min_sample_range;
        _max_sample_range = max_sample_range;
        if (sample_step)
            _sample_step = *sample_step;
    }
    void select_max_minslant_range_percent(std::optional<float> max_minslant_range_percent)
    {
        _max_minslant_range_percent = max_minslant_range_percent;
    }
    void select_min_minslant_range_percent(std::optional<float> min_minslant_range_percent)
    {
        _min_minslant_range_percent = min_minslant_range_percent;
    }
    void select_minslant_range_percent(std::optional<float> min_minslant_range_percent,
                                       std::optional<float> max_minslant_range_percent)
    {
        _min_minslant_range_percent = min_minslant_range_percent;
        _max_minslant_range_percent = max_minslant_range_percent;
    }

    void select_min_bottom_range_percent(std::optional<float> min_bottom_range_percent)
    {
        _min_bottom_range_percent = min_bottom_range_percent;
    }

    void select_max_bottom_range_percent(std::optional<float> max_bottom_range_percent)
    {
        _max_bottom_range_percent = max_bottom_range_percent;
    }

    void select_bottom_range_percent(std::optional<float> min_bottom_range_percent,
                                     std::optional<float> max_bottom_range_percent)
    {
        _min_bottom_range_percent = min_bottom_range_percent;
        _max_bottom_range_percent = max_bottom_range_percent;
    }

    void select_transmit_sectors(std::optional<std::vector<size_t>> transmit_sectors)
    {
        _transmit_sectors = transmit_sectors;
    }

    void select_transmit_sectors_by_beam_angles(std::optional<float> transmit_sector_min_beam_angle,
                                                std::optional<float> transmit_sector_max_beam_angle)
    {
        _transmit_sector_min_beam_angle = transmit_sector_min_beam_angle;
        _transmit_sector_max_beam_angle = transmit_sector_max_beam_angle;
    }

    void set_sample_step(size_t sample_step) { _sample_step = sample_step; }
    void set_beam_step(size_t beam_step) { _beam_step = beam_step; }

    // ----- from/to binary -----
    /**
     * @brief Return a PingSampleSelector read from a binary stream
     *
     * @param is input stream
     * @return PingSampleSelector
     */
    static PingSampleSelector from_stream(std::istream& is);

    /**
     * @brief Write a PingSampleSelector to a binary stream
     *
     * @param os output stream
     */
    void to_stream(std::ostream& os) const;

    // ----- printing -----
    /**
     * @brief Print function, needs __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__ macro
     * See also: classhelper/objectprinter.hpp
     *
     * @param float_precision Precision of floating point numbers
     * @return classhelper::ObjectPrinter
     */
    tools::classhelper::ObjectPrinter __printer__(unsigned int float_precision, bool superscript_exponents) const;

  private:
    void register_filter(themachinethatgoesping::tools::classhelper::ObjectPrinter& printer,
                         const std::string&                                         name,
                         const std::optional<float>&                                value,
                         const std::string&                                         unit,
                         std::string& inactive_filters) const;

    void register_filter(themachinethatgoesping::tools::classhelper::ObjectPrinter& printer,
                         const std::string&                                         name,
                         const std::optional<size_t>&                               value,
                         const std::string&                                         unit,
                         std::string& inactive_filters) const;

  public:
    // -- class helper function macros --
    // define to_binary and from_binary functions (needs to_stream and from_stream)
    __STREAM_DEFAULT_TOFROM_BINARY_FUNCTIONS__(PingSampleSelector)
    // define info_string and print functions (needs the __printer__ function)
    __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__

  private:
    template<typename ping_watercolumn_or_bottom, typename t_angles>
    std::optional<std::vector<uint_fast16_t>> get_beam_numbers_selected_by_transmit_sector(
        ping_watercolumn_or_bottom& ping_w,
        const t_angles&             beam_crosstrack_angles)
    {

        if (!_transmit_sector_min_beam_angle.has_value() &&
            !_transmit_sector_max_beam_angle.has_value() && !_transmit_sectors.has_value())
            return std::nullopt;

        // init transmit_sector_selection
        const auto                 tx_sector_per_beam = ping_w.get_tx_sector_per_beam_legacy();
        std::vector<uint_fast16_t> transmit_sector_selection;

        if (tx_sector_per_beam.size() != beam_crosstrack_angles.size())
            throw std::runtime_error(
                fmt::format("Number of transmit sectors per beam ({}) is "
                            "different from the number of beam crosstrack angles ({})",
                            tx_sector_per_beam.size(),
                            beam_crosstrack_angles.size()));

        if (_transmit_sector_min_beam_angle.has_value() or
            _transmit_sector_max_beam_angle.has_value())
        {
            float min_ba =
                _transmit_sector_min_beam_angle.value_or(std::numeric_limits<float>::lowest());
            float max_ba =
                _transmit_sector_max_beam_angle.value_or(std::numeric_limits<float>::max());

            for (unsigned int bn = 0; bn < tx_sector_per_beam.size(); ++bn)
            {
                if (!transmit_sector_selection.empty())
                    if (tx_sector_per_beam[bn] == transmit_sector_selection.back())
                        continue;

                if (beam_crosstrack_angles.unchecked(bn) >= min_ba &&
                    beam_crosstrack_angles.unchecked(bn) <= max_ba)
                {
                    if (_transmit_sectors.has_value())
                        if (std::find(_transmit_sectors->begin(),
                                      _transmit_sectors->end(),
                                      tx_sector_per_beam[bn]) == _transmit_sectors->end())
                            continue;

                    transmit_sector_selection.push_back(tx_sector_per_beam[bn]);
                }
            }
        }
        else
        {
            std::copy(tx_sector_per_beam.begin(),
                      tx_sector_per_beam.end(),
                      std::back_inserter(transmit_sector_selection));
        }

        std::vector<uint_fast16_t> beam_number_is_selected;
        beam_number_is_selected.resize(tx_sector_per_beam.size());

        for (unsigned int bn = 0; bn < tx_sector_per_beam.size(); ++bn)
            if (std::find(transmit_sector_selection.begin(),
                          transmit_sector_selection.end(),
                          tx_sector_per_beam[bn]) != transmit_sector_selection.end())
                beam_number_is_selected[bn] = 1;

        return beam_number_is_selected;
    }
};

} // namespace pingtools
} // namespace pingprocessing
} // namespace themachinethatgoesping