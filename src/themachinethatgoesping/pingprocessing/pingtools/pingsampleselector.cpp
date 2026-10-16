// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "pingsampleselector.hpp"

#include <algorithm>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>
#include <themachinethatgoesping/tools/classhelper/stream.hpp>
#include <themachinethatgoesping/tools/pyhelper/pyindexer.hpp>
#include <xtensor/core/xmath.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pingtools {

algorithms::pingtools::BeamSampleSelection PingSampleSelector::apply_selection(
    pingprocessing::datatypes::I_PingWatercolumn& ping_watercolumn)
{
    algorithms::pingtools::BeamSampleSelection selection;

    // select beams according to the options
    const auto number_of_beams            = ping_watercolumn.get_number_of_beams_legacy();
    
    // Handle empty beams case - return empty selection
    if (number_of_beams == 0)
    {
        selection.set_sample_step_ensemble(_sample_step);
        return selection;
    }
    
    auto       number_of_samples_per_beam   = ping_watercolumn.get_number_of_samples_per_beam();
    const auto first_sample_offset_pb       = ping_watercolumn.get_first_sample_offset_per_beam();
    auto       min_sample_nr_per_beam       = first_sample_offset_pb; // absolute first sample per beam

    const auto beam_crosstrack_angles = ping_watercolumn.get_beam_crosstrack_angles_legacy();
    if (beam_crosstrack_angles.size() < number_of_beams)
        throw std::runtime_error(fmt::format(
            "Number of beam crosstrack angles ({}) is smaller than the number of beams ({})",
            beam_crosstrack_angles.size(),
            number_of_beams));

    if (_max_bottom_range_percent)
    {
        // get_bottom_range_samples() returns absolute sample numbers; subtract first_sample_offset
        // to get relative counts (clamped to 0 when bottom is before beam start)
        const xt::xtensor<uint32_t, 1> bottom_abs = ping_watercolumn.get_bottom_range_samples() *
                                                     (_max_bottom_range_percent.value() * 0.01f);
        const xt::xtensor<uint32_t, 1> bottom_rel = xt::eval(
            xt::where(bottom_abs > first_sample_offset_pb,
                      bottom_abs - first_sample_offset_pb,
                      xt::zeros_like(bottom_abs)));
        number_of_samples_per_beam = xt::minimum(number_of_samples_per_beam, bottom_rel);
    }

    if (_min_bottom_range_percent)
    {
        // keep in absolute domain to match min_sample_nr_per_beam
        const xt::xtensor<uint32_t, 1> bottom_abs = ping_watercolumn.get_bottom_range_samples() *
                                                     (_min_bottom_range_percent.value() * 0.01f);
        min_sample_nr_per_beam = xt::maximum(min_sample_nr_per_beam, bottom_abs);
    }

    // convert min/max beam numbers to indices (if set, and according to python negative
    // indexing)
    size_t min_beam_number = _min_beam_number ? *_min_beam_number : 0;
    size_t max_beam_number = _max_beam_number ? *_max_beam_number : number_of_beams - 1;

    // initialize transmit sector selection
    auto beam_numbers_selected_by_transmit_sector =
        get_beam_numbers_selected_by_transmit_sector(ping_watercolumn, beam_crosstrack_angles);

    // create beam indexer
    tools::pyhelper::PyIndexer beam_indexer(
        number_of_beams, min_beam_number, max_beam_number + 1, _beam_step);

    std::optional<size_t> max_sample_nr_ping, min_sample_nr_ping;
    if (_max_minslant_range_percent)
    {
        size_t sn =
            (*_max_minslant_range_percent * 0.01f) * ping_watercolumn.get_minslant_sample_nr();
        if (_max_sample_number)
            max_sample_nr_ping = std::min(sn, *_max_sample_number);
        else
            max_sample_nr_ping = sn;
    }
    else
        max_sample_nr_ping = _max_sample_number;

    if (_min_minslant_range_percent)
    {
        size_t sn =
            (*_min_minslant_range_percent * 0.01f) * ping_watercolumn.get_minslant_sample_nr();
        if (_min_sample_number)
            min_sample_nr_ping = std::max(sn, *_min_sample_number);
        else
            min_sample_nr_ping = sn;
    }
    else
        min_sample_nr_ping = _min_sample_number;

    for (unsigned int counter = 0; counter < beam_indexer.size(); ++counter)
    {
        auto bn = beam_indexer(counter);

        // check transmit sector selection
        if (beam_numbers_selected_by_transmit_sector.has_value())
            if (!beam_numbers_selected_by_transmit_sector.value()[bn])
                continue;

        if (_min_beam_angle && beam_crosstrack_angles.unchecked(bn) < *_min_beam_angle)
            continue;
        if (_max_beam_angle && beam_crosstrack_angles.unchecked(bn) > *_max_beam_angle)
            continue;

        const size_t first_sample_offset = first_sample_offset_pb.unchecked(bn);
        size_t       number_of_samples  = number_of_samples_per_beam.unchecked(bn);

        // Absolute sample numbers for this beam
        size_t min_sample_number =
            min_sample_nr_ping.value_or(min_sample_nr_per_beam.unchecked(bn));
        size_t max_sample_number =
            max_sample_nr_ping.value_or(first_sample_offset + number_of_samples - 1);

        const size_t beam_last = first_sample_offset + number_of_samples - 1;
        if (min_sample_number > beam_last)
            continue;
        if (min_sample_number < min_sample_nr_per_beam.unchecked(bn))
            min_sample_number = min_sample_nr_per_beam.unchecked(bn);
        if (max_sample_number > beam_last)
            max_sample_number = beam_last;
        if (min_sample_number > max_sample_number)
            continue;

        // Convert to 0-based indices within beam storage for PyIndexer
        const size_t rel_min = min_sample_number - first_sample_offset;
        const size_t rel_max = max_sample_number - first_sample_offset;

        tools::pyhelper::PyIndexer sample_indexer(
            number_of_samples, rel_min, rel_max + 1, _sample_step);

        selection.add_beam(
            bn, first_sample_offset + sample_indexer(0), first_sample_offset + sample_indexer(-1));

        bn++;
    }

    // select samples according to the options
    selection.set_sample_step_ensemble(_sample_step);

    return selection;
}

algorithms::pingtools::BeamSelection PingSampleSelector::apply_selection(pingprocessing::datatypes::I_PingBottom& ping_bottom)
{
    algorithms::pingtools::BeamSelection selection;

    // select beams according to the options
    const auto number_of_beams = ping_bottom.get_number_of_beams_legacy();

    // Handle empty beams case - return empty selection
    if (number_of_beams == 0)
        return selection;

    const auto beam_crosstrack_angles = ping_bottom.get_beam_crosstrack_angles_legacy();
    if (beam_crosstrack_angles.size() < number_of_beams)
        throw std::runtime_error(fmt::format(
            "Number of beam crosstrack angles ({}) is smaller than the number of beams ({})",
            beam_crosstrack_angles.size(),
            number_of_beams));

    // convert min/max beam numbers to indices (if set, and according to python negative
    // indexing)
    size_t min_beam_number = _min_beam_number ? *_min_beam_number : 0;
    size_t max_beam_number = _max_beam_number ? *_max_beam_number : number_of_beams - 1;

    // initialize transmit sector selection
    auto beam_numbers_selected_by_transmit_sector =
        get_beam_numbers_selected_by_transmit_sector(ping_bottom, beam_crosstrack_angles);

    tools::pyhelper::PyIndexer beam_indexer(
        number_of_beams, min_beam_number, max_beam_number + 1, _beam_step);

    for (unsigned int counter = 0; counter < beam_indexer.size(); ++counter)
    {
        auto bn = beam_indexer(counter);

        // check transmit sector selection
        if (beam_numbers_selected_by_transmit_sector.has_value())
            if (!beam_numbers_selected_by_transmit_sector.value()[bn])
                continue;

        if (_min_beam_angle && beam_crosstrack_angles.unchecked(bn) < *_min_beam_angle)
            continue;
        if (_max_beam_angle && beam_crosstrack_angles.unchecked(bn) > *_max_beam_angle)
            continue;

        selection.add_beam(bn);
        bn++;
    }

    return selection;
}

PingSampleSelector PingSampleSelector::from_stream(std::istream& is)
{
    using themachinethatgoesping::tools::classhelper::stream::optional_container_from_stream;
    using themachinethatgoesping::tools::classhelper::stream::optional_from_stream;
    using themachinethatgoesping::tools::classhelper::stream::optional_set_from_stream;

    PingSampleSelector object;
    object._min_beam_number            = optional_from_stream<size_t>(is);
    object._max_beam_number            = optional_from_stream<size_t>(is);
    object._min_sample_number          = optional_from_stream<size_t>(is);
    object._max_sample_number          = optional_from_stream<size_t>(is);
    object._min_beam_angle             = optional_from_stream<float>(is);
    object._max_beam_angle             = optional_from_stream<float>(is);
    object._min_sample_range           = optional_from_stream<float>(is);
    object._max_sample_range           = optional_from_stream<float>(is);
    object._min_minslant_range_percent = optional_from_stream<float>(is);
    object._max_minslant_range_percent = optional_from_stream<float>(is);
    object._min_bottom_range_percent   = optional_from_stream<float>(is);
    object._max_bottom_range_percent   = optional_from_stream<float>(is);
    object._transmit_sectors = optional_container_from_stream<std::vector<size_t>>(is);
    object._transmit_sector_min_beam_angle = optional_from_stream<float>(is);
    object._transmit_sector_max_beam_angle = optional_from_stream<float>(is);

    is.read(reinterpret_cast<char*>(&object._beam_step), sizeof(object._beam_step));
    is.read(reinterpret_cast<char*>(&object._sample_step), sizeof(object._sample_step));

    return object;
}

void PingSampleSelector::to_stream(std::ostream& os) const
{
    using themachinethatgoesping::tools::classhelper::stream::optional_container_to_stream;
    using themachinethatgoesping::tools::classhelper::stream::optional_set_to_stream;
    using themachinethatgoesping::tools::classhelper::stream::optional_to_stream;

    optional_to_stream(os, _min_beam_number);
    optional_to_stream(os, _max_beam_number);
    optional_to_stream(os, _min_sample_number);
    optional_to_stream(os, _max_sample_number);
    optional_to_stream(os, _min_beam_angle);
    optional_to_stream(os, _max_beam_angle);
    optional_to_stream(os, _min_sample_range);
    optional_to_stream(os, _max_sample_range);
    optional_to_stream(os, _min_minslant_range_percent);
    optional_to_stream(os, _max_minslant_range_percent);
    optional_to_stream(os, _min_bottom_range_percent);
    optional_to_stream(os, _max_bottom_range_percent);
    optional_container_to_stream(os, _transmit_sectors);
    optional_to_stream(os, _transmit_sector_min_beam_angle);
    optional_to_stream(os, _transmit_sector_max_beam_angle);
    os.write(reinterpret_cast<const char*>(&_beam_step), sizeof(_beam_step));
    os.write(reinterpret_cast<const char*>(&_sample_step), sizeof(_sample_step));
}

tools::classhelper::ObjectPrinter PingSampleSelector::__printer__(unsigned int float_precision, bool superscript_exponents) const
{
    using themachinethatgoesping::tools::classhelper::ObjectPrinter;

    ObjectPrinter printer("PingSampleSelector", float_precision, superscript_exponents);

    std::string inactive_filters;

    printer.register_section("Active beam/sample filters");
    register_filter(printer, "min_beam_number", _min_beam_number, "", inactive_filters);
    register_filter(printer, "max_beam_number", _max_beam_number, "", inactive_filters);
    register_filter(printer, "min_sample_number", _min_sample_number, "", inactive_filters);
    register_filter(printer, "max_sample_number", _max_sample_number, "", inactive_filters);
    register_filter(printer, "min_beam_angle", _min_beam_angle, "°", inactive_filters);
    register_filter(printer, "max_beam_angle", _max_beam_angle, "°", inactive_filters);
    register_filter(printer, "min_sample_range", _min_sample_range, "m", inactive_filters);
    register_filter(printer, "max_sample_range", _max_sample_range, "m", inactive_filters);

    register_filter(printer,
                    "min_minslant_range_percent",
                    _min_minslant_range_percent,
                    "%",
                    inactive_filters);

    register_filter(printer,
                    "max_minslant_range_percent",
                    _max_minslant_range_percent,
                    "%",
                    inactive_filters);

    register_filter(
        printer, "min_bottom_range_percent", _min_bottom_range_percent, "%", inactive_filters);

    register_filter(
        printer, "max_bottom_range_percent", _max_bottom_range_percent, "%", inactive_filters);

    if (_transmit_sectors)
        printer.register_container("transmit_sectors", *_transmit_sectors);
    else
        inactive_filters += "transmit_sectors, ";

    register_filter(printer,
                    "transmit_sector_min_beam_angle",
                    _transmit_sector_min_beam_angle,
                    "°",
                    inactive_filters);

    register_filter(printer,
                    "transmit_sector_max_beam_angle",
                    _transmit_sector_max_beam_angle,
                    "°",
                    inactive_filters);

    printer.register_value("beam_step", _beam_step);
    printer.register_value("sample_step", _sample_step);

    printer.register_section("Inactive beam/sample filters");
    if (inactive_filters.size() > 2)
        inactive_filters = inactive_filters.substr(0, inactive_filters.size() - 2);
    else
        inactive_filters = "None";
    printer.register_string("inactive_filters", inactive_filters);

    return printer;
}

void PingSampleSelector::register_filter(themachinethatgoesping::tools::classhelper::ObjectPrinter& printer,
                     const std::string&                                         name,
                     const std::optional<float>&                                value,
                     const std::string&                                         unit,
                     std::string& inactive_filters) const
{
    if (value)
        printer.register_value(name, *value, unit);
    else
    {
        printer.register_string(name, "None");
        inactive_filters += name + ", ";
    }
}

void PingSampleSelector::register_filter(themachinethatgoesping::tools::classhelper::ObjectPrinter& printer,
                     const std::string&                                         name,
                     const std::optional<size_t>&                               value,
                     const std::string&                                         unit,
                     std::string& inactive_filters) const
{
    if (value)
        printer.register_value(name, *value, unit);
    else
    {
        printer.register_string(name, "None");
        inactive_filters += name + ", ";
    }
}

} // namespace pingtools
} // namespace pingprocessing
} // namespace themachinethatgoesping
