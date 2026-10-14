// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "amplitudecalibration.hpp"

#include <cmath>
#include <istream>
#include <limits>
#include <ostream>

#include <boost/iostreams/stream.hpp>
#include <xxhash.hpp>

#include <themachinethatgoesping/algorithms/amplitudecorrection/functions.hpp>
#include <themachinethatgoesping/tools/classhelper/objectprinter.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {
namespace calibration {

AmplitudeCalibration::AmplitudeCalibration(float system_offset)
    : _system_offset(system_offset)
{
}

bool AmplitudeCalibration::operator==(const AmplitudeCalibration& other) const
{
    if (_offset_per_beamangle != other._offset_per_beamangle)
        return false;

    if (_offset_per_range != other._offset_per_range)
        return false;

    if (_offset_per_beamangle_and_range != other._offset_per_beamangle_and_range)
        return false;

    if (std::isnan(_system_offset) && std::isnan(other._system_offset))
        return true;

    if (_system_offset == other._system_offset)
        return true;

    return false;
}

void AmplitudeCalibration::set_offset_per_beamangle(const std::vector<float>& beamangle,
                                                    const std::vector<float>& offset)
{
    _offset_per_beamangle.set_data_XY(beamangle, offset);
}

void AmplitudeCalibration::set_offset_per_range(const std::vector<float>& range,
                                                const std::vector<float>& offset)
{
    _offset_per_range.set_data_XY(range, offset);
}

void AmplitudeCalibration::set_offset_per_beamangle_and_range(
    const std::vector<float>&              beamangle,
    const std::vector<float>&              range,
    const std::vector<std::vector<float>>& values)
{
    _offset_per_beamangle_and_range.clear();
    for (unsigned int bi = 0; bi < beamangle.size(); ++bi)
        _offset_per_beamangle_and_range.append_row(beamangle[bi], range, values[bi]);
}

auto AmplitudeCalibration::get_offset_per_beamangle(const std::vector<float>& beamangles)
{
    return _offset_per_beamangle(beamangles);
}

float AmplitudeCalibration::get_offset_per_beamangle(float beamangle) const
{
    return _offset_per_beamangle.get_y(beamangle);
}

auto AmplitudeCalibration::get_offset_per_range(const std::vector<float>& ranges)
{
    return _offset_per_range(ranges);
}

float AmplitudeCalibration::get_offset_per_range(float range) const
{
    return _offset_per_range.get_y(range);
}

auto AmplitudeCalibration::get_offset_per_beamangle_and_range(
    const std::vector<float>& beamangles,
    const std::vector<float>& ranges) const
{
    return _offset_per_beamangle_and_range(beamangles, ranges);
}

AmplitudeCalibration AmplitudeCalibration::from_stream(std::istream& is)
{
    AmplitudeCalibration calibration;

    is.read(reinterpret_cast<char*>(&calibration._system_offset), sizeof(float));

    calibration._offset_per_beamangle = calibration._offset_per_beamangle.from_stream(is);
    calibration._offset_per_range     = calibration._offset_per_range.from_stream(is);
    calibration._offset_per_beamangle_and_range =
        calibration._offset_per_beamangle_and_range.from_stream(is);

    return calibration;
}

void AmplitudeCalibration::to_stream(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(&_system_offset), sizeof(float));

    _offset_per_beamangle.to_stream(os);
    _offset_per_range.to_stream(os);
    _offset_per_beamangle_and_range.to_stream(os);
}

tools::classhelper::ObjectPrinter AmplitudeCalibration::__printer__(
    unsigned int float_precision,
    bool         superscript_exponents) const
{
    tools::classhelper::ObjectPrinter printer(
        "AmplitudeCalibration", float_precision, superscript_exponents);

    printer.register_section("System offsets");
    if (!std::isnan(_system_offset))
        printer.register_value("system_offset", _system_offset, "dB");
    if (!_offset_per_beamangle.empty())
        printer.append(_offset_per_beamangle.__printer__(float_precision, superscript_exponents));
    if (!_offset_per_range.empty())
        printer.append(_offset_per_range.__printer__(float_precision, superscript_exponents));
    if (!_offset_per_beamangle_and_range.empty())
        printer.append(
            _offset_per_beamangle_and_range.__printer__(float_precision, superscript_exponents));

    return printer;
}

xxh::hash_t<64> AmplitudeCalibration::binary_hash() const
{
    xxh::hash3_state_t<64>               hash;
    boost::iostreams::stream<XXHashSink> stream(hash);

    add_hash(stream);

    stream.flush();
    return hash.digest();
}

void AmplitudeCalibration::add_hash(boost::iostreams::stream<XXHashSink>& hash_stream) const
{
    if (has_system_offset())
        hash_stream.write(reinterpret_cast<const char*>(&_system_offset), sizeof(float));

    if (has_offset_per_beamangle())
        _offset_per_beamangle.to_stream(hash_stream);

    if (has_offset_per_range())
        _offset_per_range.to_stream(hash_stream);

    if (has_offset_per_beamangle_and_range())
        _offset_per_beamangle_and_range.to_stream(hash_stream);
}

} // namespace calibration
} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping
