// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "watercolumncalibration.hpp"

#include <cmath>
#include <limits>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {
namespace calibration {

// copy constructor implementation
WaterColumnCalibration::WaterColumnCalibration(const WaterColumnCalibration& other)
    : _absorption_db_m(other._absorption_db_m)
    , _tvg_absorption_db_m(other._tvg_absorption_db_m)
    , _tvg_factor(other._tvg_factor)
{
    if (other._power_calibration)
        _power_calibration = std::make_unique<AmplitudeCalibration>(*other._power_calibration);
    if (other._ap_calibration)
        _ap_calibration = std::make_unique<AmplitudeCalibration>(*other._ap_calibration);
    if (other._av_calibration)
        _av_calibration = std::make_unique<AmplitudeCalibration>(*other._av_calibration);
    if (other._sp_calibration)
        _sp_calibration = std::make_unique<AmplitudeCalibration>(*other._sp_calibration);
    if (other._sv_calibration)
        _sv_calibration = std::make_unique<AmplitudeCalibration>(*other._sv_calibration);
}

std::optional<float> WaterColumnCalibration::get_absorption_to_apply(
    std::optional<float> absorption_db_m) const
{
    if (absorption_db_m.has_value())
    {
        float diff = absorption_db_m.value() - _tvg_absorption_db_m;
        if (std::fabs(diff) > 0.0000001f)
            return diff;
    }
    else if (_absorption_db_m.has_value())
    {
        float diff = _absorption_db_m.value() - _tvg_absorption_db_m;
        if (std::fabs(diff) > 0.0000001f)
            return diff;
    }

    return std::nullopt;
}

bool WaterColumnCalibration::has_valid_absorption_db_m() const
{
    if (_absorption_db_m.has_value())
        return std::isfinite(_absorption_db_m.value());

    return _tvg_absorption_db_m != 0.0f;
}

std::optional<float> WaterColumnCalibration::get_tvg_factor_to_apply(float tvg_factor) const
{
    this->check_initialized();

    float diff = tvg_factor - _tvg_factor;
    if (std::fabs(diff) > 0.0000001f)
        return diff;

    return std::nullopt;
}

// getters / setters
float WaterColumnCalibration::get_absorption_db_m() const
{
    if (_absorption_db_m.has_value())
        return _absorption_db_m.value();
    return _tvg_absorption_db_m;
}

float WaterColumnCalibration::get_tvg_absorption_db_m() const
{
    this->check_initialized();
    return _tvg_absorption_db_m;
}

float WaterColumnCalibration::get_tvg_factor() const
{
    this->check_initialized();
    return _tvg_factor;
}

// has calibration functions
bool WaterColumnCalibration::has_power_calibration() const { return bool(_power_calibration); }
bool WaterColumnCalibration::has_ap_calibration() const { return bool(_ap_calibration); }
bool WaterColumnCalibration::has_av_calibration() const { return bool(_av_calibration); }
bool WaterColumnCalibration::has_sp_calibration() const { return bool(_sp_calibration); }
bool WaterColumnCalibration::has_sv_calibration() const { return bool(_sv_calibration); }

// calibration getters
const AmplitudeCalibration& WaterColumnCalibration::get_power_calibration() const
{
    this->check_initialized();
    check_calibration_initialized(__func__, "Power calibration", _power_calibration);
    return *_power_calibration;
}

const AmplitudeCalibration& WaterColumnCalibration::get_ap_calibration() const
{
    this->check_initialized();
    check_calibration_initialized(__func__, "Ap calibration", _ap_calibration);
    return *_ap_calibration;
}

const AmplitudeCalibration& WaterColumnCalibration::get_av_calibration() const
{
    this->check_initialized();
    check_calibration_initialized(__func__, "Av calibration", _av_calibration);
    return *_av_calibration;
}

const AmplitudeCalibration& WaterColumnCalibration::get_sp_calibration() const
{
    this->check_initialized();
    check_calibration_initialized(__func__, "Sp calibration", _sp_calibration);
    return *_sp_calibration;
}

const AmplitudeCalibration& WaterColumnCalibration::get_sv_calibration() const
{
    this->check_initialized();
    check_calibration_initialized(__func__, "Sv calibration", _sv_calibration);
    return *_sv_calibration;
}

// calibration setters
void WaterColumnCalibration::set_power_calibration(const AmplitudeCalibration& calibration)
{
    check_modifying_base_calibration_allowed();
    _power_calibration = std::make_unique<AmplitudeCalibration>(calibration);
}

void WaterColumnCalibration::set_ap_calibration(const AmplitudeCalibration& calibration)
{
    check_modifying_base_calibration_allowed();
    _ap_calibration = std::make_unique<AmplitudeCalibration>(calibration);
}

void WaterColumnCalibration::set_av_calibration(const AmplitudeCalibration& calibration)
{
    check_modifying_base_calibration_allowed();
    _av_calibration = std::make_unique<AmplitudeCalibration>(calibration);
}

void WaterColumnCalibration::set_sp_calibration(const AmplitudeCalibration& calibration)
{
    _sp_calibration = std::make_unique<AmplitudeCalibration>(calibration);
}

void WaterColumnCalibration::set_sv_calibration(const AmplitudeCalibration& calibration)
{
    _sv_calibration = std::make_unique<AmplitudeCalibration>(calibration);
}

// operator overloads
WaterColumnCalibration WaterColumnCalibration::operator=(const WaterColumnCalibration& other)
{
    return WaterColumnCalibration(other);
}

bool WaterColumnCalibration::operator==(const WaterColumnCalibration& other) const
{
    if (bool(_power_calibration) != bool(other._power_calibration))
        return false;

    if (bool(_ap_calibration) != bool(other._ap_calibration))
        return false;

    if (bool(_av_calibration) != bool(other._av_calibration))
        return false;

    if (bool(_sp_calibration) != bool(other._sp_calibration))
        return false;

    if (bool(_sv_calibration) != bool(other._sv_calibration))
        return false;

    if (bool(_power_calibration))
        if (*_power_calibration != *other._power_calibration)
            return false;

    if (bool(_ap_calibration))
        if (*_ap_calibration != *other._ap_calibration)
            return false;

    if (bool(_av_calibration))
        if (*_av_calibration != *other._av_calibration)
            return false;

    if (bool(_sp_calibration))
        if (*_sp_calibration != *other._sp_calibration)
            return false;

    if (bool(_sv_calibration))
        if (*_sv_calibration != *other._sv_calibration)
            return false;

    if (_absorption_db_m.has_value() != other._absorption_db_m.has_value())
        return false;
    if (_absorption_db_m.has_value())
        if (!tools::helper::float_equals(_absorption_db_m.value(),
                                         other._absorption_db_m.value()))
            return false;

    if (!tools::helper::float_equals(_tvg_absorption_db_m, other._tvg_absorption_db_m))
        return false;

    if (!tools::helper::float_equals(_tvg_factor, other._tvg_factor))
        return false;

    return true;
}

// stream i/o
WaterColumnCalibration WaterColumnCalibration::from_stream(std::istream& is)
{
    WaterColumnCalibration calibration;

    std::vector<uint8_t> initialized_calibrations = { false, false, false, false, false };
    is.read(reinterpret_cast<char*>(initialized_calibrations.data()),
            sizeof(bool) * initialized_calibrations.size());

    if (initialized_calibrations[0])
        calibration._power_calibration =
            std::make_unique<AmplitudeCalibration>(AmplitudeCalibration::from_stream(is));
    if (initialized_calibrations[1])
        calibration._ap_calibration =
            std::make_unique<AmplitudeCalibration>(AmplitudeCalibration::from_stream(is));
    if (initialized_calibrations[2])
        calibration._av_calibration =
            std::make_unique<AmplitudeCalibration>(AmplitudeCalibration::from_stream(is));
    if (initialized_calibrations[3])
        calibration._sp_calibration =
            std::make_unique<AmplitudeCalibration>(AmplitudeCalibration::from_stream(is));
    if (initialized_calibrations[4])
        calibration._sv_calibration =
            std::make_unique<AmplitudeCalibration>(AmplitudeCalibration::from_stream(is));

    float absorption_db_m;
    is.read(reinterpret_cast<char*>(&absorption_db_m), sizeof(float));
    if (std::isfinite(absorption_db_m))
        calibration.set_absorption_db_m(absorption_db_m);

    is.read(reinterpret_cast<char*>(&calibration._tvg_absorption_db_m), sizeof(float) * 2);

    return calibration;
}

void WaterColumnCalibration::to_stream(std::ostream& os) const
{
    std::vector<uint8_t> initialized_calibrations = { bool(_power_calibration),
                                                      bool(_ap_calibration),
                                                      bool(_av_calibration),
                                                      bool(_sp_calibration),
                                                      bool(_sv_calibration) };
    os.write(reinterpret_cast<const char*>(initialized_calibrations.data()),
             sizeof(bool) * initialized_calibrations.size());

    if (_power_calibration)
        _power_calibration->to_stream(os);
    if (_ap_calibration)
        _ap_calibration->to_stream(os);
    if (_av_calibration)
        _av_calibration->to_stream(os);
    if (_sp_calibration)
        _sp_calibration->to_stream(os);
    if (_sv_calibration)
        _sv_calibration->to_stream(os);

    float absorption_db_m = _absorption_db_m.value_or(std::numeric_limits<float>::quiet_NaN());
    os.write(reinterpret_cast<const char*>(&absorption_db_m), sizeof(float));

    os.write(reinterpret_cast<const char*>(&_tvg_absorption_db_m), sizeof(float) * 2);
}

// objectprinter
tools::classhelper::ObjectPrinter WaterColumnCalibration::__printer__(unsigned int float_precision,
                                              bool         superscript_exponents) const
{
    tools::classhelper::ObjectPrinter printer(
        "WaterColumnCalibration", float_precision, superscript_exponents);

    printer.register_section("Absorption");
    printer.register_value(
        "Absorption", _absorption_db_m.value_or(_tvg_absorption_db_m), "dB/m");
    printer.register_value(
        "Absorption_to_apply", get_absorption_to_apply().value_or(NAN), "dB/m");
    printer.register_section("TVG applied to raw data");
    printer.register_value("TVG Absorption", _tvg_absorption_db_m, "dB/m");
    printer.register_value("TVG Factor", _tvg_factor, "log(r)");

    if (_power_calibration)
    {
        printer.register_section("Power Calibration");
        printer.append(_power_calibration->__printer__(float_precision, superscript_exponents));
    }
    if (_ap_calibration)
    {
        printer.register_section("Ap Calibration (Uncompensated uncalibrated TS)");
        printer.append(_ap_calibration->__printer__(float_precision, superscript_exponents));
    }
    if (_av_calibration)
    {
        printer.register_section("Av Calibration (Uncalibrated volume scattering)");
        printer.append(_av_calibration->__printer__(float_precision, superscript_exponents));
    }
    if (_sp_calibration)
    {
        printer.register_section("Sp Calibration (Uncompensated TS)");
        printer.append(_sp_calibration->__printer__(float_precision, superscript_exponents));
    }
    if (_sv_calibration)
    {
        printer.register_section("Sv Calibration (Volume scattering)");
        printer.append(_sv_calibration->__printer__(float_precision, superscript_exponents));
    }

    return printer;
}

// private helper function
void WaterColumnCalibration::check_calibration_initialized(std::string_view function_name,
                                   std::string_view calibration_name,
                                   const std::unique_ptr<AmplitudeCalibration>& ptr) const
{
    if (!ptr)
        throw std::runtime_error(
            fmt::format("ERROR[{}]:{} not initialized", function_name, calibration_name));
}

} // namespace calibration
} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping

template class themachinethatgoesping::tools::classhelper::OptionFrozen<
    themachinethatgoesping::pingprocessing::datatypes::calibration::t_calibration_type,
    themachinethatgoesping::pingprocessing::datatypes::calibration::
        t_calibration_type_values.size(),
    themachinethatgoesping::pingprocessing::datatypes::calibration::
        t_calibration_type_values,
    themachinethatgoesping::pingprocessing::datatypes::calibration::
        t_calibration_type_names,
    themachinethatgoesping::pingprocessing::datatypes::calibration::
        t_calibration_type_alt_names>;
