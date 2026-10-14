// SPDX-FileCopyrightText: 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
//

#pragma once

/* generated doc strings */
#include ".docstrings/watercolumncalibration.doc.hpp"

#include <fmt/format.h>
#include <themachinethatgoesping/algorithms/amplitudecorrection/functions.hpp>
#include <themachinethatgoesping/tools/classhelper/objectprinter.hpp>
#include <themachinethatgoesping/tools/classhelper/option_frozen.hpp>
#include <themachinethatgoesping/tools/helper/floatcompare.hpp>

#include "amplitudecalibration.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {
namespace calibration {

/**
 * @brief Calibration type for water column data processing
 */
enum class t_calibration_type : uint8_t
{
    power, ///< power calibration (Received amplitudes computed to power, no absorption, no
           ///< spreading loss)
    rp,    ///< power derived point scattering (uncalibrated TS without parameter and absorption
           ///< compensation)
    rv,    ///< power derived volume scattering (uncalibrated SV without parameter and absorption
           ///< compensation)
    pp,    ///< power derived point scattering (uncalibrated TS without parameter compensation)
    pv,    ///< power derived volume scattering (uncalibrated SV without parameter compensation)
    ap,    ///< amplitude derived point scattering (uncalibrated TS)
    av,    ///< amplitude derived volume scattering (uncalibrated SV)
    sp,    ///< point scattering (uncompensated TS)
    sv     ///< volume scattering SV
};

inline constexpr std::array<t_calibration_type, 9> t_calibration_type_values = {
    t_calibration_type::power, t_calibration_type::rp, t_calibration_type::rv,
    t_calibration_type::pp,    t_calibration_type::pv, t_calibration_type::ap,
    t_calibration_type::av,    t_calibration_type::sp, t_calibration_type::sv
};

inline constexpr std::array<std::string_view, 9> t_calibration_type_names = { "power", "rp", "rv",
                                                                              "pp",    "pv", "ap",
                                                                              "av",    "sp", "sv" };

inline constexpr std::array<std::string_view, 9> t_calibration_type_alt_names = {
    "Power",
    "Power, no absorption (point backscattering)",
    "Power, no absorption (volume backscattering)",
    "Power (point backscattering)",
    "Power (volume backscattering)",
    "Point backscattering (uncalibrated TS)",
    "Volume backscattering (uncalibrated SV)",
    "Point backscattering (uncompensated TS)",
    "Volume backscattering (SV)"
};

// ============================================================================
// Compile-time properties for calibration types
// ============================================================================

/**
 * @brief Get the target TVG factor for a given calibration type.
 * @param ct The calibration type.
 * @return The TVG factor (0, 20, or 40).
 */
constexpr float get_target_tvg_factor(t_calibration_type ct)
{
    switch (ct)
    {
        case t_calibration_type::power:
            return 0.0f;
        case t_calibration_type::rp:
        case t_calibration_type::pp:
        case t_calibration_type::ap:
        case t_calibration_type::sp:
            return 40.0f;
        case t_calibration_type::rv:
        case t_calibration_type::pv:
        case t_calibration_type::av:
        case t_calibration_type::sv:
            return 20.0f;
    }
    return 0.0f; // unreachable
}

/**
 * @brief Check if absorption should be applied for a given calibration type.
 *
 * For power, rp, rv: absorption is explicitly NOT applied (use 0).
 * For all others: absorption is applied.
 *
 * @param ct The calibration type.
 * @return true if absorption should be applied, false otherwise.
 */
constexpr bool applies_absorption(t_calibration_type ct)
{
    switch (ct)
    {
        case t_calibration_type::power:
        case t_calibration_type::rp:
        case t_calibration_type::rv:
            return false;
        default:
            return true;
    }
}

/**
 * @brief Calibration kind enum for selecting which AmplitudeCalibration to use.
 */
enum class t_calibration_kind : uint8_t
{
    power,
    ap,
    av,
    sp,
    sv
};

/**
 * @brief Get the calibration kind (which AmplitudeCalibration pointer) for a given calibration
 * type.
 * @param ct The calibration type.
 * @return The calibration kind.
 */
constexpr t_calibration_kind get_calibration_kind(t_calibration_type ct)
{
    switch (ct)
    {
        case t_calibration_type::power:
        case t_calibration_type::rp:
        case t_calibration_type::rv:
        case t_calibration_type::pp:
        case t_calibration_type::pv:
            return t_calibration_kind::power;
        case t_calibration_type::ap:
            return t_calibration_kind::ap;
        case t_calibration_type::av:
            return t_calibration_kind::av;
        case t_calibration_type::sp:
            return t_calibration_kind::sp;
        case t_calibration_type::sv:
            return t_calibration_kind::sv;
    }
    return t_calibration_kind::power; // unreachable
}

/**
 * @brief Get the name of a calibration kind for error messages.
 * @param ck The calibration kind.
 * @return Human-readable name.
 */
constexpr std::string_view get_calibration_kind_name(t_calibration_kind ck)
{
    switch (ck)
    {
        case t_calibration_kind::power:
            return "Power calibration";
        case t_calibration_kind::ap:
            return "Ap calibration";
        case t_calibration_kind::av:
            return "Av calibration";
        case t_calibration_kind::sp:
            return "Sp calibration";
        case t_calibration_kind::sv:
            return "Sv calibration";
    }
    return "Unknown calibration"; // unreachable
}

using o_calibration_type =
    themachinethatgoesping::tools::classhelper::OptionFrozen<t_calibration_type,
                                                             t_calibration_type_values.size(),
                                                             t_calibration_type_values,
                                                             t_calibration_type_names,
                                                             t_calibration_type_alt_names>;

class WaterColumnCalibration
{
  protected:
    // different types of offsets
    std::unique_ptr<AmplitudeCalibration> _power_calibration;
    std::unique_ptr<AmplitudeCalibration> _ap_calibration;
    std::unique_ptr<AmplitudeCalibration> _av_calibration;
    std::unique_ptr<AmplitudeCalibration> _sp_calibration;
    std::unique_ptr<AmplitudeCalibration> _sv_calibration;

    // water column corrections to be applied
    std::optional<float> _absorption_db_m; // new absorption that should be
                                           // applied to the data in dB/m

    // factors that have been applied to the raw data by a TVG function
    float _tvg_absorption_db_m = 0.0f; // the absorption applied to the data in dB/m
    float _tvg_factor = 0.0f; // the TVG factor applied to the data (typically 0, 20 or 40 (log(r)))

  public:
    // Type alias for backwards compatibility (enum is now at namespace level)
    using t_calibration_type = calibration::t_calibration_type;

    WaterColumnCalibration(float tvg_absorption_db_m = 0.0f, float tvg_factor = 0.0f)
        : _tvg_absorption_db_m(tvg_absorption_db_m)
        , _tvg_factor(tvg_factor)
    {
    }
    WaterColumnCalibration(const AmplitudeCalibration& power_calibration,
                           const AmplitudeCalibration& ap_calibration,
                           const AmplitudeCalibration& av_calibration,
                           float                       tvg_absorption_db_m = 0.0f,
                           float                       tvg_factor          = 0.0f)
        : _power_calibration(std::make_unique<AmplitudeCalibration>(power_calibration))
        , _ap_calibration(std::make_unique<AmplitudeCalibration>(ap_calibration))
        , _av_calibration(std::make_unique<AmplitudeCalibration>(av_calibration))
        , _tvg_absorption_db_m(tvg_absorption_db_m)
        , _tvg_factor(tvg_factor)
    {
    }
    WaterColumnCalibration(const WaterColumnCalibration& other);

    virtual ~WaterColumnCalibration() = default;

    // this is used to check if parent classes are initialized
    virtual void check_initialized() const {}
    virtual void check_modifying_base_calibration_allowed() const {}

    // operator overloads

    void set_absorption_db_m(std::optional<float> value) { _absorption_db_m = value; }

    std::optional<float> get_absorption_to_apply(
        std::optional<float> absorption_db_m = std::nullopt) const;
    bool has_valid_absorption_db_m() const;

    std::optional<float> get_tvg_factor_to_apply(float tvg_factor) const;

  protected:
    /**
     * @brief Get the calibration pointer for a given calibration kind.
     * @tparam kind The calibration kind (constexpr).
     * @return Reference to the unique_ptr for that calibration.
     */
    template<t_calibration_kind kind>
    const std::unique_ptr<AmplitudeCalibration>& get_calibration_ptr() const
    {
        if constexpr (kind == t_calibration_kind::power)
            return _power_calibration;
        else if constexpr (kind == t_calibration_kind::ap)
            return _ap_calibration;
        else if constexpr (kind == t_calibration_kind::av)
            return _av_calibration;
        else if constexpr (kind == t_calibration_kind::sp)
            return _sp_calibration;
        else if constexpr (kind == t_calibration_kind::sv)
            return _sv_calibration;
    }

  public:
    template<t_calibration_type       calibration_type,
             tools::helper::c_xtensor t_xtensor_2d,
             tools::helper::c_xtensor t_xtensor_1d>
    t_xtensor_2d apply_beam_sample_correction(const t_xtensor_2d& wci,
                                              const t_xtensor_1d& beam_angles,
                                              const t_xtensor_1d& ranges,
                                              int                 mp_cores = 1) const
    {
        this->check_initialized();

        constexpr auto kind = get_calibration_kind(calibration_type);
        const auto&    cal  = get_calibration_ptr<kind>();

        check_calibration_initialized(__func__, get_calibration_kind_name(kind), cal);

        // For power/rp/rv, absorption is explicitly 0; otherwise use configured absorption
        constexpr bool use_absorption = applies_absorption(calibration_type);
        auto           absorption     = use_absorption ? get_absorption_to_apply() : get_absorption_to_apply(0);

        constexpr float target_tvg = get_target_tvg_factor(calibration_type);
        auto            tvg        = get_tvg_factor_to_apply(target_tvg);

        return cal->apply_beam_sample_correction(wci, beam_angles, ranges, absorption, tvg, mp_cores);
    }

    /**
     * @brief Apply beam and sample correction with per-beam absorption coefficients.
     *
     * This overload supports per-beam absorption for multi-sector sonars (e.g., Kongsberg)
     * where each transmit sector may have a different absorption value.
     *
     * @note For calibration types that don't apply absorption (power, rp, rv), this
     *       function ignores the per-beam absorption parameter.
     */
    template<t_calibration_type       calibration_type,
             tools::helper::c_xtensor t_xtensor_2d,
             tools::helper::c_xtensor t_xtensor_1d>
    t_xtensor_2d apply_beam_sample_correction(const t_xtensor_2d& wci,
                                              const t_xtensor_1d& beam_angles,
                                              const t_xtensor_1d& ranges,
                                              const t_xtensor_1d& absorption_db_m_per_beam,
                                              int                 mp_cores = 1) const
    {
        this->check_initialized();

        constexpr auto kind = get_calibration_kind(calibration_type);
        const auto&    cal  = get_calibration_ptr<kind>();

        check_calibration_initialized(__func__, get_calibration_kind_name(kind), cal);

        constexpr float target_tvg = get_target_tvg_factor(calibration_type);
        auto            tvg        = get_tvg_factor_to_apply(target_tvg);

        // For power/rp/rv, absorption is explicitly 0 - use scalar version
        constexpr bool use_absorption = applies_absorption(calibration_type);
        if constexpr (!use_absorption)
        {
            return cal->apply_beam_sample_correction(
                wci, beam_angles, ranges, get_absorption_to_apply(0), tvg, mp_cores);
        }
        else
        {
            return cal->apply_beam_sample_correction(
                wci, beam_angles, ranges, absorption_db_m_per_beam, tvg, mp_cores);
        }
    }

    template<t_calibration_type       calibration_type,
             tools::helper::c_xtensor t_xtensor_2d,
             tools::helper::c_xtensor t_xtensor_1d>
    void inplace_beam_sample_correction(t_xtensor_2d&         wci,
                                        const t_xtensor_1d&   beam_angles,
                                        const t_xtensor_1d&   ranges,
                                        std::optional<size_t> min_beam_index = std::nullopt,
                                        std::optional<size_t> max_beam_index = std::nullopt,
                                        int                   mp_cores       = 1) const
    {
        this->check_initialized();

        constexpr auto kind = get_calibration_kind(calibration_type);
        const auto&    cal  = get_calibration_ptr<kind>();

        check_calibration_initialized(__func__, get_calibration_kind_name(kind), cal);

        // For power/rp/rv, absorption is explicitly 0; otherwise use configured absorption
        constexpr bool use_absorption = applies_absorption(calibration_type);
        auto           absorption     = use_absorption ? get_absorption_to_apply() : get_absorption_to_apply(0);

        constexpr float target_tvg = get_target_tvg_factor(calibration_type);
        auto            tvg        = get_tvg_factor_to_apply(target_tvg);

        cal->inplace_beam_sample_correction(
            wci, beam_angles, ranges, absorption, tvg, min_beam_index, max_beam_index, mp_cores);
    }

    /**
     * @brief Inplace apply beam and sample correction with per-beam absorption coefficients.
     *
     * This overload supports per-beam absorption for multi-sector sonars (e.g., Kongsberg)
     * where each transmit sector may have a different absorption value.
     *
     * @note For calibration types that don't apply absorption (power, rp, rv), this
     *       function ignores the per-beam absorption parameter.
     */
    template<t_calibration_type       calibration_type,
             tools::helper::c_xtensor t_xtensor_2d,
             tools::helper::c_xtensor t_xtensor_1d>
    void inplace_beam_sample_correction(t_xtensor_2d&         wci,
                                        const t_xtensor_1d&   beam_angles,
                                        const t_xtensor_1d&   ranges,
                                        const t_xtensor_1d&   absorption_db_m_per_beam,
                                        std::optional<size_t> min_beam_index = std::nullopt,
                                        std::optional<size_t> max_beam_index = std::nullopt,
                                        int                   mp_cores       = 1) const
    {
        this->check_initialized();

        constexpr auto kind = get_calibration_kind(calibration_type);
        const auto&    cal  = get_calibration_ptr<kind>();

        check_calibration_initialized(__func__, get_calibration_kind_name(kind), cal);

        constexpr float target_tvg = get_target_tvg_factor(calibration_type);
        auto            tvg        = get_tvg_factor_to_apply(target_tvg);

        // For power/rp/rv, absorption is explicitly 0 - use scalar version
        constexpr bool use_absorption = applies_absorption(calibration_type);
        if constexpr (!use_absorption)
        {
            cal->inplace_beam_sample_correction(wci,
                                                beam_angles,
                                                ranges,
                                                get_absorption_to_apply(0),
                                                tvg,
                                                min_beam_index,
                                                max_beam_index,
                                                mp_cores);
        }
        else
        {
            cal->inplace_beam_sample_correction(wci,
                                                beam_angles,
                                                ranges,
                                                absorption_db_m_per_beam,
                                                tvg,
                                                min_beam_index,
                                                max_beam_index,
                                                mp_cores);
        }
    }

    // getters / setters
    float get_absorption_db_m() const;

    float get_tvg_absorption_db_m() const;
    float get_tvg_factor() const;

    // has calibration
    bool has_power_calibration() const;
    bool has_ap_calibration() const;
    bool has_av_calibration() const;
    bool has_sp_calibration() const;
    bool has_sv_calibration() const;

    // getters / setters
    const AmplitudeCalibration& get_power_calibration() const;
    const AmplitudeCalibration& get_ap_calibration() const;
    const AmplitudeCalibration& get_av_calibration() const;
    const AmplitudeCalibration& get_sp_calibration() const;
    const AmplitudeCalibration& get_sv_calibration() const;
    void                        set_power_calibration(const AmplitudeCalibration& calibration);

    void set_ap_calibration(const AmplitudeCalibration& calibration);

    void set_av_calibration(const AmplitudeCalibration& calibration);
    void set_sp_calibration(const AmplitudeCalibration& calibration);

    void set_sv_calibration(const AmplitudeCalibration& calibration);

    // operator overloads
    WaterColumnCalibration operator=(const WaterColumnCalibration& other);

    bool operator==(const WaterColumnCalibration& other) const;

    // stream i/o
    static WaterColumnCalibration from_stream(std::istream& is);

    void to_stream(std::ostream& os) const;

    // ----- objectprinter -----
    tools::classhelper::ObjectPrinter __printer__(unsigned int float_precision,
                                                  bool         superscript_exponents) const;

    // ----- class helper macros -----
    __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__
    __STREAM_DEFAULT_TOFROM_BINARY_FUNCTIONS__(WaterColumnCalibration)

  private:
    void check_calibration_initialized(std::string_view function_name,
                                       std::string_view calibration_name,
                                       const std::unique_ptr<AmplitudeCalibration>& ptr) const;
};

// boost hash
// IGNORE_DOC:mkd_doc_themachinethatgoesping_pingprocessing_datatypes_calibration_hash_value
inline std::size_t hash_value(const WaterColumnCalibration& arg)
{
    return arg.binary_hash();
}
}
}
}
}

extern template class themachinethatgoesping::tools::classhelper::OptionFrozen<
    themachinethatgoesping::pingprocessing::datatypes::calibration::t_calibration_type,
    themachinethatgoesping::pingprocessing::datatypes::calibration::
        t_calibration_type_values.size(),
    themachinethatgoesping::pingprocessing::datatypes::calibration::
        t_calibration_type_values,
    themachinethatgoesping::pingprocessing::datatypes::calibration::
        t_calibration_type_names,
    themachinethatgoesping::pingprocessing::datatypes::calibration::
        t_calibration_type_alt_names>;