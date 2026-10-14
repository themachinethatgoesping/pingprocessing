// SPDX-FileCopyrightText: 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
//

#pragma once

/* generated doc strings */
#include ".docstrings/calibrationinfo.doc.hpp"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {
namespace calibration {

/**
 * @brief Base class for storing calibration documentation / provenance information.
 *
 * Records two kinds of information:
 *  - **setup_steps**: Python script snippets that document how each calibration
 *    parameter was set (e.g. from a datagram object or manually).
 *  - **formulas**: Python script snippets that document the exact math applied to
 *    the raw amplitude data (e.g. power, av, sv calibration formulas).
 *
 * A suppress-guard mechanism ensures that when an outer method (e.g.
 * `set_environment_parameters(XML_Environment)`) registers a step and then calls
 * the value-based overload internally, the inner call does not overwrite
 * the outer registration.
 *
 * Steps are stored in insertion order.  Registering a step with an existing name
 * **replaces** the previous entry (unless suppressed).
 */
class CalibrationInfo
{
  public:
    // -----------------------------------------------------------------
    // RAII suppress guard
    // -----------------------------------------------------------------

    /**
     * @brief RAII guard that temporarily suppresses step registration.
     *
     * Create via `CalibrationInfo::suppress()`.  While the guard is alive,
     * all calls to `register_step()` are silently ignored.  Nesting is
     * supported (counted).
     */
    struct SuppressGuard
    {
        CalibrationInfo* info;

        explicit SuppressGuard(CalibrationInfo* i)
            : info(i)
        {
            if (info)
                ++info->_suppress_depth;
        }

        ~SuppressGuard()
        {
            if (info)
                --info->_suppress_depth;
        }

        // movable (transfers ownership)
        SuppressGuard(SuppressGuard&& o) noexcept
            : info(o.info)
        {
            o.info = nullptr;
        }

        // not copyable
        SuppressGuard(const SuppressGuard&)            = delete;
        SuppressGuard& operator=(const SuppressGuard&) = delete;
        SuppressGuard& operator=(SuppressGuard&&)      = delete;
    };

  private:
    int _suppress_depth = 0;

    // ordered lists: (name, script)
    std::vector<std::pair<std::string, std::string>> _setup_steps;
    std::vector<std::pair<std::string, std::string>> _formulas;

  public:
    CalibrationInfo()          = default;
    virtual ~CalibrationInfo() = default;

    // -----------------------------------------------------------------
    // Suppress mechanism
    // -----------------------------------------------------------------

    /** @brief Returns true when step registration is currently suppressed. */
    bool is_suppressed() const { return _suppress_depth > 0; }

    /**
     * @brief Create a guard that suppresses step registration until destroyed.
     *
     * Typical usage:
     * @code
     * _cal_info.register_step("foo", script);   // registers
     * auto guard = _cal_info.suppress();
     * inner_method();                             // inner register_step calls are suppressed
     * @endcode
     */
    SuppressGuard suppress() { return SuppressGuard(this); }

    // -----------------------------------------------------------------
    // Step registration (replace-by-name, subject to suppression)
    // -----------------------------------------------------------------

    /**
     * @brief Register (or replace) a setup step.
     *
     * If a step with the same @p name already exists it is overwritten
     * in-place (preserving insertion order).  No-op when suppressed.
     */
    void register_step(const std::string& name, const std::string& script)
    {
        if (is_suppressed())
            return;

        for (auto& [n, s] : _setup_steps)
        {
            if (n == name)
            {
                s = script;
                return;
            }
        }
        _setup_steps.emplace_back(name, script);
    }

    // -----------------------------------------------------------------
    // Formula registration (replace-by-name, NOT subject to suppression)
    // -----------------------------------------------------------------

    /**
     * @brief Register (or replace) a formula description.
     */
    void register_formula(const std::string& name, const std::string& script)
    {
        for (auto& [n, s] : _formulas)
        {
            if (n == name)
            {
                s = script;
                return;
            }
        }
        _formulas.emplace_back(name, script);
    }

    // -----------------------------------------------------------------
    // Retrieval — individual
    // -----------------------------------------------------------------

    /** @brief Get the steps list (name→script pairs, in insertion order). */
    const std::vector<std::pair<std::string, std::string>>& get_setup_steps() const
    {
        return _setup_steps;
    }

    /** @brief Get the formulas list (name→script pairs, in insertion order). */
    const std::vector<std::pair<std::string, std::string>>& get_formulas() const
    {
        return _formulas;
    }

    /** @brief Get script for a single setup step by name (empty if not found). */
    std::string get_step_script(const std::string& name) const
    {
        for (const auto& [n, s] : _setup_steps)
            if (n == name)
                return s;
        return {};
    }

    /** @brief Get script for a single formula by name (empty if not found). */
    std::string get_formula_script(const std::string& name) const
    {
        for (const auto& [n, s] : _formulas)
            if (n == name)
                return s;
        return {};
    }

    // -----------------------------------------------------------------
    // Retrieval — combined scripts
    // -----------------------------------------------------------------

    /** @brief Concatenate all setup steps into one script. */
    std::string get_setup_script() const
    {
        std::ostringstream os;
        for (const auto& [name, script] : _setup_steps)
        {
            os << "# === " << name << " ===\n";
            os << script << "\n\n";
        }
        return os.str();
    }

    /** @brief Concatenate all formulas into one script. */
    std::string get_formulas_script() const
    {
        std::ostringstream os;
        for (const auto& [name, script] : _formulas)
        {
            os << "# === " << name << " ===\n";
            os << script << "\n\n";
        }
        return os.str();
    }

    /** @brief Concatenate everything (setup steps + formulas). */
    std::string get_full_script() const
    {
        std::ostringstream os;

        if (!_setup_steps.empty())
        {
            os << "# ============================\n";
            os << "# Calibration Setup Steps\n";
            os << "# ============================\n\n";
            os << get_setup_script();
        }

        if (!_formulas.empty())
        {
            os << "# ============================\n";
            os << "# Calibration Formulas\n";
            os << "# ============================\n\n";
            os << get_formulas_script();
        }

        return os.str();
    }

    // -----------------------------------------------------------------
    // Queries & housekeeping
    // -----------------------------------------------------------------

    bool has_steps() const { return !_setup_steps.empty(); }
    bool has_formulas() const { return !_formulas.empty(); }
    bool has_step(const std::string& name) const
    {
        for (const auto& [n, _] : _setup_steps)
            if (n == name)
                return true;
        return false;
    }
    bool has_formula(const std::string& name) const
    {
        for (const auto& [n, _] : _formulas)
            if (n == name)
                return true;
        return false;
    }

    void clear_steps() { _setup_steps.clear(); }
    void clear_formulas() { _formulas.clear(); }
    void clear()
    {
        _setup_steps.clear();
        _formulas.clear();
    }
};

} // namespace calibration
} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping
