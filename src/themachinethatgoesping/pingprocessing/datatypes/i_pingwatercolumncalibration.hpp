// SPDX-FileCopyrightText: 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

/**
 * @brief CRTP helper bases that give the format-specific ping water column classes a water column
 * calibration accessor with a *covariant* return type.
 *
 * The generic I_PingWatercolumn exposes get_watercolumn_calibration() /
 * get_multisectorwatercolumn_calibration() returning the generic base calibration types, so it can
 * be used generically. The echosounder implementations, however, want the SAME call to return their
 * specialized calibration type (e.g. S7KWaterColumnCalibration) when used through the concrete
 * class. C++ covariant return types do exactly this, but writing the override in every format class
 * is boilerplate. These CRTP bases provide the override once, parametrized on the derived type and
 * its calibration type(s):
 *
 *   - I_PingWatercolumnCalibration<Derived, Cal>: adds the single-sector get_watercolumn_calibration()
 *   - I_PingWatercolumnMultiSectorCalibration<Derived, Cal, MultiCal>: additionally adds the
 *     multi-sector get_multisectorwatercolumn_calibration() and update_calibration()
 *
 * A format derives from the matching base (passing itself as Derived) instead of directly from
 * I_PingWatercolumn; the generic interface keeps returning the generic types, the concrete class
 * returns the specialized ones, and no per-format override has to be written by hand.
 *
 * Requirements on the derived class (all current *PingWatercolumn classes already satisfy them):
 *   - a public `file_data()` accessor returning the format's ping-file-data object, which in turn
 *     provides `init_watercolumn_calibration()`, `get_watercolumn_calibration()` and (for the
 *     multi-sector base) `get_multisector_calibration()` / `update_calibration()`.
 */

#pragma once

/* generated doc strings */
#include ".docstrings/i_pingwatercolumncalibration.doc.hpp"

/* std includes */
#include <memory>
#include <utility>

#include <boost/flyweight.hpp>

#include "i_pingwatercolumn.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

/**
 * @brief CRTP base adding a covariant single-sector water column calibration accessor.
 *
 * @tparam t_derived the concrete ping water column class (must expose a `file_data()` accessor)
 * @tparam t_watercolumncalibration the format's water column calibration type (derived from
 * WaterColumnCalibration)
 */
template<typename t_derived, typename t_watercolumncalibration>
class I_PingWatercolumnCalibration : public I_PingWatercolumn
{
  public:
    explicit I_PingWatercolumnCalibration(std::shared_ptr<PingInfo> ping_info = nullptr)
        : I_PingWatercolumn(std::move(ping_info))
    {
    }

    // keep the generic (sector-indexed) overload reachable next to the specialized no-argument one
    using I_PingWatercolumn::get_watercolumn_calibration;

    /**
     * @brief Water column calibration of this ping.
     *
     * Covariant override: returns the format-specific calibration type through the concrete class
     * and the generic WaterColumnCalibration through the I_PingWatercolumn interface.
     */
    const t_watercolumncalibration& get_watercolumn_calibration() const override
    {
        return calibration_file_data().get_watercolumn_calibration();
    }

  protected:
    /**
     * @brief Lazily initialize the calibration (a cache held by the shared file data) and return the
     * (non-const) file data.
     *
     * The ping water column handle is never truly const - the calibration is a lazily built cache on
     * the shared file-data object - so the const_cast here only re-selects the non-const file_data()
     * accessor, mirroring the pre-CRTP behavior where the lazy build was triggered through the
     * shared_ptr (which propagates const only shallowly).
     */
    auto& calibration_file_data() const
    {
        auto& file_data = const_cast<t_derived*>(static_cast<const t_derived*>(this))->file_data();
        file_data.init_watercolumn_calibration();
        return file_data;
    }
};

/**
 * @brief CRTP base adding covariant single- and multi-sector water column calibration accessors.
 *
 * @tparam t_derived the concrete ping water column class (must expose a `file_data()` accessor)
 * @tparam t_watercolumncalibration the format's water column calibration type
 * @tparam t_multisectorcalibration the format's multi-sector water column calibration type (derived
 * from I_MultiSectorCalibration)
 */
template<typename t_derived, typename t_watercolumncalibration, typename t_multisectorcalibration>
class I_PingWatercolumnMultiSectorCalibration
    : public I_PingWatercolumnCalibration<t_derived, t_watercolumncalibration>
{
    using t_base = I_PingWatercolumnCalibration<t_derived, t_watercolumncalibration>;

  public:
    explicit I_PingWatercolumnMultiSectorCalibration(std::shared_ptr<PingInfo> ping_info = nullptr)
        : t_base(std::move(ping_info))
    {
    }

    /**
     * @brief Multi-sector water column calibration of this ping.
     *
     * Covariant override: returns the format-specific multi-sector calibration type through the
     * concrete class and the generic I_MultiSectorCalibration through the I_PingWatercolumn
     * interface.
     */
    const t_multisectorcalibration& get_multisectorwatercolumn_calibration() const override
    {
        return this->calibration_file_data().get_multisector_calibration();
    }

    /**
     * @brief Replace the ping's calibration with the given (deduplicated) multi-sector calibration.
     */
    void update_calibration(const boost::flyweight<t_multisectorcalibration>& calibration)
    {
        static_cast<t_derived*>(this)->file_data().update_calibration(calibration);
    }
};

} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping
