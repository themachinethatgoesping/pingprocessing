// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>
#include <xtensor-python/nanobind/pytensor.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/i_pingwatercolumn.hpp>
#include "c_i_mbesrawdetectioninfos.hpp"
#include "c_i_pingcommon.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {
namespace py_i_pingwatercolumn {

#define DOC_I_PingWatercolumn(ARG)                                                                 \
    DOC(themachinethatgoesping, pingprocessing, datatypes, I_PingWatercolumn, ARG)

namespace pingtools   = themachinethatgoesping::algorithms::pingtools;
namespace calibration = themachinethatgoesping::pingprocessing::datatypes::calibration;

template<typename T_BaseClass, typename T_PyClass>
void I_PingWatercolumn_add_interface(T_PyClass& cls)
{
    namespace nb = nanobind;

    py_i_pingcommon::I_PingCommon_add_interface<T_BaseClass>(cls);

    // I_MBESRawDetectionInfos interface (nanobind has no multiple inheritance)
    py_i_mbesrawdetectioninfos::add_interface<T_BaseClass>(cls);

    cls.def("has_tx_signal_parameters_legacy",
            &T_BaseClass::has_tx_signal_parameters_legacy,
            DOC_I_PingWatercolumn(has_tx_signal_parameters_legacy));
    cls.def("has_number_of_tx_sectors_legacy",
            &T_BaseClass::has_number_of_tx_sectors_legacy,
            DOC_I_PingWatercolumn(has_number_of_tx_sectors_legacy));
    cls.def("has_beam_numbers_per_tx_sector_legacy",
            &T_BaseClass::has_beam_numbers_per_tx_sector_legacy,
            DOC_I_PingWatercolumn(has_beam_numbers_per_tx_sector_legacy));
    cls.def("has_beam_selection_all",
            &T_BaseClass::has_beam_selection_all,
            DOC_I_PingWatercolumn(has_beam_selection_all));
    cls.def("has_number_of_beams_legacy",
            &T_BaseClass::has_number_of_beams_legacy,
            DOC_I_PingWatercolumn(has_number_of_beams_legacy));
    cls.def("has_tx_sector_per_beam_legacy",
            &T_BaseClass::has_tx_sector_per_beam_legacy,
            DOC_I_PingWatercolumn(has_tx_sector_per_beam_legacy));

    cls.def("get_tx_signal_parameters_legacy",
            &T_BaseClass::get_tx_signal_parameters_legacy,
            DOC_I_PingWatercolumn(get_tx_signal_parameters_legacy));
    cls.def("get_number_of_tx_sectors_legacy",
            &T_BaseClass::get_number_of_tx_sectors_legacy,
            DOC_I_PingWatercolumn(get_number_of_tx_sectors_legacy));

    cls.def("get_watercolumn_calibration",
            nb::overload_cast<>(&T_BaseClass::get_watercolumn_calibration, nb::const_),
            DOC_I_PingWatercolumn(get_watercolumn_calibration),
            nb::rv_policy::reference_internal);
    cls.def("get_watercolumn_calibration",
            nb::overload_cast<size_t>(&T_BaseClass::get_watercolumn_calibration, nb::const_),
            DOC_I_PingWatercolumn(get_watercolumn_calibration_2),
            nb::rv_policy::reference_internal,
            nb::arg("sector_nr"));
    cls.def("get_multisectorwatercolumn_calibration",
            &T_BaseClass::get_multisectorwatercolumn_calibration,
            DOC_I_PingWatercolumn(get_multisectorwatercolumn_calibration),
            nb::rv_policy::reference_internal);

    cls.def("get_beam_selection_all",
            nb::overload_cast<>(&T_BaseClass::get_beam_selection_all),
            DOC_I_PingWatercolumn(get_beam_selection_all));
    cls.def("get_number_of_beams_legacy",
            &T_BaseClass::get_number_of_beams_legacy,
            DOC_I_PingWatercolumn(get_number_of_beams_legacy));
    cls.def("get_beam_crosstrack_angles_legacy",
            nb::overload_cast<>(&T_BaseClass::get_beam_crosstrack_angles_legacy),
            DOC_I_PingWatercolumn(get_beam_crosstrack_angles_legacy));
    cls.def("get_beam_crosstrack_angles_legacy",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(
                &T_BaseClass::get_beam_crosstrack_angles_legacy),
            DOC_I_PingWatercolumn(get_beam_crosstrack_angles_legacy),
            nb::arg("beam_selection"));
    cls.def("has_beam_crosstrack_angles_in_world_frame",
            &T_BaseClass::has_beam_crosstrack_angles_in_world_frame,
            DOC_I_PingWatercolumn(has_beam_crosstrack_angles_in_world_frame));
    cls.def("get_beam_alongtrack_angles",
            nb::overload_cast<>(&T_BaseClass::get_beam_alongtrack_angles),
            DOC_I_PingWatercolumn(get_beam_alongtrack_angles));
    cls.def("get_beam_alongtrack_angles",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(
                &T_BaseClass::get_beam_alongtrack_angles),
            DOC_I_PingWatercolumn(get_beam_alongtrack_angles),
            nb::arg("beam_selection"));

    cls.def("get_beam_sample_selection_all",
            nb::overload_cast<>(&T_BaseClass::get_beam_sample_selection_all),
            DOC_I_PingWatercolumn(get_beam_sample_selection_all),
            nb::rv_policy::reference_internal);
    cls.def("get_first_sample_offset_per_beam",
            &T_BaseClass::get_first_sample_offset_per_beam,
            DOC_I_PingWatercolumn(get_first_sample_offset_per_beam));
    cls.def("get_number_of_samples_per_beam",
            nb::overload_cast<>(&T_BaseClass::get_number_of_samples_per_beam),
            DOC_I_PingWatercolumn(get_number_of_samples_per_beam));
    cls.def("get_number_of_samples_per_beam",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(
                &T_BaseClass::get_number_of_samples_per_beam),
            DOC_I_PingWatercolumn(get_number_of_samples_per_beam_2),
            nb::arg("beam_selection"));

    cls.def("get_tx_sector_per_beam_legacy",
            nb::overload_cast<>(&T_BaseClass::get_tx_sector_per_beam_legacy),
            DOC_I_PingWatercolumn(get_tx_sector_per_beam_legacy));
    cls.def(
        "get_tx_sector_per_beam_legacy",
        nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_tx_sector_per_beam_legacy),
        DOC_I_PingWatercolumn(get_tx_sector_per_beam_legacy_2),
        nb::arg("beam_selection"));
    cls.def("get_beam_numbers_per_tx_sector_legacy",
            nb::overload_cast<>(&T_BaseClass::get_beam_numbers_per_tx_sector_legacy),
            DOC_I_PingWatercolumn(get_beam_numbers_per_tx_sector_legacy));
    cls.def("get_beam_numbers_per_tx_sector_legacy",
            nb::overload_cast<const algorithms::pingtools::BeamSelection&>(
                &T_BaseClass::get_beam_numbers_per_tx_sector_legacy),
            DOC_I_PingWatercolumn(get_beam_numbers_per_tx_sector_legacy),
            nb::arg("beam_selection"));

    cls.def("get_approximate_ranges",
            nb::overload_cast<>(&T_BaseClass::get_approximate_ranges),
            DOC_I_PingWatercolumn(get_approximate_ranges));
    cls.def("get_approximate_ranges",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&>(
                &T_BaseClass::get_approximate_ranges),
            DOC_I_PingWatercolumn(get_approximate_ranges),
            nb::arg("beam_sample_selection"));

    cls.def("get_sound_speed_at_transducer_legacy",
            &T_BaseClass::get_sound_speed_at_transducer_legacy,
            DOC_I_PingWatercolumn(get_sound_speed_at_transducer_legacy));
    cls.def("get_sample_interval",
            &T_BaseClass::get_sample_interval,
            DOC_I_PingWatercolumn(get_sample_interval));

    cls.def("has_amplitudes",
            nb::overload_cast<>(&T_BaseClass::has_amplitudes, nb::const_),
            DOC_I_PingWatercolumn(has_amplitudes));
    cls.def("has_ap",
            nb::overload_cast<>(&T_BaseClass::has_ap, nb::const_),
            DOC_I_PingWatercolumn(has_ap));
    cls.def("has_av",
            nb::overload_cast<>(&T_BaseClass::has_av, nb::const_),
            DOC_I_PingWatercolumn(has_av));
    cls.def("has_power",
            nb::overload_cast<>(&T_BaseClass::has_power, nb::const_),
            DOC_I_PingWatercolumn(has_power));
    cls.def("has_rp",
            nb::overload_cast<>(&T_BaseClass::has_rp, nb::const_),
            DOC_I_PingWatercolumn(has_rp));
    cls.def("has_rv",
            nb::overload_cast<>(&T_BaseClass::has_rv, nb::const_),
            DOC_I_PingWatercolumn(has_rv));
    cls.def("has_pp",
            nb::overload_cast<>(&T_BaseClass::has_pp, nb::const_),
            DOC_I_PingWatercolumn(has_pp));
    cls.def("has_pv",
            nb::overload_cast<>(&T_BaseClass::has_pv, nb::const_),
            DOC_I_PingWatercolumn(has_pv));
    cls.def("has_sp",
            nb::overload_cast<>(&T_BaseClass::has_sp, nb::const_),
            DOC_I_PingWatercolumn(has_sp));
    cls.def("has_sv",
            nb::overload_cast<>(&T_BaseClass::has_sv, nb::const_),
            DOC_I_PingWatercolumn(has_sv));
    cls.def("has_watercolumn_calibration",
            nb::overload_cast<>(&T_BaseClass::has_watercolumn_calibration, nb::const_),
            DOC_I_PingWatercolumn(has_watercolumn_calibration));
    cls.def("has_multisectorwatercolumn_calibration",
            nb::overload_cast<>(&T_BaseClass::has_multisectorwatercolumn_calibration, nb::const_),
            DOC_I_PingWatercolumn(has_multisectorwatercolumn_calibration));

    cls.def("get_amplitudes",
            nb::overload_cast<int>(&T_BaseClass::get_amplitudes),
            DOC_I_PingWatercolumn(get_amplitudes),
            nb::arg("mp_cores") = 1);
    cls.def(
        "get_amplitudes",
        nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_amplitudes),
        DOC_I_PingWatercolumn(get_amplitudes),
        nb::arg("beam_selection"),
        nb::arg("mp_cores") = 1);

    cls.def("get_rp",
            nb::overload_cast<int>(&T_BaseClass::get_rp),
            DOC_I_PingWatercolumn(get_rp),
            nb::arg("mp_cores") = 1);
    cls.def("get_rp",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_rp),
            DOC_I_PingWatercolumn(get_rp),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("get_rv",
            nb::overload_cast<int>(&T_BaseClass::get_rv),
            DOC_I_PingWatercolumn(get_rv),
            nb::arg("mp_cores") = 1);
    cls.def("get_rv",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_rv),
            DOC_I_PingWatercolumn(get_rv),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("get_pp",
            nb::overload_cast<int>(&T_BaseClass::get_pp),
            DOC_I_PingWatercolumn(get_pp),
            nb::arg("mp_cores") = 1);
    cls.def("get_pp",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_pp),
            DOC_I_PingWatercolumn(get_pp),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("get_pv",
            nb::overload_cast<int>(&T_BaseClass::get_pv),
            DOC_I_PingWatercolumn(get_pv),
            nb::arg("mp_cores") = 1);
    cls.def("get_pv",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_pv),
            DOC_I_PingWatercolumn(get_pv),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("get_ap",
            nb::overload_cast<int>(&T_BaseClass::get_ap),
            DOC_I_PingWatercolumn(get_ap),
            nb::arg("mp_cores") = 1);
    cls.def("get_ap",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_ap),
            DOC_I_PingWatercolumn(get_ap),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("get_av",
            nb::overload_cast<int>(&T_BaseClass::get_av),
            DOC_I_PingWatercolumn(get_av),
            nb::arg("mp_cores") = 1);
    cls.def("get_av",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_av),
            DOC_I_PingWatercolumn(get_av),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("get_power",
            nb::overload_cast<int>(&T_BaseClass::get_power),
            DOC_I_PingWatercolumn(get_power),
            nb::arg("mp_cores") = 1);
    cls.def("get_power",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_power),
            DOC_I_PingWatercolumn(get_power),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("get_sp",
            nb::overload_cast<int>(&T_BaseClass::get_sp),
            DOC_I_PingWatercolumn(get_sp),
            nb::arg("mp_cores") = 1);
    cls.def("get_sp",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_sp),
            DOC_I_PingWatercolumn(get_sp),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("get_sv",
            nb::overload_cast<int>(&T_BaseClass::get_sv),
            DOC_I_PingWatercolumn(get_sv),
            nb::arg("mp_cores") = 1);
    cls.def("get_sv",
            nb::overload_cast<const algorithms::pingtools::BeamSampleSelection&, int>(&T_BaseClass::get_sv),
            DOC_I_PingWatercolumn(get_sv),
            nb::arg("beam_selection"),
            nb::arg("mp_cores") = 1);

    cls.def("has_bottom_range_samples",
            nb::overload_cast<>(&T_BaseClass::has_bottom_range_samples, nb::const_),
            DOC_I_PingWatercolumn(has_bottom_range_samples));
    cls.def("get_bottom_range_samples",
            nb::overload_cast<>(&T_BaseClass::get_bottom_range_samples),
            DOC_I_PingWatercolumn(get_bottom_range_samples));
    cls.def(
        "get_bottom_range_samples",
        nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_bottom_range_samples),
        DOC_I_PingWatercolumn(get_bottom_range_samples),
        nb::arg("beam_selection"));

    cls.def("get_minslant_sample_nr",
            nb::overload_cast<>(&T_BaseClass::get_minslant_sample_nr),
            DOC_I_PingWatercolumn(get_minslant_sample_nr));
    cls.def(
        "get_minslant_sample_nr",
        nb::overload_cast<const algorithms::pingtools::BeamSelection&>(&T_BaseClass::get_minslant_sample_nr),
        DOC_I_PingWatercolumn(get_minslant_sample_nr),
        nb::arg("beam_selection"));

    cls.def_prop_rw("_test_mode",
                    &T_BaseClass::get_test_mode,
                    &T_BaseClass::set_test_mode,
                    DOC_I_PingWatercolumn(test_mode));

    // --- get_wci_correction functions ---
    using calibration::o_calibration_type;
    using calibration::t_calibration_type;
    cls.def(
        "get_wci_correction",
        [](T_BaseClass&             self,
           const o_calibration_type calibration_type,
           const o_calibration_type calibration_base,
           int                      mp_cores) {
            return self.get_wci_correction(calibration_type, calibration_base, mp_cores);
        },
        DOC_I_PingWatercolumn(get_wci_correction),
        nb::arg("calibration_type"),
        nb::arg("calibration_base") = t_calibration_type::power,
        nb::arg("mp_cores")         = 1);
    cls.def(
        "get_wci_correction",
        [](T_BaseClass&                          self,
           const algorithms::pingtools::BeamSampleSelection& beam_sample_selection,
           const o_calibration_type              calibration_type,
           const o_calibration_type              calibration_base,
           int                                   mp_cores) {
            return self.get_wci_correction(
                beam_sample_selection, calibration_type, calibration_base, mp_cores);
        },
        DOC_I_PingWatercolumn(get_wci_correction),
        nb::arg("beam_sample_selection"),
        nb::arg("calibration_type"),
        nb::arg("calibration_base") = t_calibration_type::power,
        nb::arg("mp_cores")         = 1);

    // --- get_wci functions ---
    cls.def(
        "get_wci",
        [](T_BaseClass& self, const o_calibration_type calibration_type, int mp_cores) {
            return self.get_wci(calibration_type, mp_cores);
        },
        DOC_I_PingWatercolumn(get_wci),
        nb::arg("calibration_type"),
        nb::arg("mp_cores") = 1);
    cls.def(
        "get_wci",
        [](T_BaseClass&                          self,
           const algorithms::pingtools::BeamSampleSelection& beam_sample_selection,
           const o_calibration_type              calibration_type,
           int                                   mp_cores) {
            return self.get_wci(beam_sample_selection, calibration_type, mp_cores);
        },
        DOC_I_PingWatercolumn(get_wci),
        nb::arg("beam_sample_selection"),
        nb::arg("calibration_type"),
        nb::arg("mp_cores") = 1);
}

}
}
}
}
}
