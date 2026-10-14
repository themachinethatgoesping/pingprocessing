// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <boost/flyweight.hpp>

// xtensor python includes
#include <xtensor-python/nanobind/pytensor.hpp> // Numpy bindings

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/calibration/multisectorwatercolumncalibration.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {
namespace py_calibration {

#define DOC_I_MultiSectorCalibration(ARG)                                                          \
    DOC(themachinethatgoesping,                                                                    \
        pingprocessing,                                                                              \
        datatypes,                                                                                 \
        calibration,                                                                               \
        I_MultiSectorCalibration,                                                                  \
        ARG)

#define DOC_T_MultiSectorCalibration(ARG)                                                          \
    DOC(themachinethatgoesping,                                                                    \
        pingprocessing,                                                                              \
        datatypes,                                                                                 \
        calibration,                                                                               \
        T_MultiSectorCalibration,                                                                  \
        ARG)

template<typename t_calibration, typename t_float, typename t_pyclass>
void add_apply_calibration_functions(t_pyclass& c)
{
    namespace nb = nanobind;
    using namespace themachinethatgoesping::pingprocessing::datatypes::calibration;
    using pytensor1d = xt::nanobind::pytensor<t_float, 1>;
    using pytensor2d = xt::nanobind::pytensor<t_float, 2>;
    using t_calib_type = typename t_calibration::t_calibration_type;
    using t_multi_calib = T_MultiSectorCalibration<t_calibration>;

    // Standard apply_beam_sample_correction methods (absorption from calibration)
    // Using lambda wrappers to disambiguate overloads
    c.def(
        "apply_beam_sample_correction_power",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::power>(
                wci, beam_angles, ranges, beam_numbers_per_tx_sector, mp_cores);
        },
        DOC_I_MultiSectorCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_pp",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::pp>(
                wci, beam_angles, ranges, beam_numbers_per_tx_sector, mp_cores);
        },
        DOC_I_MultiSectorCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_pv",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::pv>(
                wci, beam_angles, ranges, beam_numbers_per_tx_sector, mp_cores);
        },
        DOC_I_MultiSectorCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_ap",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::ap>(
                wci, beam_angles, ranges, beam_numbers_per_tx_sector, mp_cores);
        },
        DOC_I_MultiSectorCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_av",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::av>(
                wci, beam_angles, ranges, beam_numbers_per_tx_sector, mp_cores);
        },
        DOC_I_MultiSectorCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_sp",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::sp>(
                wci, beam_angles, ranges, beam_numbers_per_tx_sector, mp_cores);
        },
        DOC_I_MultiSectorCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_sv",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::sv>(
                wci, beam_angles, ranges, beam_numbers_per_tx_sector, mp_cores);
        },
        DOC_I_MultiSectorCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);

    c.def(
        "inplace_beam_sample_correction_av",
        [](const t_multi_calib&                     self,
           pytensor2d&                              wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            self.template inplace_beam_sample_correction<t_calib_type::av>(
                wci, beam_angles, ranges, beam_numbers_per_tx_sector, mp_cores);
        },
        DOC_I_MultiSectorCalibration(inplace_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);

    // Per-beam absorption apply_beam_sample_correction methods
    c.def(
        "apply_beam_sample_correction_pp_per_beam",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const pytensor1d&                        absorption_db_m_per_beam,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::pp>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, beam_numbers_per_tx_sector, mp_cores);
        },
        "Apply pp calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_pv_per_beam",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const pytensor1d&                        absorption_db_m_per_beam,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::pv>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, beam_numbers_per_tx_sector, mp_cores);
        },
        "Apply pv calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_ap_per_beam",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const pytensor1d&                        absorption_db_m_per_beam,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::ap>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, beam_numbers_per_tx_sector, mp_cores);
        },
        "Apply ap calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_av_per_beam",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const pytensor1d&                        absorption_db_m_per_beam,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::av>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, beam_numbers_per_tx_sector, mp_cores);
        },
        "Apply av calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_sp_per_beam",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const pytensor1d&                        absorption_db_m_per_beam,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::sp>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, beam_numbers_per_tx_sector, mp_cores);
        },
        "Apply sp calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_sv_per_beam",
        [](const t_multi_calib&                     self,
           const pytensor2d&                        wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const pytensor1d&                        absorption_db_m_per_beam,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            return self.template apply_beam_sample_correction<t_calib_type::sv>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, beam_numbers_per_tx_sector, mp_cores);
        },
        "Apply sv calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);

    // Per-beam absorption inplace_beam_sample_correction methods
    c.def(
        "inplace_beam_sample_correction_av_per_beam",
        [](const t_multi_calib&                     self,
           pytensor2d&                              wci,
           const pytensor1d&                        beam_angles,
           const pytensor1d&                        ranges,
           const pytensor1d&                        absorption_db_m_per_beam,
           const std::vector<std::vector<size_t>>&  beam_numbers_per_tx_sector,
           int                                      mp_cores) {
            self.template inplace_beam_sample_correction<t_calib_type::av>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, beam_numbers_per_tx_sector, mp_cores);
        },
        "Inplace apply av calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("beam_numbers_per_tx_sector"),
        nb::arg("mp_cores") = 1);
}

template<typename t_calibration>
void init_multisectorcalibration(nanobind::module_& m, const std::string& name)
{
    namespace nb = nanobind;
    using namespace themachinethatgoesping::pingprocessing::datatypes::calibration;

    // add flyweight class
    nb::class_<boost::flyweight<T_MultiSectorCalibration<t_calibration>>>(
        m,
        (name + "_flyweight").c_str(),
        "Flyweight wrapper around the MultiSectorCalibration class.")
        //.def(nb::init<>(), "Create a flyweight class.")
        .def(nb::init<const T_MultiSectorCalibration<t_calibration>&>(),
             "Create a flyweight from a MultiSectorCalibration object.",
             nb::arg("other"))
        .def(
            "get",
            [](const boost::flyweight<T_MultiSectorCalibration<t_calibration>>& self) {
                return self.get();
            },
            "Get the underlying T_MultiSectorCalibration object.")
        // // default copy functions
        // __PYCLASS_DEFAULT_COPY__(boost::flyweight<T_MultiSectorCalibration<t_calibration>>)
        // // default binary functions
        // __PYCLASS_DEFAULT_BINARY__(boost::flyweight<T_MultiSectorCalibration<t_calibration>>)
        // // default printing functions
        // __PYCLASS_DEFAULT_PRINTING__(boost::flyweight<T_MultiSectorCalibration<t_calibration>>)
        // end flyweight
        ;

    // add class
    auto c = nb::class_<T_MultiSectorCalibration<t_calibration>>(m,
                                                                 name.c_str(),
                                                                 DOC(themachinethatgoesping, pingprocessing, datatypes, calibration,
                                                                     T_MultiSectorCalibration))
                 .def(nb::init<>(), DOC_T_MultiSectorCalibration(T_MultiSectorCalibration))
                 .def(nb::init<std::vector<t_calibration>>(),
                      DOC_T_MultiSectorCalibration(T_MultiSectorCalibration),
                      nb::arg("calibration_per_sector"))
                 .def(nb::init<const T_MultiSectorCalibration<t_calibration>&>(),
                      DOC_T_MultiSectorCalibration(T_MultiSectorCalibration),
                      nb::arg("other"))

                 // --- getters / setters ---
                 .def("has_power_calibration",
                      &T_MultiSectorCalibration<t_calibration>::has_power_calibration,
                      DOC_I_MultiSectorCalibration(has_power_calibration))
                 .def("has_ap_calibration",
                      &T_MultiSectorCalibration<t_calibration>::has_ap_calibration,
                      DOC_I_MultiSectorCalibration(has_ap_calibration))
                 .def("has_av_calibration",
                      &T_MultiSectorCalibration<t_calibration>::has_av_calibration,
                      DOC_I_MultiSectorCalibration(has_av_calibration))
                 .def("has_sp_calibration",
                      &T_MultiSectorCalibration<t_calibration>::has_sp_calibration,
                      DOC_I_MultiSectorCalibration(has_sp_calibration))
                 .def("has_sv_calibration",
                      &T_MultiSectorCalibration<t_calibration>::has_sv_calibration,
                      DOC_I_MultiSectorCalibration(has_sv_calibration))
                 .def("has_valid_absorption_db_m",
                      &T_MultiSectorCalibration<t_calibration>::has_valid_absorption_db_m,
                      DOC_I_MultiSectorCalibration(has_valid_absorption_db_m))

                 .def("get_number_of_sectors",
                      &T_MultiSectorCalibration<t_calibration>::get_number_of_sectors,
                      DOC_I_MultiSectorCalibration(get_number_of_sectors))
                 .def("__len__",
                      &T_MultiSectorCalibration<t_calibration>::get_number_of_sectors,
                      DOC_I_MultiSectorCalibration(get_number_of_sectors))

                 .def("get_calibrations",
                      &T_MultiSectorCalibration<t_calibration>::get_calibrations,
                      DOC_T_MultiSectorCalibration(get_calibrations))

                 .def(
                     "pre_hashed",
                     [](const T_MultiSectorCalibration<t_calibration>& self) {
                         return boost::flyweight<T_MultiSectorCalibration<t_calibration>>(self);
                     },
                     "Return this class as a flyweight with computed hash. "
                     "This faster when updating the ping calibration.")

                 // ----- operators -----
                 .def("__eq__",
                      &T_MultiSectorCalibration<t_calibration>::operator==,
                      DOC_I_MultiSectorCalibration(operator_eq),
                      nb::arg("other"))
             // ----- nanobind macros -----
             // default copy functions
             __PYCLASS_DEFAULT_COPY__(T_MultiSectorCalibration<t_calibration>)
             // default binary functions
             __PYCLASS_DEFAULT_BINARY__(T_MultiSectorCalibration<t_calibration>)
             // default printing functions
             __PYCLASS_DEFAULT_PRINTING__(T_MultiSectorCalibration<t_calibration>)
        // end T_MultiSectorCalibration
        ;

    // add apply calibration functions
    add_apply_calibration_functions<t_calibration, float>(c);
    add_apply_calibration_functions<t_calibration, double>(c);
}

}
}
}
}
}
