// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

// xtensor python includes
//#include <xtensor-python/nanobind/pytensor.hpp> // Numpy bindings
#include <xtensor-python/nanobind/pytensor.hpp> // Numpy bindings

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>
#include <themachinethatgoesping/tools_nanobind/enumhelper.hpp>


#include <themachinethatgoesping/pingprocessing/datatypes/calibration/watercolumncalibration.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {
namespace py_calibration {

namespace nb = nanobind;

#define DOC_WaterColumnCalibration(ARG)                                                            \
    DOC(themachinethatgoesping,                                                                    \
        pingprocessing,                                                                              \
        datatypes,                                                                                 \
        calibration,                                                                               \
        WaterColumnCalibration,                                                                    \
        ARG)

#define DOC_t_calibration_type(ARG)                                                                \
    DOC(themachinethatgoesping,                                                                    \
        pingprocessing,                                                                              \
        datatypes,                                                                                 \
        calibration,                                                                               \
        t_calibration_type,                                                                        \
        ARG)

template<typename t_float, typename t_pyclass>
void add_apply_calibration_functions(t_pyclass& c)
{
    using namespace themachinethatgoesping::pingprocessing::datatypes::calibration;
    using xt::nanobind::pytensor;
    using pytensor1d = pytensor<t_float, 1>;
    using pytensor2d = pytensor<t_float, 2>;

    // Standard apply_beam_sample_correction methods (absorption from calibration)
    // Using lambda wrappers to disambiguate overloads
    c.def(
        "apply_beam_sample_correction_power",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::power>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_rp",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::rp>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_rv",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::rv>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_pp",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::pp>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_pv",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::pv>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_ap",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::ap>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_av",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::av>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_sv",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::sv>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_sp",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::sp>(
                wci, beam_angles, ranges, mp_cores);
        },
        DOC_WaterColumnCalibration(apply_beam_sample_correction),
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("mp_cores") = 1);

    // Per-beam absorption apply_beam_sample_correction methods
    // (for calibration types that apply absorption: pp, pv, ap, av, sp, sv)
    c.def(
        "apply_beam_sample_correction_pp_per_beam",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           const pytensor1d&             absorption_db_m_per_beam,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::pp>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, mp_cores);
        },
        "Apply pp calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_pv_per_beam",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           const pytensor1d&             absorption_db_m_per_beam,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::pv>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, mp_cores);
        },
        "Apply pv calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_ap_per_beam",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           const pytensor1d&             absorption_db_m_per_beam,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::ap>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, mp_cores);
        },
        "Apply ap calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_av_per_beam",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           const pytensor1d&             absorption_db_m_per_beam,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::av>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, mp_cores);
        },
        "Apply av calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_sp_per_beam",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           const pytensor1d&             absorption_db_m_per_beam,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::sp>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, mp_cores);
        },
        "Apply sp calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("mp_cores") = 1);
    c.def(
        "apply_beam_sample_correction_sv_per_beam",
        [](const WaterColumnCalibration& self,
           const pytensor2d&             wci,
           const pytensor1d&             beam_angles,
           const pytensor1d&             ranges,
           const pytensor1d&             absorption_db_m_per_beam,
           int                           mp_cores) {
            return self.template apply_beam_sample_correction<t_calibration_type::sv>(
                wci, beam_angles, ranges, absorption_db_m_per_beam, mp_cores);
        },
        "Apply sv calibration with per-beam absorption coefficients.",
        nb::arg("wci"),
        nb::arg("beam_angles"),
        nb::arg("ranges"),
        nb::arg("absorption_db_m_per_beam"),
        nb::arg("mp_cores") = 1);
}

void init_c_watercolumncalibration(nanobind::module_& m)
{
    using namespace themachinethatgoesping::pingprocessing::datatypes::calibration;

    // add enum
    nb::enum_<t_calibration_type>(
        m,
        "t_calibration_type",
        DOC(themachinethatgoesping, pingprocessing, datatypes, calibration,
            t_calibration_type))
        .value("power", t_calibration_type::power, DOC_t_calibration_type(power))
        .value("rp", t_calibration_type::rp, DOC_t_calibration_type(rp))
        .value("rv", t_calibration_type::rv, DOC_t_calibration_type(rv))
        .value("pp", t_calibration_type::pp, DOC_t_calibration_type(pp))
        .value("pv", t_calibration_type::pv, DOC_t_calibration_type(pv))
        .value("ap", t_calibration_type::ap, DOC_t_calibration_type(ap))
        .value("av", t_calibration_type::av, DOC_t_calibration_type(av))
        .value("sp", t_calibration_type::sp, DOC_t_calibration_type(sp))
        .value("sv", t_calibration_type::sv, DOC_t_calibration_type(sv));

    // add option class
    themachinethatgoesping::tools::nanobind_helper::make_option_class<o_calibration_type>(
        m, "o_calibration_type");

    // add class
    auto c = nb::class_<WaterColumnCalibration>(
                 m,
                 "WaterColumnCalibration",
                 DOC(themachinethatgoesping, pingprocessing, datatypes, calibration,
                     WaterColumnCalibration))
                 .def(nb::init<float, float>(),
                      DOC_WaterColumnCalibration(WaterColumnCalibration),
                      nb::arg("tvg_absorption_db_m") = 0.0f,
                      nb::arg("tvg_factor")          = 0.0f)
                 .def(nb::init<const AmplitudeCalibration&,
                               const AmplitudeCalibration&,
                               const AmplitudeCalibration&,
                               float,
                               float>(),
                      DOC_WaterColumnCalibration(WaterColumnCalibration),
                      nb::arg("power_calibration")   = AmplitudeCalibration(),
                      nb::arg("ap_calibration")      = AmplitudeCalibration(),
                      nb::arg("av_calibration")      = AmplitudeCalibration(),
                      nb::arg("tvg_absorption_db_m") = 0.0f,
                      nb::arg("tvg_factor")          = 0.0f)

                 // --- getters / setters ---
                 .def("set_absorption_db_m",
                      &WaterColumnCalibration::set_absorption_db_m,
                      DOC_WaterColumnCalibration(set_absorption_db_m),
                      nb::arg("absorption_db_m"))
                 .def("get_absorption_db_m",
                      &WaterColumnCalibration::get_absorption_db_m,
                      DOC_WaterColumnCalibration(get_absorption_db_m))
                 .def("has_valid_absorption_db_m",
                      &WaterColumnCalibration::has_valid_absorption_db_m,
                      DOC_WaterColumnCalibration(has_valid_absorption_db_m))
                 .def("get_tvg_absorption_db_m",
                      &WaterColumnCalibration::get_tvg_absorption_db_m,
                      DOC_WaterColumnCalibration(get_tvg_absorption_db_m))
                 .def("get_tvg_factor",
                      &WaterColumnCalibration::get_tvg_factor,
                      DOC_WaterColumnCalibration(get_tvg_factor))

                 .def("get_absorption_to_apply",
                      &WaterColumnCalibration::get_absorption_to_apply,
                      DOC_WaterColumnCalibration(get_absorption_to_apply),
                      nb::arg("absorption_db_m") = std::nullopt)
                 .def("get_tvg_factor_to_apply",
                      &WaterColumnCalibration::get_tvg_factor_to_apply,
                      DOC_WaterColumnCalibration(get_tvg_factor_to_apply),
                      nb::arg("tvg_factor"))

                 // --- convenient data access ---
                 .def("has_power_calibration",
                      &WaterColumnCalibration::has_power_calibration,
                      DOC_WaterColumnCalibration(has_power_calibration))
                 .def("has_ap_calibration",
                      &WaterColumnCalibration::has_ap_calibration,
                      DOC_WaterColumnCalibration(has_ap_calibration))
                 .def("has_av_calibration",
                      &WaterColumnCalibration::has_av_calibration,
                      DOC_WaterColumnCalibration(has_av_calibration))
                 .def("has_sp_calibration",
                      &WaterColumnCalibration::has_sp_calibration,
                      DOC_WaterColumnCalibration(has_sp_calibration))
                 .def("has_sv_calibration",
                      &WaterColumnCalibration::has_sv_calibration,
                      DOC_WaterColumnCalibration(has_sv_calibration))
                 .def("get_power_calibration",
                      &WaterColumnCalibration::get_power_calibration,
                      DOC_WaterColumnCalibration(get_power_calibration))
                 .def("get_ap_calibration",
                      &WaterColumnCalibration::get_ap_calibration,
                      DOC_WaterColumnCalibration(get_ap_calibration))
                 .def("get_av_calibration",
                      &WaterColumnCalibration::get_av_calibration,
                      DOC_WaterColumnCalibration(get_av_calibration))
                 .def("get_sp_calibration",
                      &WaterColumnCalibration::get_sp_calibration,
                      DOC_WaterColumnCalibration(get_sp_calibration))
                 .def("get_sv_calibration",
                      &WaterColumnCalibration::get_sv_calibration,
                      DOC_WaterColumnCalibration(get_sv_calibration))
                 .def("set_power_calibration",
                      &WaterColumnCalibration::set_power_calibration,
                      DOC_WaterColumnCalibration(set_power_calibration),
                      nb::arg("calibration"))
                 .def("set_ap_calibration",
                      &WaterColumnCalibration::set_ap_calibration,
                      DOC_WaterColumnCalibration(set_ap_calibration),
                      nb::arg("calibration"))
                 .def("set_av_calibration",
                      &WaterColumnCalibration::set_av_calibration,
                      DOC_WaterColumnCalibration(set_av_calibration),
                      nb::arg("calibration"))
                 .def("set_sp_calibration",
                      &WaterColumnCalibration::set_sp_calibration,
                      DOC_WaterColumnCalibration(set_sp_calibration),
                      nb::arg("calibration"))
                 .def("set_sv_calibration",
                      &WaterColumnCalibration::set_sv_calibration,
                      DOC_WaterColumnCalibration(set_sv_calibration),
                      nb::arg("calibration"))

                 .def("check_initialized",
                      &WaterColumnCalibration::check_initialized,
                      DOC_WaterColumnCalibration(check_initialized))
                 .def("check_modifying_base_calibration_allowed",
                      &WaterColumnCalibration::check_modifying_base_calibration_allowed,
                      DOC_WaterColumnCalibration(check_modifying_base_calibration_allowed))

                 // ----- operators -----
                 .def("__eq__",
                      &WaterColumnCalibration::operator==,
                      DOC_WaterColumnCalibration(operator_eq),
                      nb::arg("other"))
             // ----- nanobind macros -----
             // default copy functions
             __PYCLASS_DEFAULT_COPY__(WaterColumnCalibration)
             // default binary functions
             __PYCLASS_DEFAULT_BINARY__(WaterColumnCalibration)
             // default printing functions
             __PYCLASS_DEFAULT_PRINTING__(WaterColumnCalibration)
        // end WaterColumnCalibration
        ;

    add_apply_calibration_functions<float>(c);
    add_apply_calibration_functions<double>(c);
}
}
}
}
}
}
