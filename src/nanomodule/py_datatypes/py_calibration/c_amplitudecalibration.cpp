// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

// xtensor python includes
#include <xtensor-python/nanobind/pytensor.hpp> // Numpy bindings

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/calibration/amplitudecalibration.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {
namespace py_calibration {

namespace nb = nanobind;

#define DOC_AmplitudeCalibration(ARG)                                                              \
    DOC(themachinethatgoesping,                                                                    \
        pingprocessing,                                                                              \
        datatypes,                                                                                 \
        calibration,                                                                               \
        AmplitudeCalibration,                                                                      \
        ARG)

void init_c_amplitudecalibration(nanobind::module_& m)
{
    using namespace themachinethatgoesping::pingprocessing::datatypes::calibration;
    using BiVectorOffset = tools::vectorinterpolators::BiVectorInterpolator<
        tools::vectorinterpolators::AkimaInterpolator<float>>;

    // add class
    auto c =
        nb::class_<AmplitudeCalibration>(m,
                                         "AmplitudeCalibration",
                                         DOC(themachinethatgoesping, pingprocessing, datatypes, calibration,
                                             AmplitudeCalibration))
            .def(nb::init<>(), DOC_AmplitudeCalibration(AmplitudeCalibration))
            .def(nb::init<float>(),
                 DOC_AmplitudeCalibration(AmplitudeCalibration),
                 nb::arg("system_offset"))
            .def(nb::init<const AmplitudeCalibration&>(),
                 DOC_AmplitudeCalibration(AmplitudeCalibration))

            // --- get offsets ---
            .def("get_per_beam_offsets",
                 &AmplitudeCalibration::get_per_beam_offsets<xt::nanobind::pytensor<float, 1>>,
                 DOC_AmplitudeCalibration(get_per_beam_offsets),
                 nb::arg("beamangles"))
            .def("get_per_sample_offsets",
                 &AmplitudeCalibration::get_per_sample_offsets<xt::nanobind::pytensor<float, 1>>,
                 DOC_AmplitudeCalibration(get_per_sample_offsets),
                 nb::arg("ranges"))

            // --- apply_beam_sample_correction with scalar absorption ---
            .def(
                "apply_beam_sample_correction",
                [](const AmplitudeCalibration&              self,
                   const xt::nanobind::pytensor<float, 2>&  wci,
                   const xt::nanobind::pytensor<float, 1>&  beam_angles,
                   const xt::nanobind::pytensor<float, 1>&  ranges,
                   std::optional<float>                     absorption_db_m,
                   std::optional<float>                     tvg_factor,
                   int                                      mp_cores) {
                    return self.apply_beam_sample_correction<xt::nanobind::pytensor<float, 2>,
                                                             xt::nanobind::pytensor<float, 1>>(
                        wci, beam_angles, ranges, absorption_db_m, tvg_factor, mp_cores);
                },
                DOC_AmplitudeCalibration(apply_beam_sample_correction),
                nb::arg("wci"),
                nb::arg("beam_angles"),
                nb::arg("ranges"),
                nb::arg("absorption_db_m"),
                nb::arg("tvg_factor"),
                nb::arg("mp_cores") = 1)

            // --- apply_beam_sample_correction with per-beam absorption ---
            .def(
                "apply_beam_sample_correction_per_beam",
                [](const AmplitudeCalibration&              self,
                   const xt::nanobind::pytensor<float, 2>&  wci,
                   const xt::nanobind::pytensor<float, 1>&  beam_angles,
                   const xt::nanobind::pytensor<float, 1>&  ranges,
                   const xt::nanobind::pytensor<float, 1>&  absorption_db_m_per_beam,
                   std::optional<float>                     tvg_factor,
                   int                                      mp_cores) {
                    return self.apply_beam_sample_correction<xt::nanobind::pytensor<float, 2>,
                                                             xt::nanobind::pytensor<float, 1>>(
                        wci,
                        beam_angles,
                        ranges,
                        absorption_db_m_per_beam,
                        tvg_factor,
                        mp_cores);
                },
                DOC_AmplitudeCalibration(apply_beam_sample_correction_2),
                nb::arg("wci"),
                nb::arg("beam_angles"),
                nb::arg("ranges"),
                nb::arg("absorption_db_m_per_beam"),
                nb::arg("tvg_factor"),
                nb::arg("mp_cores") = 1)

            // --- inplace_beam_sample_correction with scalar absorption ---
            .def(
                "inplace_beam_sample_correction",
                [](const AmplitudeCalibration&             self,
                   xt::nanobind::pytensor<float, 2>&       wci,
                   const xt::nanobind::pytensor<float, 1>& beam_angles,
                   const xt::nanobind::pytensor<float, 1>& ranges,
                   std::optional<float>                    absorption_db_m,
                   std::optional<float>                    tvg_factor,
                   std::optional<size_t>                   min_beam_index,
                   std::optional<size_t>                   max_beam_index,
                   int                                     mp_cores) {
                    self.inplace_beam_sample_correction<xt::nanobind::pytensor<float, 2>,
                                                        xt::nanobind::pytensor<float, 1>>(
                        wci,
                        beam_angles,
                        ranges,
                        absorption_db_m,
                        tvg_factor,
                        min_beam_index,
                        max_beam_index,
                        mp_cores);
                },
                DOC_AmplitudeCalibration(inplace_beam_sample_correction),
                nb::arg("wci"),
                nb::arg("beam_angles"),
                nb::arg("ranges"),
                nb::arg("absorption_db_m"),
                nb::arg("tvg_factor"),
                nb::arg("min_beam_index") = std::nullopt,
                nb::arg("max_beam_index") = std::nullopt,
                nb::arg("mp_cores")       = 1)

            // --- inplace_beam_sample_correction with per-beam absorption ---
            .def(
                "inplace_beam_sample_correction_per_beam",
                [](const AmplitudeCalibration&             self,
                   xt::nanobind::pytensor<float, 2>&       wci,
                   const xt::nanobind::pytensor<float, 1>& beam_angles,
                   const xt::nanobind::pytensor<float, 1>& ranges,
                   const xt::nanobind::pytensor<float, 1>& absorption_db_m_per_beam,
                   std::optional<float>                    tvg_factor,
                   std::optional<size_t>                   min_beam_index,
                   std::optional<size_t>                   max_beam_index,
                   int                                     mp_cores) {
                    self.inplace_beam_sample_correction<xt::nanobind::pytensor<float, 2>,
                                                        xt::nanobind::pytensor<float, 1>>(
                        wci,
                        beam_angles,
                        ranges,
                        absorption_db_m_per_beam,
                        tvg_factor,
                        min_beam_index,
                        max_beam_index,
                        mp_cores);
                },
                DOC_AmplitudeCalibration(inplace_beam_sample_correction_2),
                nb::arg("wci"),
                nb::arg("beam_angles"),
                nb::arg("ranges"),
                nb::arg("absorption_db_m_per_beam"),
                nb::arg("tvg_factor"),
                nb::arg("min_beam_index") = std::nullopt,
                nb::arg("max_beam_index") = std::nullopt,
                nb::arg("mp_cores")       = 1)

            // --- convenient data access ---
            .def("get_system_offset",
                 &AmplitudeCalibration::get_system_offset,
                 DOC_AmplitudeCalibration(get_system_offset))
            .def("has_system_offset",
                 &AmplitudeCalibration::has_system_offset,
                 DOC_AmplitudeCalibration(has_system_offset))
            .def("set_system_offset",
                 &AmplitudeCalibration::set_system_offset,
                 DOC_AmplitudeCalibration(set_system_offset),
                 nb::arg("value"))
            .def("set_offset_per_beamangle",
                 nb::overload_cast<const xt::nanobind::pytensor<float, 1>&,
                                   const xt::nanobind::pytensor<float, 1>&>(
                     &AmplitudeCalibration::set_offset_per_beamangle<
                         const xt::nanobind::pytensor<float, 1>>),
                 DOC_AmplitudeCalibration(set_offset_per_beamangle),
                 nb::arg("beamangle"),
                 nb::arg("offset"))
            .def("has_offset_per_beamangle",
                 &AmplitudeCalibration::has_offset_per_beamangle,
                 DOC_AmplitudeCalibration(has_offset_per_beamangle))
            .def("set_offset_per_range",
                 nb::overload_cast<const xt::nanobind::pytensor<float, 1>&,
                                   const xt::nanobind::pytensor<float, 1>&>(
                     &AmplitudeCalibration::set_offset_per_range<
                         const xt::nanobind::pytensor<float, 1>>),
                 DOC_AmplitudeCalibration(set_offset_per_range),
                 nb::arg("range"),
                 nb::arg("offset"))
            .def("has_offset_per_range",
                 &AmplitudeCalibration::has_offset_per_range,
                 DOC_AmplitudeCalibration(has_offset_per_range))
            .def("set_offset_per_beamangle_and_range",
                 nb::overload_cast<const xt::nanobind::pytensor<float, 1>&,
                                   const xt::nanobind::pytensor<float, 1>&,
                                   const xt::nanobind::pytensor<float, 2>&>(
                     &AmplitudeCalibration::template set_offset_per_beamangle_and_range<
                         xt::nanobind::pytensor<float, 1>,
                         xt::nanobind::pytensor<float, 2>>),
                 DOC_AmplitudeCalibration(set_offset_per_beamangle_and_range),
                 nb::arg("beamangles"),
                 nb::arg("ranges"),
                 nb::arg("values"))
            .def("set_offset_per_beamangle_and_range",
                 static_cast<void (AmplitudeCalibration::*)(const BiVectorOffset&)>(
                     &AmplitudeCalibration::set_offset_per_beamangle_and_range),
                 DOC_AmplitudeCalibration(set_offset_per_beamangle_and_range),
                 nb::arg("offset_per_beamangle_and_range"))
            .def("has_offset_per_beamangle_and_range",
                 &AmplitudeCalibration::has_offset_per_beamangle_and_range,
                 DOC_AmplitudeCalibration(has_offset_per_beamangle_and_range))

            .def("get_interpolator_offset_per_beamangle",
                 &AmplitudeCalibration::get_interpolator_offset_per_beamangle,
                 DOC_AmplitudeCalibration(get_interpolator_offset_per_beamangle))
            .def("get_interpolator_offset_per_range",
                 &AmplitudeCalibration::get_interpolator_offset_per_range,
                 DOC_AmplitudeCalibration(get_interpolator_offset_per_range))
            .def("get_interpolator_offset_per_beamangle_and_range",
                 &AmplitudeCalibration::get_interpolator_offset_per_beamangle_and_range,
                 DOC_AmplitudeCalibration(get_interpolator_offset_per_beamangle_and_range))

            //   .def("get_offset_per_beamangle",
            //        nb::overload_cast<const std::vector<float>&>(
            //            &AmplitudeCalibration::get_offset_per_beamangle, nb::const_),
            //        DOC_AmplitudeCalibration(get_offset_per_beamangle),
            //        nb::arg("beamangle"))
            //   .def("get_offset_per_beamangle",
            //        nb::overload_cast<float>(&AmplitudeCalibration::get_offset_per_beamangle,
            //                                 nb::const_),
            //        DOC_AmplitudeCalibration(get_offset_per_beamangle),
            //        nb::arg("beamangle"))
            //   .def("get_offset_per_range",
            //        nb::overload_cast<const std::vector<float>&>(
            //            &AmplitudeCalibration::get_offset_per_range),
            //        DOC_AmplitudeCalibration(get_offset_per_range),
            //      nb::arg("range"))
            //   .def("get_offset_per_range",
            //        nb::overload_cast<float>(&AmplitudeCalibration::get_offset_per_range,
            //        nb::const_), DOC_AmplitudeCalibration(get_offset_per_range), nb::arg("range"))

            // ----- operators -----
            .def("__eq__",
                 &AmplitudeCalibration::operator==,
                 DOC_AmplitudeCalibration(operator_eq),
                 nb::arg("other"))
        // ----- nanobind macros -----
        // default copy functions
        __PYCLASS_DEFAULT_COPY__(AmplitudeCalibration)
        // default binary functions
        __PYCLASS_DEFAULT_BINARY__(AmplitudeCalibration)
        // default printing functions
        __PYCLASS_DEFAULT_PRINTING__(AmplitudeCalibration)
        // end AmplitudeCalibration
        ;
}
}
}
}
}
}
