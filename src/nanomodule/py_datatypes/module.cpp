// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "module.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {

// -- submodule declarations --
void init_c_i_pingcommon(nb::module_& m);      // py_datatypes/c_i_pingcommon.cpp
void init_c_i_pingfiledata(nb::module_& m);    // py_datatypes/c_i_pingfiledata.cpp
void init_c_i_ping(nb::module_& m);            // py_datatypes/c_i_ping.cpp
void init_c_i_pingbottom(nb::module_& m);      // py_datatypes/c_i_pingbottom.cpp
void init_c_i_pingwatercolumn(nb::module_& m); // py_datatypes/c_i_pingwatercolumn.cpp
namespace py_calibration {
void init_c_amplitudecalibration(nb::module_& m);
void init_c_watercolumncalibration(nb::module_& m);
void init_c_multisectorwatercolumncalibration(nb::module_& m);
}

// -- create submodule --
void init_m_datatypes(nb::module_& m)
{
    // module description
    auto subm = m.def_submodule(
        "datatypes", "Format-independent ping data type interfaces and calibration classes");

    // calibration classes (must be registered before the ping interfaces that return them)
    py_calibration::init_c_amplitudecalibration(subm);
    py_calibration::init_c_watercolumncalibration(subm);
    py_calibration::init_c_multisectorwatercolumncalibration(subm);

    // ping interfaces
    init_c_i_pingcommon(subm);
    init_c_i_pingfiledata(subm);
    init_c_i_pingbottom(subm);
    init_c_i_pingwatercolumn(subm);
    init_c_i_ping(subm);
}

}
}
}
}
