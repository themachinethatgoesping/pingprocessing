// SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "module.hpp"

#include <pybind11/pybind11.h>

#include <themachinethatgoesping/pingprocessing/watercolumn/wci/make_wci.hpp>
namespace py = pybind11;

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_watercolumn {
namespace py_wci {

#define DOC_M_WCI(ARG) DOC(themachinethatgoesping, pingprocessing, watercolumn, wci, ARG)

// -- create submodule --
void init_m_wci(py::module& m)
{
    // module description
    auto subm = m.def_submodule("wci", "Functions for creating waterolumn images.");

    // functions
    subm.def("make_wci",
             &themachinethatgoesping::pingprocessing::watercolumn::wci::make_wci,
             DOC_M_WCI(make_wci),
             py::arg("ping"));
}

}
}
}
}
}