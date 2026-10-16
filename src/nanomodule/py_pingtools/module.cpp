// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "module.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_pingtools {

// -- submodule declarations --
void init_c_pingsampleselector(nanobind::module_& m); // c_pingsampleselector.cpp

// -- create submodule --
void init_m_pingtools(nb::module_& m)
{
    // module description
    auto subm = m.def_submodule("pingtools", "Ping-aware tools (e.g. PingSampleSelector).");

    // pingtool classes
    init_c_pingsampleselector(subm);
}

}
}
}
}
