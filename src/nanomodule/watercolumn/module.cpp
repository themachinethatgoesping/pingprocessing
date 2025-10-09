// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "module.hpp"
#include "image/module.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_watercolumn {

// -- submodule declarations -

// -- create submodule --
void init_m_watercolumn(nb::module_& m)
{
    // module description
    auto subm = m.def_submodule("watercolumn", "Functions for processing watercolumn data.");

    py_image::init_m_image(subm);
}

}
}
}
}