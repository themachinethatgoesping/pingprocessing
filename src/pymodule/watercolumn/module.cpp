// SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "module.hpp"
#include "image/module.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_watercolumn {

// -- submodule declarations -

// -- create submodule --
void init_m_watercolumn(py::module& m)
{
    // module description
    auto subm = m.def_submodule("watercolumn", "Functions for processing watercolumn data.");

    py_image::init_m_image(subm);
}

}
}
}
}