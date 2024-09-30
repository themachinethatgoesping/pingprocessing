// SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY // this is needed for xtensor-python but must only be included once

#include <xtensor-python/pytensor.hpp> // Numpy bindings

#include "watercolumn/module.hpp"

// declare modules
// void init_m_navtools(pybind11::module& m); // m_navtools.cpp
namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {

PYBIND11_MODULE(MODULE_NAME, m)
{
    xt::import_numpy(); // import numpy for xtensor (otherwise there will be weird segfaults)

    pybind11::add_ostream_redirect(m, "ostream_redirect");

    m.doc()               = "Python module process pings from echosounders.";
    m.attr("__version__") = MODULE_VERSION;

    py_watercolumn::init_m_watercolumn(m);
}

}
}
}