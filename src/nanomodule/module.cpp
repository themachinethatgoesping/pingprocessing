// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/nanobind.h>

#include <xtensor-python/nanobind/pytensor.hpp> // Numpy bindings

#include "watercolumn/module.hpp"

// declare modules
// void init_m_navtools(nanobind::module_& m); // m_navtools.cpp
namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {

NB_MODULE(MODULE_NAME, m)
{

    auto echosounder_module = nanobind::module_::import_("themachinethatgoesping.echosounders_nanopy");

    m.doc()               = "Python module process pings from echosounders.";
    m.attr("__version__") = MODULE_VERSION;

    py_watercolumn::init_m_watercolumn(m);
}

}
}
}