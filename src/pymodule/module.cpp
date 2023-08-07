// SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

// declare modules
//void init_m_navtools(pybind11::module& m); // m_navtools.cpp

PYBIND11_MODULE(MODULE_NAME, m)
{
    pybind11::add_ostream_redirect(m, "ostream_redirect");

    m.doc() = "Python module process ping data, e.g. apply absorption, spreading loss, compute "
              "range/depth, raytrace ...";
    m.attr("__version__") = MODULE_VERSION;

    // init_m_navtools(m);
}
