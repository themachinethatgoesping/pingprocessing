// SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <pybind11/pybind11.h>


namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {

namespace py_watercolumn {

// -- initialize module --
void init_m_watercolumn(pybind11::module& m);

}
}
}
}