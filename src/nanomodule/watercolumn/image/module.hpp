// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <nanobind/nanobind.h>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {

namespace py_watercolumn {
namespace py_image {

// -- initialize module --
void init_m_image(nanobind::module_& m);

}
}
}
}
}