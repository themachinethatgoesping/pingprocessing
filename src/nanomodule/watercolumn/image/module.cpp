// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "module.hpp"

#include <nanobind/nanobind.h>

#include <themachinethatgoesping/pingprocessing/watercolumn/image/make_wci.hpp>
namespace nb = nanobind;

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_watercolumn {
namespace py_image {

#define DOC_M_WCI(ARG) DOC(themachinethatgoesping, pingprocessing, watercolumn, image, ARG)

// -- create submodule --
void init_m_image(nb::module_& m)
{
    // module description
    auto subm = m.def_submodule("image", "Functions for creating waterolumn images.");

    // functions
    subm.def("make_wci",
             &themachinethatgoesping::pingprocessing::watercolumn::image::make_wci,
             DOC_M_WCI(make_wci),
             nb::arg("ping"));
}

}
}
}
}
}