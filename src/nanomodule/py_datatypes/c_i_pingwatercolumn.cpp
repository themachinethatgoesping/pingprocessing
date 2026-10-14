// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>
#include <xtensor-python/nanobind/pytensor.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/i_pingwatercolumn.hpp>
#include "i_pingwatercolumn.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {

#define DOC_I_PingWatercolumn(ARG)                                                                 \
    DOC(themachinethatgoesping, pingprocessing, datatypes, I_PingWatercolumn, ARG)

void init_c_i_pingwatercolumn(nanobind::module_& m)
{
    namespace nb = nanobind;
    using namespace themachinethatgoesping::pingprocessing::datatypes;

    auto cls =
        nb::class_<I_PingWatercolumn, I_PingCommon>(
            m,
            "I_PingWatercolumn",
            DOC(themachinethatgoesping, pingprocessing, datatypes, I_PingWatercolumn));

    py_i_pingwatercolumn::I_PingWatercolumn_add_interface<I_PingWatercolumn>(cls);

    cls __PYCLASS_DEFAULT_COPY__(I_PingWatercolumn);
}

}
}
}
}
