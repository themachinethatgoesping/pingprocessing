// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>
#include <themachinethatgoesping/tools_nanobind/datetime.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/i_ping.hpp>
#include "c_i_ping.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {

#define DOC_I_Ping(ARG)                                                                            \
    DOC(themachinethatgoesping, pingprocessing, datatypes, I_Ping, ARG)

void init_c_i_ping(nanobind::module_& m)
{
    namespace nb = nanobind;
    using namespace themachinethatgoesping::pingprocessing::datatypes;

    auto cls =
        nb::class_<I_Ping, I_PingCommon>(
            m,
            "I_Ping",
            DOC(themachinethatgoesping, pingprocessing, datatypes, I_Ping));

    cls.def(nb::init<>(), DOC_I_Ping(I_Ping));

    py_i_ping::I_Ping_add_interface<I_Ping>(cls);

    cls __PYCLASS_DEFAULT_COPY__(I_Ping)
        __PYCLASS_DEFAULT_PRINTING__(I_Ping);
}

}
}
}
}
