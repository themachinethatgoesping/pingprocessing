// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/i_pingfiledata.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {
namespace py_i_pingfiledata {

#define DOC_I_PingFileData(ARG)                                                                     \
    DOC(themachinethatgoesping, pingprocessing, datatypes, I_PingFileData, ARG)

template<typename T_BaseClass, typename T_PyClass>
void I_PingFileData_add_interface(T_PyClass& cls)
{
    namespace nb = nanobind;

    cls.def("get_file_numbers",
            &T_BaseClass::get_file_numbers,
            DOC_I_PingFileData(get_file_numbers));
    cls.def("get_primary_file_path",
            &T_BaseClass::get_primary_file_path,
            DOC_I_PingFileData(get_primary_file_path));
    cls.def("get_file_paths",
            &T_BaseClass::get_file_paths,
            DOC_I_PingFileData(get_file_paths));

    cls.def("get_primary_file_nr",
            &T_BaseClass::get_primary_file_nr,
            DOC_I_PingFileData(get_primary_file_nr));
    cls.def("set_primary_file_nr",
            &T_BaseClass::set_primary_file_nr,
            DOC_I_PingFileData(set_primary_file_nr),
            nb::arg("file_nr"));

    cls.def("get_file_ping_counter",
            &T_BaseClass::get_file_ping_counter,
            DOC_I_PingFileData(get_file_ping_counter));
    cls.def("set_file_ping_counter",
            &T_BaseClass::set_file_ping_counter,
            DOC_I_PingFileData(set_file_ping_counter),
            nb::arg("file_ping_counter"));

    cls __PYCLASS_DEFAULT_COPY__(T_BaseClass)
        __PYCLASS_DEFAULT_PRINTING__(T_BaseClass);
}

}
}
}
}
}
