// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
//
// SPDX-License-Identifier: MPL-2.0

#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

// xtensor python includes
#include <xtensor-python/nanobind/pytensor.hpp> // Numpy bindings

#include <themachinethatgoesping/tools_nanobind/classhelper.hpp>

#include <themachinethatgoesping/pingprocessing/datatypes/calibration/multisectorwatercolumncalibration.hpp>

#include "c_multisectorwatercolumncalibration.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace pymodule {
namespace py_datatypes {
namespace py_calibration {

namespace nb = nanobind;


void init_c_multisectorwatercolumncalibration(nanobind::module_& m)
{
    using namespace themachinethatgoesping::pingprocessing::datatypes::calibration;

    init_multisectorcalibration<WaterColumnCalibration>(m, "MultiSectorWaterColumnCalibration");
}

}
}
}
}
}
