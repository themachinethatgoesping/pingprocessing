// Kiel SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

/* generated doc strings */
#include ".docstrings/echogrambase.doc.hpp"

/* generated doc strings */
#include ".docstrings/make_wci.doc.hpp"

#include <memory>


#include <xtensor/containers/xtensor.hpp>
#include <xtensor-python/pytensor.hpp> // Numpy bindings

#include <themachinethatgoesping/echosounders/filetemplates/datatypes/i_ping.hpp>

/**
 * @brief Some dummy
 *
 */

namespace themachinethatgoesping {
namespace pingprocessing {
namespace watercolumn {
namespace echograms {

/**
 * @brief Class to convert 2d water column ping coordinates to 2d echogram images
 * Allows for converting between ping_nr/sample_nr to time/depth/range and vice versa
 * Allows for adding parameters in different units (e.g. time, depth, range) and convert them to echogram coordinates
 */
class EchogramBase {
    
};

}


}
}
}
}