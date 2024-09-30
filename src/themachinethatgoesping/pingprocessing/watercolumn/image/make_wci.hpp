// Kiel SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

/* generated doc strings */
#include ".docstrings/make_wci.doc.hpp"

#include <memory>

#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor-python/pytensor.hpp> // Numpy bindings

#include <themachinethatgoesping/echosounders/filetemplates/datatypes/i_ping.hpp>

/**
 * @brief Some dummy
 *
 */

namespace themachinethatgoesping {
namespace pingprocessing {
namespace watercolumn {
namespace image {


xt::xtensor<float, 2> make_wci(
    std::shared_ptr<echosounders::filetemplates::datatypes::I_Ping> ping)
{
    // check ping
    if (!ping->has_watercolumn())
    {
        throw std::runtime_error("Ping does not contain watercolumn data.");
    }

    size_t samples = xt::amax(ping->watercolumn().get_number_of_samples_per_beam())();
    size_t beams = ping->watercolumn().get_number_of_beams();

    std::array<size_t, 2> shape = { samples, beams};

    return xt::zeros<float>(shape);
}

}
}
}
}