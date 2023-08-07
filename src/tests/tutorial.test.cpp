// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>

// using namespace testing;
using namespace std;
// using namespace themachinethatgoesping;

namespace tutorial_tests {

unsigned int Factorial(unsigned int number)
{
    return number <= 1 ? number : Factorial(number - 1) * number;
}

TEST_CASE("Tutorial: Factorials are computed", "[factorial]")
{
    REQUIRE_FALSE(Factorial(0) == 1);
    REQUIRE(Factorial(1) == 1);
    REQUIRE(Factorial(2) == 2);
    REQUIRE(Factorial(3) == 6);
    REQUIRE(Factorial(10) == 3628800);
}

}
