// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <themachinethatgoesping/pingprocessing/pingtools/pingsampleselector.hpp>

// using namespace testing;
using namespace std;
using namespace themachinethatgoesping::pingprocessing::pingtools;
#define TESTTAG "[pingtools]"

TEST_CASE("PingSampleselector should support common functions", TESTTAG)
{
    // initialize class structure
    auto obj = PingSampleSelector();

    // make sure binary_hash is stable for empty object
    // the hash tests need to be updated in case the object structure changes
    CHECK(obj.binary_hash() == 11393313688947811519ULL);

    // set some variables
    obj.select_beam_range_by_numbers(1, -1, 1);
    REQUIRE(obj.get_min_beam_number() == 1);
    REQUIRE(obj.get_max_beam_number() == -1);
    REQUIRE(obj.get_beam_step() == 1);

    // test binary (inbetween)
    REQUIRE(obj == PingSampleSelector(obj.from_binary(obj.to_binary())));
    obj.select_sample_range_by_numbers(-2, 2, 2);
    REQUIRE(obj.get_min_sample_number() == -2);
    REQUIRE(obj.get_max_sample_number() == 2);
    REQUIRE(obj.get_sample_step() == 2);

    // test binary (inbetween)
    REQUIRE(obj == PingSampleSelector(obj.from_binary(obj.to_binary())));
    obj.select_beam_range_by_angles(-3.3, 4.4, 10);
    REQUIRE(obj.get_min_beam_angle() == Catch::Approx(-3.3));
    REQUIRE(obj.get_max_beam_angle() == Catch::Approx(4.4));
    REQUIRE(obj.get_beam_step() == 10);

    // test binary (inbetween)
    REQUIRE(obj == PingSampleSelector(obj.from_binary(obj.to_binary())));
    obj.select_sample_range_by_ranges(4.3, -5.4, 20);
    REQUIRE(obj.get_min_sample_range() == Catch::Approx(4.3));
    REQUIRE(obj.get_max_sample_range() == Catch::Approx(-5.4));
    REQUIRE(obj.get_sample_step() == 20);

    // test binary (inbetween)
    REQUIRE(obj == PingSampleSelector(obj.from_binary(obj.to_binary())));
    obj.select_transmit_sectors(std::vector<size_t>{ 3, 1 });
    REQUIRE(obj.get_transmit_sectors().value().at(0) == 3);
    REQUIRE(obj.get_transmit_sectors().value().at(1) == 1);

    // test binary (inbetween)
    REQUIRE(obj == PingSampleSelector(obj.from_binary(obj.to_binary())));
    obj.select_transmit_sectors_by_beam_angles(-2.3, 6.4);
    REQUIRE(obj.get_transmit_sector_min_beam_angle() == Catch::Approx(-2.3));
    REQUIRE(obj.get_transmit_sector_max_beam_angle() == Catch::Approx(6.4));

    // test binary (inbetween)
    REQUIRE(obj == PingSampleSelector(obj.from_binary(obj.to_binary())));
    obj.select_min_minslant_range_percent(0.95);
    REQUIRE(obj.get_min_minslant_range_percent() == Catch::Approx(0.95));
    obj.select_max_minslant_range_percent(0.30);
    REQUIRE(obj.get_max_minslant_range_percent() == Catch::Approx(0.30));
    
    // test binary (inbetween)
    REQUIRE(obj == PingSampleSelector(obj.from_binary(obj.to_binary())));
    obj.select_min_bottom_range_percent(0.96);
    REQUIRE(obj.get_min_bottom_range_percent() == Catch::Approx(0.96));
    obj.select_max_bottom_range_percent(0.31);
    REQUIRE(obj.get_max_bottom_range_percent() == Catch::Approx(0.31));

    obj.set_sample_step(12);
    REQUIRE(obj.get_sample_step() == 12);
    obj.set_beam_step(13);
    REQUIRE(obj.get_beam_step() == 13);

    CHECK(obj.binary_hash() == 15382176863757348378ULL);

    // test inequality
    REQUIRE(obj != PingSampleSelector());

    // test copy
    REQUIRE(obj == PingSampleSelector(obj));

    // test binary (inbetween)
    REQUIRE(obj == PingSampleSelector(obj.from_binary(obj.to_binary())));

    // test stream
    std::stringstream buffer;
    obj.to_stream(buffer);
    REQUIRE(obj == PingSampleSelector(obj.from_stream(buffer)));

    // test print does not crash
    REQUIRE(obj.info_string().size() != 0);

    //--- Class concept ---
    obj.clear();
    REQUIRE(obj == PingSampleSelector());
}
