#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <Eigen/Dense>
#include "floatvectormatcher.hpp"

using namespace Eigen;


TEST_CASE("Eigen with ranges") {
    MatrixXi m1(2,3);

    m1 << 1, 2, 3,
          4, 5, 6;

//    const auto& expected = Catch::Range::EqualsMatcher<std::vector<int>,std::vector<int>>(std::vector{1,2,3});
    const auto& row0 = m1.row(0);
    const std::vector<int> vect_expected{1,2,3};
    const auto& expected = Catch::Range::EqualsMatcher<std::vector<int>,decltype(row0)>(vect_expected);
    CHECK_THAT( row0, expected );
}