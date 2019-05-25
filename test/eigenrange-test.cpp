#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <limits>
#include <Eigen/Dense>
#include <vector>
#include <range/v3/all.hpp>
#include "floatvectormatcher.hpp"

using namespace Eigen;
using namespace ranges;

TEST_CASE("Extract indices rows from a matrix")
{
      MatrixXi m(10, 4);
      m.reshaped() = VectorXi::LinSpaced(40, 1, 40);
      std::vector<size_t> indices{3, 5, 8};
      m = m(indices, all).eval();
      std::cout << m << std::endl;
}
TEST_CASE("Enumerate and find")
{
      std::vector<std::string> v{"blah", "blih", "blouh", "machin", "truc"};
      auto venum = v | view::enumerate | to_vector;
      auto i = find_if(venum, [](auto &s) { return s.second == "machin"; });
      CHECK(i->first == 3);
}

TEST_CASE("Shuffle and split")
{
      std::default_random_engine gen;
      size_t n = 10;
      size_t s = 3;
      auto tosplit = view::ints(static_cast<size_t>(0), n) | to_vector | action::shuffle(gen);
      auto a = tosplit | view::take(s);
      auto b = tosplit | view::slice(s, n);
      CHECK(distance(a) == 3);
      CHECK(distance(b) == 7);
}

TEST_CASE("Eigen with ranges")
{
      MatrixXi m1(2, 3);

      m1 << 1, std::numeric_limits<int>::quiet_NaN(), 3,
          4, 5, 6;

      //    const auto& expected = Catch::Range::EqualsMatcher<std::vector<int>,std::vector<int>>(std::vector{1,2,3});
      const auto &row0 = m1.row(0);
      const std::vector<int> vect_expected{1, std::numeric_limits<int>::quiet_NaN(), 3};
      const auto &expected = Catch::Matchers::Equals<std::vector<int>, decltype(row0)>(vect_expected);
      CHECK_THAT(row0, expected);

      MatrixXd m2(2, 3);
      // m2 << 1.0, std::numeric_limits<double>::quiet_NaN(), 3.0,
      m2 << 1.0, 2.0, 3.0,
          4.0, 5.0, 6.0;

      const auto &row1 = m2.row(0);
      const std::vector<double> vect_expected2{1.0, 2.0, 3.0};
      const auto &expected2 = Catch::Matchers::Approx<std::vector<double>, decltype(row1)>(vect_expected2);
      CHECK_THAT(row1, expected2);

      MatrixXd m3(2, 3);
      m3 << 1.0, std::numeric_limits<double>::quiet_NaN(), 3.0,
          4.0, 5.0, 6.0;

      const auto &row2 = m3.row(0);
      const std::vector<double> vect_expected3{1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
      const auto &expected3 = Catch::Matchers::Approx<std::vector<double>, decltype(row2)>(vect_expected3);
      CHECK_THAT(row2, expected3);

      MatrixXd m4(2, 3);
      m3 << 1.0, 2.001, 3.0,
          4.0, 5.0, 6.0;

      const auto &row3 = m3.row(0);
      CHECK_THAT(row3, !expected2);

      RowVectorXd vec1(3);
      vec1 << 1, 2, 3;
      VectorXd res(10);
      res << 1, 2, 3 ,4 ,5, 7, 9, 10, 12, 15;

 // https://stackoverflow.com/questions/36820639/how-do-i-write-a-range-pipeline-that-uses-temporary-containers     
    auto restmp = res(seq(3,7)).array(); 
      auto diff = restmp
            | view::sliding(2)
            | view::transform([](const auto& l) -> double { 
                    return (2.0 * std::abs(l[1]-l[0])/(l[1]+l[0]) ) ; })
            | to_vector;


}
