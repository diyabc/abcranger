#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "matutils.hpp"
#include "readreftable.hpp"

// #include <numeric>
// #include <boost/accumulators/accumulators.hpp>
// #include <boost/accumulators/statistics/stats.hpp>
// #include <boost/accumulators/statistics/mean.hpp>
// #include <boost/accumulators/statistics/variance.hpp>
// using namespace boost::accumulators;


// BOOST_AUTO_TEST_CASE(UnifDistrib, * boost::unit_test::tolerance(0.01)) {
//     accumulator_set<double,stats<tag::mean, tag::variance>> acc;
//     for(auto i = 0; i < 10000; i++) acc(dis(rng));
//     BOOST_TEST( mean(acc) == 0.5);
//     BOOST_TEST( variance(acc) == 1.0/12.0);
// }

TEST_CASE("Matutils : add linear Combination") {
    MatrixXi m1(2,3);
    MatrixXi m2(3,2);
    MatrixXi m3(2,5);

    m1 << 1, 2, 3,
          4, 5, 6;

    m2 << 1, 2,
          3, 4, 
          5, 6; 

    m3 << 1, 2, 3, 22, 28, 
          4, 5, 6, 49, 64;
    addLinearComb(m1,m1,m2);
    CHECK( (m3 - m1).lpNorm<Infinity>() == 0 );
}

// TEST_CASE(NoiseNames) {
//     auto myread = readreftable("headerRF.txt","reftableRF.bin");
//     size_t noisecols = 5;
//     size_t nrow = myread.nrec;
//     // size_t oldsize = myread.stats.size();
//     auto stats_names(myread.stats_names);
//     // for(auto i = 0; i < noisecols; i++) {
//     //     stats_names.push_back("NOISE" + std::to_string(i));
//     // }
//     // addNoiseCols(myread,noisecols);
//     BOOST_TEST ( 1 == 1 );
//     // BOOST_TEST( myread.stats.size() == (oldsize + noisecols * nrow) );
//     // BOOST_TEST( myread.stats_names == stats_names );
// }

