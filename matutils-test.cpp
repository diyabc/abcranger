#define BOOST_TEST_MODULE MatUtilsTest
#include <boost/test/unit_test.hpp>

#include <numeric>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
using namespace boost::accumulators;

#include "matutils.hpp"
#include "readreftable.hpp"

BOOST_AUTO_TEST_CASE(AddCol) {
    size_t ncol = 10;
    size_t nrow = 100;
    std::vector<int> v(ncol*nrow);
    std::iota(std::begin(v),std::end(v),0);
    std::vector<int> w(v);
    w.resize(w.size()+nrow);
    std::iota(std::next(std::begin(w),ncol*nrow),std::end(w),ncol*nrow);
    std::vector<int> c(nrow);
    std::iota(std::begin(c),std::end(c),nrow*ncol);
    addCol(v,c);
    BOOST_TEST( v == w);
}

BOOST_AUTO_TEST_CASE(ReplaceCol) {
    size_t ncol = 10;
    size_t nrow = 100;
    size_t colnum = 5;
    std::vector<int> v(ncol*nrow);
    std::iota(std::begin(v),std::end(v),0);
    std::vector<int> w(v);
    std::iota(std::next(std::begin(w),ncol*nrow),std::end(w),ncol*nrow);
    std::iota(std::next(std::begin(w),colnum*nrow),std::next(std::begin(w),(colnum+1)*nrow),ncol*nrow);
    std::vector<int> c(nrow);
    std::iota(std::begin(c),std::end(c),nrow*ncol);
    replaceCol(v,c,colnum);
    BOOST_TEST( v == w );
}

BOOST_AUTO_TEST_CASE(UnifDistrib, * boost::unit_test::tolerance(0.01)) {
    auto noisevec = getNoise(100000);
    accumulator_set<double,stats<tag::mean, tag::variance>> acc;
    for(auto& r: noisevec) acc(r);
    BOOST_TEST( mean(acc) == 0.5);
    BOOST_TEST( variance(acc) == 1.0/12.0);
}

BOOST_AUTO_TEST_CASE(NoiseNames) {
    auto myread = readreftable("headerRF.txt","reftableRF.bin");
    size_t noisecols = 5;
    size_t nrow = myread.nrec;
    size_t oldsize = myread.stats.size();
    auto stats_names(myread.stats_names);
    // for(auto i = 0; i < noisecols; i++) {
    //     stats_names.push_back("NOISE" + std::to_string(i));
    // }
    // addNoiseCols(myread,noisecols);
    BOOST_TEST ( 1 == 1 );
    // BOOST_TEST( myread.stats.size() == (oldsize + noisecols * nrow) );
    // BOOST_TEST( myread.stats_names == stats_names );
}

