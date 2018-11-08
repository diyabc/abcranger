#define BOOST_TEST_MODULE PLSEigenTest
#include <boost/test/unit_test.hpp>

#include "pls-eigen.hpp"

BOOST_AUTO_TEST_CASE(PLSEigenTestSimple, *boost::unit_test::tolerance(1e-7)) {
    BOOST_CHECK(1 == 1);
}