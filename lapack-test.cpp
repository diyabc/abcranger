// https://www.seehuhn.de/pages/matrixfn.html
#define BOOST_TEST_MODULE LapackTest
#include <boost/test/unit_test.hpp>


#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/ublas/banded.hpp>

using namespace boost::numeric::ublas;
using namespace boost::numeric::bindings;

static const int N = 7;

BOOST_AUTO_TEST_CASE(LapackEigen, * boost::unit_test::tolerance(1e-10))
{

    matrix<double, column_major> mo(N,N),m(N,N);
    m = zero_matrix<double>(N,N);

    // banded_adaptor<matrix<double, column_major>> ba(m,1,1);
    // for (signed i = 0; i < signed (ba.size1 ()); ++ i)
    //     for (signed j = std::max (i - 1, 0); j < std::min (i + 2, signed (ba.size2 ())); ++ j)
    //         ba (i, j) = i == j ? 2 : -1;

    for(auto i = 0; i <N-1; i++) m(i+1,i) = -1;
    for(auto i = 0; i <N-1; i++) m(i,i+1) = -1;
    for(auto i = 1; i <N-1; i++) m(i,i) = 2;
    m(0,0) = 1;
    m(N-1,N-1) = 1; 
 
    std::cout << m << std::endl;
    mo = m;
    vector<double> smd(N);
    lapack::syev('V','U',m,smd);
    std::transform(std::begin(smd),std::end(smd), std::begin(smd),[](double d) { return sqrt(d); });
    matrix<double> sm = prod(prod<matrix<double>>(m,diagonal_matrix<double>(smd.size(),smd.data())),trans(m));
    matrix<double> md = prod(sm, sm);
    // std::transform(md.data().begin(),md.data().end(), md.data().begin(),[](double d) { return abs(d) < 1.0e-10 ? 0 : d; });
    BOOST_TEST(md.data() == mo.data(), boost::test_tools::per_element());
}
