// https://www.seehuhn.de/pages/matrixfn.html
#define BOOST_TEST_MODULE EigenTest
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

static const int N = 7;

BOOST_AUTO_TEST_CASE(EigenSimple, *boost::unit_test::tolerance(1e-10))
{
    vector<double> mbuf(N*N);
    vector<double> mdbuf(N*N);
    Map<MatrixXd> m(mbuf.data(), N,N);
    Map<MatrixXd> md(mdbuf.data(),N,N);

    m = MatrixXd::Zero(N, N);

    m.diagonal(1).setConstant(-1);
    m.diagonal(-1).setConstant(-1);
    m.diagonal().setConstant(2);
    m(0, 0) = 1;
    m(N - 1, N - 1) = 1;

    cout << "Here is the matrix m:" << endl
         << m << endl;

    JacobiSVD<MatrixXd> svd(m, ComputeThinU);
    auto D = svd.singularValues().array().sqrt().matrix().asDiagonal();
    auto U = svd.matrixU();
    // SelfAdjointEigenSolver<MatrixXd> svd(m);
    // auto D = svd.eigenvalues().array().sqrt().matrix().asDiagonal();
    // auto U = svd.eigenvectors();

    auto d = U * D * U.transpose(); 
    md  = d * d;
    cout << "Here is the result of d * d" << endl;
    cout << md << endl;
    BOOST_TEST(mbuf == mdbuf, boost::test_tools::per_element());


}
