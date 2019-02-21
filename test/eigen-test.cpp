#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// https://www.seehuhn.de/pages/matrixfn.html
#include <iostream>
#include <Eigen/Dense>
#include "floatvectormatcher.hpp"

using namespace Eigen;
using namespace std;

static const int N = 7;


TEST_CASE( "Simple eigen square root" )
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

    JacobiSVD<MatrixXd> svd(m, ComputeFullU);
    auto D = svd.singularValues().array().sqrt().matrix().asDiagonal();
    auto U = svd.matrixU();
    // SelfAdjointEigenSolver<MatrixXd> svd(m);
    // auto D = svd.eigenvalues().array().sqrt().matrix().asDiagonal();
    // auto U = svd.eigenvectors();

    auto d = U * D * U.transpose(); 
    md  = d * d;
    cout << "Here is the result of d * d" << endl;
    cout << md << endl;
    CHECK_THAT( mbuf, Catch::Matchers::Approx(mdbuf).margin(1e-10) );


}
