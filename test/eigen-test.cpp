#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

// https://www.seehuhn.de/pages/matrixfn.html
#include <iostream>
#include <Eigen/Dense>
// #include "floatvectormatcher.hpp"

using namespace Eigen;

static const int N = 7;

TEST_CASE( "Column mean")
{
    MatrixXi m(3,3);
    m << 1,2,3,
         4,5,6,
         7,8,9;

    CHECK( m.col(1).mean() == 5.0);
}

TEST_CASE( "Simple eigen square root" )
{
    std::vector<double> mbuf(N*N);
    std::vector<double> mdbuf(N*N);
    Map<MatrixXd> m(mbuf.data(), N,N);
    Map<MatrixXd> md(mdbuf.data(),N,N);

    m = MatrixXd::Zero(N, N);

    m.diagonal(1).setConstant(-1);
    m.diagonal(-1).setConstant(-1);
    m.diagonal().setConstant(2);
    m(0, 0) = 1;
    m(N - 1, N - 1) = 1;

    std::cout << "Here is the matrix m:" << std::endl
         << m << std::endl;

    JacobiSVD<MatrixXd> svd(m, ComputeFullU);
    auto D = svd.singularValues().array().sqrt().matrix().asDiagonal();
    auto U = svd.matrixU();

    // SelfAdjointEigenSolver<MatrixXd> svd(m);
    // auto D = svd.eigenvalues().array().sqrt().matrix().asDiagonal();
    // auto U = svd.eigenvectors();

    auto d = U * D * U.transpose(); 
    md  = d * d;
    std::cout << "Here is the result of d * d" << std::endl;
    std::cout << md << std::endl;
    const auto expected = Catch::Matchers::Approx(mdbuf).margin(1e-10);
    CHECK_THAT( mbuf, expected );


}
