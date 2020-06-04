/**
 * @brief Linear Discriminant Analysis with Eigen
 * 
 * Dead simple, blazing fast LDA, \cite friedman2001elements
 * 
 * @file lda.hpp
 * @author François-David Collin <Francois-David.Collin@umontpellier.fr>
 * @date 2018-08-31
 */
#pragma once

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef Matrix<size_t, Dynamic, 1> VectorXs;

/**
 * @brief Computes lda with Trevor/hastie algorithm
 * 
 * @param x data
 * @param y labels from 0 to K 
 * @param Ld computes matrix whoses columns are Ld vectors
 */
template<class Derived>
std::vector<int> lda(const MatrixBase<Derived> &x,
         const VectorXs &y,
         MatrixXd& Ld)
{
    auto n = x.rows();
    auto K = y.maxCoeff() + 1;
    auto p = x.cols();

    // M = Centroids
    MatrixXd M = MatrixXd::Zero(K, p);
    VectorXd m = VectorXd::Zero(p);
    VectorXs d = VectorXs::Zero(K);

    for (auto i = 0; i < n; i++)
    {
        auto c = y[i];
        auto r = x.row(i);
        d[c]++;
        M.row(c) += r;
        m += r;
    }

    for (auto c = 0; c < K; c++)
    {
        M.row(c) /= static_cast<double>(d[c]);
    }

    // m = Global mean
    m /= static_cast<double>(n);

    // D is x centered data and sorted by classes
    MatrixXd D(n, p);

    // W = Within-class covariance matrix
    MatrixXd Wraw = MatrixXd::Zero(p, p);
    size_t slicepos = 0;
    for (auto c = 0; c < K; c++)
    {
        // Dc is a single class subrange of x
        MatrixXd Dc = x.block(slicepos, 0, d[c], p);
        // Centering
        Dc.rowwise() -= M.row(c);
        Wraw += Dc.transpose() * Dc;
        slicepos += d[c];
    }
    Wraw /= static_cast<double>(n - K);
    auto validvars = std::vector<int>();
    for(auto i = 0; i< x.cols(); i++) {
        if (std::abs(Wraw(i,i)) >= 1.0e-8) validvars.push_back(i);
    }
    auto W = Wraw(validvars,validvars);
    // [4,4]((0.265008,0.0927211,0.167514,0.0384014),(0.0927211,0.115388,0.0552435,0.0327102),(0.167514,0.0552435,0.185188,0.0426653),(0.0384014,0.0327102,0.0426653,0.0418816))

    // Calculate pseudo-inverse square root for W with SVD
    // https://math.stackexchange.com/a/1176942
    JacobiSVD<MatrixXd> svd(W,ComputeFullU|ComputeFullV);
    double tolerance = std::numeric_limits<double>::epsilon() * std::max(W.cols(), W.rows()) * svd.singularValues().array().abs().maxCoeff();
    auto W12 = svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse().sqrt(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
    // SelfAdjointEigenSolver<MatrixXd> svd(W);
    // if (svd.info() != Success) {
    //     throw std::runtime_error("LDA's first solver failed");
    // }
    //auto W12 = svd.matrixU() * svd.eigenvalues().array().inverse().sqrt().matrix().asDiagonal() * svd.matrixU().transpose();
//    auto W12 = svd.operatorInverseSqrt();
    MatrixXd Mvalid = M(all,validvars);
    MatrixXd Mstar = Mvalid * W12;
    VectorXd mstar = VectorXd::Zero(validvars.size());
    for (auto c = 0; c < K; c++)
        mstar += Mstar.row(c);
    mstar /= static_cast<double>(K);

    Mstar.rowwise() -= mstar.transpose();
    auto Bstar = Mstar.transpose() * Mstar / static_cast<double>(K - 1);

    // We get the eigenvectors of B* via SVD
    JacobiSVD<MatrixXd> svd2(Bstar,ComputeThinU);
    MatrixXd Vl = W12 * svd2.matrixU();
    Ld = Vl.leftCols(K-1);
    return validvars;
}