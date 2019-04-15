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


/**
 * @brief Computes lda with Trevor/hastie algorithm
 * 
 * @param x data
 * @param y labels from 0 to K 
 * @param Ld computes matrix whoses columns are Ld vectors
 */
template<class Derived>
void lda(const MatrixBase<Derived> &x,
         const Matrix<size_t, -1, 1> &y,
         MatrixXd& Ld)
{
    auto n = x.rows();
    auto p = x.cols();
    auto K = y.maxCoeff() + 1;

    // M = Centroids
    MatrixXd M = MatrixXd::Zero(K, p);
    VectorXd m = VectorXd::Zero(p);
    Matrix<size_t, -1, 1> d = Matrix<size_t, -1, 1>::Zero(K);

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
    MatrixXd W = MatrixXd::Zero(p, p);
    size_t slicepos = 0;
    for (auto c = 0; c < K; c++)
    {
        // Dc is a single class subrange of x
        MatrixXd Dc = x.block(slicepos, 0, d[c], p);
        // Centering
        Dc.rowwise() -= M.row(c);
        W += Dc.transpose() * Dc;
        slicepos += d[c];
    }
    W /= static_cast<double>(n - K);
    // [4,4]((0.265008,0.0927211,0.167514,0.0384014),(0.0927211,0.115388,0.0552435,0.0327102),(0.167514,0.0552435,0.185188,0.0426653),(0.0384014,0.0327102,0.0426653,0.0418816))

    SelfAdjointEigenSolver<MatrixXd> svd(W);
    if (svd.info() != Success) {
        throw std::runtime_error("LDA's first solver failed");
    }
//    auto W12 = svd.eigenvectors() * svd.eigenvalues().array().inverse().sqrt().matrix().asDiagonal() * svd.eigenvectors().transpose();
    auto W12 = svd.operatorInverseSqrt();
    // [4,4]((2.87992,-0.787459,-1.37944,0.163224),(-0.787459,3.57306,0.106208,-0.914195),(-1.37944,0.106208,3.46983,-0.913961),(0.163224,-0.914195,-0.913961,5.92239))

    MatrixXd Mstar = M * W12;
    VectorXd mstar = VectorXd::Zero(p);
    for (auto c = 0; c < K; c++)
        mstar += Mstar.row(c);
    mstar /= static_cast<double>(K);

    Mstar.rowwise() -= mstar.transpose();
    auto Bstar = Mstar.transpose() * Mstar / static_cast<double>(K - 1);

    SelfAdjointEigenSolver<MatrixXd> svd2(Bstar,ComputeEigenvectors);
    if (svd2.info() != Success) {
        throw std::runtime_error("LDA's second solver failed");
    }

    MatrixXd Vl = W12 * svd2.eigenvectors();

    Ld = Vl.rowwise().reverse().block(0,0,p,K-1);
}