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

template<typename _Matrix_Type_>
bool pseudoInverseSqrt(const _Matrix_Type_ &a, _Matrix_Type_ &result, double epsilon = std::numeric_limits<double>::epsilon())
{
  if(a.rows() < a.cols())
   return false;
  Eigen::JacobiSVD< _Matrix_Type_ > svd = a.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs().maxCoeff();
  result = svd.matrixV() * _Matrix_Type_( (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse().sqrt(), 0) ).asDiagonal() * svd.matrixU().adjoint();
}

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
    // m = means by features
    VectorXd m = VectorXd::Zero(p);
    // d = counts by class
    Matrix<size_t, -1, 1> d = Matrix<size_t, -1, 1>::Zero(K);

    // Loop for M, m and d
    for (auto i = 0; i < n; i++)
    {
        auto c = y[i];
        auto r = x.row(i);
        d[c]++;
        M.row(c) += r;
        m += r;
    }

    // Finalize M 
    for (auto c = 0; c < K; c++)
    {
        M.row(c) /= static_cast<double>(d[c]);
    }

    // Finalize m
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

    // Calculate inverse square root for W with eigen solver
    JacobiSVD<MatrixXd> svd(W,ComputeFullU|ComputeFullV);
    double tolerance = std::numeric_limits<double>::epsilon() * std::max(W.cols(), W.rows()) * svd.singularValues().array().abs().maxCoeff();
    auto W12 = svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse().sqrt(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
    // SelfAdjointEigenSolver<MatrixXd> svd(W);
    // if (svd.info() != Success) {
    //     throw std::runtime_error("LDA's first solver failed");
    // }
    //auto W12 = svd.matrixU() * svd.eigenvalues().array().inverse().sqrt().matrix().asDiagonal() * svd.matrixU().transpose();
//    auto W12 = svd.operatorInverseSqrt();

    // M* from friedman2001elements
    MatrixXd Mstar = M * W12;

    // Now we calculate B* covariance matrix of M*
    VectorXd mstar = VectorXd::Zero(p);
    for (auto c = 0; c < K; c++)
        mstar += Mstar.row(c);
    mstar /= static_cast<double>(K);

    Mstar.rowwise() -= mstar.transpose();
    auto Bstar = Mstar.transpose() * Mstar / static_cast<double>(K - 1);

    // We get the eigenvectors of B* via SVD
    // https://math.stackexchange.com/a/1176942
    JacobiSVD<MatrixXd> svd2(Bstar,ComputeThinU);
    MatrixXd Vl = W12 * svd2.matrixU();
    Ld = Vl.block(0,0,p,K-1);
}