/**
 * @brief Partial Least Squares Regression with Eigen
 * 
 * Same source as LDA, \cite friedman2001elements
 * 
 * @file pls-eigen.hpp
 * @author François-David Collin <Francois-David.Collin@umontpellier.fr>
 * @brief 
 * @version 0.1
 * @date 2018-11-08
 * 
 * @copyright Copyright (c) 2018
 * 
 */
#pragma once

#include "various.hpp"
#include <Eigen/Dense>
#include <list>
#include <algorithm>
#include <range/v3/all.hpp>

using namespace Eigen;
using namespace std;
using namespace ranges;

template<class Derived>
std::vector<size_t> filterConstantVars(const MatrixBase<Derived>& xr) {
    auto meanr = xr.colwise().mean();
    auto stdr = ((xr.rowwise() - meanr).array().square().colwise().sum() / (xr.rows() - 1)).sqrt();;
    std::vector<size_t> validvars;
    for(size_t i = 0; i< xr.cols(); i++) {
        if (stdr(i) >= 1.0e-8) validvars.push_back(i);
    }
    return validvars;
}

template<class Derived, class OtherDerived>
VectorXd pls(const MatrixBase<Derived>& x,
         const MatrixBase<OtherDerived>& y,
         size_t ncomp,
         MatrixXd& Projection,
         RowVectorXd& mean,
         RowVectorXd& std,
         bool stopping = false)
{
    auto n = x.rows();
    auto p = x.cols();
    mean = x.colwise().mean();
    std = ((x.rowwise() - mean).array().square().colwise().sum() / (x.rows() - 1)).sqrt();;
    MatrixXd X = (x.rowwise() - mean).array().rowwise() / std.array();
    MatrixXd Xo = X;
    MatrixXd XX = X.transpose() * X;
    MatrixXd Y = MatrixXd::Zero(n, ncomp + 1);
    MatrixXd P(p,ncomp);
    MatrixXd R(p,ncomp);
    VectorXd XY(n); 
    VectorXd w(p);
    VectorXd r(p);
    VectorXd t(p);
    VectorXd res(ncomp);
    size_t window_size = std::max(2_z,ncomp/10_z);

    std::list<unsigned char> stopping_criterium(window_size,0);
    // Y.col(0) = y.rowwise() - y.colwise().mean();
    double ymean = y.mean();
    Y.col(0).array() = ymean;
    double SSTO = (y.array() - ymean).array().square().sum();
    int m = 0;
    // for (auto m = 0; m < ncomp; m++)
    while (m < ncomp)
    {
        XY = X.transpose() * y;
        SelfAdjointEigenSolver<MatrixXd> es( XY.transpose() * XY );
        auto abs_eigenvalues = es.eigenvalues().array().abs();
        size_t max_eigenvalue_indice = ranges::distance(abs_eigenvalues.begin(),ranges::max_element(abs_eigenvalues));
        auto q = es.eigenvectors().col(max_eigenvalue_indice);
        w = XY * q;
        w /= sqrt((w.transpose()*w)(0,0));
        r=w;
        for (auto j=0; j<=m-1;j++)
        {
            r -= (P.col(j).transpose()*w)(0,0)*R.col(j);
        }
        R.col(m) = r;
        t = Xo * r;
        P.col(m) = XX *r/(t.transpose()*t)(0,0);
        VectorXd Zm = X * XY;
        double Znorm = Zm.array().square().sum();
        double Thetam = Zm.dot(y) / Znorm;
        Y.col(m + 1) = Y.col(m) + Thetam * Zm;
        X -= Zm.rowwise().replicate(p) * ((Zm/Znorm).transpose() * X).asDiagonal();
        res(m) = (Y.col(m + 1).array() - ymean).array().square().sum() / SSTO;
        if ((m >= 2) && stopping) {
            auto lastdiff = res(m) - res(m-1);
            auto lastmean = (res(m) + res(m-1))/2.0;
            size_t remains = ncomp - m;
            stopping_criterium.pop_front();
            stopping_criterium.push_back(lastmean >= 0.99 * remains * lastdiff);
            auto wcrit = ranges::accumulate(stopping_criterium,0);
            if (wcrit == window_size) break;
        }
        m++;
    }
    if (m < ncomp) {
        m--;
        res = res(seq(0,m)).eval();
    }
    Projection = R;
//    VectorXd res = (Y.block(0,1,n,ncomp).array() - ymean).array().square().colwise().sum() / SSTO;
    return res;
}