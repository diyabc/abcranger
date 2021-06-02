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
#include "tqdm.hpp"

using namespace Eigen;
using namespace std;
using namespace ranges;

template<class Derived>
std::vector<size_t> filterConstantVars(const MatrixBase<Derived>& xr) {
    RowVectorXd meanr = xr.colwise().mean();
    VectorXd stdr = ((xr.rowwise() - meanr).array().square().colwise().sum() / (xr.rows() - 1)).sqrt();;
    std::vector<size_t> validvars(xr.cols());
    size_t m = 0;
    for(size_t i = 0; i< xr.cols(); i++) {
        if (stdr(i) >= 1.0e-8) validvars[m++] = i;
    }
    validvars.resize(m-1);
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
    size_t n = x.rows();
    size_t p = x.cols();
    mean = x.colwise().mean();
    ncomp = std::min(std::min(n,p),ncomp);
    std = ((x.rowwise() - mean).array().square().colwise().sum() / (x.rows() - 1)).sqrt();;
    MatrixXd X = (x.rowwise() - mean).array().rowwise() / std.array();
    MatrixXd X0 = X;
    MatrixXd Ptilde(ncomp,p);
    MatrixXd Wstar(p,ncomp);
    VectorXd res(ncomp);
    size_t window_size = std::max(2_z,ncomp/10_z);

    std::list<unsigned char> stopping_criterium(window_size,0);
    double ymean = y.mean();
    MatrixXd w_k, t_k, p_k, y_k;
    y_k = y;
    double SSTO = (y.array() - ymean).array().square().sum();
    int m = 0;
    tqdm bar;
    
    while (m < ncomp)
    {
        bar.progress(m,ncomp);
        // $w_{k}=\frac{X_{k-1}^{T} y_{k-1}}{\left\|X_{k-1}^{T} y_{k-1}\right\|}$ 
        w_k = X.transpose() * y;   //  (p)   
        w_k /= sqrt((w_k.transpose()*w_k)(0,0));
        Wstar.col(m) = w_k;
        // $t_{k}=X_{k-1}w_{k}$
        t_k = X * w_k; // (n)
        double t_k_s = (t_k.transpose() * t_k)(0,0);
        // $p_{k}=\frac{X_{k-1}^{T} t_{k}}{t_{k}^{T} t_{k}}$
        p_k = (X.transpose() * t_k) / t_k_s; // (p)
        // $\widetilde{\mathbf{P}}_{K \times p}=\mathbf{t}\left[\widetilde{p}_{1}, \ldots, \widetilde{p}_{K}\right]$
        Ptilde.row(m) = p_k.transpose();
        // $q_{k}=\frac{y_{k-1}^{T} t_{k}}{t_{k}^{T} t_{k}}$
        double q_k = (y_k.transpose() * t_k)(0,0) / t_k_s; //(n,n)
        // $y_{k}=y_{k-1}-q_{k} t_{k}$
        y_k -= q_k * t_k;
        // $X_{k}=X_{k-1}-t_{k} p_{k}^{T}$
        X -= (t_k * p_k.transpose());
        res(m) = 1 - (y_k.array() - ymean).array().square().sum() / SSTO;
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

    // $\mathbf{W}=\mathbf{W}^{*}\left(\widetilde{\mathbf{P}} \mathbf{W}^{*}\right)^{-1}$
    Projection = Wstar*(Ptilde*Wstar).inverse();
    return res;
}

// Tenenhaus, M. L’approche PLS. Revue de statistique appliquée 47, 5–40 (1999).
// Vancolen, S. La Régression PLS. Mémoire Postgrade en Statistiques, University of Neuchâtel (Switzerland) 1--28 (2004).
// Wold, S., Sjöström, M. & Eriksson, L. PLS-regression: a basic tool of chemometrics. Chemometrics and intelligent laboratory systems 58, 109–130 (2001).
// Mémoire m2 de ghislain : https://plmbox.math.cnrs.fr/f/1192b14f90ea44a1b26a/
// Krämer, N. An overview on the shrinkage properties of partial least squares regression. Computational Statistics 22, 249–273 (2007).