/**
 * @brief Linear Discriminant Analysis
 * 
 * Dead simple, blazing fast LDA, \cite friedman2001elements
 * 
 * @file lda.hpp
 * @author François-David Collin <Francois-David.Collin@umontpellier.fr>
 * @date 2018-08-31
 */
#pragma once

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>


#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>


using namespace boost::numeric::ublas;
using namespace boost::numeric::bindings;

/**
 * @brief Computes lda with Trevor/hastie algorithm
 * 
 * @param x data
 * @param y labels from 0 to K 
 * @param Ld computes matrix whoses columns are Ld vectors
 */
void lda(const matrix<double> &x,
        const vector<size_t> &y,
        matrix<double> &Ld)
{
    auto n = x.size1();
    auto p = x.size2();
    auto K = *std::max_element(y.begin(), y.end()) + 1;

    // M = Centroids 
    matrix<double> M = zero_matrix(K,p);
    vector<double> m = zero_vector(p);
    vector<size_t> d = scalar_vector<size_t>(p,0);

    for(auto i = 0; i < n; i++) {
        auto c = y[i];
        auto r = row(x,i);
        d[c]++;
        row(M,c) += r;
        m += r;
    }
    for(auto c = 0; c < K; c++) {
        row(M,c) /= static_cast<double>(d[c]);
    }

    // m = Global mean
    m /= static_cast<double>(n);

    // D is x centered data and sorted by classes
    matrix<double> D(n,p);
    vector<size_t> countrows = scalar_vector<size_t>(K,0);

    // W = Within-class covariance matrix
    size_t slicepos = 0;
    matrix<double> W = zero_matrix(p,p);
    for(auto c = 0; c < K; c++) {
        // Dc is a single class subrange of x
        matrix<double> Dc = project(x,range(slicepos,slicepos+d[c]),range(0,p));
        // Centering
        noalias(Dc) -= outer_prod(scalar_vector(d[c],1.0),row(M,c));
        W += prod(trans(Dc),Dc);
        slicepos += d[c];
    }
    W /= (n - K);
    // Eigen decomposition of W, Wv vectors, Wb values
    matrix<double, column_major> Wv(p,p);
    Wv = W;
    vector<double> Wb  = zero_vector(p);
    lapack::syev('V','U',Wv,Wb);

    // W12 = inverse square root of W
    transform(begin(Wb),end(Wb),begin(Wb),
        [](auto& a) { return 1.0/sqrt(a); } );
    matrix<double> W12 = prod<matrix<double>>(Wv,prod<matrix<double>>(diagonal_matrix<double>(p,Wb.data()),trans(Wv)));

    // Mstar = spherified M
    matrix<double> Mstar = prod(M,W12);
    // mstar = Mean of Mstar
    vector<double> mstar = zero_vector(p);
    for(auto c = 0; c < K; c++) mstar += row(Mstar,c);
    mstar /= K;
    // Bstar = covariance matrix of Mstar
    noalias(Mstar) -= outer_prod(scalar_vector(K,1.0),mstar);
    matrix<double> Bstar = prod(trans(Mstar),Mstar)/(K-1);

    // Eigen decomposition of Bstar, Vstar = vectors, Db = values
    matrix<double, column_major> Vstar(p,p);
    Vstar = Bstar;
    vector<double> Db(p);
    lapack::syev('V','U',Vstar,Db);
    // Finally we "unsphere" the eigenvectors of Db
    matrix<double> Vl = prod(W12,Vstar);
    Ld.resize(p,K-1);
    for(auto c = 0; c < K - 1; c++) {
        column(Ld,c) = column(Vl,p-1-c);
    }
}
