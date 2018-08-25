#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>

#include <stdio.h>
#include <stdlib.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>
// inverse of matrix
#include <boost/numeric/ublas/matrix_expression.hpp>

#include <boost/numeric/ublas/lu.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>

using namespace std;
using namespace boost::numeric::ublas;

template <typename C>
using uvector = boost::numeric::ublas::vector<C>;

double norm(const boost::numeric::ublas::vector<double>& v1, const boost::numeric::ublas::vector<double>& v2) {
    assert (v1.size() == v2.size());
    double sum = 0;
    for (size_t i=0; i<v1.size(); ++i)
    {
        double minus = v1(i) - v2(i);
        double r = minus * minus;
        sum += r;
    }

    return sqrt(sum);
}

namespace lda {
    void generate_k_random_integers(set<size_t>& s, int upper_bound, int k) {
        boost::mt19937 rng;   
        boost::uniform_int<> g(1, upper_bound);
        for (size_t i=0; i<(size_t)k; ++i) {
            int x = g(rng) - 1;
            s.insert(x);
        }
    }

    void compute_centroids(const matrix<double>& x,
                           const std::vector<set<size_t>>& mapping,
                           matrix<double>& W) {
        auto p = x.size2();
        auto K = mapping.size();
        W = zero_matrix(K,p);
        uvector<double> zero_vec = zero_vector<double>(p);
        for(auto i = 0; i < mapping.size(); i++) {
            auto& s = mapping[i];
            matrix_row<matrix<double>> c(W,i);
            c = transform_reduce(execution::par,
                                 begin(s),end(s),
                                 zero_vec,
                                 [] (auto& a, 
                                     auto& b) { return a + b; },
                                 [&x] (size_t j) { return row(x,j); });
            c /= s.size();
        }
    }

    void compute_means(const matrix<double>& x,
                       uvector<double>& m) {
        auto p = x.size2();
        m.resize(p);
        std::vector<size_t> l(p);
        std::iota(begin(l),end(l),0);
        transform(execution::par,
                  begin(l),
                  end(l),
                  m.begin(),
                  [&x] (auto j) { auto col = column(x,j);
                                  return }
                  )
        foreach(execution::par,begin(m),end(m), [&x] (auto& c) {
            c = 
        }) {

        }
        for(auto& i = 0; i < mapping) {
            m[i] = 
        }
    }

    // adjust data according to the mean
    void adjust_data(boost::numeric::ublas::matrix<double>& x, 
            const map<int, set<size_t> >& mapping,
            map<int, boost::numeric::ublas::vector<double> >& cat_mean_map) {
        map<int, set<size_t> >::const_iterator it = mapping.begin();

        size_t dim = x.size2();
        for(; it != mapping.end(); ++it) { // for each category
            const set<size_t>& rows = it->second;
            boost::numeric::ublas::vector<double>& mean = cat_mean_map[it->first];
            set<size_t>::const_iterator r_it(rows.begin());
            for (; r_it != rows.end(); ++r_it) { // for each training instance in this category
                for (int d=0; d<dim; ++d) { // for each dim
                    x(*r_it,d) -= mean(d);
                }
            }
        }
    }

    // compute the between class scatter, here we only consider the two classes case.
    void compute_matrix_b(const boost::numeric::ublas::vector<double>& mean1, 
            const boost::numeric::ublas::vector<double>& mean2,
            boost::numeric::ublas::matrix<double>& sb) {
        sb = outer_prod(mean1 - mean2, mean1 - mean2);
    }

    void fill_matrix(boost::numeric::ublas::matrix<double>& x) {
        for(size_t i=1; i<x.size1(); ++i) {
            for(size_t j=0; j<i; ++j)  {
                x(i,j) = x(j,i);
            }
        }
    }

    void compute_covariance(matrix<double>& x,
                            const std::vector<set<size_t>>& mapping,
                            matrix<double>& covar_matrix) {
        auto p = x.size2();
        covar_matrix = zero_matrix(p,p);
        
    }

    void compute_covariance(boost::numeric::ublas::matrix<double>& x, 
            const map<int, set<size_t> >& mapping,
            boost::numeric::ublas::matrix<double>& covar_matrix
            ) {
        size_t dim = x.size2();
        covar_matrix.resize(dim, dim);
        for(int i=0; i<dim; ++i) {
            for(int j=0; j<dim; ++j) {
                covar_matrix(i,j) = 0;
            }
        }
        map<int, set<size_t> >::const_iterator it = mapping.begin();
        // compute the s(w) = sum(s(i)) for each category
        for(; it != mapping.end(); ++it) { // for each category
            const set<size_t>& rows = it->second;
            for (int d=0; d<dim; ++d) { // for each dim
                for (int d2=d; d2<dim; ++d2) {
                    double covar = 0;
                    set<size_t>::const_iterator r_it(rows.begin());
                    for (; r_it != rows.end(); ++r_it) { // for each training instance in this category
                        covar += x(*r_it,d) * x(*r_it,d2);
                    }
                    covar /= rows.size();
                    covar_matrix(d,d2) += covar;
                }
            }

            fill_matrix(covar_matrix);
        }

    }
    
    template<class T>
        bool inverse(const boost::numeric::ublas::matrix<T> &m,
                boost::numeric::ublas::matrix<T> &inv) {
            using namespace boost::numeric::ublas;
            typedef permutation_matrix<size_t> pmatrix;
            matrix<T> A(m); // copy
            pmatrix pm(A.size1());
            int res = lu_factorize(A, pm);
            if (res != 0) return false;
            inv.assign(identity_matrix<T>(A.size1()));
            lu_substitute(A, pm, inv);
            return true;
        }

    template<class T>
        boost::numeric::ublas::matrix<T> gjinverse(const boost::numeric::ublas::matrix<T> &m,
                bool &singular)
        {
            using namespace boost::numeric::ublas;

            const int size = m.size1();

            // Cannot invert if non-square matrix or 0x0 matrix.
            // Report it as singular in these cases, and return
            // a 0x0 matrix.
            if (size != (int)(m.size2()) || size == (int)0) {
                cout << "Cannot invert if non-square matrix or 0x0 matrix " << endl;
                singular = true;
                boost::numeric::ublas::matrix<T> A(0, 0);
                return A;
            }

            // Handle 1x1 matrix edge case as general purpose
            // inverter below requires 2x2 to function properly.
            if (size == 1) {
                boost::numeric::ublas::matrix<T> A(1, 1);
                if (m(0, 0) == 0.0) {
                    cout << "size=1" << endl;
                    singular = true;
                    return A;
                }

                singular = false;
                A(0, 0) = 1/m(0, 0);
                return A;
            }

            // Create an augmented matrix A to invert. Assign the
            // matrix to be inverted to the left hand side and an
            // identity matrix to the right hand side.
            boost::numeric::ublas::matrix<T> A(size, 2*size);
            matrix_range<boost::numeric::ublas::matrix<T> > Aleft(A, range(0, size), range(0, size));
            Aleft = m;
            matrix_range<boost::numeric::ublas::matrix<T> > Aright(A, range(0, size), range(size, 2*size));
            Aright = boost::numeric::ublas::identity_matrix<T>(size);

            // Doing partial pivot
            for (int kk = 0; kk < size; kk++) {
                // Swap rows to eliminate zero diagonal elements.
                for (int k = 0; k < size; k++) {
                    if (A(k, k) == 0) // XXX: test for "small" instead
                    {
                        // Find a row(l) to swap with row(k)
                        int l = -1;
                        for (int i = k+1; i < size; i++) {
                            if (A(i, k) != 0) {
                                l = i;
                                break;
                            }
                        }

                        // Swap the rows if found
                        if (l < 0) {
                            std::cout<< "Error:" << __FUNCTION__ << ":"
                                << "Input matrix is singular, because cannot find"
                                << " a row to swap while eliminating zero-diagonal.";
                            singular = true;
                            return Aleft;
                        } else {
                            matrix_row<boost::numeric::ublas::matrix<T> > rowk(A, k);
                            matrix_row<boost::numeric::ublas::matrix<T> > rowl(A, l);
                            rowk.swap(rowl);
                        }
                    }
                }

                // normalize the current row
                for (int j = kk+1; j < 2*size; j++)
                    A(kk, j) /= A(kk, kk);
                A(kk, kk) = 1;

                // normalize other rows
                for (int i = 0; i < size; i++) {
                    if (i != kk) // other rows  // FIX: PROBLEM HERE
                    {
                        if (A(i, kk) != 0) {
                            for (int j = kk+1; j < 2*size; j++)
                                A(i, j) -= A(kk, j) * A(i, kk);
                            A(i, kk) = 0;
                        }
                    }
                }
            }

            singular = false;
            return Aright;
        }
}

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         cout << "Usage: " << argv[0] << " data_file" << endl;
//         return -1;
//     }

//     const int record_num = 10;
//     const int dim_num = 2;
//     using namespace lda;

//     boost::numeric::ublas::vector<double> y(record_num);
//     boost::numeric::ublas::matrix<double> x(record_num, dim_num);
//     SimpleDataLoader loader(record_num, dim_num);
//     loader.load_file(argv[1], y, x);

//     cout << x << endl;
//     cout << y << endl;

//     loader.dump_map();

//     map<int, set<size_t> >& m = loader.get_cat_row_map();
//     map<int, boost::numeric::ublas::vector<double> > cat_mean_map;
//     compute_means(x, m, cat_mean_map);

//     cout << cat_mean_map[0] << endl;
//     cout << cat_mean_map[1] << endl;
//     adjust_data(x, m, cat_mean_map);

//     cout << "after adjust data:" << endl;
//     cout << x << endl;

//     // compute the covariance matrix for each category
//     boost::numeric::ublas::matrix<double> covar_matrix;
//     compute_covariance(x, m, covar_matrix);
//     cout << "covariance matrix: s(w):"  << endl;
//     cout << covar_matrix << endl;


//     boost::numeric::ublas::matrix<double> covar_b;
//     compute_matrix_b(cat_mean_map[0], cat_mean_map[1], covar_b);
//     cout << "s(b):" << endl;
//     cout << covar_b << endl;
    
//     // finally, we get the weight using w = inverse(s(w) * (u1 - u2)
//     bool singular = false;
//     boost::numeric::ublas::matrix<double> inv =  gjinverse(covar_matrix, singular);
//     if (singular) {
//         cout << "The matrix is singluar, can not get the inverse matrix..." << endl;
//         return -1;
//     }
    
//     cout << "inverse of matrix:" << endl;
//     cout << inv << endl;

//     inverse(covar_matrix, inv);
//     cout << "inverse of matrix:" << endl;
//     cout << inv << endl;

//     boost::numeric::ublas::vector<double> weight(prod(inv, cat_mean_map[0] - cat_mean_map[1]));
//     cout << "the weight:" << endl;
//     cout << weight << endl;

//     return 0;
// }