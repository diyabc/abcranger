#include <vector>
#include <algorithm>
#include <random>
#include "readreftable.hpp"
#include "lda-eigen.hpp"

static std::mt19937 rng;
static std::uniform_real_distribution<> dis(0.0,1.0);

template<class Derived>
MatrixBase<Derived>& constCastAddColsMatrix(MatrixBase<Derived> const & M_, size_t n)
{
    auto ncols = M_.cols();
    MatrixBase<Derived>& M = const_cast<MatrixBase<Derived>&>(M_);
    M.derived().conservativeResize(NoChange,ncols + n);
    return M;
}

template<class Derived, class OtherDerived>
void addCols(MatrixBase<Derived> const &X_, const MatrixBase<OtherDerived>& M) {
    auto ncols = X_.cols();
    auto& X = constCastAddColsMatrix(X_,M.cols());
    X.block(0,ncols,X.rows(),M.cols()) = M;
}


template<class Derived>
void addNoiseCols(MatrixBase<Derived> const &X_, size_t n)
{
    auto ncols = X_.cols();
    auto& X = constCastAddColsMatrix(X_,n);
    X.block(0,ncols,X_.rows(),n) = X.block(0,ncols,X_.rows(),n).unaryExpr([](double x){ return dis(rng);});
}

template<class Derived>
void addNoise(Reftable& rf, MatrixBase<Derived> const &statobs, size_t n) {
    addNoiseCols(rf.stats,n);
    addNoiseCols(statobs,n);
    for(auto i = 0; i < n; i++) rf.stats_names.push_back("NOISE" + std::to_string(i+1));
}

template<class Derived, class OtherDerived>
void addLinearComb(MatrixBase<Derived> const & X_, const MatrixBase<OtherDerived>& M) {
    auto ncols = X_.cols();
    auto& X = constCastAddColsMatrix(X_,M.cols());
    X.block(0,ncols,X.rows(),M.cols()) = X.block(0,0,X.rows(),ncols) * M;    
}

template<class Derived>
void addLda(Reftable& rf, MatrixBase<Derived> const &statobs) {
    Matrix<size_t,-1,1> scen(rf.nrec);
    for(auto i = 0; i < rf.nrec; i++) scen(i) = static_cast<size_t>(rf.scenarios[i]) - 1;
    MatrixXd Ld;
    lda(rf.stats, scen, Ld);
    addLinearComb(rf.stats,Ld);
    addLinearComb(statobs,Ld);
    for(auto i = 0; i < rf.nrecscen.size() - 1; i++) {
        rf.stats_names.push_back("LDA" + std::to_string(i+1));
    }
}

void addScen(Reftable& rf) {
    addCols(rf.stats, Map<VectorXd>(rf.scenarios.data(),rf.nrec));
    rf.stats_names.push_back("Y");
}

