#pragma once
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

template<class MatrixType>
struct Reftable {
    size_t nrec;
    std::vector<size_t> nrecscen;
    std::vector<size_t> nparam;
    std::vector<std::string> params_names;
    std::vector<std::string> stats_names;
    MatrixType stats;
    MatrixType params;    
    std::vector<double> scenarios;
//    Reftable(MatrixType _stats, MatrixType _params) : stats(_stats), params(_params) {}
    Reftable(size_t _nrec,
    std::vector<size_t> _nrecscen,
    std::vector<size_t> _nparam,
    std::vector<std::string> _params_names,
    std::vector<std::string> _stats_names,
    MatrixType _stats,
    MatrixType _params,
    std::vector<double> _scenarios) : nrec(_nrec), 
                nrecscen(_nrecscen),
                nparam(_nparam),
                params_names(_params_names),
                stats_names(_stats_names),
                stats(_stats),
                params(_params),
                scenarios(_scenarios) {}

};
