#pragma once
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

struct Reftable {
    size_t nrec;
    std::vector<size_t> nrecscen;
    std::vector<size_t> nparam;
    std::vector<string> params_names;
    std::vector<string> stats_names;
    MatrixXd stats;
    MatrixXd params;    
    std::vector<double> scenarios;
};
