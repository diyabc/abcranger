#pragma once
#include <vector>

using namespace std;

struct Reftable {
    size_t nrec;
    std::vector<size_t> nrecscen;
    std::vector<size_t> nparam;
    std::vector<string> params_names;
    std::vector<string> stats_names;
    std::vector<double> stats;
    std::vector<double> params;
    std::vector<double> scenarios;
};

Reftable readreftable(string headerpath, string reftablepath, size_t N = 0);