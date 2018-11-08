#include <vector>
#include <algorithm>
#include <random>
#include "readreftable.hpp"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0,1.0);

vector<double> getNoise(size_t n) {
    vector<double> v(n);
    std::generate(std::begin(v),std::end(v),[] () { return dis(gen); });
    return v;
}

void addCol(MatrixXd& m, vector<double> c) {
    m.conservativeResize(m.rows(),m.cols()+1);
    m.col(m.cols()-1) = Map<VectorXd>(c.data(),c.size());
}

void addNoiseCols(Reftable& rf, size_t ncols) {
    for(auto i = 0; i < ncols; i++) {
        auto noise = getNoise(rf.nrec);
        addCol(rf.stats,noise);
        rf.stats_names.push_back("NOISE" + std::to_string(i));
    }
}